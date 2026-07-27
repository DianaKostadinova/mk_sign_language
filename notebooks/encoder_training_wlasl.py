"""
Signer-independent sign encoder — WLASL training scaffold (Colab / Kaggle).

Pivot from the AUTSL version (encoder_training.py): AUTSL's only openly
available skeleton release was reduced to 10 points/hand, forcing a coarse
descriptor. WLASL instead has a Kaggle-hosted release
(abd0kamel/mutemotion-output, landmarks_V3.npz) at (T, 553, 3) per video —
consistent with full MediaPipe Holistic (33 pose + 468 face + 21+21 hands),
so we get FULL 21-point hands. That means we reuse the exact same
`_hand_shape`/`_arm_feat`/`build_frame`-equivalent descriptor the Macedonian
DTW pipeline already uses — no coarse compromise, and no need to re-extract
`data/landmarks/word_templates.npz` (the original full-feature gallery
already works as-is with this encoder).

Also: WLASL's vocabulary (2000 signs) is ~9x AUTSL's (226) — per
"Representing Signs as Signs" (arxiv 2502.20171), pretraining vocabulary
diversity matters more than language match for one-shot transfer.

Signer IDs and per-video metadata come from the OFFICIAL WLASL_v0.3.json
(has signer_id, unlike the Kaggle "WLASL_parsed_data.json"). Cross-reference
by video_id (matches landmarks_V3.npz's keys).

Pipeline:
1. Load WLASL_v0.3.json (official metadata: gloss, video_id, signer_id, split).
2. Load landmarks_V3.npz (keypoints, keyed by video_id).
3. Full 21pt-hand + arm features (dtw_common._hand_shape / _arm_feat).
4. Temporal encoder (BiGRU) -> L2-normalised embedding.
5. Supervised-contrastive loss with signer-disjoint batches (official split).
6. Validate on held-out (official test-split) signers.
7. Export encoder; embed the EXISTING Macedonian word_templates.npz -> gallery.
"""

# %%
import os, math, random, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("device:", DEVICE)

# Paths come from the environment so the Colab notebook sets them once and no
# one hand-edits this file per session (hand-editing after every disconnect was
# a reliable way to lose a run to a typo).
WLASL_JSON    = os.environ.get("WLASL_JSON",  "/content/WLASL_v0.3.json")       # official metadata: has signer_id
PARSED_JSON   = os.environ.get("PARSED_JSON", "/content/wlasl/WLASL_parsed_data.json")  # Kaggle metadata: list index == landmarks_V3.npz key
LANDMARKS_NPZ = os.environ.get("LANDMARKS_NPZ", "/content/wlasl/landmarks_V3.npz")

# Everything below lives on Drive, i.e. survives a Colab reset.
CACHE_DIR = os.environ.get("WLASL_CACHE_DIR", "/content/drive/MyDrive/wlasl/cache")
CKPT_PATH = os.environ.get("WLASL_CKPT",      "/content/drive/MyDrive/wlasl/sign_encoder_ckpt.pt")

# 0 = full 2000-gloss vocabulary. Set e.g. 300 for a fast end-to-end smoke run
# before committing to the long one — the N most frequent glosses in train.
MAX_GLOSSES = int(os.environ.get("WLASL_MAX_GLOSSES", "0"))

EMBED_DIM   = 256
SEQ_LEN     = 32            # matches alphabet TEMPLATE_LEN
BATCH_SIGNS = 32            # P signs per batch
BATCH_VIEWS = 4             # K clips per sign per batch (P*K samples)
EPOCHS      = 40

# %%
# Reuse the real feature functions so WLASL and Macedonian are byte-compatible
# (this is the whole point of the pivot: no coarse descriptor needed here).
import sys; sys.path.append(".")
try:
    from dtw_common import _hand_shape, _arm_feat, HAND_SHAPE_DIM, ARM_WEIGHT, make_template, POS_DIM
except ImportError:
    raise SystemExit("Upload alphabet/dtw_common.py next to this notebook first.")

# landmarks_V3.npz layout: (T, 553, 3) = 42 (hands) + 33 (pose) + 478 (face).
# Confirmed from the dataset author's own extraction code (Kaggle notebook):
# block order is [left_hand(21), right_hand(21), pose(33), face(478)] —
# NOT the naive pose-face-hands guess, which failed the wrist-alignment
# sanity check earlier. VERIFY again below before trusting this fully.
LHAND_SLICE = slice(0, 21)
RHAND_SLICE = slice(21, 42)
POSE_SLICE  = slice(42, 75)     # 33 points, standard order -> local idx 15/16 = wrists

def frame_feature(row: np.ndarray) -> np.ndarray:
    """One (553,3) holistic frame -> (POS_DIM,) feature, matching dtw_common exactly."""
    pose = row[POSE_SLICE]                      # (33,3) — same layout _arm_feat expects
    lh   = row[LHAND_SLICE][:, :2]               # (21,2)
    rh   = row[RHAND_SLICE][:, :2]

    def hs(h):
        return _hand_shape(h) if np.any(h) else np.zeros(HAND_SHAPE_DIM, np.float32)

    return np.concatenate([hs(lh), hs(rh), _arm_feat(pose) * ARM_WEIGHT]).astype(np.float32)

def clip_to_template(seq: np.ndarray) -> np.ndarray:
    """(T,553,3) keypoint clip -> (SEQ_LEN, POS_DIM*2) template — identical
    representation to the Macedonian side (same dtw_common.make_template)."""
    frames = np.array([frame_feature(f) for f in seq], np.float32)
    return make_template(frames)

# %%
def _load_meta():
    """landmarks_V3.npz is keyed by the INDEX into WLASL_parsed_data.json
    (verified: all 21083 npz keys are valid indices into that list) — NOT
    the official video_id directly. So: npz key -> parsed[key] gives
    {gloss, video_path, split}; video_id is parsed from video_path's
    filename; that video_id is then looked up in the OFFICIAL
    WLASL_v0.3.json (the only place signer_id lives)."""
    with open(WLASL_JSON, encoding="utf-8") as f:
        wlasl = json.load(f)
    with open(PARSED_JSON, encoding="utf-8") as f:
        parsed = json.load(f)
    signer_of = {}   # video_id -> signer_id, from the official metadata
    for entry in wlasl:
        for inst in entry["instances"]:
            signer_of[str(inst["video_id"])] = inst["signer_id"]
    return parsed, signer_of

_ALLOWED_GLOSSES = None    # memo: chosen from the train split, reused for test

def _build_items(split, npz_keys):
    """-> [(npz_key, gloss, signer_id)] for one split, honouring MAX_GLOSSES."""
    global _ALLOWED_GLOSSES
    parsed, signer_of = _load_meta()
    rows = []
    for k in npz_keys:
        p = parsed[int(k)]
        vid = Path(p["video_path"]).stem         # ".../12327.mp4" -> "12327"
        signer = signer_of.get(vid)
        if signer is None:                       # both files out of sync for this clip
            continue
        rows.append((k, p["gloss"], signer, p["split"]))
    if MAX_GLOSSES and _ALLOWED_GLOSSES is None:
        cnt = {}
        for _, g, _, sp in rows:
            if sp == "train":
                cnt[g] = cnt.get(g, 0) + 1
        _ALLOWED_GLOSSES = set(sorted(cnt, key=lambda g: -cnt[g])[:MAX_GLOSSES])
    return [(k, g, s) for k, g, s, sp in rows
            if sp == split and (not MAX_GLOSSES or g in _ALLOWED_GLOSSES)]

def _shard_paths(split):
    d = Path(CACHE_DIR)
    return sorted(d.glob(f"{split}_*.npz")) if d.exists() else []

def precompute_templates(split="train", shard_size=2000):
    """Convert every clip in a split to a template ONCE, writing fixed-size
    shards to Drive as it goes.

    This is the step that used to die with the session: the old lazy in-RAM
    cache meant every reconnect re-did the whole conversion, smeared randomly
    across epoch 0, so a disconnect before the first epoch closed lost all of
    it. Shards are immutable once written, so re-running after a disconnect
    only redoes the partial shard in flight (<= shard_size clips)."""
    npz   = np.load(LANDMARKS_NPZ, allow_pickle=True)
    items = _build_items(split, list(npz.keys()))
    done  = set()
    for sp in _shard_paths(split):
        done |= set(np.load(sp, allow_pickle=True)["keys"].tolist())
    todo = [it for it in items if it[0] not in done]
    print(f"[{split}] {len(items)} clips total, {len(done)} already cached, "
          f"{len(todo)} to go", flush=True)
    if not todo:
        return
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    n  = len(_shard_paths(split))
    t0 = time.time()
    bk, bt, bg, bs = [], [], [], []
    for i, (k, g, s) in enumerate(todo, 1):
        bk.append(k); bt.append(clip_to_template(np.asarray(npz[k])))
        bg.append(g); bs.append(s)
        if len(bk) >= shard_size or i == len(todo):
            out = Path(CACHE_DIR) / f"{split}_{n:04d}.npz"
            np.savez_compressed(out, keys=np.array(bk),
                                templates=np.stack(bt).astype(np.float32),
                                glosses=np.array(bg), signers=np.array(bs))
            rate = i / max(time.time() - t0, 1e-6)
            print(f"  saved {out.name}  {i}/{len(todo)}  "
                  f"({rate:.1f} clips/s, ~{(len(todo)-i)/max(rate,1e-6)/60:.0f} min left)",
                  flush=True)
            n += 1; bk, bt, bg, bs = [], [], [], []
    print(f"[{split}] complete in {(time.time()-t0)/60:.1f} min", flush=True)

class WLASLClips(Dataset):
    """Reads only the precomputed Drive shards — no landmark decoding, no
    feature extraction. Startup after a reconnect is seconds, not an hour."""
    def __init__(self, split="train"):
        shards = _shard_paths(split)
        if not shards:
            raise SystemExit(f"No cache for '{split}'. Run precompute_templates('{split}') first.")
        T, G, S = [], [], []
        for sp in shards:
            d = np.load(sp, allow_pickle=True)
            T.append(d["templates"]); G += d["glosses"].tolist(); S += d["signers"].tolist()
        self.templates = np.concatenate(T).astype(np.float32)
        self.glosses, self.signers = G, [int(x) for x in S]

        self.by_sign = {}
        for i, gloss in enumerate(G):
            self.by_sign.setdefault(gloss, []).append(i)
        self.signs = list(self.by_sign)
        print(f"WLASLClips[{split}]: {len(G)} clips, {len(self.signs)} glosses, "
              f"{len(set(self.signers))} signers")

    def __len__(self): return len(self.glosses)

    def __getitem__(self, i):
        return self.templates[i], self.glosses[i], self.signers[i]

class PKSampler(torch.utils.data.Sampler):
    """Yield batches of P signs x K clips (varied signers) for contrastive loss."""
    def __init__(self, ds: WLASLClips, P=BATCH_SIGNS, K=BATCH_VIEWS, batches=400):
        self.ds, self.P, self.K, self.batches = ds, P, K, batches
    def __iter__(self):
        for _ in range(self.batches):
            signs = random.sample(self.ds.signs, min(self.P, len(self.ds.signs)))
            batch = []
            for s in signs:
                pool = self.ds.by_sign[s]
                batch += random.sample(pool, min(self.K, len(pool))) if len(pool) >= self.K \
                         else [random.choice(pool) for _ in range(self.K)]
            yield batch
    def __len__(self): return self.batches

# %%
class SignEncoder(nn.Module):
    def __init__(self, in_dim, hidden=256, embed=EMBED_DIM, layers=2):
        super().__init__()
        self.gru  = nn.GRU(in_dim, hidden, layers, batch_first=True,
                           bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(2 * hidden, embed), nn.ReLU(),
                                  nn.Linear(embed, embed))
    def forward(self, x):
        h, _ = self.gru(x)
        z = self.head(h.mean(dim=1))
        return F.normalize(z, dim=-1)

# %%
def supcon_loss(z, labels, temp=0.07):
    sim = z @ z.t() / temp
    sim.fill_diagonal_(-1e9)
    pos = (labels.view(-1, 1) == labels.view(1, -1)).float()
    pos.fill_diagonal_(0)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    denom = pos.sum(1).clamp(min=1)
    return -((pos * logp).sum(1) / denom).mean()

# %%
def collate(batch):
    xs, glosses, signers = zip(*batch)
    gloss_to_id = {g: i for i, g in enumerate(sorted(set(glosses)))}
    labels = torch.tensor([gloss_to_id[g] for g in glosses])
    return torch.tensor(np.stack(xs)), labels, torch.tensor(signers)

@torch.no_grad()
def evaluate_heldout(model, val_ds, gallery_per_sign=1):
    """kNN among held-out (test-split) signers: one signer's clip = gallery, others = query."""
    model.eval()
    embs, glosses, signers = [], [], []
    for x, g, sg in DataLoader(val_ds, batch_size=64,
                                collate_fn=lambda b: (torch.tensor(np.stack([i[0] for i in b])),
                                                       [i[1] for i in b],
                                                       [i[2] for i in b])):
        embs.append(model(x.to(DEVICE)).cpu())
        glosses += g; signers += sg
    E = torch.cat(embs); glosses = np.array(glosses); signers = np.array(signers)
    if len(set(signers)) < 2:
        return 0.0
    gal = signers == sorted(set(signers))[0]
    qry = ~gal
    if gal.sum() == 0 or qry.sum() == 0:
        return 0.0
    sim = E[qry] @ E[gal].t()
    pred = glosses[gal][sim.argmax(1).numpy()]
    return float((pred == glosses[qry]).mean())

PRINT_INTERVAL_S = 10    # progress line at least this often, regardless of batch speed
CKPT_INTERVAL_S  = 120   # mid-epoch checkpoint this often — first epoch (cache build) can be slow

def _save_ckpt(ckpt_path, model, opt, sched, epoch, complete):
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "epoch": epoch, "complete": complete}, ckpt_path)

def run_training(ckpt_path: str = CKPT_PATH):
    train_ds = WLASLClips("train")
    val_ds   = WLASLClips("test")
    in_dim   = train_ds[0][0].shape[-1]
    train_ld = DataLoader(train_ds, batch_sampler=PKSampler(train_ds), collate_fn=collate)

    model = SignEncoder(in_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    start_epoch = 0
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        # a mid-epoch (incomplete) checkpoint restarts that same epoch from batch 0 —
        # only the model/optimizer state carries over, not batch position.
        start_epoch = ckpt["epoch"] + 1 if ckpt.get("complete", True) else ckpt["epoch"]
        print(f"resumed from checkpoint at epoch {start_epoch} "
              f"({'complete' if ckpt.get('complete', True) else 'mid-epoch'})", flush=True)

    print(f"train samples={len(train_ds)}  val samples={len(val_ds)}  "
          f"batches/epoch={len(train_ld)}", flush=True)
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        last_print, last_ckpt = time.time(), time.time()
        for b, (x, labels, _) in enumerate(train_ld):
            x = x.to(DEVICE)
            loss = supcon_loss(model(x), labels.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()

            now = time.time()
            if now - last_print >= PRINT_INTERVAL_S:
                print(f"  epoch {epoch:02d}  batch {b+1}/{len(train_ld)}  "
                      f"loss {loss.item():.3f}", flush=True)
                last_print = now
            if ckpt_path and now - last_ckpt >= CKPT_INTERVAL_S:
                _save_ckpt(ckpt_path, model, opt, sched, epoch, complete=False)
                print(f"  (mid-epoch checkpoint saved, batch {b+1}/{len(train_ld)})", flush=True)
                last_ckpt = now
        sched.step()
        acc = evaluate_heldout(model, val_ds)
        print(f"epoch {epoch:02d}  loss {loss.item():.3f}  held-out top-1 {acc:.1%}", flush=True)
        if ckpt_path:
            _save_ckpt(ckpt_path, model, opt, sched, epoch, complete=True)
    return model

# model = run_training()
# torch.save(model.state_dict(), "sign_encoder_wlasl.pt")

# %% [markdown]
# ## Use it on Macedonian — SAME gallery as before, no re-extraction needed
#
# Because this encoder trains on full 21pt-hand features (byte-identical to
# `alphabet/dtw_common.build_frame`), the ORIGINAL `data/landmarks/
# word_templates.npz` (full-feature, already extracted) works directly —
# unlike the AUTSL version, no `word_templates_coarse.npz` step is needed.

# %%
def build_macedonian_gallery(encoder_path="sign_encoder_wlasl.pt",
                             templates_npz="data/landmarks/word_templates.npz",
                             out="data/landmarks/word_embeddings_wlasl.npz"):
    d = np.load(templates_npz, allow_pickle=True)
    templ = torch.tensor(d["templates"])
    model = SignEncoder(templ.shape[-1]).to(DEVICE)
    model.load_state_dict(torch.load(encoder_path, map_location=DEVICE, weights_only=True))
    model.eval()
    with torch.no_grad():
        emb = model(templ.to(DEVICE)).cpu().numpy()
    np.savez(out, embeddings=emb, labels=d["labels"])
    print(f"Embedded {len(emb)} word templates -> {out}")

def predict(clip_template: np.ndarray, gallery_npz="data/landmarks/word_embeddings_wlasl.npz",
            encoder_path="sign_encoder_wlasl.pt", top_k=5):
    g = np.load(gallery_npz, allow_pickle=True)
    G, labels = torch.tensor(g["embeddings"]), np.array([str(x) for x in g["labels"]])
    model = SignEncoder(clip_template.shape[-1]).to(DEVICE)
    model.load_state_dict(torch.load(encoder_path, map_location=DEVICE, weights_only=True))
    model.eval()
    with torch.no_grad():
        q = model(torch.tensor(clip_template[None]).to(DEVICE)).cpu()
    sims = (q @ G.t()).numpy()[0]
    top = sims.argsort()[::-1][:top_k]
    return [(labels[i], float(sims[i])) for i in top]

# %% [markdown]
# ## Sanity check BEFORE training: verify the 553-point layout assumption
#
# Run this first. If hand features look degenerate (near-zero variance
# across fingers), the POSE_SLICE/LHAND_SLICE/RHAND_SLICE indices above are
# wrong and need adjusting before trusting anything downstream.

# %%
def sanity_check_layout():
    npz = np.load(LANDMARKS_NPZ, allow_pickle=True)
    key = list(npz.keys())[0]
    seq = np.asarray(npz[key])
    print(f"sample '{key}' shape: {seq.shape}")
    mid = seq[len(seq) // 2]
    lh, rh = mid[LHAND_SLICE], mid[RHAND_SLICE]
    print(f"left hand block std (x,y,z): {lh.std(axis=0)}")
    print(f"right hand block std (x,y,z): {rh.std(axis=0)}")
    print("(expect non-trivial, non-zero std in x/y if this is really a hand)")
