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
    from dtw_common import (_hand_shape, _arm_feat, HAND_SHAPE_DIM, ARM_WEIGHT,
                            make_template, POS_DIM, mirror_sequence,
                            resample_sequence, smooth_sequence, VEL_WEIGHT)
except ImportError:
    raise SystemExit("Upload alphabet/dtw_common.py next to this notebook first.")

# landmarks_V3.npz layout: (T, 553, 3) = 478 (face) + 33 (pose) + 21 + 21 (hands).
#
# MEASURED, not assumed (2026-07-27, 40 random clips, all four plausible block
# orders scored by median ||pose_wrist - hand_root||, which must be ~0 for the
# correct one):
#
#   face,pose,LH,RH  (this)  L=0.041  R=0.030   <- winner, ~7x better
#   pose,LH,RH,face          L=0.245  R=0.191
#   LH,RH,pose,face          L=0.264  R=0.209   <- what this file used to assume
#   pose,face,LH,RH          L=0.302  R=0.309
#
# The earlier [LH,RH,pose,face] guess was wrong; it produced hand features that
# looked superficially reasonable but were misaligned with the body. Re-run
# sanity_check_layout() if the dataset is ever re-released.
FACE_SLICE  = slice(0, 478)
POSE_SLICE  = slice(478, 511)   # 33 points, standard order -> local idx 15/16 = wrists
LHAND_SLICE = slice(511, 532)
RHAND_SLICE = slice(532, 553)

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

_CACHE_ALLOWED = None

def _allowed_from_cache():
    """The MAX_GLOSSES most frequent glosses, counted from the train shards.
    Same set for train and test, so the two splits stay comparable.
    None when MAX_GLOSSES == 0 (use everything)."""
    global _CACHE_ALLOWED
    if not MAX_GLOSSES:
        return None
    if _CACHE_ALLOWED is None:
        cnt = {}
        for sp in _shard_paths("train"):
            for g in np.load(sp, allow_pickle=True)["glosses"]:
                cnt[str(g)] = cnt.get(str(g), 0) + 1
        _CACHE_ALLOWED = set(sorted(cnt, key=lambda g: -cnt[g])[:MAX_GLOSSES])
        print(f"MAX_GLOSSES={MAX_GLOSSES}: restricted to {len(_CACHE_ALLOWED)} glosses")
    return _CACHE_ALLOWED

def augment_template(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random view of a cached (32, POS_DIM*2) template.

    Without this the pipeline is fully deterministic — one clip always yields
    the exact same vector — so with ~7 clips per gloss the encoder just
    memorises them (measured: 71.5% train retrieval, 0.24% test). Every
    transform here is label-preserving: the sign is the same, the performance
    of it differs.

    Velocity is recomputed rather than transformed, so it always matches the
    positions it is paired with."""
    P   = POS_DIM
    H   = HAND_SHAPE_DIM
    pos = t[:, :P].astype(np.float32).copy()
    L   = len(pos)
    # a hand absent from the source clip is an exact zero block, and must stay
    # one — that is how it looks at inference time.
    present = [bool(np.any(pos[:, :H])), bool(np.any(pos[:, H:2 * H]))]

    # different speed / slightly different trim boundaries
    if rng.random() < 0.8:
        span  = int(rng.integers(int(0.7 * L), L + 1))
        start = int(rng.integers(0, L - span + 1))
        pos   = resample_sequence(pos[start:start + span], L)

    # opposite-handed signer (hand-shape descriptors are mirror-invariant)
    if rng.random() < 0.5:
        pos = mirror_sequence(pos)
        present.reverse()

    pos = pos + rng.normal(0, 0.02, pos.shape).astype(np.float32)

    # Smooth BEFORE differencing, exactly as make_template does. Skipping this
    # let the injected jitter through np.diff * VEL_WEIGHT (=8), so the noise
    # landed in the velocity half amplified ~8x — swamping the real motion
    # signal in half of every feature vector.
    pos = smooth_sequence(pos)

    # a hand the tracker lost for this take — only ever drop one of two, never
    # the last remaining hand
    if rng.random() < 0.15 and all(present):
        present[int(rng.random() < 0.5)] = False

    # zeroing happens last so dropped/absent hands are exactly zero, not noise
    for i, p in enumerate(present):
        if not p:
            pos[:, i * H:(i + 1) * H] = 0

    vel = np.diff(pos, axis=0, prepend=pos[:1]) * VEL_WEIGHT
    return np.concatenate([pos, vel], axis=1).astype(np.float32)

class WLASLClips(Dataset):
    """Reads only the precomputed Drive shards — no landmark decoding, no
    feature extraction. Startup after a reconnect is seconds, not an hour."""
    def __init__(self, split="train", augment=False):
        shards = _shard_paths(split)
        if not shards:
            raise SystemExit(f"No cache for '{split}'. Run precompute_templates('{split}') first.")
        T, G, S = [], [], []
        for sp in shards:
            d = np.load(sp, allow_pickle=True)
            T.append(d["templates"]); G += d["glosses"].tolist(); S += d["signers"].tolist()
        templates = np.concatenate(T).astype(np.float32)

        # MAX_GLOSSES has to be applied here too, not just in precompute — the
        # shards on Drive hold the full vocabulary, so filtering only at
        # precompute time would silently do nothing once the cache exists.
        allowed = _allowed_from_cache()
        if allowed is not None:
            keep = [i for i, g in enumerate(G) if g in allowed]
            templates = templates[keep]
            G = [G[i] for i in keep]; S = [S[i] for i in keep]

        self.templates = templates
        self.glosses, self.signers = G, [int(x) for x in S]
        self.augment = augment
        self._rng    = np.random.default_rng(SEED)

        self.by_sign = {}
        for i, gloss in enumerate(G):
            self.by_sign.setdefault(gloss, []).append(i)
        self.signs = list(self.by_sign)
        print(f"WLASLClips[{split}]: {len(G)} clips, {len(self.signs)} glosses, "
              f"{len(set(self.signers))} signers")

    def __len__(self): return len(self.glosses)

    def __getitem__(self, i):
        t = self.templates[i]
        if self.augment:
            t = augment_template(t, self._rng)
        return t, self.glosses[i], self.signers[i]

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
    # dropout stays at 0.1. Raising it to 0.3 together with augmentation and a
    # 100x weight-decay bump collapsed training outright: loss sat at exactly
    # ln(127) = 4.844 for 8 epochs, the closed-form value for "all embeddings
    # identical". Augmentation is the overfitting fix; regularisation should
    # only be added back one step at a time, verifying loss still falls.
    def __init__(self, in_dim, hidden=256, embed=EMBED_DIM, layers=2, dropout=0.1):
        super().__init__()
        self.gru  = nn.GRU(in_dim, hidden, layers, batch_first=True,
                           bidirectional=True, dropout=dropout)
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
def evaluate_heldout(model, val_ds, bs=128, verbose=False):
    """One-shot cross-signer retrieval over the WHOLE held-out vocabulary.

    For each gloss: one clip is enrolled as that gloss's single gallery entry
    (exactly how a Macedonian word gets enrolled from its one video), and every
    other clip of that gloss performed by a DIFFERENT signer becomes a query.
    Chance is 1/n_glosses.

    NB this replaces an earlier version that used a single signer's clips as the
    entire gallery. That was fine for AUTSL, where all 43 signers perform all 226
    signs — but on WLASL each signer contributes only a handful of the 2000
    glosses, so the gallery covered a tiny slice of the vocabulary and every
    query outside it was unanswerable. It would have reported near-zero no
    matter how well the encoder worked."""
    model.eval()
    embs = []
    for i in range(0, len(val_ds), bs):
        xb = val_ds.templates[i:i + bs]
        embs.append(model(torch.tensor(xb).to(DEVICE)).cpu())
    E       = torch.cat(embs)
    glosses = np.array(val_ds.glosses)
    signers = np.array(val_ds.signers)

    gal_idx   = np.array([idxs[0] for idxs in val_ds.by_sign.values()])
    gal_gloss = np.array(list(val_ds.by_sign.keys()))
    gal_signer_of = dict(zip(gal_gloss, signers[gal_idx]))

    qry = np.ones(len(glosses), bool)
    qry[gal_idx] = False
    qry &= np.array([signers[i] != gal_signer_of[glosses[i]] for i in range(len(glosses))])
    if not qry.any():
        return 0.0

    pred = gal_gloss[(E[qry] @ E[gal_idx].t()).argmax(1).numpy()]
    acc  = float((pred == glosses[qry]).mean())
    if verbose:
        print(f"  eval: {len(gal_idx)} glosses enrolled, {int(qry.sum())} "
              f"cross-signer queries, chance {1/len(gal_idx):.2%}")
    return acc

@torch.no_grad()
def embed_spread(model, ds, n=512):
    """Mean pairwise cosine over a sample of embeddings. ~0 = well spread,
    ~1 = collapsed to a single point. Printed every epoch because a collapse is
    otherwise invisible: the loss just parks at ln(P*K-1) and never moves."""
    model.eval()
    idx = np.random.default_rng(0).choice(len(ds.templates),
                                          min(n, len(ds.templates)), replace=False)
    E = model(torch.tensor(ds.templates[idx]).to(DEVICE))
    return float((E @ E.t()).mean())

PRINT_INTERVAL_S = 10    # progress line at least this often, regardless of batch speed
CKPT_INTERVAL_S  = 120   # mid-epoch checkpoint this often — first epoch (cache build) can be slow

def _save_ckpt(ckpt_path, model, opt, sched, epoch, complete):
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "epoch": epoch, "complete": complete}, ckpt_path)

def run_training(ckpt_path: str = CKPT_PATH):
    train_ds = WLASLClips("train", augment=True)
    val_ds   = WLASLClips("test")            # never augmented
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
        acc    = evaluate_heldout(model, val_ds, verbose=(epoch == start_epoch))
        spread = embed_spread(model, val_ds)
        warn   = "   <-- COLLAPSED, stop and fix" if spread > 0.9 else ""
        print(f"epoch {epoch:02d}  loss {loss.item():.3f}  held-out top-1 {acc:.1%}"
              f"  cos {spread:+.3f}{warn}", flush=True)
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
# Run this first, and read the verdict. The npz ships 553 unlabelled points per
# frame, so the block order is inferred, not given — and an earlier inference
# was already wrong once.
#
# The test: if the slices are right, the POSE skeleton's wrist (pose idx 15/16)
# and the HAND block's root landmark (idx 0) describe the same physical joint,
# so they must nearly coincide. A wrong layout measures the gap between two
# unrelated body parts and scores far worse. Eyeballing per-finger variance is
# NOT sufficient — a misaligned hand block still has plausible-looking variance,
# which is exactly how the earlier wrong layout slipped through.

# %%
def sanity_check_layout(n_clips=40, stride=5, thresh=0.10):
    """Score the configured slices by wrist alignment, and cross-check every
    other plausible block order. Returns True if the current layout wins."""
    npz  = np.load(LANDMARKS_NPZ, allow_pickle=True)
    keys = list(npz.keys())
    rng  = random.Random(0)
    sample = rng.sample(keys, min(n_clips, len(keys)))

    cands = {
        "face,pose,LH,RH": (slice(511, 532), slice(532, 553), slice(478, 511)),
        "LH,RH,pose,face": (slice(0, 21),    slice(21, 42),   slice(42, 75)),
        "pose,LH,RH,face": (slice(33, 54),   slice(54, 75),   slice(0, 33)),
        "pose,face,LH,RH": (slice(511, 532), slice(532, 553), slice(0, 33)),
    }
    current = (LHAND_SLICE, RHAND_SLICE, POSE_SLICE)

    print(f"sample shape: {np.asarray(npz[sample[0]]).shape}   "
          f"({len(sample)} clips, every {stride}th frame)\n")
    results = {}
    for name, (ls, rs, ps) in cands.items():
        dl, dr, nl, nr, nf = [], [], 0, 0, 0
        for k in sample:
            for f in np.asarray(npz[k])[::stride]:
                nf += 1
                lh, rh, po = f[ls][:, :2], f[rs][:, :2], f[ps][:, :2]
                if np.any(lh):
                    nl += 1
                    if np.any(po): dl.append(np.linalg.norm(po[15] - lh[0]))
                if np.any(rh):
                    nr += 1
                    if np.any(po): dr.append(np.linalg.norm(po[16] - rh[0]))
        med = lambda a: float(np.median(a)) if a else float("inf")
        results[name] = max(med(dl), med(dr))
        mark = " <- CONFIGURED" if (ls, rs, ps) == current else ""
        print(f"  {name:16s} hands present L={nl/nf:5.1%} R={nr/nf:5.1%}   "
              f"wrist err L={med(dl):.4f} R={med(dr):.4f}{mark}")

    best = min(results, key=results.get)
    cfg  = next((n for n, s in cands.items() if s == current), None)
    print()
    if cfg == best and results[best] < thresh:
        print(f"PASS — '{best}' wins at {results[best]:.4f} and is what's configured.")
        return True
    print(f"FAIL — best layout is '{best}' ({results[best]:.4f}), "
          f"configured is '{cfg}' ({results.get(cfg, float('inf')):.4f}).")
    print("Fix FACE/POSE/LHAND/RHAND_SLICE before running anything else.")
    return False
