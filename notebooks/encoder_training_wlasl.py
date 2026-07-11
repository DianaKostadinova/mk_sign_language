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

# TODO: point these at your uploaded files
WLASL_JSON      = "/content/WLASL_v0.3.json"
LANDMARKS_NPZ   = "/content/wlasl/landmarks_V3.npz"
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
class WLASLClips(Dataset):
    """Cross-references official WLASL_v0.3.json (signer_id, split) against
    the Kaggle landmarks_V3.npz (keypoints, keyed by video_id string)."""
    def __init__(self, split="train"):
        with open(WLASL_JSON, encoding="utf-8") as f:
            wlasl = json.load(f)
        self.npz = np.load(LANDMARKS_NPZ, allow_pickle=True)
        available = set(self.npz.keys())

        self.items = []   # (video_id, gloss, signer_id)
        for entry in wlasl:
            gloss = entry["gloss"]
            for inst in entry["instances"]:
                if inst["split"] != split:
                    continue
                vid = inst["video_id"]
                if vid not in available:          # not every official video has landmarks here
                    continue
                self.items.append((vid, gloss, inst["signer_id"]))

        self.by_sign = {}
        for i, (_, gloss, _) in enumerate(self.items):
            self.by_sign.setdefault(gloss, []).append(i)
        self.signs = list(self.by_sign)
        self._cache: dict[int, np.ndarray] = {}
        print(f"WLASLClips[{split}]: {len(self.items)} clips, {len(self.signs)} glosses")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        cached = self._cache.get(i)
        if cached is not None:
            vid, gloss, signer = self.items[i]
            return cached, gloss, signer
        vid, gloss, signer = self.items[i]
        seq = np.asarray(self.npz[vid])          # (T, 553, 3)
        tmpl = clip_to_template(seq)
        self._cache[i] = tmpl
        return tmpl, gloss, signer

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

CKPT_PATH        = "/content/drive/MyDrive/wlasl/sign_encoder_ckpt.pt"   # <-- EDIT
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
