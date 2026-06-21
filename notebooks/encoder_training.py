# %% [markdown]
# # Signer-independent sign encoder — training scaffold (Colab / Kaggle)
#
# **Goal.** Train a temporal encoder that maps a sign clip (a sequence of
# pose/hand features) to a fixed-length embedding where *the same sign by
# different signers* lands close together. Then we never train on the 2500
# Macedonian words — we **embed** them once as a gallery and do nearest
# neighbour. This is the FaceNet recipe applied to signs.
#
# **Why train on AUTSL and not Macedonian?** We only have ONE Macedonian
# video per word (one signer), so Macedonian cannot teach "ignore the
# signer". AUTSL has 43 signers × 226 signs — it teaches signer-invariance,
# which is language-agnostic at the motion level. The Turkish *vocabulary*
# is discarded; only the invariance transfers.
#
# **Run this in the cloud** (Kaggle Notebook or Colab) so the dataset never
# touches your laptop. Only the final encoder `.pt` (a few MB) comes home.
#
# Pipeline:
# 1. Load AUTSL **wholebody keypoints** (hands included) — NOT raw video.
# 2. Convert keypoints -> the SAME shape features as `alphabet/dtw_common.py`.
# 3. Temporal encoder (BiGRU) -> L2-normalised embedding.
# 4. Supervised-contrastive loss with **signer-disjoint** batches.
# 5. Validate on held-out signers (the real test of invariance).
# 6. Export encoder; embed Macedonian `word_templates.npz` -> gallery -> kNN.

# %%
# --- Environment -------------------------------------------------------------
# Kaggle: add the AUTSL keypoints dataset to the notebook (right panel ->
#         "Add Data"); it mounts read-only under /kaggle/input/... .
# Colab:  download the preprocessed wholebody keypoints into the VM, e.g. the
#         ChaLearn-2021 release (ustc-slr/ChaLearn-2021-ISLR-Challenge).
#
# We need ONLY keypoints + labels + signer ids. No video.
import os, math, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("device:", DEVICE)

# TODO: point these at the mounted dataset
AUTSL_KEYPOINTS_DIR = "/kaggle/input/autsl-wholebody-keypoints"   # <-- EDIT
EMBED_DIM   = 256
SEQ_LEN     = 32            # match alphabet TEMPLATE_LEN so features line up
BATCH_SIGNS = 32            # P signs per batch
BATCH_VIEWS = 4            # K clips per sign per batch (P*K samples)
EPOCHS      = 40

# %% [markdown]
# ## 1. Keypoints -> shape features (parity with the DTW pipeline)
#
# Your live demo and Macedonian gallery use the features in
# `alphabet/dtw_common.py`: per-hand joint angles + finger spreads +
# normalised fingertip distances, plus arm features. For the embedding space
# to be shared, AUTSL clips must be encoded with the **same** descriptor.
#
# The mmpose wholebody layout has 133 keypoints: body + 21 left-hand + 21
# right-hand. Map those hand keypoints to the 21-landmark order MediaPipe
# uses, then call the exact same feature code.

# %%
# Reuse the real feature functions so AUTSL and Macedonian are byte-compatible.
# On Kaggle/Colab, upload alphabet/dtw_common.py alongside this notebook.
import sys; sys.path.append(".")
try:
    from dtw_common import _hand_shape, _arm_feat, HAND_SHAPE_DIM, make_template
except ImportError:
    raise SystemExit("Upload alphabet/dtw_common.py next to this notebook first.")

# mmpose wholebody -> MediaPipe-21 index maps. VERIFY against your keypoint
# file's documented joint order before trusting these.
# TODO: confirm these indices for your specific keypoint export.
WB_LEFT_HAND  = list(range(91, 112))    # 21 left-hand keypoints
WB_RIGHT_HAND = list(range(112, 133))   # 21 right-hand keypoints
WB_POSE_FOR_ARM = {                     # body joints used by _arm_feat
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9,    "r_wrist": 10,
}

def frame_feature(wb_xy: np.ndarray) -> np.ndarray | None:
    """One wholebody frame (133,2) -> POS_DIM feature, matching dtw_common."""
    # Build a MediaPipe-style 33-pose stub holding just the joints _arm_feat
    # needs at indices 11,12,13,14,15,16 (shoulders/elbows/wrists).
    pose = np.zeros((33, 3), np.float32)
    m = WB_POSE_FOR_ARM
    pose[11, :2], pose[12, :2] = wb_xy[m["l_shoulder"]], wb_xy[m["r_shoulder"]]
    pose[13, :2], pose[14, :2] = wb_xy[m["l_elbow"]],    wb_xy[m["r_elbow"]]
    pose[15, :2], pose[16, :2] = wb_xy[m["l_wrist"]],    wb_xy[m["r_wrist"]]

    lh, rh = wb_xy[WB_LEFT_HAND], wb_xy[WB_RIGHT_HAND]
    def hs(h):
        return _hand_shape(h) if np.any(h) else np.zeros(HAND_SHAPE_DIM, np.float32)
    from dtw_common import ARM_WEIGHT
    return np.concatenate([hs(lh), hs(rh), _arm_feat(pose) * ARM_WEIGHT]).astype(np.float32)

def clip_to_template(seq_xy: np.ndarray) -> np.ndarray:
    """(T,133,2) keypoint clip -> (SEQ_LEN, POS_DIM*2) template (smoothed,
    resampled, +velocity) — identical representation to the Macedonian side."""
    frames = [frame_feature(f) for f in seq_xy]
    frames = np.array([f for f in frames if f is not None], np.float32)
    return make_template(frames)            # uses TEMPLATE_LEN==SEQ_LEN

# %% [markdown]
# ## 2. Dataset with signer-aware sampling
#
# The contrastive loss only learns invariance if a batch contains the **same
# sign performed by different signers**. So we sample P signs, and for each,
# K clips drawn from DIFFERENT signers. Held-out signers go to validation.

# %%
class AUTSLClips(Dataset):
    """TODO: implement load_index() for your keypoint files. Each item must
    expose (clip_xy[T,133,2], sign_id, signer_id)."""
    def __init__(self, split="train", val_signers=(40, 41, 42)):
        self.items = self.load_index(AUTSL_KEYPOINTS_DIR)   # [(path, sign, signer)]
        keep = (lambda s: s not in val_signers) if split == "train" else (lambda s: s in val_signers)
        self.items = [it for it in self.items if keep(it[2])]
        self.by_sign = {}
        for i, (_, sign, _) in enumerate(self.items):
            self.by_sign.setdefault(sign, []).append(i)
        self.signs = list(self.by_sign)

    def load_index(self, root):
        raise NotImplementedError(
            "Read your keypoint manifest and return [(path, sign_id, signer_id), ...]")

    def load_clip(self, path) -> np.ndarray:
        raise NotImplementedError("Load (T,133,2) keypoints from `path`.")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, sign, signer = self.items[i]
        return clip_to_template(self.load_clip(path)), sign, signer

class PKSampler(torch.utils.data.Sampler):
    """Yield batches of P signs x K clips (varied signers) for contrastive loss."""
    def __init__(self, ds: AUTSLClips, P=BATCH_SIGNS, K=BATCH_VIEWS, batches=400):
        self.ds, self.P, self.K, self.batches = ds, P, K, batches
    def __iter__(self):
        for _ in range(self.batches):
            signs = random.sample(self.ds.signs, self.P)
            batch = []
            for s in signs:
                pool = self.ds.by_sign[s]
                batch += random.sample(pool, min(self.K, len(pool))) if len(pool) >= self.K \
                         else [random.choice(pool) for _ in range(self.K)]
            yield batch
    def __len__(self): return self.batches

# %% [markdown]
# ## 3. Temporal encoder
#
# Small BiGRU over the feature sequence -> mean-pool -> projection -> L2 norm.
# Compact on purpose: the features already do most of the invariance work, so
# this needs far less data/compute than an RGB video model.

# %%
class SignEncoder(nn.Module):
    def __init__(self, in_dim, hidden=256, embed=EMBED_DIM, layers=2):
        super().__init__()
        self.gru  = nn.GRU(in_dim, hidden, layers, batch_first=True,
                           bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(2 * hidden, embed), nn.ReLU(),
                                  nn.Linear(embed, embed))
    def forward(self, x):                  # x: (B, SEQ_LEN, in_dim)
        h, _ = self.gru(x)
        z = self.head(h.mean(dim=1))
        return F.normalize(z, dim=-1)

# %% [markdown]
# ## 4. Supervised contrastive loss (SupCon / NT-Xent)
#
# Pulls same-sign embeddings together, pushes different signs apart. With
# signer-varied positives, "together" must cross signers -> invariance.

# %%
def supcon_loss(z, labels, temp=0.07):
    sim = z @ z.t() / temp
    sim.fill_diagonal_(-1e9)
    labels = labels.view(-1, 1)
    pos = (labels == labels.t()).float()
    pos.fill_diagonal_(0)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    denom = pos.sum(1).clamp(min=1)
    return -((pos * logp).sum(1) / denom).mean()

# %% [markdown]
# ## 5. Train + validate (held-out signers)

# %%
def collate(batch):
    xs, signs, signers = zip(*batch)
    return (torch.tensor(np.stack(xs)),
            torch.tensor(signs), torch.tensor(signers))

def run_training():
    train_ds = AUTSLClips("train")
    val_ds   = AUTSLClips("val")
    in_dim   = train_ds[0][0].shape[-1]            # POS_DIM*2
    train_ld = DataLoader(train_ds, batch_sampler=PKSampler(train_ds), collate_fn=collate)

    model = SignEncoder(in_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        for x, signs, _ in train_ld:
            x = x.to(DEVICE)
            loss = supcon_loss(model(x), signs.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        acc = evaluate_heldout(model, val_ds)       # real invariance signal
        print(f"epoch {epoch:02d}  loss {loss.item():.3f}  held-out top-1 {acc:.1%}")
    return model

@torch.no_grad()
def evaluate_heldout(model, val_ds, gallery_per_sign=1):
    """kNN among held-out signers: one signer's clip = gallery, others = query."""
    model.eval()
    embs, signs, signers = [], [], []
    for x, s, sg in DataLoader(val_ds, batch_size=64, collate_fn=collate):
        embs.append(model(x.to(DEVICE)).cpu()); signs += list(s.numpy()); signers += list(sg.numpy())
    E = torch.cat(embs); signs = np.array(signs); signers = np.array(signers)
    gal = signers == np.unique(signers)[0]          # one held-out signer as gallery
    qry = ~gal
    if gal.sum() == 0 or qry.sum() == 0: return 0.0
    sim = E[qry] @ E[gal].t()
    pred = signs[gal][sim.argmax(1).numpy()]
    return float((pred == signs[qry]).mean())

# model = run_training()
# torch.save(model.state_dict(), "sign_encoder.pt")   # <- the only file you bring home

# %% [markdown]
# ## 6. Use it on Macedonian — embed the gallery, then kNN
#
# Locally (back on your laptop), load the trained encoder, embed your 2500
# `word_templates.npz` templates ONCE -> the dictionary. At inference, your
# existing segmentation produces a clip -> template -> encoder -> nearest
# neighbour. DTW is replaced by cosine distance; everything else is unchanged.

# %%
def build_macedonian_gallery(encoder_path="sign_encoder.pt",
                             templates_npz="data/landmarks/word_templates.npz",
                             out="data/landmarks/word_embeddings.npz"):
    d = np.load(templates_npz, allow_pickle=True)
    templ = torch.tensor(d["templates"])            # (K, SEQ_LEN, POS_DIM*2)
    model = SignEncoder(templ.shape[-1]).to(DEVICE)
    model.load_state_dict(torch.load(encoder_path, map_location=DEVICE)); model.eval()
    with torch.no_grad():
        emb = model(templ.to(DEVICE)).cpu().numpy()
    np.savez(out, embeddings=emb, labels=d["labels"])
    print(f"Embedded {len(emb)} word templates -> {out}")

def predict(clip_template: np.ndarray, gallery_npz="data/landmarks/word_embeddings.npz",
            encoder_path="sign_encoder.pt", top_k=5):
    g = np.load(gallery_npz, allow_pickle=True)
    G, labels = torch.tensor(g["embeddings"]), np.array([str(x) for x in g["labels"]])
    model = SignEncoder(clip_template.shape[-1]).to(DEVICE)
    model.load_state_dict(torch.load(encoder_path, map_location=DEVICE)); model.eval()
    with torch.no_grad():
        q = model(torch.tensor(clip_template[None]).to(DEVICE)).cpu()
    sims = (q @ G.t()).numpy()[0]
    top = sims.argsort()[::-1][:top_k]
    return [(labels[i], float(sims[i])) for i in top]
