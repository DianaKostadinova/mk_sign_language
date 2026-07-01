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

# Point this at the SAM-SLR processed-skeleton folder (the Google Drive
# release linked from jackyjsy/CVPR21Chal-SLR): contains
# {train,test}_data_joint.npy + train_label.pkl + {train,test}_labels.csv.
AUTSL_KEYPOINTS_DIR = "/content/autsl_skeleton"   # <-- EDIT if running on Kaggle instead of Colab
EMBED_DIM   = 256
SEQ_LEN     = 32            # match alphabet TEMPLATE_LEN so features line up
BATCH_SIGNS = 32            # P signs per batch
BATCH_VIEWS = 4            # K clips per sign per batch (P*K samples)
EPOCHS      = 40

# %% [markdown]
# ## 1. Keypoints -> shape features (parity with the DTW pipeline)
#
# The released AUTSL skeleton data (SAM-SLR, `jackyjsy/CVPR21Chal-SLR`) is
# **not** the raw 133 mmpose wholebody points — it's already reduced to a
# 27-node graph: 7 body points (nose + shoulders/elbows/wrists) + 10 points
# per hand (wrist/root, then base+tip per finger, thumb tip only). That's
# fewer points than the 21-landmark MediaPipe hands `dtw_common._hand_shape`
# expects, so it can't be byte-compatible with the full descriptor.
#
# Instead both sides go through `dtw_common._hand_shape_coarse10`, a
# 10-point descriptor designed to match what AUTSL actually has.
# `COARSE10_FROM_MP21` picks the same 10 points out of Macedonian's full
# 21-point MediaPipe hands so the two sides share one embedding space —
# see `alphabet/dtw_common.py`.

# %%
# Reuse the real feature functions so AUTSL and Macedonian are byte-compatible.
# On Kaggle/Colab, upload alphabet/dtw_common.py alongside this notebook.
import sys; sys.path.append(".")
try:
    from dtw_common import _hand_shape_coarse10, _arm_feat, COARSE_HAND_DIM, ARM_WEIGHT, make_template
except ImportError:
    raise SystemExit("Upload alphabet/dtw_common.py next to this notebook first.")

# 27-node layout produced by SAM-SLR's data-prepare (see
# CVPR21Chal-SLR/SL-GCN/data_gen/sign_gendata.py, the '27' config):
#   node 0           : nose
#   nodes 1,2 / 3,4 / 5,6 : l/r shoulder, l/r elbow, l/r wrist
#   nodes 7-16        : left hand  (root, thumb_tip, idx_base,idx_tip,
#                                    mid_base,mid_tip, ring_base,ring_tip,
#                                    pinky_base,pinky_tip)
#   nodes 17-26       : right hand, same order
WB27_LEFT_HAND  = list(range(7, 17))
WB27_RIGHT_HAND = list(range(17, 27))
WB27_POSE = {"l_shoulder": 1, "r_shoulder": 2, "l_elbow": 3,
             "r_elbow": 4,    "l_wrist": 5,    "r_wrist": 6}

def frame_feature(node_xy: np.ndarray) -> np.ndarray | None:
    """One 27-node frame (27,2) -> (2*COARSE_HAND_DIM + ARM_FEAT_DIM,) feature."""
    pose = np.zeros((33, 3), np.float32)
    m = WB27_POSE
    pose[11, :2], pose[12, :2] = node_xy[m["l_shoulder"]], node_xy[m["r_shoulder"]]
    pose[13, :2], pose[14, :2] = node_xy[m["l_elbow"]],    node_xy[m["r_elbow"]]
    pose[15, :2], pose[16, :2] = node_xy[m["l_wrist"]],    node_xy[m["r_wrist"]]

    lh, rh = node_xy[WB27_LEFT_HAND], node_xy[WB27_RIGHT_HAND]
    def hs(h):
        return _hand_shape_coarse10(h) if np.any(h) else np.zeros(COARSE_HAND_DIM, np.float32)
    return np.concatenate([hs(lh), hs(rh), _arm_feat(pose) * ARM_WEIGHT]).astype(np.float32)

def clip_to_template(seq_xy: np.ndarray) -> np.ndarray:
    """(T,27,2) keypoint clip -> (SEQ_LEN, feat_dim*2) template (smoothed,
    resampled, +velocity) — same coarse representation as the Macedonian side."""
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
import pickle, re, csv
from pathlib import Path

class AUTSLClips(Dataset):
    """Reads the SAM-SLR processed-skeleton release directly:
    {train,test}_data_joint.npy ((N,3,T,27,1): x,y,conf x frames x 27 nodes)
    + names/labels, row-aligned with the npy — from *_label.pkl where it
    exists (train), else *_labels.csv (test: 'sample_name,label' rows, no
    header assumed — verify against your actual file before trusting this).

    AUTSL's official train/test split is already signer-disjoint, so "val"
    here just maps to the test split — that IS the held-out-signer set, no
    extra carving needed."""
    def __init__(self, split="train"):
        file_split = "train" if split == "train" else "test"
        root = Path(AUTSL_KEYPOINTS_DIR)
        # mmap: data_joint.npy is multiple GB, don't load it into RAM.
        self.data = np.load(root / f"{file_split}_data_joint.npy", mmap_mode="r")
        names, labels = self._load_labels(root, file_split)
        self.names  = names
        self.labels = labels
        self.signers = [self._signer_of(n) for n in self.names]
        assert len(self.names) == self.data.shape[0], "label/data row count mismatch"
        self._cache: dict[int, np.ndarray] = {}

        self.by_sign = {}
        for i, s in enumerate(self.labels):
            self.by_sign.setdefault(s, []).append(i)
        self.signs = list(self.by_sign)

    @staticmethod
    def _signer_of(name: str) -> int:
        """AUTSL sample names look like 'signer23_sample412...' — pull the id."""
        m = re.search(r"signer(\d+)", name, re.IGNORECASE)
        return int(m.group(1)) if m else -1

    @staticmethod
    def _load_labels(root: Path, file_split: str):
        pkl_path = root / f"{file_split}_label.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                names, labels = pickle.load(f)
            return list(names), list(labels)
        csv_path = root / f"{file_split}_labels.csv"
        names, labels = [], []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 2 or not row[1].strip().lstrip("-").isdigit():
                    continue                       # skip a header row if present
                names.append(row[0]); labels.append(int(row[1]))
        return names, labels

    def __len__(self): return len(self.names)

    def __getitem__(self, i):
        # Cache: a sample's template is deterministic, but PKSampler re-draws
        # the same indices across all 400 batches x 40 epochs — without this,
        # clip_to_template (pure-Python/numpy, no GPU) reruns from scratch
        # every single time, which is the actual reason a "silent" epoch can
        # take a very long time.
        cached = self._cache.get(i)
        if cached is not None:
            return cached, self.labels[i], self.signers[i]
        row = np.asarray(self.data[i])               # (3, T, 27, 1)
        xy  = np.transpose(row[:2, :, :, 0], (1, 2, 0))  # (T, 27, 2), drop confidence
        tmpl = clip_to_template(xy)
        self._cache[i] = tmpl
        return tmpl, self.labels[i], self.signers[i]

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

CKPT_PATH = "/content/drive/MyDrive/autsl/sign_encoder_ckpt.pt"   # <-- EDIT: survives a dropped Colab session

def run_training(ckpt_path: str = CKPT_PATH):
    train_ds = AUTSLClips("train")
    val_ds   = AUTSLClips("val")
    in_dim   = train_ds[0][0].shape[-1]            # POS_DIM*2
    train_ld = DataLoader(train_ds, batch_sampler=PKSampler(train_ds), collate_fn=collate)

    model = SignEncoder(in_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    start_epoch = 0
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"]); start_epoch = ckpt["epoch"] + 1
        print(f"resumed from checkpoint at epoch {start_epoch}", flush=True)

    print(f"train samples={len(train_ds)}  val samples={len(val_ds)}  "
          f"batches/epoch={len(train_ld)}", flush=True)
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        for b, (x, signs, _) in enumerate(train_ld):
            x = x.to(DEVICE)
            loss = supcon_loss(model(x), signs.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            if (b + 1) % 20 == 0:                    # first-epoch feature caching is slow; show it's alive
                print(f"  epoch {epoch:02d}  batch {b+1}/{len(train_ld)}  "
                      f"loss {loss.item():.3f}", flush=True)
        sched.step()
        acc = evaluate_heldout(model, val_ds)       # real invariance signal
        print(f"epoch {epoch:02d}  loss {loss.item():.3f}  held-out top-1 {acc:.1%}", flush=True)
        if ckpt_path:                                # so a dropped connection costs <=1 epoch, not the whole run
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch}, ckpt_path)
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
# IMPORTANT: `data/landmarks/word_templates.npz` (from
# `words/extract_word_templates.py`) is featurized with the FULL 21-point
# `_hand_shape` descriptor (POS_DIM=68) — wrong dimension for this encoder,
# which only ever sees the 10-point coarse descriptor (parity with AUTSL's
# reduced skeleton). Before this step, re-run word extraction using
# `dtw_common.build_frame_coarse` in place of `build_frame` to produce a
# `word_templates_coarse.npz` (same video set, coarse features) — that's
# the file this function actually needs. Not done yet: it's pointless until
# the encoder is trained, so do it as the step right before this one.
#
# Locally (back on your laptop), load the trained encoder, embed your 2500
# coarse-feature templates ONCE -> the dictionary. At inference, your
# existing segmentation produces a clip -> coarse template -> encoder ->
# nearest neighbour. DTW is replaced by cosine distance.

# %%
def build_macedonian_gallery(encoder_path="sign_encoder.pt",
                             templates_npz="data/landmarks/word_templates_coarse.npz",
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
