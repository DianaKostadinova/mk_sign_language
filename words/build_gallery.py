"""
Build the word embedding gallery for the AUTSL encoder.

`models/sign_encoder.pt` was trained on coarse templates built with
VEL_WEIGHT=8.0. The descriptor has since moved to 1.0 (see dtw_common), so
templates on disk no longer match what the encoder expects. Feeding it 1.0
templates is an out-of-distribution input and scores near noise — which looks
exactly like "the encoder doesn't work".

So both sides are rescaled to ENCODER_VEL here, and word_demo.py does the same
to live queries. Rescaling is exact: a template is
concat([pos, diff(pos) * w]), so changing w is a multiply on the velocity half.

    python words/build_gallery.py
    python words/build_gallery.py --limit-to data/landmarks/word_templates.npz

--limit-to restricts the gallery to the vocabulary of another template file, so
the encoder can be compared against the DTW demo on the SAME word list. Without
it the encoder faces ~2310 words while word_demo_dtw.py faces 100, and the two
numbers mean nothing next to each other.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "alphabet"))
sys.path.append(str(ROOT / "notebooks"))

from alphabet.dtw_common import VEL_WEIGHT, COARSE_POS_DIM
from encoder_training import SignEncoder, DEVICE

ENCODER_VEL = 8.0        # weighting models/sign_encoder.pt was trained with


def rescale_velocity(templates: np.ndarray, target: float,
                     current: float = VEL_WEIGHT) -> np.ndarray:
    """(K, T, D*2) -> velocity half rescaled from `current` to `target`."""
    out = templates.astype(np.float32).copy()
    half = out.shape[-1] // 2
    out[..., half:] *= (target / current)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", type=Path,
                    default=ROOT / "data" / "landmarks" / "word_templates_coarse.npz")
    ap.add_argument("--encoder", type=Path, default=ROOT / "models" / "sign_encoder.pt")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "landmarks" / "word_embeddings.npz")
    ap.add_argument("--limit-to", type=Path, default=None,
                    help="restrict gallery to the vocabulary of this template file")
    ap.add_argument("--encoder-vel", type=float, default=ENCODER_VEL)
    args = ap.parse_args()

    for p in (args.templates, args.encoder):
        if not p.exists():
            print(f"missing: {p}")
            return

    d = np.load(args.templates, allow_pickle=True)
    templates, labels = d["templates"], d["labels"].astype(str)
    if templates.shape[-1] != 2 * COARSE_POS_DIM:
        print(f"expected coarse templates of width {2*COARSE_POS_DIM}, "
              f"got {templates.shape[-1]} — is this the coarse file?")
        return

    if args.limit_to:
        vocab = set(np.load(args.limit_to, allow_pickle=True)["labels"].astype(str))
        keep = np.array([l in vocab for l in labels])
        if not keep.any():
            print("--limit-to left nothing; label sets do not overlap")
            return
        templates, labels = templates[keep], labels[keep]
        print(f"restricted to {len(set(labels))} words from {args.limit_to.name}")

    templates = rescale_velocity(templates, args.encoder_vel)
    print(f"{len(templates)} templates / {len(set(labels))} words, "
          f"velocity rescaled {VEL_WEIGHT} -> {args.encoder_vel} for the encoder")

    model = SignEncoder(templates.shape[-1]).to(DEVICE)
    model.load_state_dict(torch.load(args.encoder, map_location=DEVICE,
                                     weights_only=True))
    model.eval()

    embs = []
    with torch.no_grad():
        for i in range(0, len(templates), 512):
            batch = torch.tensor(templates[i:i + 512], dtype=torch.float32).to(DEVICE)
            embs.append(model(batch).cpu().numpy())
    emb = np.concatenate(embs)

    np.savez(args.out, embeddings=emb, labels=labels, encoder_vel=args.encoder_vel)
    print(f"wrote {args.out}  ({emb.shape})")

    # spread check: a collapsed gallery makes every query return the same words
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sub = E[np.random.default_rng(0).choice(len(E), min(500, len(E)), replace=False)]
    print(f"mean pairwise cosine {float((sub @ sub.T).mean()):+.3f}  "
          "(near 1.0 would mean a degenerate gallery)")


if __name__ == "__main__":
    main()
