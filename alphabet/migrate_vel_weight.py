"""Rescale stored templates after a VEL_WEIGHT change.

A template is concat([positions, diff(positions) * VEL_WEIGHT]), so switching
weights is an exact multiplication of the velocity half by (new / old) — no
MediaPipe re-run needed. Mixing templates built under different weights
silently corrupts matching, so every stored .npz must be migrated together.

    python alphabet/migrate_vel_weight.py --from 8.0        # -> current VEL_WEIGHT

Originals are copied to <name>.vel<old>.npz before anything is written.
"""
import argparse, shutil, sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))
from dtw_common import VEL_WEIGHT, POS_DIM, COARSE_POS_DIM

ROOT  = Path(__file__).parent.parent
FILES = ["dtw_templates.npz", "user_templates.npz",
         "word_templates.npz", "word_templates_coarse.npz"]

def half_dim(width):
    """templates are (T, POS_DIM*2); infer which descriptor this file uses"""
    for d in (POS_DIM, COARSE_POS_DIM):
        if width == 2 * d:
            return d
    raise ValueError(f"unexpected template width {width}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="old", type=float, required=True,
                    help="VEL_WEIGHT the stored templates were built with")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if abs(args.old - VEL_WEIGHT) < 1e-9:
        print(f"stored weight already matches VEL_WEIGHT={VEL_WEIGHT}; nothing to do")
        return

    scale = VEL_WEIGHT / args.old
    print(f"VEL_WEIGHT {args.old} -> {VEL_WEIGHT}   (velocity half x {scale:g})\n")

    for name in FILES:
        path = ROOT / "data" / "landmarks" / name
        if not path.exists():
            print(f"  {name:30s} missing, skipped")
            continue
        d = dict(np.load(path, allow_pickle=True))
        if "templates" not in d:
            print(f"  {name:30s} no 'templates' key, skipped")
            continue

        T = d["templates"].astype(np.float32)
        P = half_dim(T.shape[-1])
        before = float(np.abs(T[:, :, P:]).mean())
        T[:, :, P:] *= scale
        d["templates"] = T
        after = float(np.abs(T[:, :, P:]).mean())

        print(f"  {name:30s} {T.shape}  mean|vel| {before:.4f} -> {after:.4f}"
              f"{'  (dry run)' if args.dry_run else ''}")
        if not args.dry_run:
            backup = path.with_suffix(f".vel{args.old:g}.npz")
            if not backup.exists():
                shutil.copy2(path, backup)
            np.savez(path, **d)

    if not args.dry_run:
        print("\ndone. Originals kept alongside as *.vel%g.npz" % args.old)
        print("NOTE: data/landmarks/word_embeddings.npz was produced by an encoder "
              "trained on the OLD weighting and is now stale — rebuild it before use.")

if __name__ == "__main__":
    main()
