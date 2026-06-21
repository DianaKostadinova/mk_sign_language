"""
Stage-0 baseline, part 2/2 — how discriminative is the current DTW pipeline
at WORD scale?

We only have ONE video per word, so we cannot measure cross-signer accuracy
here. What we CAN measure is an UPPER BOUND on the current representation:

    gallery : one trim-variant template per word video
    query   : a DIFFERENT trim-variant of the SAME video
    metric  : nearest-neighbour DTW — is the top match the same word?

This is the easiest possible case (same signer, same recording, only the
segmentation window differs). Reading the result:

  • HIGH top-1  -> features + DTW stay discriminative across thousands of
                  words; the pipeline survives word-length sequences. The
                  ONLY thing left to solve is cross-signer -> the encoder.
  • LOW top-1   -> the representation itself doesn't scale to this many
                  classes; the encoder has a harder job and the features
                  may need rethinking before training anything.

DTW is O(N^2) in the number of words, so start with a --sample subset.

Usage:
    python words/eval_dtw_baseline.py --sample 250
    python words/eval_dtw_baseline.py            # all words (slow)
"""
import sys
import argparse
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from alphabet.dtw_common import dtw_distance

CACHE = ROOT / "data" / "landmarks" / "word_templates.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=250,
                    help="evaluate on a random subset of N words (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not CACHE.exists():
        print(f"{CACHE} not found — run words/extract_word_templates.py first.")
        return

    d = np.load(CACHE, allow_pickle=True)
    templates = d["templates"]
    labels    = np.array([str(x) for x in d["labels"]])
    vid_ids   = d["vid_ids"]
    trim_ids  = d["trim_ids"]

    # group template indices by source video
    by_vid: dict[int, list[int]] = {}
    for idx, v in enumerate(vid_ids):
        by_vid.setdefault(int(v), []).append(idx)

    # gallery = first trim of each video; query = a different trim (if any)
    gallery_idx, query_idx = [], []
    for v, idxs in by_vid.items():
        idxs = sorted(idxs, key=lambda i: trim_ids[i])
        gallery_idx.append(idxs[0])
        if len(idxs) > 1:
            query_idx.append(idxs[-1])      # last trim = most different window

    rng = np.random.default_rng(args.seed)
    if args.sample and args.sample < len(query_idx):
        query_idx = list(rng.choice(query_idx, size=args.sample, replace=False))

    gallery = templates[gallery_idx]
    g_labels = labels[gallery_idx]
    print(f"Gallery: {len(gallery)} words   |   Queries: {len(query_idx)}")
    print("Running nearest-neighbour DTW (this is the slow part)...\n")

    top1 = top5 = 0
    misses = []
    for n, qi in enumerate(query_idx, 1):
        q, q_label = templates[qi], labels[qi]
        dists = np.array([dtw_distance(q, g) for g in gallery])
        order = np.argsort(dists)
        ranked = g_labels[order]
        if ranked[0] == q_label:
            top1 += 1
        if q_label in ranked[:5]:
            top5 += 1
        elif len(misses) < 12:
            misses.append((q_label, list(ranked[:3])))
        if n % 25 == 0 or n == len(query_idx):
            print(f"  [{n:>4}/{len(query_idx)}]  "
                  f"top-1 {top1/n:5.1%}   top-5 {top5/n:5.1%}")

    N = len(query_idx)
    print(f"\n=== Stage-0 DTW word baseline (self-retrieval upper bound) ===")
    print(f"  words in gallery : {len(gallery)}")
    print(f"  queries          : {N}")
    print(f"  top-1 accuracy   : {top1/N:.1%}")
    print(f"  top-5 accuracy   : {top5/N:.1%}")
    if misses:
        print(f"\n  example misses (true -> top-3 predicted):")
        for true, pred in misses:
            print(f"    {true:<22} -> {pred}")
    print("\nInterpretation: this is the EASY case (same signer/recording).")
    print("Cross-signer real use will be LOWER — the gap is what the encoder closes.")


if __name__ == "__main__":
    main()
