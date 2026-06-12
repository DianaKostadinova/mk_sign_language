"""
Offline sanity check: for each letter, how far is its template from the
nearest *other* letter? Small distances = likely live confusions.
Run after extract_templates.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from alphabet.dtw_common import dtw_distance


def main():
    data      = np.load(Path(__file__).parent.parent / "data" / "landmarks" / "dtw_templates.npz",
                        allow_pickle=True)
    templates = data["templates"]
    labels    = data["labels"]

    print(f"{'letter':<8}{'nearest other':<16}{'dist':<8}{'2nd nearest':<16}{'dist'}")
    rows = []
    for cls in sorted(set(labels)):
        own_idx = np.where(labels == cls)[0]
        query   = templates[own_idx[0]]          # full-window variant
        other_best: dict[str, float] = {}
        for tmpl, lab in zip(templates, labels):
            if lab == cls:
                continue
            d = dtw_distance(query, tmpl)
            if d < other_best.get(lab, np.inf):
                other_best[lab] = d
        ranked = sorted(other_best.items(), key=lambda kv: kv[1])
        (l1, d1), (l2, d2) = ranked[0], ranked[1]
        rows.append((d1, cls, l1, d2, l2))
        print(f"{cls:<8}{l1:<16}{d1:<8.3f}{l2:<16}{d2:.3f}")

    rows.sort()
    print("\nTightest pairs (most likely confusions):")
    for d1, cls, l1, *_ in rows[:5]:
        print(f"  {cls} ↔ {l1}  dist={d1:.3f}")


if __name__ == "__main__":
    main()
