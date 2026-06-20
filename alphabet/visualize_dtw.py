"""
Explain the DTW matcher visually, using your OWN templates.

Produces data/dtw_explained.png with three panels:
  1. Two takes of the SAME letter   -> low DTW cost, warping path hugs diagonal
  2. SAME-letter cost matrix + path -> the actual alignment the matcher finds
  3. A DIFFERENT letter vs take 1    -> high DTW cost, path forced off-diagonal

This is exactly what classify() does: it walks the query against every
template, sums the cheapest aligned distance, and picks the smallest total.
"""
import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

sys.path.append(str(Path(__file__).parent.parent))
from alphabet.dtw_common import DTW_BAND, TEMPLATE_LEN

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "data" / "landmarks" / "user_templates.npz"
if not SRC.exists():
    SRC = ROOT / "data" / "landmarks" / "dtw_templates.npz"
OUT  = ROOT / "data" / "dtw_explained.png"


def dtw_path(a, b, band=DTW_BAND):
    """Banded DTW returning (distance, cost_matrix, warping_path)."""
    cost = cdist(a, b, metric="euclidean")
    Ta, Tb = cost.shape
    D = np.full((Ta + 1, Tb + 1), np.inf)
    D[0, 0] = 0.0
    back = {}
    for i in range(1, Ta + 1):
        for j in range(max(1, i - band), min(Tb, i + band) + 1):
            choices = [(D[i-1, j-1], (i-1, j-1)),
                       (D[i-1, j],   (i-1, j)),
                       (D[i, j-1],   (i, j-1))]
            best, prev = min(choices, key=lambda c: c[0])
            D[i, j] = cost[i-1, j-1] + best
            back[(i, j)] = prev
    # backtrack
    path, node = [], (Ta, Tb)
    while node != (0, 0):
        path.append((node[0]-1, node[1]-1))
        node = back.get(node, (0, 0))
    path.reverse()
    return D[Ta, Tb] / (Ta + Tb), cost, np.array(path)


def main():
    d = np.load(SRC, allow_pickle=True)
    tmpl, labels = d["templates"], np.array([str(x) for x in d["labels"]])

    # pick a letter that has >=2 takes for the "same letter" comparison
    same_label = None
    for lab in labels:
        if list(labels).count(lab) >= 2:
            same_label = lab
            break
    if same_label is None:                       # fallback: just use first two
        same_label = labels[0]
    same_idx = np.where(labels == same_label)[0][:2]
    diff_idx = np.where(labels != same_label)[0][0]

    A = tmpl[same_idx[0]]          # take 1 of letter X
    B = tmpl[same_idx[1]] if len(same_idx) > 1 else tmpl[same_idx[0]]
    C = tmpl[diff_idx]             # a different letter

    d_same, cost_same, path_same = dtw_path(A, B)
    d_diff, cost_diff, path_diff = dtw_path(A, C)

    same_lab = same_label
    diff_lab = labels[diff_idx]

    # a single representative feature channel to draw the 1-D signals
    ch = int(np.argmax(A.std(axis=0)))   # the most "active" feature dimension

    fig = plt.figure(figsize=(15, 5.2), dpi=150)
    fig.suptitle("How the DTW matcher works — on your own templates",
                 fontsize=15, weight="bold", y=1.02)

    # ── Panel 1: same-letter signals + alignment links ───────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    t = np.arange(TEMPLATE_LEN)
    ax1.plot(t, A[:, ch], "-o", ms=3, color="#2b6cb0", label=f"'{same_lab}' take 1 (query)")
    ax1.plot(t, B[:, ch] + 0.0, "-o", ms=3, color="#2f855a", label=f"'{same_lab}' take 2 (template)")
    for (i, j) in path_same[::2]:
        ax1.plot([i, j], [A[i, ch], B[j, ch]], color="#bbb", lw=0.6, zorder=0)
    ax1.set_title(f"Same letter '{same_lab}': DTW links the matching moments\n"
                  f"warp distance = {d_same:.3f}  (LOW = good match)", fontsize=9.5)
    ax1.set_xlabel("frame"); ax1.set_ylabel("most active feature")
    ax1.legend(fontsize=7.5, loc="best")

    # ── Panel 2: cost matrix + warping path + band ───────────────────────────
    ax2 = fig.add_subplot(1, 3, 2)
    im = ax2.imshow(cost_same.T, origin="lower", cmap="viridis", aspect="auto")
    ax2.plot(path_same[:, 0], path_same[:, 1], color="white", lw=2.2,
             label="optimal warping path")
    # draw Sakoe-Chiba band edges
    n = TEMPLATE_LEN
    ax2.plot([0, n-1], [-DTW_BAND, n-1-DTW_BAND], "--", color="red", lw=1, alpha=0.7)
    ax2.plot([0, n-1], [DTW_BAND, n-1+DTW_BAND], "--", color="red", lw=1, alpha=0.7,
             label=f"band (±{DTW_BAND} frames)")
    ax2.set_xlim(0, n-1); ax2.set_ylim(0, n-1)
    ax2.set_title("Cost matrix: every query frame vs every template frame\n"
                  "path = cheapest way to line them up", fontsize=9.5)
    ax2.set_xlabel(f"'{same_lab}' take 1 frame"); ax2.set_ylabel(f"'{same_lab}' take 2 frame")
    ax2.legend(fontsize=7.5, loc="upper left")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="per-frame distance")

    # ── Panel 3: different letter, forced off-diagonal ───────────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(cost_diff.T, origin="lower", cmap="viridis", aspect="auto")
    ax3.plot(path_diff[:, 0], path_diff[:, 1], color="white", lw=2.2)
    ax3.set_xlim(0, n-1); ax3.set_ylim(0, n-1)
    ax3.set_title(f"Different letters '{same_lab}' vs '{diff_lab}'\n"
                  f"warp distance = {d_diff:.3f}  (HIGH = rejected)", fontsize=9.5)
    ax3.set_xlabel(f"'{same_lab}' frame"); ax3.set_ylabel(f"'{diff_lab}' frame")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="per-frame distance")

    plt.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    print(f"same-letter '{same_lab}' distance = {d_same:.4f}")
    print(f"diff-letter '{same_lab}' vs '{diff_lab}' distance = {d_diff:.4f}")
    print(f"=> matcher prefers the smaller one (margin = {d_diff/d_same:.2f}x)")


if __name__ == "__main__":
    main()
