"""
Visualize the template "space".

There is no learned embedding here — the matcher compares templates with DTW
distance, not Euclidean. So we compute the full pairwise DTW distance matrix
between all templates and use MDS to lay them out in 2D such that DTW-similar
templates sit close together. Clusters = letters; overlapping clusters = the
letter pairs the matcher is most likely to confuse.

Output: models/template_map.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from alphabet.dtw_common import dtw_distance

ROOT = Path(__file__).parent.parent
data = np.load(ROOT / "data" / "landmarks" / "dtw_templates.npz", allow_pickle=True)
templates, labels = data["templates"], data["labels"]
n = len(templates)

print(f"Computing {n}x{n} pairwise DTW distances...")
D = np.zeros((n, n), dtype=np.float64)
for i in range(n):
    for j in range(i + 1, n):
        d = dtw_distance(templates[i], templates[j])
        D[i, j] = D[j, i] = d

print("Running MDS to 2D...")
coords = MDS(n_components=2, dissimilarity="precomputed",
             random_state=0, normalized_stress="auto").fit_transform(D)

# One color per letter; bold letter glyph at each cluster centroid.
classes = sorted(set(labels))
cmap    = plt.cm.get_cmap("hsv", len(classes))
color_of = {c: cmap(i) for i, c in enumerate(classes)}

fig, ax = plt.subplots(figsize=(15, 13))
for c in classes:
    idx = np.where(labels == c)[0]
    ax.scatter(coords[idx, 0], coords[idx, 1], color=color_of[c], s=60,
               alpha=0.6, edgecolors="none")
    cx, cy = coords[idx, 0].mean(), coords[idx, 1].mean()
    ax.text(cx, cy, c, fontsize=20, fontweight="bold", ha="center", va="center",
            color="black")

ax.set_title("Template map — MDS of pairwise DTW distances\n"
             "(close = matcher sees them as similar)", fontsize=14)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
out = ROOT / "models" / "template_map.png"
plt.savefig(out, dpi=130)
print(f"Saved -> {out}")
