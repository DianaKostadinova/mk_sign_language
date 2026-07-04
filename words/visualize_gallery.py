"""
Visualize the word embedding gallery.

Unlike alphabet/visualize_templates.py (which has no learned embedding and
has to compute an expensive full pairwise DTW distance matrix + MDS), this
gallery IS already a learned embedding space — PCA straight on the
embedding vectors is enough, no O(n^2) distance computation needed.

Colors by video-topic folder (data/landmarks/word_templates_coarse.npz
"topics" — same row order as the gallery, since build_macedonian_gallery()
embeds that exact templates file). Reading the result: tight, well-separated
clusters of the SAME word's multiple trim-variants = the encoder is
consistent; if unrelated words interleave heavily, the embedding isn't
discriminative enough yet.

Output: models/word_gallery_map.png
"""
import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT      = Path(__file__).parent.parent
GALLERY   = ROOT / "data" / "landmarks" / "word_embeddings.npz"
TEMPLATES = ROOT / "data" / "landmarks" / "word_templates_coarse.npz"
OUT       = ROOT / "models" / "word_gallery_map.png"


def main():
    if not GALLERY.exists():
        print(f"{GALLERY} not found — run build_macedonian_gallery() first.")
        return

    g = np.load(GALLERY, allow_pickle=True)
    emb, labels = g["embeddings"], np.array([str(x) for x in g["labels"]])

    topics = None
    if TEMPLATES.exists():
        t = np.load(TEMPLATES, allow_pickle=True)
        if "topics" in t and len(t["topics"]) == len(labels):
            topics = np.array([str(x) for x in t["topics"]])

    print(f"Embeddings: {emb.shape}  ({len(set(labels))} unique words)")
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(emb)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%} "
          f"(2 components — expect this to be low, embeddings are 256-d; "
          f"this is a rough 2D squash, not the full structure)")

    fig, ax = plt.subplots(figsize=(15, 13))
    if topics is not None:
        classes = sorted(set(topics))
        cmap = plt.cm.get_cmap("hsv", len(classes))
        color_of = {c: cmap(i) for i, c in enumerate(classes)}
        for c in classes:
            idx = np.where(topics == c)[0]
            ax.scatter(coords[idx, 0], coords[idx, 1], color=color_of[c],
                       s=12, alpha=0.5, edgecolors="none")
        title_extra = f"colored by {len(classes)} video-topic folders"
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.5, edgecolors="none")
        title_extra = "(no topic metadata found)"

    # Label a random sample of points so the map isn't unreadable with 2000+ words.
    rng = np.random.default_rng(0)
    sample = rng.choice(len(labels), size=min(60, len(labels)), replace=False)
    for i in sample:
        ax.annotate(labels[i], (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.8)

    ax.set_title(f"Macedonian word gallery — PCA of encoder embeddings\n{title_extra}",
                 fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=130)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
