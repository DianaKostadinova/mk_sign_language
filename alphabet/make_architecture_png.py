"""Render the alphabet DTW-demo architecture to a PNG."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent.parent / "data" / "architecture_alphabet_demo.png"

# colors
C_ENROLL = "#2b6cb0"
C_LIVE   = "#2f855a"
C_CORE   = "#744210"
C_BOX    = "#ffffff"
C_NOTE   = "#9b2c2c"

fig, ax = plt.subplots(figsize=(13, 8.5), dpi=150)
ax.set_xlim(0, 13); ax.set_ylim(0, 8.5); ax.axis("off")

ax.text(6.5, 8.15, "MK_Sign — Alphabet Fingerspelling Demo (DTW template matching)",
        ha="center", va="center", fontsize=15, weight="bold")
ax.text(6.5, 7.78, "Signer-dependent · no training · nearest-neighbor DTW",
        ha="center", va="center", fontsize=10, style="italic", color="#555")


def box(x, y, w, h, text, edge, fc=C_BOX, fs=9.5, weight="normal", tc="#1a1a1a"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                linewidth=1.6, edgecolor=edge, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, color=tc, wrap=True)


def arrow(x1, y1, x2, y2, color="#333", style="-|>", lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, lw=lw, color=color, linestyle=ls))


# ── Lane labels ───────────────────────────────────────────────────────────
ax.text(0.15, 6.55, "ENROLLMENT\n(once per user)", ha="left", va="center",
        fontsize=9, weight="bold", color=C_ENROLL, rotation=0)
ax.text(0.15, 3.55, "LIVE USE\n(every clip)", ha="left", va="center",
        fontsize=9, weight="bold", color=C_LIVE)

# ── Enrollment lane (top) ─────────────────────────────────────────────────
box(2.4, 6.05, 2.2, 1.0, "Webcam takes\nraise hand = REC\nhand out = save\n3 takes / letter", C_ENROLL)
box(5.2, 6.05, 2.2, 1.0, "MediaPipe\nHands (2) + Pose", C_ENROLL)
box(8.0, 6.05, 2.3, 1.0, "Shape features\nangles · spreads ·\nnorm. distances + arm", C_ENROLL)
box(10.7, 6.05, 2.0, 1.0, "user_templates\n.npz\n(smoothed, len 32,\n+ velocity)", C_ENROLL, fc="#ebf4fb")

arrow(4.6, 6.55, 5.2, 6.55, C_ENROLL)
arrow(7.4, 6.55, 8.0, 6.55, C_ENROLL)
arrow(10.3, 6.55, 10.7, 6.55, C_ENROLL)
ax.text(3.5, 5.75, "record_templates.py", ha="center", fontsize=7.5, color=C_ENROLL, style="italic")

# ── Live lane (bottom) ────────────────────────────────────────────────────
box(2.4, 3.05, 2.2, 1.0, "Record a clip\nhand-OUT between\nletters = delimiter", C_LIVE)
box(5.2, 3.05, 2.2, 1.0, "MediaPipe\nHands (2) + Pose\nper frame", C_LIVE)
box(8.0, 3.05, 2.3, 1.0, "split_into_signs()\nhand-absent gaps\ncut into segments", C_LIVE)
arrow(4.6, 3.55, 5.2, 3.55, C_LIVE)
arrow(7.4, 3.55, 8.0, 3.55, C_LIVE)
ax.text(3.5, 2.75, "sequence_demo.py", ha="center", fontsize=7.5, color=C_LIVE, style="italic")

# ── Core matcher (center bottom) ──────────────────────────────────────────
box(8.7, 1.15, 3.7, 1.45,
    "DTW nearest-neighbor  (classify)\n"
    "• try query trims + mirror\n"
    "• Sakoe-Chiba banded DTW\n"
    "• best label + confidence margin",
    C_CORE, fc="#fffaf0", fs=9)

box(4.6, 1.35, 2.7, 1.05, "Predicted letter\n+ margin\n→ append to word", C_CORE, fc="#fffaf0")

# arrows into core
arrow(9.15, 3.05, 9.7, 2.6, C_LIVE)                       # segments -> DTW
arrow(11.7, 6.05, 11.2, 2.6, C_ENROLL, ls="--")          # templates -> DTW
ax.text(11.95, 4.3, "templates\n(reference set)", ha="center", fontsize=7.5,
        color=C_ENROLL, rotation=90, style="italic")
arrow(8.7, 1.9, 7.3, 1.9, C_CORE)                         # DTW -> output

# ── Status / limitation note ──────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.4, 0.2), 7.6, 0.85,
             boxstyle="round,pad=0.05,rounding_size=0.1",
             linewidth=1.4, edgecolor=C_NOTE, facecolor="#fff5f5"))
ax.text(0.6, 0.62,
        "Status: works when matched against the USER's own takes (signer-dependent).\n"
        "Matching a new user against the reference signer's videos fails (domain gap).\n"
        "Next: contrastive encoder on a multi-signer corpus → swap DTW for embedding distance → zero enrollment.",
        ha="left", va="center", fontsize=8, color=C_NOTE)

plt.tight_layout()
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
