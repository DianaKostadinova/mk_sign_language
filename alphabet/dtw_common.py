"""
Shared code for the DTW template matcher.

A "template" is one full sign performance: detected landmark frames,
resampled to a fixed length, with velocity features appended.
Classification is nearest-neighbor DTW against one or more templates
per letter — no training involved.

Used by extract_templates.py (builds templates from the reference
videos) and dtw_demo.py (matches live webcam segments against them).
"""

import numpy as np
from scipy.spatial.distance import cdist

# ── Feature configuration ─────────────────────────────────────────────────────
TEMPLATE_LEN = 32     # every sign (template or live segment) is resampled to this
COORDS       = 2      # (x, y) only — MediaPipe z is too noisy across cameras
VEL_WEIGHT   = 1.5    # how strongly velocity counts vs. position in the distance
DTW_BAND     = 8      # Sakoe-Chiba band half-width (frames)

HAND_POINTS = 21
ARM_POINTS  = 6
POS_DIM     = (HAND_POINTS + ARM_POINTS) * COORDS   # 54


def frame_feature(hand_vec: np.ndarray, arm_vec: np.ndarray) -> np.ndarray:
    """
    Build one frame's position feature from the normalized vectors
    produced by utils.normalize_landmarks (63,) / normalize_arm_landmarks (18,).
    Keeps only the first COORDS coordinates of each landmark. -> (POS_DIM,)
    """
    hand = hand_vec.reshape(HAND_POINTS, 3)[:, :COORDS]
    arm  = arm_vec.reshape(ARM_POINTS, 3)[:, :COORDS]
    return np.concatenate([hand.ravel(), arm.ravel()]).astype(np.float32)


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """Flip x of every landmark — matches a left-handed signer against
    right-handed templates (and vice versa). seq: (T, POS_DIM)."""
    out = seq.copy()
    out[:, 0::COORDS] *= -1
    return out


def resample_sequence(seq: np.ndarray, length: int = TEMPLATE_LEN) -> np.ndarray:
    """Linearly resample (T, D) to (length, D) along the time axis."""
    T = len(seq)
    if T == 1:
        return np.repeat(seq, length, axis=0)
    t_src = np.linspace(0, T - 1, length)
    lo    = np.floor(t_src).astype(int)
    hi    = np.minimum(lo + 1, T - 1)
    alpha = (t_src - lo)[:, None]
    return (seq[lo] * (1 - alpha) + seq[hi] * alpha).astype(np.float32)


def make_template(frames: np.ndarray) -> np.ndarray:
    """
    (T, POS_DIM) raw detected frames -> (TEMPLATE_LEN, POS_DIM * 2) template.
    Resamples to fixed length, then appends frame-to-frame velocity so signs
    that differ mainly in motion direction stay separable.
    """
    seq = resample_sequence(np.asarray(frames, dtype=np.float32))
    vel = np.diff(seq, axis=0, prepend=seq[:1]) * VEL_WEIGHT * TEMPLATE_LEN
    return np.concatenate([seq, vel], axis=1)


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int = DTW_BAND) -> float:
    """
    DTW distance between two equal-length feature sequences with a
    Sakoe-Chiba band, normalized by path length.
    """
    cost = cdist(a, b, metric="euclidean")
    Ta, Tb = cost.shape
    D = np.full((Ta + 1, Tb + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, Ta + 1):
        j_lo = max(1, i - band)
        j_hi = min(Tb, i + band)
        for j in range(j_lo, j_hi + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j - 1],
                                               D[i - 1, j],
                                               D[i, j - 1])
    return D[Ta, Tb] / (Ta + Tb)


def classify(frames: np.ndarray,
             templates: np.ndarray,
             labels: np.ndarray) -> tuple[str, float, float]:
    """
    Match one captured segment against all templates.

    frames    : (T, POS_DIM) raw detected position frames (any T >= 2)
    templates : (K, TEMPLATE_LEN, POS_DIM * 2)
    labels    : (K,) label per template (several templates may share a label)

    Returns (best_label, best_distance, margin) where margin is
    second_best_class_distance / best_class_distance — higher = more confident.
    """
    frames = np.asarray(frames, dtype=np.float32)
    queries = [make_template(frames), make_template(mirror_sequence(frames))]

    class_best: dict[str, float] = {}
    for tmpl, label in zip(templates, labels):
        d = min(dtw_distance(q, tmpl) for q in queries)
        if d < class_best.get(label, np.inf):
            class_best[label] = d

    ranked = sorted(class_best.items(), key=lambda kv: kv[1])
    best_label, best_dist = ranked[0]
    margin = (ranked[1][1] / best_dist) if len(ranked) > 1 and best_dist > 1e-9 else np.inf
    return best_label, best_dist, margin
