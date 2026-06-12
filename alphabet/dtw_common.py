"""
Shared code for the DTW template matcher.

A "template" is one full sign performance: detected landmark frames,
smoothed, resampled to a fixed length, with velocity features appended.
Classification is nearest-neighbor DTW against one or more templates
per letter — no training involved.

Features cover BOTH hands: two hand slots (left-of-body, right-of-body)
plus arm landmarks. A missing hand is a zero block, so one- and
two-handed signs are directly comparable.

Used by extract_templates.py (builds templates from the reference
videos), record_templates.py (your own webcam takes) and dtw_demo.py
(matches live webcam segments against them).
"""

import sys
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

sys.path.append(str(Path(__file__).parent.parent))
from utils import (normalize_landmarks, normalize_arm_landmarks,
                   pose_landmarks_from_result)

# ── Feature configuration ─────────────────────────────────────────────────────
TEMPLATE_LEN = 32     # every sign (template or live segment) is resampled to this
COORDS       = 2      # (x, y) only — MediaPipe z is too noisy across cameras
SMOOTH_WIN   = 3      # moving-average window over raw frames (kills landmark jitter)
VEL_WEIGHT   = 8.0    # velocity contribution vs. position in the distance
ARM_WEIGHT   = 0.3    # arm features downweighted — webcam framing differs from videos
DTW_BAND     = 8      # Sakoe-Chiba band half-width (frames)

# Query trim variants: live segmentation often captures the approach phase
# (hand raising into position) that the video templates don't include, so we
# also try matching with the start of the segment cut off.
QUERY_TRIMS = [(0.00, 1.00), (0.25, 1.00)]

HAND_POINTS = 21
ARM_POINTS  = 6
HAND_FEAT   = HAND_POINTS * COORDS                  # 42 per hand slot
POS_DIM     = 2 * HAND_FEAT + ARM_POINTS * COORDS   # 96

# Arm landmark order (utils.ARM_INDICES): Lshoulder, Rshoulder, Lelbow,
# Relbow, Lwrist, Rwrist — these pairs swap when mirroring.
_ARM_MIRROR_ORDER = [1, 0, 3, 2, 5, 4]


def build_frame(hand_result, pose_result) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    One frame's feature from raw MediaPipe results.

    Returns (feature (POS_DIM,), wrist_xy (2,)) — wrist_xy is the mean raw
    wrist position of the detected hands (used for motion segmentation).
    Returns (None, None) if no pose or no hands were detected.

    Hand slot assignment: with two hands, sorted by image x (left slot =
    leftmost). With one hand, the slot is chosen by which side of the body
    midline (shoulder midpoint) the wrist is on. The other slot is zeros.
    """
    pose_raw = pose_landmarks_from_result(pose_result)
    if pose_raw is None or not hand_result.hand_landmarks:
        return None, None

    hands = [np.array([[lm.x, lm.y, lm.z] for lm in h], dtype=np.float32)
             for h in hand_result.hand_landmarks[:2]]

    left = right = None
    if len(hands) == 2:
        hands.sort(key=lambda h: float(h[0, 0]))
        left, right = hands
    else:
        mid_x = float(pose_raw[11, 0] + pose_raw[12, 0]) / 2
        if float(hands[0][0, 0]) < mid_x:
            left = hands[0]
        else:
            right = hands[0]

    def hand_feat(h):
        if h is None:
            return np.zeros(HAND_FEAT, dtype=np.float32)
        return normalize_landmarks(h).reshape(HAND_POINTS, 3)[:, :COORDS].ravel()

    arm = (normalize_arm_landmarks(pose_raw)
           .reshape(ARM_POINTS, 3)[:, :COORDS].ravel() * ARM_WEIGHT)

    feature  = np.concatenate([hand_feat(left), hand_feat(right), arm]).astype(np.float32)
    wrist_xy = np.mean([h[0, :2] for h in hands], axis=0)
    return feature, wrist_xy


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """Mirror left<->right: swap the hand slots, swap left/right arm points,
    and flip every x coordinate. Matches an opposite-handed signer."""
    out = seq.copy()
    T   = len(out)

    left_block  = out[:, :HAND_FEAT].copy()
    out[:, :HAND_FEAT]               = out[:, HAND_FEAT:2 * HAND_FEAT]
    out[:, HAND_FEAT:2 * HAND_FEAT]  = left_block

    arm = out[:, 2 * HAND_FEAT:].reshape(T, ARM_POINTS, COORDS)
    out[:, 2 * HAND_FEAT:] = arm[:, _ARM_MIRROR_ORDER, :].reshape(T, -1)

    out[:, 0::COORDS] *= -1   # flip x of every landmark (hands and arm)
    return out


def smooth_sequence(seq: np.ndarray, win: int = SMOOTH_WIN) -> np.ndarray:
    """Moving average over time. Velocity features amplify per-frame landmark
    jitter, so smoothing first matters more than it looks."""
    if len(seq) < win:
        return seq
    kernel = np.ones(win, dtype=np.float32) / win
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"),
                               0, seq).astype(np.float32)


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
    Smooths, resamples to fixed length, then appends frame-to-frame velocity
    so signs that differ mainly in motion direction stay separable.
    """
    seq = smooth_sequence(np.asarray(frames, dtype=np.float32))
    seq = resample_sequence(seq)
    vel = np.diff(seq, axis=0, prepend=seq[:1]) * VEL_WEIGHT
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
             labels: np.ndarray,
             top_k: int = 3) -> tuple[list[tuple[str, float]], float]:
    """
    Match one captured segment against all templates.

    frames    : (T, POS_DIM) raw detected position frames (any T >= 2)
    templates : (K, TEMPLATE_LEN, POS_DIM * 2)
    labels    : (K,) label per template (several templates may share a label)

    Returns (ranked, margin):
      ranked : top_k list of (label, distance), best first
      margin : second_best / best distance — higher = more confident
    """
    frames = np.asarray(frames, dtype=np.float32)
    T = len(frames)

    queries = []
    for lo, hi in QUERY_TRIMS:
        a, b = int(T * lo), int(T * hi)
        if b - a >= 2:
            sub = frames[a:b]
            queries.append(make_template(sub))
            queries.append(make_template(mirror_sequence(sub)))

    class_best: dict[str, float] = {}
    for tmpl, label in zip(templates, labels):
        d = min(dtw_distance(q, tmpl) for q in queries)
        if d < class_best.get(label, np.inf):
            class_best[label] = d

    ranked = sorted(class_best.items(), key=lambda kv: kv[1])
    best_dist = ranked[0][1]
    margin = (ranked[1][1] / best_dist) if len(ranked) > 1 and best_dist > 1e-9 else np.inf
    return ranked[:top_k], margin
