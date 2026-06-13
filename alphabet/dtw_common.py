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
from utils import pose_landmarks_from_result

# ── Feature configuration ─────────────────────────────────────────────────────
# Features describe hand SHAPE (joint angles, finger spreads, normalized
# fingertip distances) rather than raw landmark coordinates. Shape descriptors
# are scale-, translation- and rotation-invariant — and, being built from
# unsigned angles and distances, identical for a left vs. right hand forming
# the same shape — so they transfer across different signers and cameras far
# better than coordinates. This is what makes cross-person matching possible.
TEMPLATE_LEN = 32     # every sign (template or live segment) is resampled to this
SMOOTH_WIN   = 3      # moving-average window over raw frames (kills landmark jitter)
VEL_WEIGHT   = 8.0    # velocity contribution vs. position in the distance
ARM_WEIGHT   = 0.5    # arm/body-pose features weight relative to hand shape
DTW_BAND     = 8      # Sakoe-Chiba band half-width (frames)

# Query trim variants: live segmentation often captures the approach phase
# (hand raising into position) that the video templates don't include, so we
# also try matching with the start of the segment cut off.
QUERY_TRIMS = [(0.00, 1.00), (0.25, 1.00)]

HAND_SHAPE_DIM = 31   # 15 joint angles + 4 finger spreads + 12 distances
ARM_FEAT_DIM   = 6    # 2 elbow angles + 2 wrist positions (x,y) rel. to body
POS_DIM        = 2 * HAND_SHAPE_DIM + ARM_FEAT_DIM   # 68

# Joint-angle triplets (angle measured at the middle landmark): 3 per finger.
_TRIPLETS = [(0, 1, 2), (1, 2, 3), (2, 3, 4),          # thumb
             (0, 5, 6), (5, 6, 7), (6, 7, 8),          # index
             (0, 9, 10), (9, 10, 11), (10, 11, 12),    # middle
             (0, 13, 14), (13, 14, 15), (14, 15, 16),  # ring
             (0, 17, 18), (17, 18, 19), (18, 19, 20)]  # pinky
# Finger direction vectors (MCP -> tip) for inter-finger spread angles.
_FINGER_DIR = [(2, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
_SPREAD     = [(0, 1), (1, 2), (2, 3), (3, 4)]   # adjacent finger pairs
# Normalized distances: thumb-to-tips, adjacent tips, wrist-to-tips.
_DIST_PAIRS = [(4, 8), (4, 12), (4, 16), (4, 20),
               (8, 12), (12, 16), (16, 20),
               (0, 4), (0, 8), (0, 12), (0, 16), (0, 20)]


def _cos_at(p: np.ndarray, a: int, b: int, c: int) -> float:
    """Cosine of the angle at landmark b in triangle (a, b, c). 2D points."""
    v1, v2 = p[a] - p[b], p[c] - p[b]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))


def _hand_shape(p: np.ndarray) -> np.ndarray:
    """(21, 2) hand landmarks -> (HAND_SHAPE_DIM,) signer-invariant descriptor."""
    angles = [_cos_at(p, a, b, c) for a, b, c in _TRIPLETS]

    dirs = [p[t] - p[m] for m, t in _FINGER_DIR]
    spreads = []
    for i, j in _SPREAD:
        n1, n2 = np.linalg.norm(dirs[i]), np.linalg.norm(dirs[j])
        spreads.append(0.0 if n1 < 1e-6 or n2 < 1e-6
                       else float(np.clip(np.dot(dirs[i], dirs[j]) / (n1 * n2), -1.0, 1.0)))

    scale = np.linalg.norm(p[0] - p[9])          # wrist -> middle-MCP
    if scale < 1e-6:
        scale = 1.0
    dists = [float(np.linalg.norm(p[a] - p[b]) / scale) for a, b in _DIST_PAIRS]

    return np.array(angles + spreads + dists, dtype=np.float32)


def _arm_feat(pose: np.ndarray) -> np.ndarray:
    """(33, 3) pose -> (ARM_FEAT_DIM,): elbow angles + wrist positions relative
    to shoulder midpoint, normalized by shoulder width (so body-size invariant)."""
    p = pose[:, :2]
    sw = np.linalg.norm(p[11] - p[12])
    if sw < 1e-6:
        sw = 1.0
    mid  = (p[11] + p[12]) / 2
    cosL = _cos_at(p, 11, 13, 15)
    cosR = _cos_at(p, 12, 14, 16)
    wl   = (p[15] - mid) / sw
    wr   = (p[16] - mid) / sw
    return np.array([cosL, cosR, wl[0], wl[1], wr[0], wr[1]], dtype=np.float32)


def build_frame(hand_result, pose_result) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    One frame's feature from raw MediaPipe results.

    Returns (feature (POS_DIM,), wrist_xy (2,)) — wrist_xy is the mean raw
    wrist position of the detected hands (used for motion segmentation).
    Returns (None, None) if no pose or no hands were detected.

    Hand slot assignment: with two hands, sorted by image x (slot 0 =
    leftmost). With one hand, the slot is chosen by which side of the body
    midline (shoulder midpoint) the wrist is on. The other slot is zeros.
    """
    pose_raw = pose_landmarks_from_result(pose_result)
    if pose_raw is None or not hand_result.hand_landmarks:
        return None, None

    hands = [np.array([[lm.x, lm.y] for lm in h], dtype=np.float32)
             for h in hand_result.hand_landmarks[:2]]

    slot0 = slot1 = None
    if len(hands) == 2:
        hands.sort(key=lambda h: float(h[0, 0]))
        slot0, slot1 = hands
    else:
        mid_x = float(pose_raw[11, 0] + pose_raw[12, 0]) / 2
        if float(hands[0][0, 0]) < mid_x:
            slot0 = hands[0]
        else:
            slot1 = hands[0]

    def hs(h):
        return _hand_shape(h) if h is not None else np.zeros(HAND_SHAPE_DIM, dtype=np.float32)

    feature  = np.concatenate([hs(slot0), hs(slot1), _arm_feat(pose_raw) * ARM_WEIGHT])
    wrist_xy = np.mean([h[0] for h in hands], axis=0)
    return feature.astype(np.float32), wrist_xy


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Mirror left<->right to match an opposite-handed signer. Hand-shape
    descriptors are mirror-invariant, so this just swaps the two hand slots
    and the left/right arm features (negating the horizontal wrist offset).
    """
    out = seq.copy()
    H   = HAND_SHAPE_DIM

    slot0 = out[:, :H].copy()
    out[:, :H]      = out[:, H:2 * H]
    out[:, H:2 * H] = slot0

    a = 2 * H   # arm block: [cosL, cosR, dxL, dyL, dxR, dyR]
    cosL, cosR = out[:, a].copy(),     out[:, a + 1].copy()
    dxL,  dyL  = out[:, a + 2].copy(), out[:, a + 3].copy()
    dxR,  dyR  = out[:, a + 4].copy(), out[:, a + 5].copy()
    out[:, a],     out[:, a + 1] = cosR, cosL
    out[:, a + 2], out[:, a + 3] = -dxR, dyR
    out[:, a + 4], out[:, a + 5] = -dxL, dyL
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


# ── Offline sequence splitting ────────────────────────────────────────────────
# Used by sequence_demo.py: record a whole clip, then cut it into individual
# signs here, where the entire motion profile is visible at once.
# Thresholds are in SECONDS / per-second units so they hold at any frame rate;
# split_into_signs converts them to frame counts using the measured fps.
SEQ_MIN_GAP_S    = 0.25    # hand-absent duration that marks a between-sign boundary
SEQ_PAUSE_SPEED  = 0.18    # wrist speed (normalized units / second) below = "still"
SEQ_PAUSE_DUR_S  = 0.20    # still duration that splits one sign from the next
SEQ_MIN_SEG_S    = 0.25    # signs shorter than this are dropped as noise
_SPEED_EMA       = 0.4


def _ema_speed(wrists: list[np.ndarray], fps: float) -> np.ndarray:
    """Per-frame EMA-smoothed wrist speed in normalized-units-per-SECOND."""
    speeds = np.zeros(len(wrists), dtype=np.float32)
    ema = 0.0
    for i in range(1, len(wrists)):
        s   = float(np.linalg.norm(wrists[i] - wrists[i - 1])) * fps
        ema = _SPEED_EMA * s + (1 - _SPEED_EMA) * ema
        speeds[i] = ema
    return speeds


def _split_on_pauses(feats: list[np.ndarray], wrists: list[np.ndarray],
                     pause_speed: float, pause_frames: int,
                     fps: float) -> list[list[np.ndarray]]:
    """Split one hand-present block at low-motion pauses, dropping the still
    frames themselves (they're the gap between signs, not part of either)."""
    n = len(feats)
    low = _ema_speed(wrists, fps) < pause_speed

    is_pause = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if low[i]:
            j = i
            while j < n and low[j]:
                j += 1
            if j - i >= pause_frames:
                is_pause[i:j] = True
            i = j
        else:
            i += 1

    # Safety: if the entire block is low-motion (e.g. a held static letter with
    # a near-stationary wrist), pause-splitting would drop the whole thing.
    # Keep it as a single sign instead — never delete a letter via pauses.
    if is_pause.all():
        return [feats]

    segments, cur = [], []
    for i in range(n):
        if is_pause[i]:
            if cur:
                segments.append(cur)
                cur = []
        else:
            cur.append(feats[i])
    if cur:
        segments.append(cur)
    return segments if segments else [feats]


def split_into_signs(timeline: list,
                     fps: float,
                     min_gap_s: float = SEQ_MIN_GAP_S,
                     pause_speed: float = SEQ_PAUSE_SPEED,
                     pause_dur_s: float = SEQ_PAUSE_DUR_S,
                     min_seg_s: float = SEQ_MIN_SEG_S,
                     pause_split: bool = False,
                     debug: bool = False) -> list[np.ndarray]:
    """
    Cut a recorded clip into individual signs.

    timeline : list where each element is (feature, wrist_xy) for a frame with
               a detected hand, or None for a frame with no hand.
    fps      : measured processing frame rate, used to turn the second-based
               thresholds into frame counts (so they hold at any frame rate).

    Primary boundary is hand-absent gaps >= min_gap_s — reliable and the
    recommended way to separate letters. Pause-splitting (low-motion gaps
    within a block) is OFF by default: during fingerspelling the wrist holds
    still *within* a letter, so it shatters single letters into fragments.
    Enable it only for clearly motion-based signs. Returns one array per sign.
    """
    min_gap      = max(2, round(min_gap_s   * fps))
    pause_frames = max(2, round(pause_dur_s * fps))
    min_seg      = max(3, round(min_seg_s   * fps))

    blocks, cur_f, cur_w, absent = [], [], [], 0
    for item in timeline:
        if item is None:
            absent += 1
            if absent >= min_gap and cur_f:
                blocks.append((cur_f, cur_w))
                cur_f, cur_w = [], []
        else:
            absent = 0
            cur_f.append(item[0])
            cur_w.append(item[1])
    if cur_f:
        blocks.append((cur_f, cur_w))

    signs, dropped = [], 0
    for feats, wrists in blocks:
        parts = (_split_on_pauses(feats, wrists, pause_speed, pause_frames, fps)
                 if pause_split else [feats])
        for seg in parts:
            if len(seg) >= min_seg:
                signs.append(np.array(seg, dtype=np.float32))
            else:
                dropped += 1

    if debug:
        present = sum(1 for t in timeline if t is not None)
        mode = f"pause>={pause_frames}f" if pause_split else "pause=OFF"
        print(f"  [split] fps={fps:.1f}  frames={len(timeline)} "
              f"(present={present})  gap>={min_gap}f {mode} "
              f"min_seg>={min_seg}f  ->  blocks={len(blocks)} signs={len(signs)} "
              f"dropped_short={dropped}")
    return signs


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
