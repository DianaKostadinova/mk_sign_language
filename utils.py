"""
Shared utilities — imported by every script in the project.
Keeping normalization in one place ensures training and live demo
are always identical.
"""

import numpy as np


# ── Landmark normalization ────────────────────────────────────────────────────

# MediaPipe hand landmark indices
WRIST        = 0
MIDDLE_MCP   = 9   # base of middle finger — used as scale reference


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Make landmarks position- and scale-invariant.

    Input:  (21, 3) array of raw (x, y, z) landmark coordinates
    Output: (63,)   flattened normalized array

    Steps:
      1. Translate so wrist (landmark 0) is at the origin
      2. Scale so the wrist-to-middle-MCP distance = 1
      3. Flatten to a 1D vector
    """
    landmarks = landmarks.copy().astype(np.float32)

    # 1. Translate — subtract wrist from every point
    landmarks -= landmarks[WRIST]

    # 2. Scale — divide by wrist-to-middle-finger-base distance
    scale = np.linalg.norm(landmarks[MIDDLE_MCP])
    if scale > 1e-6:          # avoid division by zero if hand not visible
        landmarks /= scale

    # 3. Flatten: (21, 3) → (63,)
    return landmarks.flatten()


def landmarks_from_result(result) -> np.ndarray | None:
    """
    Extract a (21, 3) numpy array from a MediaPipe HandLandmarker result.
    If multiple hands are detected, picks the one with the highest confidence.
    Returns None if no hand was detected.
    """
    if not result.hand_landmarks:
        return None

    if len(result.hand_landmarks) == 1:
        hand = result.hand_landmarks[0]
    else:
        # Pick the hand with the highest handedness score
        scores = [result.handedness[i][0].score for i in range(len(result.hand_landmarks))]
        hand   = result.hand_landmarks[int(np.argmax(scores))]

    return np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
