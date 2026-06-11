"""
Shared utilities — imported by every script in the project.
Keeping normalization in one place ensures training and live demo
are always identical.
"""

import numpy as np


# ── Hand landmark normalization ───────────────────────────────────────────────

# MediaPipe hand landmark indices
WRIST        = 0
MIDDLE_MCP   = 9   # base of middle finger — used as scale reference


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Make hand landmarks position- and scale-invariant.

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


# ── Arm / pose landmark normalization ─────────────────────────────────────────

# Upper-body pose landmark indices (MediaPipe 33-point convention)
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
ARM_INDICES     = [11, 12, 13, 14, 15, 16]   # shoulders, elbows, wrists
ARM_DIM         = len(ARM_INDICES) * 3        # 18


def normalize_arm_landmarks(pose_landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize upper-body arm landmarks so they are position- and scale-invariant.

    Input:  (33, 3) full pose landmark array
    Output: (18,)   flattened normalized array

    Centering reference : midpoint of left + right shoulders
    Scale reference     : shoulder-to-shoulder distance
    """
    pose_landmarks = pose_landmarks.copy().astype(np.float32)
    arm = pose_landmarks[ARM_INDICES]           # (6, 3)

    shoulder_mid   = (pose_landmarks[_LEFT_SHOULDER] + pose_landmarks[_RIGHT_SHOULDER]) / 2
    arm           -= shoulder_mid

    shoulder_width = np.linalg.norm(
        pose_landmarks[_LEFT_SHOULDER] - pose_landmarks[_RIGHT_SHOULDER]
    )
    if shoulder_width > 1e-6:
        arm /= shoulder_width

    return arm.flatten()   # (18,)


def pose_landmarks_from_result(result) -> np.ndarray | None:
    """
    Extract a (33, 3) numpy array from a MediaPipe PoseLandmarker result.
    Returns None if no pose was detected.
    """
    if not result.pose_landmarks:
        return None
    pose = result.pose_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in pose], dtype=np.float32)
