
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.append(str(Path(__file__).parent.parent))
from alphabet.dtw_common import build_frame, make_template, TEMPLATE_LEN

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent.parent
VIDEO_DIR       = ROOT / "videos" / "азбука"
HAND_MODEL_PATH = ROOT / "data" / "hand_landmarker.task"
POSE_MODEL_PATH = ROOT / "data" / "pose_landmarker.task"
OUTPUT_DIR      = ROOT / "data" / "landmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only use the middle of the video — skips the approach/release phases
WINDOW_START = 0.15
WINDOW_END   = 0.85

# Minimum detected frames needed to build a template at all
MIN_FRAMES = 12

# Each video yields several templates: the full detected window plus trimmed
# variants, so live segmentation that clips a sign slightly early/late still
# finds a close match.
TRIM_VARIANTS = [(0.00, 1.00), (0.10, 0.90), (0.00, 0.80), (0.20, 1.00)]


# ── MediaPipe setup ───────────────────────────────────────────────────────────
def make_hand_detector():
    base_options = mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def make_pose_detector():
    base_options = mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


# ── Per-video extraction ──────────────────────────────────────────────────────
def extract_detected_frames(video_path: Path, hand_detector, pose_detector) -> np.ndarray:
    """All detected position-feature frames in the middle window. (T, POS_DIM)"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Could not open {video_path.name}")
        return np.empty((0,))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame  = int(total_frames * WINDOW_START)
    end_frame    = int(total_frames * WINDOW_END)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(end_frame - start_frame):
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        feature, _ = build_frame(hand_detector.detect(mp_image),
                                 pose_detector.detect(mp_image))
        if feature is not None:
            frames.append(feature)

    cap.release()
    return np.array(frames, dtype=np.float32)


def templates_from_frames(frames: np.ndarray) -> list[np.ndarray]:
    """Build the trim-variant templates for one video's detected frames."""
    T = len(frames)
    templates = []
    for lo, hi in TRIM_VARIANTS:
        a, b = int(T * lo), int(T * hi)
        if b - a >= MIN_FRAMES:
            templates.append(make_template(frames[a:b]))
    return templates


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    letter_videos = sorted([
        p for p in VIDEO_DIR.glob("*.mp4")
        if "– буква" in p.name
    ])
    if not letter_videos:
        print(f"No letter videos found in {VIDEO_DIR}")
        return

    print(f"Found {len(letter_videos)} letter videos  |  TEMPLATE_LEN={TEMPLATE_LEN}\n")

    hand_detector = make_hand_detector()
    pose_detector = make_pose_detector()
    all_templates, all_labels, failed = [], [], []

    for i, video_path in enumerate(letter_videos, 1):
        label  = video_path.stem.split(" – ")[0].strip()
        frames = extract_detected_frames(video_path, hand_detector, pose_detector)
        templates = templates_from_frames(frames) if len(frames) >= MIN_FRAMES else []
        print(f"[{i:02d}/{len(letter_videos)}] {label}  → "
              f"{len(frames)} frames, {len(templates)} templates")
        if templates:
            all_templates.extend(templates)
            all_labels.extend([label] * len(templates))
        else:
            failed.append(label)

    templates_arr = np.stack(all_templates).astype(np.float32)  # (K, TEMPLATE_LEN, D*2)
    labels_arr    = np.array(all_labels)

    out_path = OUTPUT_DIR / "dtw_templates.npz"
    np.savez(out_path, templates=templates_arr, labels=labels_arr)
    print(f"\nSaved {len(all_templates)} templates for "
          f"{len(set(all_labels))} letters  shape={templates_arr.shape}")
    print(f"Output → {out_path}")
    if failed:
        print(f"\nFailed: {failed}")


if __name__ == "__main__":
    main()
