
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

# Allow imports from project root (utils.py)
sys.path.append(str(Path(__file__).parent.parent))
from utils import normalize_landmarks, landmarks_from_result

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
VIDEO_DIR    = ROOT / "videos" / "азбука"
MODEL_PATH   = ROOT / "data" / "hand_landmarker.task"
OUTPUT_DIR   = ROOT / "data" / "landmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── MediaPipe setup ───────────────────────────────────────────────────────────
def make_detector() -> mp_vision.HandLandmarker:
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ── Best-frame selection ──────────────────────────────────────────────────────
def best_frame_landmarks(video_path: Path, detector: mp_vision.HandLandmarker) -> np.ndarray | None:

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Could not open {video_path.name}")
        return None

    best_landmarks = None
    best_score     = -1.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # MediaPipe expects RGB, OpenCV gives BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result    = detector.detect(mp_image)

        raw = landmarks_from_result(result)
        if raw is None:
            continue   # no hand detected in this frame

        # Use handedness confidence as the quality score
        score = result.handedness[0][0].score if result.handedness else 0.0

        if score > best_score:
            best_score     = score
            best_landmarks = raw

    cap.release()

    if best_landmarks is None:
        print(f"  [WARN] No hand detected in any frame: {video_path.name}")
        return None

    return normalize_landmarks(best_landmarks)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Filter: only files with "– буква" in the name
    letter_videos = sorted([
        p for p in VIDEO_DIR.glob("*.mp4")
        if "– буква" in p.name
    ])

    if not letter_videos:
        print(f"No letter videos found in {VIDEO_DIR}")
        print("Make sure the scraper has downloaded the азбука category.")
        return

    print(f"Found {len(letter_videos)} letter videos\n")

    detector   = make_detector()
    landmarks  = []
    labels     = []
    failed     = []

    for i, video_path in enumerate(letter_videos, 1):
        # Label = just the letter, e.g. "А – буква.mp4" → "А"
        label = video_path.stem.split(" – ")[0].strip()
        print(f"[{i:02d}/{len(letter_videos)}] {label:3s}  ({video_path.name})")

        result = best_frame_landmarks(video_path, detector)

        if result is not None:
            landmarks.append(result)
            labels.append(label)
        else:
            failed.append(label)

    # ── Save ──────────────────────────────────────────────────────────────────
    landmarks_arr = np.array(landmarks, dtype=np.float32)   # (N, 63)
    labels_arr    = np.array(labels)                         # (N,)

    out_landmarks = OUTPUT_DIR / "letters.npy"
    out_labels    = OUTPUT_DIR / "letters_labels.npy"

    np.save(out_landmarks, landmarks_arr)
    np.save(out_labels,    labels_arr)

    print(f"\nSaved {len(landmarks)} letters  →  {out_landmarks}")
    print(f"Saved labels              →  {out_labels}")
    print(f"Array shape: {landmarks_arr.shape}")

    if failed:
        print(f"\nFailed ({len(failed)} letters — no hand detected):")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
