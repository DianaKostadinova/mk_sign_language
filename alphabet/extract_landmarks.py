
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
from utils import normalize_landmarks, landmarks_from_result

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
VIDEO_DIR  = ROOT / "videos" / "азбука"
MODEL_PATH = ROOT / "data" / "hand_landmarker.task"
OUTPUT_DIR = ROOT / "data" / "landmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# How many frames per sign sequence to save
SEQ_LEN = 20

# Only use the middle 60% of the video — skips the approach/release phases
WINDOW_START = 0.20
WINDOW_END   = 0.80

# Sliding-window extraction: number of sequences to pull per video
SEQS_PER_VIDEO = 30
# Step size between window starts (in detected-frame indices)
WINDOW_STEP    = 3


# ── MediaPipe setup ───────────────────────────────────────────────────────────
def make_detector():
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ── Sequence extraction ───────────────────────────────────────────────────────
def extract_all_sequences(video_path: Path, detector) -> list[np.ndarray]:
    """
    Extract multiple fixed-length sequences from a single video using a
    sliding window over the detected frames.

    Returns a list of (SEQ_LEN, 63) arrays; empty list if detection fails.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Could not open {video_path.name}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame  = int(total_frames * WINDOW_START)
    end_frame    = int(total_frames * WINDOW_END)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    detected_frames = []
    for _ in range(end_frame - start_frame):
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result    = detector.detect(mp_image)
        raw       = landmarks_from_result(result)
        if raw is not None:
            detected_frames.append(normalize_landmarks(raw))

    cap.release()

    if len(detected_frames) < SEQ_LEN:
        print(f"  [WARN] Too few detected frames: {video_path.name} ({len(detected_frames)})")
        return []

    sequences = []
    start = 0
    while start + SEQ_LEN <= len(detected_frames) and len(sequences) < SEQS_PER_VIDEO:
        window = detected_frames[start : start + SEQ_LEN]
        sequences.append(np.array(window, dtype=np.float32))
        start += WINDOW_STEP

    return sequences


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    letter_videos = sorted([
        p for p in VIDEO_DIR.glob("*.mp4")
        if "– буква" in p.name
    ])

    if not letter_videos:
        print(f"No letter videos found in {VIDEO_DIR}")
        return

    print(f"Found {len(letter_videos)} letter videos  |  SEQ_LEN={SEQ_LEN}\n")

    detector  = make_detector()
    sequences = []
    labels    = []
    failed    = []

    for i, video_path in enumerate(letter_videos, 1):
        label = video_path.stem.split(" – ")[0].strip()
        seqs = extract_all_sequences(video_path, detector)
        print(f"[{i:02d}/{len(letter_videos)}] {label}  → {len(seqs)} sequences")
        if seqs:
            sequences.extend(seqs)
            labels.extend([label] * len(seqs))
        else:
            failed.append(label)

    sequences_arr = np.array(sequences, dtype=np.float32)  # (N, SEQ_LEN, 63)
    labels_arr    = np.array(labels)

    np.save(OUTPUT_DIR / "letters_seq.npy",    sequences_arr)
    np.save(OUTPUT_DIR / "letters_labels.npy", labels_arr)

    print(f"\nSaved {len(sequences)} sequences  shape={sequences_arr.shape}")
    print(f"Output → {OUTPUT_DIR}")

    if failed:
        print(f"\nFailed: {failed}")


if __name__ == "__main__":
    main()
