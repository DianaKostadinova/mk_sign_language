
"""
Record your own webcam templates for the DTW matcher.

The reference videos are a different signer, camera, and framing than your
webcam — adding even one or two takes of yourself per letter usually improves
live accuracy more than any tuning. Recorded takes are saved to
data/landmarks/user_templates.npz and loaded automatically by dtw_demo.py.

Controls:
  SPACE  start / stop recording a take of the current letter
  D      delete the takes recorded for the current letter (this session)
  N / B  next / previous letter
  Q      save everything and quit
"""

import sys
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(Path(__file__).parent.parent))
from alphabet.dtw_common import build_frame, make_template

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent.parent
HAND_MODEL_PATH = ROOT / "data" / "hand_landmarker.task"
POSE_MODEL_PATH = ROOT / "data" / "pose_landmarker.task"
REF_TEMPLATES   = ROOT / "data" / "landmarks" / "dtw_templates.npz"
USER_TEMPLATES  = ROOT / "data" / "landmarks" / "user_templates.npz"

MIN_TAKE_FRAMES = 10


def make_hand_detector():
    base_options = mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def make_pose_detector():
    base_options = mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


_FONT_CACHE: dict[int, object] = {}


def _get_font(size: int):
    if size not in _FONT_CACHE:
        font = None
        for path in ["arial.ttf", "C:/Windows/Fonts/arial.ttf",
                     "C:/Windows/Fonts/arialuni.ttf", "C:/Windows/Fonts/seguiui.ttf"]:
            try:
                font = ImageFont.truetype(path, size)
                break
            except (IOError, OSError):
                continue
        _FONT_CACHE[size] = font or ImageFont.load_default()
    return _FONT_CACHE[size]


def draw_cyrillic(frame, texts):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    for text, pos, size, color in texts:
        draw.text(pos, text, font=_get_font(size), fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    if not REF_TEMPLATES.exists():
        print(f"Run extract_templates.py first ({REF_TEMPLATES} not found).")
        return
    letters = sorted(set(np.load(REF_TEMPLATES, allow_pickle=True)["labels"]))

    # Existing user takes get extended, not overwritten
    user_templates: list[np.ndarray] = []
    user_labels:    list[str] = []
    if USER_TEMPLATES.exists():
        prev = np.load(USER_TEMPLATES, allow_pickle=True)
        user_templates = list(prev["templates"])
        user_labels    = list(prev["labels"])
        print(f"Loaded {len(user_labels)} existing user takes")

    hand_detector = make_hand_detector()
    pose_detector = make_pose_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    idx          = 0
    recording    = False
    take_frames: list[np.ndarray] = []
    timestamp_ms = 0
    flash_msg, flash_until = "", 0.0

    import time
    print("SPACE record/stop · N next · B back · D delete letter takes · Q save+quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        timestamp_ms += 33

        frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        feature, _ = build_frame(hand_detector.detect_for_video(mp_image, timestamp_ms),
                                 pose_detector.detect_for_video(mp_image, timestamp_ms))

        detected = feature is not None
        if detected and recording:
            take_frames.append(feature)

        letter = letters[idx]
        n_takes = user_labels.count(letter)
        texts = [(f"{letter}", (30, 10), 90,
                  (255, 80, 80) if recording else (0, 255, 0)),
                 (f"{idx + 1}/{len(letters)}   takes: {n_takes}", (30, 115), 26,
                  (255, 255, 255))]
        if recording:
            texts.append((f"● REC  {len(take_frames)} frames", (30, 150), 26, (255, 80, 80)))
        elif not detected:
            texts.append(("no hand detected", (30, 150), 26, (255, 0, 0)))
        if time.time() < flash_until:
            texts.append((flash_msg, (30, 185), 26, (0, 200, 255)))
        frame = draw_cyrillic(frame, texts)

        cv2.imshow("Record templates", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if not recording:
                recording, take_frames = True, []
            else:
                recording = False
                if len(take_frames) >= MIN_TAKE_FRAMES:
                    user_templates.append(make_template(np.array(take_frames, np.float32)))
                    user_labels.append(letter)
                    flash_msg, flash_until = f"saved take ({len(take_frames)} frames)", time.time() + 2
                else:
                    flash_msg, flash_until = "too short, discarded", time.time() + 2
        elif key == ord("n"):
            idx, recording = (idx + 1) % len(letters), False
        elif key == ord("b"):
            idx, recording = (idx - 1) % len(letters), False
        elif key == ord("d"):
            keep = [(t, l) for t, l in zip(user_templates, user_labels) if l != letter]
            user_templates = [t for t, _ in keep]
            user_labels    = [l for _, l in keep]
            flash_msg, flash_until = f"deleted takes for {letter}", time.time() + 2
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if user_templates:
        np.savez(USER_TEMPLATES,
                 templates=np.stack(user_templates).astype(np.float32),
                 labels=np.array(user_labels))
        print(f"Saved {len(user_labels)} takes for "
              f"{len(set(user_labels))} letters → {USER_TEMPLATES}")
    else:
        print("No takes recorded.")


if __name__ == "__main__":
    main()
