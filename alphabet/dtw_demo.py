
"""
Live webcam demo for the DTW template matcher.

Instead of classifying a rolling window every frame, this segments signs
by hand motion:

  1. Hand appears and starts moving  -> start recording the segment
  2. Hand goes still (or disappears) -> segment ends
  3. The segment is resampled and matched against the reference
     templates with DTW; the closest letter is shown.

Run extract_templates.py first to build data/landmarks/dtw_templates.npz.
Press Q to quit.
"""

import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(Path(__file__).parent.parent))
from utils import (normalize_landmarks, landmarks_from_result,
                   normalize_arm_landmarks, pose_landmarks_from_result)
from alphabet.dtw_common import frame_feature, classify

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent.parent
HAND_MODEL_PATH = ROOT / "data" / "hand_landmarker.task"
POSE_MODEL_PATH = ROOT / "data" / "pose_landmarker.task"
TEMPLATES_PATH  = ROOT / "data" / "landmarks" / "dtw_templates.npz"

# ── Segmentation tuning ───────────────────────────────────────────────────────
# Speed = wrist movement per frame in normalized image coords, EMA-smoothed.
START_SPEED    = 0.015   # moving faster than this starts a segment
END_SPEED      = 0.008   # slower than this counts as "still"
END_HOLD       = 10      # consecutive still frames that end a segment
PRE_ROLL       = 5       # frames kept from just before motion started
MIN_SEG_FRAMES = 12      # shorter segments are discarded as noise
MAX_SEG_FRAMES = 150     # safety cap (~5 s) — classify and reset
EMA_ALPHA      = 0.4

# ── Matching tuning ───────────────────────────────────────────────────────────
MARGIN_OK    = 1.10     # 2nd-best class must be 10%+ farther to count as confident
RESULT_HOLD  = 3.0      # seconds the last prediction stays on screen


# ── MediaPipe detectors ───────────────────────────────────────────────────────
def make_hand_detector():
    base_options = mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def make_pose_detector():
    base_options = mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


# ── Drawing helpers ───────────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm
]


def _find_font(size: int):
    candidates = [
        "arial.ttf", "arialuni.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialuni.ttf",
        "C:/Windows/Fonts/seguiui.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def put_text_unicode(frame, text, pos, size, color):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img_pil).text(pos, text, font=_find_font(size), fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_skeleton(frame, pts):
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)


# ── Segment recorder ──────────────────────────────────────────────────────────
class SignSegmenter:
    """Speed-based start/stop detection for one sign performance."""

    def __init__(self):
        self.pre_roll   = deque(maxlen=PRE_ROLL)
        self.segment    = []
        self.recording  = False
        self.still_run  = 0
        self.ema_speed  = 0.0
        self.prev_wrist = None

    def _update_speed(self, wrist_xy: np.ndarray) -> float:
        if self.prev_wrist is None:
            speed = 0.0
        else:
            speed = float(np.linalg.norm(wrist_xy - self.prev_wrist))
        self.prev_wrist = wrist_xy
        self.ema_speed  = EMA_ALPHA * speed + (1 - EMA_ALPHA) * self.ema_speed
        return self.ema_speed

    def _finish(self) -> np.ndarray | None:
        seg = self.segment
        self.segment, self.recording, self.still_run = [], False, 0
        if len(seg) >= MIN_SEG_FRAMES:
            return np.array(seg, dtype=np.float32)
        return None

    def update(self, feature: np.ndarray | None,
               wrist_xy: np.ndarray | None) -> np.ndarray | None:
        """
        Feed one frame. feature/wrist_xy are None when the hand wasn't detected.
        Returns a completed segment (T, POS_DIM) when one just ended, else None.
        """
        if feature is None:
            self.prev_wrist = None
            self.ema_speed  = 0.0
            self.pre_roll.clear()
            return self._finish() if self.recording else None

        speed = self._update_speed(wrist_xy)

        if not self.recording:
            self.pre_roll.append(feature)
            if speed > START_SPEED:
                self.segment   = list(self.pre_roll)
                self.recording = True
                self.still_run = 0
            return None

        self.segment.append(feature)

        if speed < END_SPEED:
            self.still_run += 1
        else:
            self.still_run = 0

        if self.still_run >= END_HOLD:
            # Drop most of the trailing still frames — they're the hold, not the sign
            self.segment = self.segment[:-(END_HOLD - 2)]
            return self._finish()

        if len(self.segment) >= MAX_SEG_FRAMES:
            return self._finish()

        return None


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    if not TEMPLATES_PATH.exists():
        print(f"Templates not found: {TEMPLATES_PATH}")
        print("Run extract_templates.py first.")
        return

    data      = np.load(TEMPLATES_PATH, allow_pickle=True)
    templates = data["templates"]
    labels    = data["labels"]
    print(f"Loaded {len(templates)} templates for {len(set(labels))} letters")

    hand_detector = make_hand_detector()
    pose_detector = make_pose_detector()
    segmenter     = SignSegmenter()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam open. Perform a sign, then hold still to classify. Press Q to quit.\n")

    last_label, last_margin, last_time = None, 0.0, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        hand_result = hand_detector.detect(mp_image)
        pose_result = pose_detector.detect(mp_image)
        hand_raw    = landmarks_from_result(hand_result)
        pose_raw    = pose_landmarks_from_result(pose_result)

        feature, wrist_xy, pts = None, None, None
        if hand_raw is not None and pose_raw is not None:
            feature  = frame_feature(normalize_landmarks(hand_raw),
                                     normalize_arm_landmarks(pose_raw))
            wrist_xy = hand_raw[0, :2]
            h, w = frame.shape[:2]
            pts  = [(int(lm.x * w), int(lm.y * h)) for lm in hand_result.hand_landmarks[0]]

        segment = segmenter.update(feature, wrist_xy)
        if segment is not None:
            label, dist, margin = classify(segment, templates, labels)
            last_label, last_margin, last_time = label, margin, time.time()
            print(f"Segment ({len(segment)} frames) → {label}  "
                  f"dist={dist:.3f}  margin={margin:.2f}")

        # ── Overlay ──────────────────────────────────────────────────────────
        if pts:
            draw_skeleton(frame, pts)

        if segmenter.recording:
            status, status_color = f"● recording ({len(segmenter.segment)})", (255, 80, 80)
        elif feature is not None:
            status, status_color = "ready — start signing", (200, 200, 200)
        else:
            status, status_color = "no hand detected", (0, 0, 255)
        frame = put_text_unicode(frame, status, (30, frame.shape[0] - 50), 26, status_color)

        if last_label is not None and time.time() - last_time < RESULT_HOLD:
            confident = last_margin >= MARGIN_OK
            color = (0, 255, 0) if confident else (0, 160, 255)
            frame = put_text_unicode(frame, last_label, (30, 20), 80, color)
            frame = put_text_unicode(frame, f"margin {last_margin:.2f}", (30, 115), 24,
                                     (255, 255, 255))
            if not confident:
                frame = put_text_unicode(frame, "(uncertain)", (170, 115), 24, color)

        cv2.imshow("MK Sign Language — Alphabet (DTW)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
