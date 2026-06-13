
"""
Sequence translator for the DTW template matcher.

Not a live per-frame translator. Instead:

  1. Raise your hand — recording starts.
  2. Sign a letter, then either remove your hand briefly OR pause in place,
     then sign the next letter. Either gap marks a boundary.
  3. When you're done, take your hand out of frame and leave it out for ~2s.
     Recording stops, the whole clip is split into individual signs, each is
     matched, and the recognized string is shown.

Because the entire clip is processed at once (not frame-by-frame in real time),
segmentation is far more reliable than the live demo.

Run extract_templates.py first. Press Q to quit, R to reset the current take.
"""

import sys
import time
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
from alphabet.dtw_common import build_frame, classify, split_into_signs

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent.parent
HAND_MODEL_PATH = ROOT / "data" / "hand_landmarker.task"
POSE_MODEL_PATH = ROOT / "data" / "pose_landmarker.task"
TEMPLATES_PATH  = ROOT / "data" / "landmarks" / "dtw_templates.npz"
USER_TEMPLATES  = ROOT / "data" / "landmarks" / "user_templates.npz"

# ── Recording control ─────────────────────────────────────────────────────────
STOP_ABSENT_S = 1.8    # hand gone this long (seconds) ends recording + translates
MARGIN_OK     = 1.10   # below this, the letter is shown but flagged uncertain


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


CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]


def draw_skeleton(frame, pts):
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)


def process_take(timeline, fps, templates, labels):
    """Split a recorded timeline into signs, classify each. Returns
    (result_string, per_sign list of (label, margin))."""
    print(f"\n--- processing take ---")
    signs = split_into_signs(timeline, fps, debug=True)
    result, details = [], []
    for i, seg in enumerate(signs, 1):
        ranked, margin = classify(seg, templates, labels)
        label = ranked[0][0]
        result.append(label)
        details.append((label, margin))
        flag = "" if margin >= MARGIN_OK else "  (uncertain)"
        top  = "  ".join(f"{l}={d:.2f}" for l, d in ranked)
        print(f"  sign {i:2d} ({len(seg):3d} frames): {top}  margin={margin:.2f}{flag}")
    print(f"--- result: {''.join(result)} ---\n")
    return "".join(result), details


def main():
    if not TEMPLATES_PATH.exists():
        print(f"Run extract_templates.py first ({TEMPLATES_PATH} not found).")
        return
    # The reference-video templates are a different signer/camera and do not
    # match a webcam well (proven: they self-match perfectly but fail live).
    # So once you've recorded your own takes, match against ONLY those — they
    # are in-domain ground truth. Reference templates are the fallback.
    if USER_TEMPLATES.exists():
        user      = np.load(USER_TEMPLATES, allow_pickle=True)
        templates = user["templates"]
        labels    = user["labels"]
        print(f"Matching against YOUR {len(labels)} recorded takes "
              f"({len(set(labels))} letters). Reference videos ignored.")
        print("Letters you haven't recorded yet cannot be predicted.")
    else:
        data      = np.load(TEMPLATES_PATH, allow_pickle=True)
        templates = data["templates"]
        labels    = data["labels"]
        print(f"Matching against {len(templates)} REFERENCE templates "
              f"({len(set(labels))} letters).")
        print("These are a different signer — expect poor webcam accuracy.")
        print("Run record_templates.py to record your own; then this matches those.")

    hand_detector = make_hand_detector()
    pose_detector = make_pose_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Raise your hand to start recording.")
    print("BETWEEN LETTERS: take your hand out of frame briefly, then bring it")
    print("  back for the next letter. (A still pause works too, but hand-out is")
    print("  far more reliable for fingerspelling.)")
    print(f"TO FINISH: take your hand out and leave it out for ~{STOP_ABSENT_S:.0f}s.")
    print("Q quits, R resets the current take.\n")

    timeline: list = []
    frame_times: list = []      # wall-clock time of each recorded frame
    recording    = False
    absent_secs  = 0.0
    last_result  = ""
    last_details: list = []
    timestamp_ms = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now   = time.time()
        frame = cv2.flip(frame, 1)
        timestamp_ms += 33

        frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)

        feature, wrist_xy = build_frame(hand_result, pose_result)
        present = feature is not None

        hand_pts = []
        if present:
            h, w = frame.shape[:2]
            hand_pts = [[(int(lm.x * w), int(lm.y * h)) for lm in hand]
                        for hand in hand_result.hand_landmarks[:2]]

        # ── Recording state machine ─────────────────────────────────────────
        if not recording:
            if present:
                recording   = True
                timeline    = [(feature, wrist_xy)]
                frame_times = [now]
                absent_secs = 0.0
        else:
            timeline.append((feature, wrist_xy) if present else None)
            frame_times.append(now)
            if present:
                absent_secs = 0.0
            else:
                absent_secs += now - frame_times[-2]
                if absent_secs >= STOP_ABSENT_S:
                    span = frame_times[-1] - frame_times[0]
                    fps  = (len(frame_times) - 1) / span if span > 0 else 15.0
                    last_result, last_details = process_take(
                        timeline, fps, templates, labels)
                    recording, timeline, frame_times, absent_secs = False, [], [], 0.0

        # ── Overlay ──────────────────────────────────────────────────────────
        for pts in hand_pts:
            draw_skeleton(frame, pts)

        if recording:
            n_present = sum(1 for t in timeline if t is not None)
            if present:
                status, color = f"* RECORDING  {n_present} frames", (80, 80, 255)
            else:
                left = max(0.0, STOP_ABSENT_S - absent_secs)
                status, color = f"hand out - finishing in {left:.1f}s...", (0, 180, 255)
            cv2.putText(frame, status, (30, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, "raise hand to start", (30, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        texts = []
        if last_result:
            texts.append((last_result, (30, 15), 70, (0, 255, 0)))
            uncertain = [l for l, m in last_details if m < MARGIN_OK]
            if uncertain:
                texts.append(("uncertain: " + " ".join(uncertain), (30, 100), 22,
                              (0, 160, 255)))
        if texts:
            frame = draw_cyrillic(frame, texts)

        cv2.imshow("MK Sign Language — Sequence (DTW)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            recording, timeline, frame_times, absent_secs = False, [], [], 0.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
