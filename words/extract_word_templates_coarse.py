"""
Stage-1 baseline, part 1 — batch-extract COARSE-feature templates for the
WORD videos, for the AUTSL-trained encoder pathway only.

The AUTSL-trained encoder (notebooks/encoder_training.py) only ever sees the
10-point-per-hand coarse descriptor (dtw_common._hand_shape_coarse10), since
that's all AUTSL's released skeleton data has. To embed the Macedonian words
in the SAME space, they must go through the identical coarse descriptor —
hence this separate script (build_frame_coarse) instead of reusing
data/landmarks/word_templates.npz, which is featurized with the full
21-point descriptor for the DTW production pipeline and is NOT compatible
with the encoder's input dimension.

Usage:
    python words/extract_word_templates_coarse.py                # all word videos
    python words/extract_word_templates_coarse.py --limit 300    # quick subset first
"""
import sys
import argparse
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from alphabet.extract_templates import (        # reuse, don't reinvent
    make_hand_detector, make_pose_detector,
    extract_detected_frames, templates_from_frames, MIN_FRAMES,
)
from words.extract_word_templates import find_word_videos, VIDEO_ROOT

OUT_PATH = ROOT / "data" / "landmarks" / "word_templates_coarse.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N videos (0 = all)")
    args = ap.parse_args()

    videos = find_word_videos()
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        print(f"No word videos found under {VIDEO_ROOT}")
        return
    print(f"Found {len(videos)} word videos.", flush=True)

    full_index = {str(vp): i for i, vp in enumerate(find_word_videos())}

    templates, labels, vid_ids, trim_ids, topics, paths = [], [], [], [], [], []
    done_paths: set[str] = set()
    if OUT_PATH.exists():
        prev = np.load(OUT_PATH, allow_pickle=True)
        if "paths" in prev:
            templates = list(prev["templates"]); labels  = list(prev["labels"])
            vid_ids   = list(prev["vid_ids"]);   trim_ids = list(prev["trim_ids"])
            topics    = list(prev["topics"]);    paths    = list(prev["paths"])
            done_paths = set(str(p) for p in paths)
            print(f"Resuming: {len(done_paths)} videos already done.", flush=True)

    def checkpoint():
        if not templates:
            return
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(OUT_PATH,
                 templates=np.stack(templates).astype(np.float32),
                 labels=np.array(labels),
                 vid_ids=np.array(vid_ids, dtype=np.int32),
                 trim_ids=np.array(trim_ids, dtype=np.int32),
                 topics=np.array(topics),
                 paths=np.array(paths))

    hand_detector = make_hand_detector()
    pose_detector = make_pose_detector()

    failed, processed = [], 0
    todo = [vp for vp in videos if str(vp) not in done_paths]
    print(f"{len(todo)} videos left to process.\n", flush=True)
    for vp in todo:
        label, topic, vid = vp.stem.strip(), vp.parent.name, full_index[str(vp)]
        frames = extract_detected_frames(vp, hand_detector, pose_detector, coarse=True)
        tmpls  = templates_from_frames(frames) if len(frames) >= MIN_FRAMES else []
        if not tmpls:
            failed.append(label)
        for k, t in enumerate(tmpls):
            templates.append(t); labels.append(label); vid_ids.append(vid)
            trim_ids.append(k);  topics.append(topic); paths.append(str(vp))
        processed += 1
        if processed % 50 == 0 or processed == len(todo):
            checkpoint()
            print(f"  [{processed:>4}/{len(todo)}] "
                  f"{len(set(vid_ids))} videos, {len(templates)} templates "
                  f"(checkpointed)", flush=True)

    checkpoint()
    if not templates:
        print("No templates extracted.")
        return
    print(f"\nSaved {len(templates)} templates from "
          f"{len(set(vid_ids))} videos → {OUT_PATH}")
    if failed:
        print(f"{len(failed)} videos yielded no template "
              f"(too few detected frames): {failed[:10]}"
              f"{' ...' if len(failed) > 10 else ''}")


if __name__ == "__main__":
    main()
