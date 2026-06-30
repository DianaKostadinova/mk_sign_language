"""
Stage-0 baseline, part 1/2 — batch-extract DTW templates for the WORD videos.

Reuses the EXACT alphabet pipeline (alphabet/extract_templates.py): same
MediaPipe detectors, same signer-invariant shape features, same trim-variant
templates. The only differences here are:
  • we walk every topic folder under videos/ (not just азбука),
  • the label is the video's file name (the word),
  • we remember which video + which trim each template came from, so the
    evaluator can do a fair query-vs-gallery split.

The MediaPipe pass over ~2400 videos is the slow part, so its output is
cached to data/landmarks/word_templates.npz. Run this once; iterate on the
evaluator (eval_dtw_baseline.py) as much as you like without re-extracting.

Usage:
    python words/extract_word_templates.py                # all word videos
    python words/extract_word_templates.py --limit 300    # quick subset first
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

VIDEO_ROOT = ROOT / "videos"
OUT_PATH   = ROOT / "data" / "landmarks" / "word_templates.npz"

# Folders that are NOT words (letters / test clips / empty)
SKIP_DIRS = {"азбука", "test-znaci", "misc"}


def find_word_videos() -> list[Path]:
    vids = []
    for folder in sorted(p for p in VIDEO_ROOT.iterdir() if p.is_dir()):
        if folder.name in SKIP_DIRS:
            continue
        vids.extend(sorted(folder.glob("*.mp4")))
    return vids


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

    # vid_id is the video's index in the full sorted list — stable across runs,
    # so resuming keeps the same grouping the evaluator relies on.
    full_index = {str(vp): i for i, vp in enumerate(find_word_videos())}

    # Resume: load any existing cache and skip videos already processed (by path).
    templates, labels, vid_ids, trim_ids, topics, paths = [], [], [], [], [], []
    done_paths: set[str] = set()
    if OUT_PATH.exists():
        prev = np.load(OUT_PATH, allow_pickle=True)
        if "paths" in prev:                       # only resume from a v2 cache
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
        frames = extract_detected_frames(vp, hand_detector, pose_detector)
        tmpls  = templates_from_frames(frames) if len(frames) >= MIN_FRAMES else []
        if not tmpls:
            failed.append(label)
        for k, t in enumerate(tmpls):
            templates.append(t); labels.append(label); vid_ids.append(vid)
            trim_ids.append(k);  topics.append(topic); paths.append(str(vp))
        processed += 1
        if processed % 50 == 0 or processed == len(todo):
            checkpoint()                          # crash-safe periodic save
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
