#!/usr/bin/env python3
"""Build the Ultralytics-format symlink view (Milestone 2, step 3 of 3).

Reads labels.csv + splits.json + frames/, materialises:

    yolo/
        stage1/{train,val,test}/{healthy,failure}/
        stage2/{train,val,test}/{bed_adhesion,spaghetti_or_shift,other}/

Each leaf entry is a symlink: yolo/.../<pid>_<frame>.jpg → frames/<pid>/<frame>.jpg

The `eval` split is intentionally NOT materialised here — it is curated
separately by scripts/curate_eval_set.py (hand-picked hard cases).

Frame-labeling rule (matches docs/DATASET_STRUCTURE.md):

    Success print, frame N         → stage1: healthy ;  stage2: <none>
    Failure print, N <  fail-AMB  → drop (conservative; pre-failure healthy)
    Failure print, fail-AMB ≤ N < fail → drop (ambiguity / "warning zone")
    Failure print, N ≥ fail        → stage1: failure ;  stage2: <merged class>

Where AMB = --ambiguity-layers (default 5).

Stage-2 class mapping (constant in this file — edit STAGE2_MAP to change):

    bed_adhesion → bed_adhesion
    spaghetti    → spaghetti_or_shift   ┐ merged per user decision
    layer_shift  → spaghetti_or_shift   ┘
    other        → other                (kept separate; 13 examples is small)

Usage:

    python scripts/build_yolo_views.py
    python scripts/build_yolo_views.py --wipe        # nuke yolo/ and rebuild
    python scripts/build_yolo_views.py --ambiguity-layers 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

# Edit this map to change the stage-2 class set.
STAGE2_MAP: dict[str, str] = {
    "bed_adhesion": "bed_adhesion",
    "spaghetti":    "spaghetti_or_shift",
    "layer_shift":  "spaghetti_or_shift",
    "other":        "other",
}
STAGE2_CLASSES = sorted(set(STAGE2_MAP.values()))


def default_link_mode() -> str:
    """symlink on POSIX, hardlink on Windows (no admin/dev-mode required)."""
    return "hardlink" if os.name == "nt" else "symlink"


def make_link(target: Path, link: Path, mode: str) -> None:
    """Idempotent link via the chosen strategy.

    - symlink:  cheap, relative; requires admin / Developer Mode on Windows
    - hardlink: identical to the source from a reader's POV; same volume only
    - copy:     duplicates bytes; portable; uses ~10 GB for this dataset
    """
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        return
    if mode == "symlink":
        link.symlink_to(os.path.relpath(target, link.parent))
    elif mode == "hardlink":
        os.link(target, link)
    elif mode == "copy":
        shutil.copy2(target, link)
    else:
        raise ValueError(f"unknown link mode: {mode}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels",     default=Path("labels.csv"),  type=Path)
    p.add_argument("--splits",     default=Path("splits.json"), type=Path)
    p.add_argument("--frames-dir", default=Path("frames"),      type=Path)
    p.add_argument("--out-dir",    default=Path("yolo"),        type=Path)
    p.add_argument("--ambiguity-layers", default=5, type=int)
    p.add_argument("--wipe", action="store_true",
                   help="delete out-dir before building")
    p.add_argument("--link-mode", default=default_link_mode(),
                   choices=("symlink", "hardlink", "copy"),
                   help="how to materialise yolo entries (default: symlink on "
                        "POSIX, hardlink on Windows)")
    args = p.parse_args()
    print(f"Link mode: {args.link_mode}")

    if args.wipe and args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    splits = json.loads(args.splits.read_text())
    # eval is excluded; curate_eval_set.py handles it
    pid_to_split: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for pid in splits[split_name]:
            pid_to_split[pid] = split_name

    with open(args.labels, encoding="utf-8") as f:
        rows = {r["print_id"]: r for r in csv.DictReader(f)}

    counts = {
        "stage1": {s: Counter() for s in ("train", "val", "test")},
        "stage2": {s: Counter() for s in ("train", "val", "test")},
    }
    frames_seen      = 0
    frames_linked    = 0
    missing_prints   = []
    bad_failure_type = Counter()

    for pid, split in pid_to_split.items():
        row = rows.get(pid)
        if not row:
            continue
        print_frames_dir = args.frames_dir / pid
        if not print_frames_dir.is_dir():
            missing_prints.append(pid)
            continue

        is_failure = (row["outcome"] == "failure")
        failure_at = int(row["failure_at_frame"]) if is_failure and row["failure_at_frame"] else None
        failure_type = (row.get("failure_type") or "").strip()
        s2_class = STAGE2_MAP.get(failure_type) if is_failure else None
        if is_failure and not s2_class:
            bad_failure_type[failure_type] += 1

        for frame_path in sorted(print_frames_dir.glob("*.jpg")):
            frames_seen += 1
            try:
                frame_idx = int(frame_path.stem)
            except ValueError:
                continue

            if not is_failure:
                # Healthy print: every frame → stage1 healthy
                link = args.out_dir / "stage1" / split / "healthy" / f"{pid}_{frame_idx:06d}.jpg"
                make_link(frame_path.resolve(), link, args.link_mode)
                counts["stage1"][split]["healthy"] += 1
                frames_linked += 1
                continue

            # Failure print: apply the rule
            if frame_idx < failure_at - args.ambiguity_layers:
                continue  # pre-failure: conservative drop
            if frame_idx < failure_at:
                continue  # ambiguity window: drop

            # frame_idx ≥ failure_at  → use for both stages
            link_name = f"{pid}_{frame_idx:06d}.jpg"

            s1_link = args.out_dir / "stage1" / split / "failure" / link_name
            make_link(frame_path.resolve(), s1_link, args.link_mode)
            counts["stage1"][split]["failure"] += 1
            frames_linked += 1

            if s2_class:
                s2_link = args.out_dir / "stage2" / split / s2_class / link_name
                make_link(frame_path.resolve(), s2_link, args.link_mode)
                counts["stage2"][split][s2_class] += 1

    # Report
    print(f"Frames seen:    {frames_seen}")
    print(f"Frames linked:  {frames_linked}")
    print(f"Frames dropped: {frames_seen - frames_linked}   "
          f"(pre-failure + ambiguity window across all failure prints)")
    print()
    if missing_prints:
        print(f"⚠ {len(missing_prints)} print(s) had no frames/ directory on disk:")
        for pid in missing_prints[:6]:
            print(f"    {pid}")
        if len(missing_prints) > 6:
            print(f"    ... and {len(missing_prints) - 6} more. Run extract_frames.py first.")
        print()
    if bad_failure_type:
        print(f"⚠ unrecognised failure_type values (not in STAGE2_MAP): {dict(bad_failure_type)}")
        print()

    print("Stage 1 (healthy / failure):")
    for split in ("train", "val", "test"):
        h = counts["stage1"][split]["healthy"]
        f_ = counts["stage1"][split]["failure"]
        print(f"  {split:6s}: {h:7d} healthy / {f_:7d} failure")
    print()

    print("Stage 2 (failure-only multi-class):")
    header = f"  {'split':<6}  " + "  ".join(f"{c:<20}" for c in STAGE2_CLASSES)
    print(header)
    for split in ("train", "val", "test"):
        line = f"  {split:<6}  " + "  ".join(
            f"{counts['stage2'][split][c]:<20}" for c in STAGE2_CLASSES
        )
        print(line)
    print()

    print(f"YOLO view → {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
