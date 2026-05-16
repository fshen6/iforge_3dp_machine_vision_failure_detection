#!/usr/bin/env python3
"""Frame extraction (Milestone 2, step 1 of 3).

Decodes every raw timelapse to per-print JPEG frames. One JPEG per video
frame (= one printed layer, by convention). Frames are center-cropped to
square and resized to 256x256, saved as JPEG quality 90.

Idempotent: a print that already has the right number of frames on disk
is skipped. Safe to re-run after adding new timelapses to the dataset.

Layout produced:

    frames/
        print_0001/
            000000.jpg
            000001.jpg
            ...
        print_0002/
            ...

Uses ffmpeg via subprocess (must be on PATH). Parallelised across CPUs.

Usage:

    python scripts/extract_frames.py --videos-root "3dp computer vision training data"

Add --workers N to control parallelism (default: half of CPU count).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def extract_one(args: tuple[str, Path, Path, int, int]) -> tuple[str, int, str]:
    """Decode video → JPEG frames. Returns (print_id, frames_written, status).

    Idempotent: if the count of .jpg files in out_dir matches the video's
    frame count (probed via ffprobe), skip without re-decoding.
    """
    print_id, src, out_dir, size, quality = args

    if not src.exists():
        return (print_id, 0, f"MISSING: {src}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Cheap idempotency check: existing frame count vs ffprobed frame count.
    existing = sorted(out_dir.glob("*.jpg"))
    if existing:
        try:
            n = int(subprocess.check_output(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-count_packets", "-show_entries", "stream=nb_read_packets",
                 "-of", "csv=p=0", str(src)],
                stderr=subprocess.DEVNULL, timeout=60).decode().strip())
            if n == len(existing):
                return (print_id, 0, "skip-cached")
        except Exception:
            pass  # fall through, re-decode

    # Wipe out_dir and re-decode (handles partial previous runs)
    for f in existing:
        f.unlink()

    # ffmpeg filter: center crop to square, then scale to size×size
    vf = f"crop='min(iw,ih)':'min(iw,ih)':(iw-min(iw\\,ih))/2:(ih-min(iw\\,ih))/2,scale={size}:{size}"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-q:v", str(_jpeg_q_for_quality(quality)),
        "-start_number", "0",
        str(out_dir / "%06d.jpg"),
    ]
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, timeout=600)
    except subprocess.CalledProcessError as e:
        return (print_id, 0, f"FFMPEG_ERROR: {e.stderr.decode()[:200]}")
    except subprocess.TimeoutExpired:
        return (print_id, 0, "FFMPEG_TIMEOUT")

    written = len(list(out_dir.glob("*.jpg")))
    return (print_id, written, "ok")


def _jpeg_q_for_quality(quality_0_100: int) -> int:
    """ffmpeg uses 1-31 (1=best, 31=worst). Map 0-100 → 31..1."""
    return max(1, min(31, round(31 - (quality_0_100 / 100) * 30)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=Path("labels.csv"), type=Path)
    p.add_argument("--videos-root", required=True, type=Path,
                   help="Local path containing raw/successful/... and raw/failed/...")
    p.add_argument("--frames-dir", default=Path("frames"), type=Path)
    p.add_argument("--size", default=256, type=int)
    p.add_argument("--quality", default=90, type=int)
    p.add_argument("--workers", default=max(1, (os.cpu_count() or 4) // 2), type=int)
    args = p.parse_args()

    with open(args.labels, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    tasks = []
    for r in rows:
        # source_file is "raw/successful/Tolstoy/foo.mp4" → join with --videos-root
        src = args.videos_root / r["source_file"]
        out_dir = args.frames_dir / r["print_id"]
        tasks.append((r["print_id"], src, out_dir, args.size, args.quality))

    print(f"Extracting frames for {len(tasks)} prints "
          f"using {args.workers} workers...", flush=True)

    ok = 0
    skipped = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(extract_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            pid, n, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip-cached":
                skipped += 1
            else:
                failed += 1
                print(f"  [{pid}] {status}")
            if i % 50 == 0:
                print(f"  progress: {i}/{len(tasks)}  "
                      f"(ok={ok} skipped={skipped} failed={failed})", flush=True)

    print()
    print(f"Done. ok={ok}  skipped(cached)={skipped}  failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
