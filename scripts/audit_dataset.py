#!/usr/bin/env python3
"""Dataset audit (Milestone 1).

Walks raw/successful/ and raw/failed/ under --videos-root, parses filenames
into (print_id, printer_id, failure_type), ffprobes each file for duration
and resolution, and produces:

  1. labels.csv  — the dataset's single source of truth (idempotent: any
                   failure_at_seconds and started_at values already present
                   for a print_id are preserved on re-audit)
  2. DATASET_AUDIT.md — one-page human-readable summary
  3. stdout summary matching the format in docs/ENGINEERING_PLAN.md M1

Usage:
    python scripts/audit_dataset.py \\
        --videos-root /Volumes/External/print_data

Filename conventions (see docs/DATASET_STRUCTURE.md):
    raw/successful/print_001_printer03.mp4
    raw/failed/print_456_printer02_bed_adhesion.mp4

Acceptance: completes without error, prints per-class counts, surfaces any
class with < 30 examples (decision gate for whether stage 2 is viable).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable

CANONICAL_FAILURE_TYPES = ("bed_adhesion", "spaghetti", "layer_shift", "other")
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}

# print_001_printer03[_<failure_type>]  (printer_03 / printer03 both accepted)
FILENAME_RE = re.compile(
    r"^(?P<print_id>print_\d+)_printer_?(?P<printer_num>\d+)(?:_(?P<rest>.+))?$",
    re.IGNORECASE,
)

# Min examples per failure class before stage 2 is viable (engineering plan M1 gate).
MIN_CLASS_EXAMPLES = 30


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--videos-root",
        type=Path,
        required=True,
        help="Directory containing raw/successful/ and raw/failed/.",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Output path for labels.csv (default: <videos-root>/labels.csv).",
    )
    p.add_argument(
        "--audit-md",
        type=Path,
        default=repo_root / "docs" / "DATASET_AUDIT.md",
        help="Output path for the human-readable audit report.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel ffprobe workers (default: 8).",
    )
    p.add_argument("--ffprobe", default="ffprobe", help="Path to the ffprobe binary.")
    return p.parse_args()


def normalize_printer_id(num: str) -> str:
    return f"printer_{int(num):02d}"


def normalize_failure_type(raw: str | None) -> tuple[str, bool]:
    """Return (canonical_type, was_recognized). Unknown types fall back to 'other'."""
    if not raw:
        return ("other", False)
    s = raw.strip().lower()
    if s in CANONICAL_FAILURE_TYPES:
        return (s, True)
    if s == "other_failure":
        return ("other", True)
    return ("other", False)


def parse_filename(stem: str, outcome: str) -> tuple[str, str, str | None, list[str]]:
    """Parse a filename stem into (print_id, printer_id, failure_type, warnings)."""
    m = FILENAME_RE.match(stem)
    warnings: list[str] = []
    if not m:
        return ("", "", None, [f"could not parse filename: {stem!r}"])
    print_id = m.group("print_id").lower()
    printer_id = normalize_printer_id(m.group("printer_num"))
    rest = m.group("rest")
    if outcome == "success":
        if rest:
            warnings.append(f"{stem!r}: unexpected trailing tokens for success file: {rest!r}")
        return (print_id, printer_id, None, warnings)
    # failure
    failure_type, recognized = normalize_failure_type(rest)
    if not recognized:
        warnings.append(
            f"{stem!r}: unrecognized failure type {rest!r}, normalized to 'other'"
        )
    return (print_id, printer_id, failure_type, warnings)


def ffprobe(ffprobe_bin: str, path: Path) -> dict:
    """Return {duration, width, height, fps, creation_time} or {} on failure."""
    try:
        proc = subprocess.run(
            [
                ffprobe_bin,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                "-show_entries", "format=duration:format_tags=creation_time",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        return {"error": str(e)}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"ffprobe JSON parse failed: {e}"}
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    out = {}
    try:
        out["duration"] = float(fmt.get("duration", 0)) or None
    except (TypeError, ValueError):
        out["duration"] = None
    out["width"] = stream.get("width")
    out["height"] = stream.get("height")
    rfr = stream.get("r_frame_rate", "")
    try:
        if rfr and "/" in rfr:
            n, d = rfr.split("/", 1)
            out["fps"] = float(n) / float(d) if float(d) != 0 else None
        else:
            out["fps"] = float(rfr) if rfr else None
    except (TypeError, ValueError, ZeroDivisionError):
        out["fps"] = None
    try:
        out["nb_frames"] = int(stream.get("nb_frames")) if stream.get("nb_frames") else None
    except (TypeError, ValueError):
        out["nb_frames"] = None
    out["creation_time"] = (fmt.get("tags") or {}).get("creation_time")
    return out


def find_videos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def load_existing_labels(labels_path: Path) -> dict[str, dict]:
    if not labels_path.exists():
        return {}
    with labels_path.open(newline="") as f:
        return {r["print_id"]: r for r in csv.DictReader(f) if r.get("print_id")}


def save_labels_atomic(labels_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".labels.", suffix=".csv.tmp", dir=str(labels_path.parent))
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, labels_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def format_resolution(w: int | None, h: int | None) -> str:
    if not w or not h:
        return "unknown"
    return f"{w}x{h}"


def summarize(rows: list[dict]) -> dict:
    successful = [r for r in rows if r["outcome"] == "success"]
    failed = [r for r in rows if r["outcome"] == "failure"]
    failure_counts = Counter(r["failure_type"] for r in failed)
    printer_counts = Counter(r["printer_id"] for r in rows)
    printer_failure_counts = Counter(r["printer_id"] for r in failed)

    resolutions = Counter(r["_resolution"] for r in rows if r.get("_resolution"))
    fps_values = [r["_fps"] for r in rows if r.get("_fps")]
    seconds_per_frame_values = [1.0 / f for f in fps_values if f and f > 0]
    durations = [float(r["duration_seconds"]) for r in rows if r.get("duration_seconds")]

    return {
        "total_successful": len(successful),
        "total_failed": len(failed),
        "failure_counts": failure_counts,
        "printer_counts": printer_counts,
        "printer_failure_counts": printer_failure_counts,
        "resolutions": resolutions,
        "total": len(rows),
        "avg_seconds_per_frame": mean(seconds_per_frame_values) if seconds_per_frame_values else None,
        "avg_fps": mean(fps_values) if fps_values else None,
        "avg_duration_seconds": mean(durations) if durations else None,
        "total_duration_hours": sum(durations) / 3600 if durations else 0,
    }


def render_stdout_summary(s: dict) -> str:
    lines = []
    lines.append("=== DATASET AUDIT ===")
    lines.append(f"Total successful timelapses: {s['total_successful']}")
    lines.append(f"Total failed timelapses: {s['total_failed']}")
    for ft in CANONICAL_FAILURE_TYPES:
        lines.append(f"  - {ft}: {s['failure_counts'].get(ft, 0)}")
    lines.append("")
    if s["avg_seconds_per_frame"]:
        lines.append(
            f"Frame rate (avg per timelapse): 1 frame / {s['avg_seconds_per_frame']:.1f}s "
            f"(encoded fps avg: {s['avg_fps']:.2f})"
        )
    else:
        lines.append("Frame rate: unknown (ffprobe could not read fps)")
    if s["resolutions"]:
        total = sum(s["resolutions"].values())
        parts = []
        for res, n in s["resolutions"].most_common():
            parts.append(f"{res} ({n / total:.0%})")
        lines.append("Resolution: " + ", ".join(parts))
        if len(s["resolutions"]) > 1:
            lines.append("  -- mixed resolutions; normalize at decode")
    lines.append("")
    lines.append("Per-printer distribution:")
    if s["printer_counts"]:
        # one printer per line — easier to scan than a single comma-joined line
        for printer in sorted(s["printer_counts"]):
            total_n = s["printer_counts"][printer]
            failures_n = s["printer_failure_counts"].get(printer, 0)
            warn = "  ← NO FAILURES" if failures_n == 0 else ""
            lines.append(f"  {printer}: {total_n} prints ({failures_n} failures){warn}")
    lines.append("")
    if s["avg_duration_seconds"]:
        lines.append(
            f"Avg timelapse duration: {s['avg_duration_seconds']/60:.1f} min "
            f"(total archive: {s['total_duration_hours']:.1f} hours)"
        )
    lines.append("")
    lines.append("Failure timestamp precision: TBD — run M1.5 (label_failure_boundaries.py) next.")
    lines.append("")
    lines.append("Decision gates:")
    thin_classes = [ft for ft in CANONICAL_FAILURE_TYPES if s["failure_counts"].get(ft, 0) < MIN_CLASS_EXAMPLES]
    if thin_classes:
        lines.append(
            f"  ⚠ Classes below {MIN_CLASS_EXAMPLES} examples: {', '.join(thin_classes)} "
            f"— drop from stage 2 or merge into 'other'."
        )
    else:
        lines.append(f"  ✓ All failure classes have ≥ {MIN_CLASS_EXAMPLES} examples — stage 2 viable.")
    return "\n".join(lines)


def render_audit_md(s: dict, generated_at: str, videos_root: Path) -> str:
    lines = ["# Dataset Audit", ""]
    lines.append(f"_Generated {generated_at} from `{videos_root}` by `scripts/audit_dataset.py`._")
    lines.append("")
    lines.append("## Class counts")
    lines.append("")
    lines.append("| Class | Count |")
    lines.append("|---|---|")
    lines.append(f"| successful | {s['total_successful']} |")
    lines.append(f"| failed (all) | {s['total_failed']} |")
    for ft in CANONICAL_FAILURE_TYPES:
        lines.append(f"| failed / {ft} | {s['failure_counts'].get(ft, 0)} |")
    lines.append("")
    lines.append("## Video format")
    lines.append("")
    if s["resolutions"]:
        total = sum(s["resolutions"].values())
        lines.append("| Resolution | Count | Share |")
        lines.append("|---|---|---|")
        for res, n in s["resolutions"].most_common():
            lines.append(f"| {res} | {n} | {n / total:.0%} |")
        lines.append("")
    if s["avg_seconds_per_frame"]:
        lines.append(f"- Avg frame interval: **1 frame / {s['avg_seconds_per_frame']:.1f}s**")
        lines.append(f"- Avg encoded fps: {s['avg_fps']:.2f}")
    if s["avg_duration_seconds"]:
        lines.append(f"- Avg timelapse duration: **{s['avg_duration_seconds']/60:.1f} min**")
        lines.append(f"- Total archive duration: {s['total_duration_hours']:.1f} hours")
    lines.append("")
    lines.append("## Per-printer distribution")
    lines.append("")
    lines.append("| Printer | Prints | Failures |")
    lines.append("|---|---|---|")
    for printer in sorted(s["printer_counts"]):
        lines.append(
            f"| {printer} | {s['printer_counts'][printer]} | "
            f"{s['printer_failure_counts'].get(printer, 0)} |"
        )
    lines.append("")
    lines.append("## Decision gates")
    lines.append("")
    thin = [ft for ft in CANONICAL_FAILURE_TYPES if s["failure_counts"].get(ft, 0) < MIN_CLASS_EXAMPLES]
    if thin:
        lines.append(
            f"- ⚠ Failure classes below {MIN_CLASS_EXAMPLES} examples: "
            f"**{', '.join(thin)}** — drop from stage 2 or merge into `other`."
        )
    else:
        lines.append(f"- ✓ All failure classes have ≥ {MIN_CLASS_EXAMPLES} examples — stage 2 viable.")
    lines.append("- `failure_at_seconds` precision: TBD — run M1.5 next.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    videos_root = args.videos_root.resolve()
    labels_path = (args.labels or (videos_root / "labels.csv")).resolve()

    successful_dir = videos_root / "raw" / "successful"
    failed_dir = videos_root / "raw" / "failed"
    if not successful_dir.exists() and not failed_dir.exists():
        sys.exit(
            f"error: neither {successful_dir} nor {failed_dir} exists. "
            f"Check --videos-root."
        )

    # Discover and parse filenames.
    discoveries: list[tuple[Path, str]] = (
        [(p, "success") for p in find_videos(successful_dir)]
        + [(p, "failure") for p in find_videos(failed_dir)]
    )
    if not discoveries:
        sys.exit(f"error: no video files found under {videos_root}/raw/")

    warnings: list[str] = []
    parsed: dict[str, dict] = {}  # print_id -> partial row
    for video_path, outcome in discoveries:
        print_id, printer_id, failure_type, file_warnings = parse_filename(video_path.stem, outcome)
        warnings.extend(file_warnings)
        if not print_id:
            continue
        if print_id in parsed:
            warnings.append(
                f"duplicate print_id {print_id!r} — kept first ({parsed[print_id]['source_file']}), "
                f"dropped {video_path.relative_to(videos_root)}"
            )
            continue
        parsed[print_id] = {
            "print_id": print_id,
            "source_file": str(video_path.relative_to(videos_root)),
            "printer_id": printer_id,
            "started_at": "",
            "outcome": outcome,
            "failure_type": failure_type or "",
            "failure_at_seconds": "",
            "duration_seconds": "",
            "_path": video_path,
        }

    # ffprobe in parallel.
    print(f"ffprobing {len(parsed)} videos with {args.workers} workers...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(ffprobe, args.ffprobe, r["_path"]): pid for pid, r in parsed.items()}
        completed = 0
        for fut in as_completed(futures):
            pid = futures[fut]
            info = fut.result()
            completed += 1
            if completed % 50 == 0 or completed == len(futures):
                print(f"  ffprobed {completed}/{len(futures)}", file=sys.stderr)
            row = parsed[pid]
            if info.get("error"):
                warnings.append(f"{row['source_file']}: ffprobe failed: {info['error']}")
                row["_resolution"] = None
                row["_fps"] = None
                continue
            row["duration_seconds"] = str(int(round(info["duration"]))) if info.get("duration") else ""
            row["_resolution"] = format_resolution(info.get("width"), info.get("height"))
            row["_fps"] = info.get("fps")
            if info.get("creation_time"):
                row["started_at"] = info["creation_time"]

    # Idempotent merge: preserve failure_at_seconds, started_at (if better), failure_type
    # overrides from a prior labels.csv.
    existing = load_existing_labels(labels_path)
    for pid, row in parsed.items():
        prev = existing.get(pid)
        if not prev:
            continue
        if prev.get("failure_at_seconds", "").strip():
            row["failure_at_seconds"] = prev["failure_at_seconds"]
        # Prefer human-curated started_at over ffprobe creation_time.
        if prev.get("started_at", "").strip():
            row["started_at"] = prev["started_at"]
        # If a human re-categorized a failure (e.g. corrected from filename), keep it.
        if prev.get("failure_type", "").strip() and prev.get("outcome") == "failure":
            if prev["failure_type"] != row["failure_type"]:
                warnings.append(
                    f"{pid}: failure_type from filename ({row['failure_type']!r}) "
                    f"differs from labels.csv ({prev['failure_type']!r}); keeping labels.csv value"
                )
            row["failure_type"] = prev["failure_type"]

    # Build the rows list in a stable order (success first, then failure, both sorted by print_id).
    fieldnames = [
        "print_id", "source_file", "printer_id", "started_at",
        "outcome", "failure_type", "failure_at_seconds", "duration_seconds",
    ]
    ordered = sorted(parsed.values(), key=lambda r: (r["outcome"] != "success", r["print_id"]))
    csv_rows = [{k: r.get(k, "") for k in fieldnames} for r in ordered]
    save_labels_atomic(labels_path, csv_rows, fieldnames)

    # Summarize.
    summary = summarize(ordered)
    stdout = render_stdout_summary(summary)
    print(stdout)

    audit_md = render_audit_md(
        summary,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        videos_root=videos_root,
    )
    args.audit_md.parent.mkdir(parents=True, exist_ok=True)
    args.audit_md.write_text(audit_md)

    print(f"\nWrote {labels_path} ({len(csv_rows)} rows)")
    print(f"Wrote {args.audit_md}")

    if warnings:
        print(f"\n{len(warnings)} warning(s):", file=sys.stderr)
        for w in warnings[:40]:
            print(f"  - {w}", file=sys.stderr)
        if len(warnings) > 40:
            print(f"  ... and {len(warnings) - 40} more", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
