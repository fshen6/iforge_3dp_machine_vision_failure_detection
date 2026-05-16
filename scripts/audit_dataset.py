#!/usr/bin/env python3
"""Dataset audit (Milestone 1).

Walks the canonical dataset layout under --videos-root, ffprobes each video
in parallel for duration and resolution, and produces:

  1. labels.csv  — the dataset's single source of truth. Idempotent across
                   re-runs: print_ids and failure_at_frame values are
                   preserved by matching source_file (or printer_id+basename
                   as a fallback, so a file moved between failure-type
                   folders keeps its label).
  2. DATASET_AUDIT.md — one-page human-readable summary.
  3. stdout summary matching the format in docs/ENGINEERING_PLAN.md M1.

Expected layout (folders are the source of truth for outcome, failure_type,
and printer_id — filenames themselves are not parsed):

    <videos-root>/raw/
        successful/
            Tolstoy/   *.mp4
            Bell/      *.mp4
            ...
        failed/
            bed_adhesion/
                Tolstoy/   *.mp4
                Bell/      *.mp4
                ...
            spaghetti/
                <printer>/ *.mp4
            layer_shift/
            other/

Usage (cross-platform):

    python scripts/audit_dataset.py --videos-root D:\\print_data
    python scripts/audit_dataset.py --videos-root /Volumes/Drive/print_data

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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

CANONICAL_PRINTERS = (
    "Tolstoy", "Bell", "Socrates", "Einstein",
    "Beethoven", "Watt", "Hypatia", "Picasso",
)
CANONICAL_FAILURE_TYPES = ("bed_adhesion", "spaghetti", "layer_shift", "other")
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}

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
        help="Directory containing raw/successful/<printer>/ and raw/failed/<type>/<printer>/.",
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


def find_videos(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def walk_dataset(videos_root: Path) -> tuple[list[dict], list[str]]:
    """Return ([{path, outcome, failure_type, printer_id, source_file}, ...], warnings)."""
    successful_root = videos_root / "raw" / "successful"
    failed_root = videos_root / "raw" / "failed"
    discoveries: list[dict] = []
    warnings: list[str] = []

    canonical_printer_set = set(CANONICAL_PRINTERS)
    canonical_failure_set = set(CANONICAL_FAILURE_TYPES)

    # raw/successful/<printer>/*.mp4
    if successful_root.exists():
        for printer_dir in sorted(p for p in successful_root.iterdir() if p.is_dir()):
            printer_id = printer_dir.name
            if printer_id not in canonical_printer_set:
                warnings.append(
                    f"unknown printer folder {printer_id!r} under {successful_root.as_posix()} "
                    f"(canonical: {', '.join(CANONICAL_PRINTERS)})"
                )
            for video in find_videos(printer_dir):
                discoveries.append({
                    "_path": video,
                    "outcome": "success",
                    "failure_type": None,
                    "printer_id": printer_id,
                    "source_file": video.relative_to(videos_root).as_posix(),
                })

    # raw/failed/<failure_type>/<printer>/*.mp4
    if failed_root.exists():
        for ft_dir in sorted(p for p in failed_root.iterdir() if p.is_dir()):
            failure_type = ft_dir.name
            if failure_type not in canonical_failure_set:
                warnings.append(
                    f"unknown failure_type folder {failure_type!r} under {failed_root.as_posix()} "
                    f"(canonical: {', '.join(CANONICAL_FAILURE_TYPES)}); skipping its contents"
                )
                continue
            for printer_dir in sorted(p for p in ft_dir.iterdir() if p.is_dir()):
                printer_id = printer_dir.name
                if printer_id not in canonical_printer_set:
                    warnings.append(
                        f"unknown printer folder {printer_id!r} under {ft_dir.as_posix()}"
                    )
                for video in find_videos(printer_dir):
                    discoveries.append({
                        "_path": video,
                        "outcome": "failure",
                        "failure_type": failure_type,
                        "printer_id": printer_id,
                        "source_file": video.relative_to(videos_root).as_posix(),
                    })
            # Also catch videos placed directly under raw/failed/<failure_type>/ without a printer subfolder.
            stray = find_videos(ft_dir)
            if stray:
                warnings.append(
                    f"{len(stray)} video(s) under {ft_dir.as_posix()} are missing a printer subfolder; ignored"
                )

    return discoveries, warnings


def ffprobe(ffprobe_bin: str, path: Path) -> dict:
    """Return {duration, width, height, fps, creation_time} or {error: ...} on failure."""
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
    out: dict = {}
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


def load_existing_labels(labels_path: Path) -> list[dict]:
    if not labels_path.exists():
        return []
    # utf-8-sig transparently strips a BOM if Excel ever round-tripped this file.
    with labels_path.open(newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r.get("print_id")]


def assign_print_ids(rows: list[dict], existing: list[dict]) -> None:
    """Assign stable print_ids. Reuse from existing labels.csv by source_file or
    (printer_id, basename) fallback. New rows get the next available print_0NNN."""
    by_source = {r["source_file"]: r for r in existing}
    by_printer_basename = {(r["printer_id"], Path(r["source_file"]).name): r for r in existing}
    used: set[str] = set()
    pid_re = re.compile(r"^print_(\d+)$")

    for row in rows:
        prev = by_source.get(row["source_file"]) or by_printer_basename.get(
            (row["printer_id"], Path(row["source_file"]).name)
        )
        if prev and prev.get("print_id"):
            row["print_id"] = prev["print_id"]
            used.add(prev["print_id"])

    # Highest existing numeric ID; new IDs start above it.
    max_n = 0
    for r in existing:
        m = pid_re.match(r.get("print_id", ""))
        if m:
            max_n = max(max_n, int(m.group(1)))

    next_n = max_n + 1
    for row in rows:
        if row.get("print_id"):
            continue
        while f"print_{next_n:04d}" in used:
            next_n += 1
        row["print_id"] = f"print_{next_n:04d}"
        used.add(row["print_id"])
        next_n += 1


def merge_preserved_fields(rows: list[dict], existing: list[dict]) -> list[str]:
    """Copy human-curated fields from existing rows to new rows by source_file
    or (printer_id, basename) fallback. Returns warnings."""
    by_source = {r["source_file"]: r for r in existing}
    by_printer_basename = {(r["printer_id"], Path(r["source_file"]).name): r for r in existing}
    warnings: list[str] = []

    for row in rows:
        prev = by_source.get(row["source_file"]) or by_printer_basename.get(
            (row["printer_id"], Path(row["source_file"]).name)
        )
        if not prev:
            continue
        if (prev.get("failure_at_frame") or "").strip():
            row["failure_at_frame"] = prev["failure_at_frame"]
        # Back-compat: if an older labels.csv from before the rename still has
        # failure_at_seconds set, carry the value into failure_at_frame is impossible
        # without per-video fps, so we drop it silently and the user re-labels.
        if (prev.get("started_at") or "").strip():
            row["started_at"] = prev["started_at"]
        # If a human reclassified failure_type in labels.csv (vs the folder), flag the conflict
        # but trust the folder — it's the new source of truth.
        if (
            row["outcome"] == "failure"
            and (prev.get("failure_type") or "").strip()
            and prev["failure_type"] != row["failure_type"]
        ):
            warnings.append(
                f"{row['print_id']} ({row['source_file']}): failure_type changed "
                f"{prev['failure_type']!r} → {row['failure_type']!r} based on folder; if the "
                f"old value was correct, move the file back."
            )
    return warnings


def save_labels_atomic(labels_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".labels.", suffix=".csv.tmp", dir=str(labels_path.parent))
    try:
        # Force UTF-8: Windows would otherwise pick cp1252 from the locale and choke on
        # non-ASCII characters (em dashes, accented chars, ffprobe timezone glyphs).
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
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
        parts = [f"{res} ({n / total:.0%})" for res, n in s["resolutions"].most_common()]
        lines.append("Resolution: " + ", ".join(parts))
        if len(s["resolutions"]) > 1:
            lines.append("  -- mixed resolutions; normalize at decode")
    lines.append("")
    lines.append("Per-printer distribution:")
    if s["printer_counts"]:
        # Sort canonical printers first (in their declared order), then any unknowns alphabetically.
        canonical_seen = [p for p in CANONICAL_PRINTERS if p in s["printer_counts"]]
        unknown = sorted(p for p in s["printer_counts"] if p not in CANONICAL_PRINTERS)
        missing = [p for p in CANONICAL_PRINTERS if p not in s["printer_counts"]]
        for printer in canonical_seen + unknown:
            total_n = s["printer_counts"][printer]
            failures_n = s["printer_failure_counts"].get(printer, 0)
            warn = "  ← NO FAILURES" if failures_n == 0 else ""
            unk = "  (unknown printer)" if printer not in CANONICAL_PRINTERS else ""
            lines.append(f"  {printer}: {total_n} prints ({failures_n} failures){warn}{unk}")
        for printer in missing:
            lines.append(f"  {printer}: 0 prints  ← NOT SEEN")
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
    lines.append(f"_Generated {generated_at} from `{videos_root.as_posix()}` by `scripts/audit_dataset.py`._")
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
    canonical_seen = [p for p in CANONICAL_PRINTERS if p in s["printer_counts"]]
    unknown = sorted(p for p in s["printer_counts"] if p not in CANONICAL_PRINTERS)
    missing = [p for p in CANONICAL_PRINTERS if p not in s["printer_counts"]]
    for printer in canonical_seen + unknown:
        lines.append(
            f"| {printer} | {s['printer_counts'][printer]} | "
            f"{s['printer_failure_counts'].get(printer, 0)} |"
        )
    for printer in missing:
        lines.append(f"| {printer} _(missing)_ | 0 | 0 |")
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
    lines.append("- `failure_at_frame` precision: TBD — run M1.5 next.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    videos_root = args.videos_root.resolve()
    labels_path = (args.labels or (videos_root / "labels.csv")).resolve()

    successful_root = videos_root / "raw" / "successful"
    failed_root = videos_root / "raw" / "failed"
    if not successful_root.exists() and not failed_root.exists():
        sys.exit(
            f"error: neither {successful_root} nor {failed_root} exists. "
            f"Expected layout:\n"
            f"  raw/successful/<Printer>/*.mp4\n"
            f"  raw/failed/<failure_type>/<Printer>/*.mp4\n"
            f"  Printers: {', '.join(CANONICAL_PRINTERS)}\n"
            f"  failure_types: {', '.join(CANONICAL_FAILURE_TYPES)}"
        )

    discoveries, walk_warnings = walk_dataset(videos_root)
    if not discoveries:
        sys.exit(f"error: no video files found under {videos_root.as_posix()}/raw/")

    warnings = list(walk_warnings)

    # ffprobe in parallel
    print(f"ffprobing {len(discoveries)} videos with {args.workers} workers...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(ffprobe, args.ffprobe, d["_path"]): d for d in discoveries}
        completed = 0
        for fut in as_completed(futures):
            d = futures[fut]
            info = fut.result()
            completed += 1
            if completed % 50 == 0 or completed == len(futures):
                print(f"  ffprobed {completed}/{len(futures)}", file=sys.stderr)
            if info.get("error"):
                warnings.append(f"{d['source_file']}: ffprobe failed: {info['error']}")
                d["_resolution"] = None
                d["_fps"] = None
                d["duration_seconds"] = ""
                d["started_at"] = ""
                continue
            d["duration_seconds"] = str(int(round(info["duration"]))) if info.get("duration") else ""
            d["_resolution"] = format_resolution(info.get("width"), info.get("height"))
            d["_fps"] = info.get("fps")
            d["started_at"] = info.get("creation_time") or ""

    # Stable ordering: success before failure, then by failure_type, then by printer, then by source_file.
    failure_order = {ft: i for i, ft in enumerate(CANONICAL_FAILURE_TYPES)}
    printer_order = {p: i for i, p in enumerate(CANONICAL_PRINTERS)}

    def sort_key(d: dict) -> tuple:
        return (
            0 if d["outcome"] == "success" else 1,
            failure_order.get(d.get("failure_type") or "", 99),
            printer_order.get(d["printer_id"], 99),
            d["source_file"],
        )

    discoveries.sort(key=sort_key)

    # Initialize CSV-shaped fields on every row.
    for d in discoveries:
        d.setdefault("failure_at_frame", "")
        d.setdefault("failure_type", d.get("failure_type") or "")
        d.setdefault("started_at", d.get("started_at") or "")

    # Idempotent merge with existing labels.csv: print_ids and human fields.
    existing = load_existing_labels(labels_path)
    assign_print_ids(discoveries, existing)
    warnings.extend(merge_preserved_fields(discoveries, existing))

    fieldnames = [
        "print_id", "source_file", "printer_id", "started_at",
        "outcome", "failure_type", "failure_at_frame", "duration_seconds",
    ]
    csv_rows = [{k: r.get(k, "") for k in fieldnames} for r in discoveries]
    save_labels_atomic(labels_path, csv_rows, fieldnames)

    summary = summarize(discoveries)
    print(render_stdout_summary(summary))

    audit_md = render_audit_md(
        summary,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        videos_root=videos_root,
    )
    args.audit_md.parent.mkdir(parents=True, exist_ok=True)
    args.audit_md.write_text(audit_md, encoding="utf-8")

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
