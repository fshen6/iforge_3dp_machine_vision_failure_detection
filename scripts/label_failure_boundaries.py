#!/usr/bin/env python3
"""Interactive failure-boundary labeling helper (Milestone 1.5).

Reads labels.csv, finds failed timelapses without a failure_at_seconds value,
opens each in mpv, and lets you press Enter at the moment the failure visually
starts. The current playhead time is saved back to labels.csv as an integer
number of seconds.

Usage:
    python scripts/label_failure_boundaries.py \\
        --labels print_data/labels.csv \\
        --videos-root /Volumes/External/print_data

The script is resumable: rerunning skips rows that already have a value, so
you can label in 50-video batches and take breaks.

Operational definition of "failure visually starts" — see
docs/ENGINEERING_PLAN.md M1.5. ±5s precision is enough; pick the earlier
candidate if torn between two frames.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Embedded mpv Lua script. Receives the sidecar output path via
# `--script-opts=labeler-out=<path>`, writes a single line of the form
# "<command>:<value>" then quits mpv.
LUA_SCRIPT = r"""
local opts = require 'mp.options'
local options = { out = "" }
opts.read_options(options, "labeler")

local function write(cmd, value)
    if options.out == "" then return end
    local f = io.open(options.out, "w")
    if f == nil then return end
    f:write(cmd .. ":" .. (value or "") .. "\n")
    f:close()
end

mp.add_key_binding("ENTER", "labeler-save", function()
    local t = mp.get_property_number("time-pos") or 0
    write("save", string.format("%.3f", t))
    mp.osd_message(string.format("SAVED failure_at_seconds = %.0fs", t), 1)
    mp.add_timeout(0.4, function() mp.command("quit 0") end)
end)

mp.add_key_binding("Q", "labeler-quit", function()
    write("quit", "")
    mp.osd_message("Quitting session — progress saved", 1)
    mp.add_timeout(0.4, function() mp.command("quit 0") end)
end)

mp.add_key_binding("Z", "labeler-back-30",  function() mp.command("seek -30") end)
mp.add_key_binding("X", "labeler-fwd-30",   function() mp.command("seek 30")  end)
mp.add_key_binding("Ctrl+z", "labeler-back-120", function() mp.command("seek -120") end)
mp.add_key_binding("Ctrl+x", "labeler-fwd-120",  function() mp.command("seek 120")  end)
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, required=True, help="Path to labels.csv")
    p.add_argument(
        "--videos-root",
        type=Path,
        required=True,
        help="Directory that source_file paths in labels.csv are relative to "
             "(e.g. the print_data/ root containing raw/failed/...)",
    )
    p.add_argument("--mpv", default="mpv", help="Path to the mpv binary (default: mpv on PATH)")
    p.add_argument("--start-at", default=None, help="Optional print_id to resume from (skips earlier rows)")
    return p.parse_args()


def load_rows(labels_path: Path) -> tuple[list[dict], list[str]]:
    if not labels_path.exists():
        sys.exit(
            f"error: {labels_path} does not exist.\n"
            f"Run Milestone 1 (scripts/audit_dataset.py) first to bootstrap labels.csv "
            f"from the raw/ filenames. See docs/ENGINEERING_PLAN.md."
        )
    with labels_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    required = {"print_id", "source_file", "outcome", "failure_type", "failure_at_seconds"}
    missing = required - set(fieldnames)
    if missing:
        sys.exit(f"error: labels.csv missing required columns: {sorted(missing)}")
    return rows, fieldnames


def save_rows_atomic(labels_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    fd, tmp_path = tempfile.mkstemp(prefix=".labels.", suffix=".csv.tmp", dir=str(labels_path.parent))
    try:
        # Force UTF-8 — Windows defaults to cp1252 and would crash on non-ASCII fields.
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, labels_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def is_pending(row: dict) -> bool:
    return row.get("outcome", "").strip() == "failure" and not (row.get("failure_at_seconds") or "").strip()


def launch_mpv(mpv: str, video_path: Path, title: str, lua_path: Path, sidecar: Path) -> None:
    cmd = [
        mpv,
        f"--force-media-title={title}",
        f"--script={lua_path}",
        f"--script-opts=labeler-out={sidecar}",
        "--osd-level=2",
        "--osd-duration=2000",
        "--keep-open=yes",
        str(video_path),
    ]
    subprocess.run(cmd)


def parse_sidecar(sidecar: Path) -> tuple[str, str]:
    if not sidecar.exists() or sidecar.stat().st_size == 0:
        return ("skip", "")
    line = sidecar.read_text().strip()
    if ":" not in line:
        return ("skip", "")
    cmd, _, value = line.partition(":")
    return (cmd, value)


def main() -> int:
    args = parse_args()

    if not shutil.which(args.mpv):
        sys.exit(f"error: '{args.mpv}' not found on PATH. Install mpv (brew install mpv).")

    rows, fieldnames = load_rows(args.labels)

    pending_indices = [i for i, r in enumerate(rows) if is_pending(r)]
    if args.start_at:
        try:
            cutoff = next(i for i in pending_indices if rows[i]["print_id"] == args.start_at)
            pending_indices = pending_indices[pending_indices.index(cutoff):]
        except StopIteration:
            sys.exit(f"error: --start-at {args.start_at!r} not found among pending rows")

    if not pending_indices:
        print("Nothing to label — every failed print already has failure_at_seconds.")
        return 0

    print(f"Labeling {len(pending_indices)} failed timelapses. mpv keys:")
    print("  ←/→: ±5s   shift+←/→: ±1s   ↑/↓: ±60s   Z/X: ±30s   ctrl+Z/X: ±2min")
    print("  ,/.: frame step   [/]: speed   space: pause")
    print("  ENTER: save current time and advance")
    print("  q: skip this video (mpv default close)")
    print("  shift+Q: quit session (progress already saved)")
    print()

    # Write the Lua script to a temp file once per session.
    lua_fd, lua_path_str = tempfile.mkstemp(prefix="labeler-", suffix=".lua")
    with os.fdopen(lua_fd, "w") as f:
        f.write(LUA_SCRIPT)
    lua_path = Path(lua_path_str)

    saved = 0
    skipped = 0
    missing = 0
    try:
        for n, idx in enumerate(pending_indices, 1):
            row = rows[idx]
            video_path = (args.videos_root / row["source_file"]).resolve()
            tag = f"[{n}/{len(pending_indices)}] {row['print_id']} ({row['failure_type']})"

            if not video_path.exists():
                print(f"{tag}  MISSING: {video_path}")
                missing += 1
                continue

            title = f"{n}/{len(pending_indices)} | {row['print_id']} | {row['failure_type']}"
            sidecar_fd, sidecar_path_str = tempfile.mkstemp(prefix="labeler-out-", suffix=".txt")
            os.close(sidecar_fd)
            sidecar = Path(sidecar_path_str)
            # Truncate so we can distinguish "no write" from a stale value.
            sidecar.write_text("")

            try:
                launch_mpv(args.mpv, video_path, title, lua_path, sidecar)
                cmd, value = parse_sidecar(sidecar)
            finally:
                if sidecar.exists():
                    sidecar.unlink()

            if cmd == "save":
                seconds = int(round(float(value)))
                row["failure_at_seconds"] = str(seconds)
                save_rows_atomic(args.labels, rows, fieldnames)
                saved += 1
                print(f"{tag}  saved failure_at_seconds = {seconds}s")
            elif cmd == "quit":
                print(f"{tag}  session quit — resume by rerunning the script")
                break
            else:
                skipped += 1
                print(f"{tag}  skipped")
    finally:
        if lua_path.exists():
            lua_path.unlink()

    remaining = sum(1 for r in rows if is_pending(r))
    print()
    print(f"Session done: saved {saved}, skipped {skipped}, missing files {missing}.")
    print(f"Remaining unlabeled failed prints: {remaining}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
