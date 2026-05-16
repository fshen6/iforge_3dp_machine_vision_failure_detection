# Dataset Structure

Conventions for organizing the training data so Milestones 2-4 can run without re-decoding 1000+ timelapses every time you change your mind about a label or a split.

## Principle

**One decode, multiple views.** Decode timelapses to frames once and never again. Build YOLO-format folder structures with **symlinks** so you can experiment with class definitions and splits cheaply.

Three layers:
1. **Raw** — original timelapses, untouched, immutable
2. **Canonical** — decoded frames organized by `print_id` + a single CSV that holds all labels
3. **YOLO views** — symlink-based folder structures in the shape Ultralytics expects, regenerable in seconds

## Directory layout

```
print_data/
├── raw/                                  # ORIGINAL FILES, NEVER MODIFY
│   ├── successful/
│   │   ├── print_001_printer03.mp4
│   │   ├── print_002_printer07.mp4
│   │   └── ...                            (~1000 files)
│   └── failed/
│       ├── print_456_printer02_bed_adhesion.mp4
│       ├── print_457_printer05_spaghetti.mp4
│       └── ...                            (~287 files)
│
├── labels.csv                            # SINGLE SOURCE OF TRUTH for metadata
│
├── frames/                               # DECODED ONCE, grouped by print_id (no class folders)
│   ├── print_0001/
│   │   ├── 000000.jpg                    (frame 0 = layer 1)
│   │   ├── 000001.jpg                    (frame 1 = layer 2)
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── print_0002/
│   └── ...
│
├── splits.json                           # print_id → train | val | test | eval
│
└── yolo/                                 # ULTRALYTICS-FORMAT VIEWS, all symlinks
    ├── stage1/                            (binary: healthy / failure)
    │   ├── train/
    │   │   ├── healthy/
    │   │   │   ├── print_001_000000.jpg  → ../../../../frames/print_001/000000.jpg
    │   │   │   └── ...
    │   │   └── failure/
    │   │       ├── print_456_001830.jpg  → ../../../../frames/print_456/001830.jpg
    │   │       └── ...
    │   ├── val/
    │   │   ├── healthy/
    │   │   └── failure/
    │   └── test/
    │       ├── healthy/
    │       └── failure/
    │
    ├── stage2/                            (multi-class: spaghetti / bed_adhesion / layer_shift / other)
    │   ├── train/
    │   │   ├── spaghetti/
    │   │   ├── bed_adhesion/
    │   │   ├── layer_shift/
    │   │   └── other/
    │   ├── val/  ...
    │   └── test/ ...
    │
    └── eval/                              # HAND-CURATED, held out forever
        ├── stage1/
        │   ├── healthy/                  (~50-100 hand-picked hard frames)
        │   └── failure/
        └── stage2/
            ├── spaghetti/
            ├── bed_adhesion/
            ├── layer_shift/
            └── other/
```

## labels.csv — single source of truth

This is the one file you edit when you find a mislabeled timelapse. Everything else regenerates from it.

```csv
print_id,source_file,printer_id,started_at,outcome,failure_type,failure_at_frame,frame_count,duration_seconds
print_0001,raw/successful/Tolstoy/DSC_0001.mp4,Tolstoy,2026-02-14T09:12:00,success,,,480,8
print_0002,raw/successful/Bell/IMG_77.mp4,Bell,2026-02-14T09:45:00,success,,,240,10
print_0456,raw/failed/bed_adhesion/Tolstoy/oops.mp4,Tolstoy,2026-03-22T11:30:00,failure,bed_adhesion,47,212,4
print_0457,raw/failed/spaghetti/Bell/spag1.mp4,Bell,2026-03-22T14:15:00,failure,spaghetti,830,1100,18
```

| Column | Meaning |
|---|---|
| `print_id` | Unique ID per print, auto-assigned by `audit_dataset.py`. Use this for splits, never the filename. |
| `printer_id` | Which physical printer (`Tolstoy`, `Bell`, `Socrates`, `Einstein`, `Beethoven`, `Watt`, `Hypatia`, `Picasso`). Used for stratification. |
| `outcome` | `success` or `failure`. Drives stage 1 labels. |
| `failure_type` | `spaghetti` / `bed_adhesion` / `layer_shift` / `other`. Drives stage 2 labels. NULL for successes. |
| `failure_at_frame` | Frame index where the failure visually starts (mpv's `estimated-frame-number`, 0-based). Equivalent to layer index since timelapses capture one frame per layer. NULL for successes. |
| `frame_count` | Total frames in the video = total layers in the print. Use this for size stratification. |
| `duration_seconds` | Encoded playback duration. Cosmetic — it's just `frame_count / fps` and varies with the (unimportant) encoded fps. |

> **One frame = one print layer.** Timelapses are captured one frame per printed layer. The video's encoded fps (25 or 60 in the current archive) is purely a playback-speed choice and has no effect on labels — `failure_at_frame = 47` means the failure starts at layer 48 (frame indices are 0-based) regardless of fps. Don't try to convert frame index to seconds: physical print time isn't recoverable from the video alone.

## The frame-labeling rule (the part most people get wrong)

A failed timelapse contains frames that span healthy → failure. You cannot label every frame in a failed file as a failure example — most of them are healthy, the failure happened at some point.

Use `failure_at_frame` to split each failed timelapse's frames. The ambiguity window is a fixed frame count `AMBIGUITY_LAYERS` — the number of layers immediately before the called failure that are dropped as ambiguous "warning sign" frames. Default: **5 layers**. Tune up if you find the model learning early-warning patterns that don't generalize:

| Frame (layer) index | Stage 1 label | Stage 2 label |
|---|---|---|
| `frame < failure_at_frame − AMBIGUITY_LAYERS` | candidate `healthy` | n/a |
| `failure_at_frame − AMBIGUITY_LAYERS ≤ frame < failure_at_frame` | **drop** — ambiguous "warning sign" layers | n/a |
| `frame ≥ failure_at_frame` | `failure` | `<failure_type>` |

For successful prints: every frame is `healthy` for stage 1, excluded from stage 2.

Since one frame = one printed layer, the rule is comparable across all your videos without any fps conversion. The same `AMBIGUITY_LAYERS` applies whether a video is encoded at 25 fps or 60 fps.

**Decoding implication.** The 10s-cadence inference plan in this doc was written assuming fixed-rate timelapses. With layer-per-frame capture, "10s cadence at inference time" is the printer's wall-clock layer time during a print, not a property of the training videos. For training, you can use **every raw frame** as a candidate sample — there's no decode/subsample step needed. The training pipeline reads frames directly from the .mp4s.

## Split strategy

**By `print_id`, NEVER by frame.** If frames from `print_456` appear in both train and val, your validation accuracy is a lie — the model memorizes that specific print's lighting and camera position.

```json
{
  "train": ["print_001", "print_002", ...],
  "val":   ["print_011", "print_012", ...],
  "test":  ["print_021", "print_022", ...],
  "eval":  ["print_031", "print_032", ...]
}
```

Target ratios: 70% train / 15% val / 15% test, plus a ~5-10% subset of test pulled out as `eval`.

**Stratification:** when you assign prints to splits, balance by `(printer_id, outcome)` joint. Every printer should appear in every split. Every class should appear in every split. Use `sklearn.model_selection.StratifiedShuffleSplit` with the joint key.

The `eval` split is held out from training **and** from threshold tuning. It exists to grade v2 vs v1 honestly.

## Image format

- **Cadence:** one decoded JPEG per raw video frame = one per printed layer. No subsampling. Inference at runtime polls one frame per layer-completion event, matching training cadence exactly.
- **Format:** JPEG quality 90 — fine for vision, much smaller than PNG.
- **Resize at decode time:** 256×256 — Ultralytics will random-crop to 224×224 during training.
- **Aspect:** preserve original via letterbox OR center-crop to square — pick one and stay consistent. Center-crop is simpler if camera framing is uniform.

## Why this design

- **Re-labeling is cheap.** Find a mislabeled timelapse? Edit `labels.csv`, rerun `build_yolo_views.py`. ~30 seconds.
- **New classes are cheap.** Decide to split `other_failure` into `clog` and `under_extrusion`? Update labels.csv, regenerate. No re-decode.
- **Splits are reproducible.** `splits.json` is committed (it's small text). Anyone can rebuild the exact same training set.
- **Disk is bounded.** Frames decoded once. YOLO views are symlinks — they cost ~0 bytes on disk.
- **Bad-frame quarantine is easy.** Add a `quarantined` column to labels.csv, exclude in build script. Don't delete the file.

## Disk budget estimates

Assumptions: ~1300 timelapses, average ~400 layers per print = ~520K frames decoded total (run `audit_dataset.py` to see actual `frame_count` distribution).

| Folder | Size estimate |
|---|---|
| `raw/` | 50-100 GB (your current archive) |
| `frames/` | 10-15 GB (256×256 JPEG q90, ~25 KB per frame × ~520K frames) |
| `yolo/` | < 100 MB (all symlinks) |
| Total active | ~70-115 GB |

Use an SSD for `frames/` if you can — random reads during training data loading are the hot path. Mechanical disks work but slow down each epoch noticeably.
