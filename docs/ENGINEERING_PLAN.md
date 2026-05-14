# Engineering Plan — Print Farm Failure Detection v1

Generated 2026-05-14
References: design doc (2026-05-11), eng review (2026-05-11)

Stack (locked):
- Language: Python 3.11+
- Env: `uv` (fast, modern Python project manager)
- Training: Ultralytics YOLOv8n-cls (NOT v11 — better Hailo compatibility)
- Inference: HailoRT runtime on Raspberry Pi 5 8GB + Hailo-8 AI HAT (26 TOPS)
- Model export: Ultralytics `yolo export ... format=hailo` OR explicit ONNX → Hailo Dataflow Compiler → HEF
- Service: FastAPI (for the ntfy webhook + future operator UI)
- DB: SQLite (via stdlib `sqlite3` or `sqlmodel` if you want types)
- Notifications: ntfy (self-hosted or hosted)
- Config: YAML (one file, one source of truth)
- Process supervision: systemd
- Frame storage: **256 GB** external USB 3.0 SSD on the Pi 5 with **tiered retention** (cannot stack PCIe NVMe HAT with AI HAT)

Total active dev time: ~4 weeks of building + 2 weeks of observation.

---

## Repo structure (target)

```
print-vision/
├── pyproject.toml                 # uv project file
├── README.md
├── config.example.yaml            # all tunable params
├── scripts/                       # one-off dev/ops scripts
│   ├── audit_dataset.py           # count per-class, frame rates, timestamps
│   ├── extract_frames.py          # timelapses -> frames at 10s cadence
│   ├── build_splits.py            # print_id assignment + stratified split
│   ├── curate_eval_set.py         # interactive picker for held-out eval frames
│   ├── retention_cleanup.py       # tiered retention: hot->warm at 7d, warm->delete at 90d
│   └── healthy_sampler.py         # marks 1-in-100 healthy frames as 'sampled' for retraining diversity
├── training/
│   ├── train_stage1.py            # binary healthy/failure
│   ├── train_stage2.py            # multi-class on failure-only
│   └── grade_against_eval.py      # score a checkpoint against eval set
├── service/
│   ├── __init__.py
│   ├── main.py                    # entry point, async orchestrator
│   ├── config.py                  # YAML loader, schema validation
│   ├── octoprint_client.py        # /api/job poll, snapshot pull, pause API
│   ├── inference.py               # load engines, stage1 + stage2 predict
│   ├── decision.py                # rolling-window vote, threshold checks
│   ├── action.py                  # pause with retry, notification dispatch
│   ├── notify.py                  # ntfy client
│   ├── webhook.py                 # FastAPI app — ntfy button-tap webhook
│   ├── db.py                      # SQLite schema + queries
│   └── heartbeat.py               # 5-min alive ping to ntfy
├── models/                        # gitignored — .pt and .engine files
├── tests/
│   ├── test_octoprint_client.py
│   ├── test_decision.py
│   ├── test_action.py
│   └── fixtures/
└── deploy/
    ├── print-vision.service       # systemd unit
    └── ntfy-actions.md            # action button JSON format reference
```

---

## Milestone 0 — Project Setup (0.5 day)

Deliverables:
- Empty git repo, GitHub or self-hosted
- `pyproject.toml` with `uv` initialized
- `pre-commit` with `ruff` (lint + format)
- Skeleton README

Steps:
1. `uv init print-vision && cd print-vision`
2. `uv add fastapi uvicorn pydantic httpx pyyaml ultralytics opencv-python-headless pillow`
3. `uv add --dev pytest ruff pre-commit`
4. `git init && pre-commit install`
5. Write `.gitignore` (models/, *.pt, *.engine, *.db, frames/, .venv/)
6. First commit

Acceptance: `uv run python -c "import ultralytics; print('ok')"` succeeds.

---

## Milestone 1 — Dataset Audit (1-2 days)

Goal: know exactly what you have before writing training code.

Deliverables:
- `scripts/audit_dataset.py`
- One-page `DATASET_AUDIT.md` with per-class counts, frame rates, timestamp precision

Output the script should produce:
```
=== DATASET AUDIT ===
Total successful timelapses: 1023
Total failed timelapses: 287
  - bed_adhesion: 94
  - spaghetti: 81
  - layer_shift: 52
  - other_failure: 60

Frame rate (avg per timelapse): 1 frame / 8.2s
Resolution: 1920x1080 (87%), 1280x720 (13%) -- need to normalize at decode

Per-printer distribution:
  printer_01: 145, printer_02: 132, ... (verify no printer has zero failures)

Failure timestamp precision: layer-level (yes/no) -- ___
```

Steps:
1. Walk the dataset folders, parse filenames/metadata
2. For each timelapse, sample 1 frame to check resolution
3. For failed timelapses, parse the failure-reason and timestamp from your existing labels
4. Print summary, write `DATASET_AUDIT.md`

Acceptance:
- Audit completes without error on the full dataset
- You know per-class counts and can decide: is stage 2 viable, or should v1 ship as binary-only?

**Decision gate after this milestone:** If any failure class has < 30 examples, drop it from stage 2 or merge into `other_failure`.

---

## Milestone 2 — Dataset Prep (2-3 days)

Goal: timelapses → labeled frame folders ready for YOLOv11-cls.

Deliverables:
- `scripts/extract_frames.py` — decode timelapses to frames at 10s cadence, resize to 224x224
- `scripts/build_splits.py` — assign `print_id`, split train/val/test at print level, stratify by printer
- `scripts/curate_eval_set.py` — interactive helper: opens N candidate frames per class, you keep/skip
- `print_data/` folder structure populated (see design doc training section)

Steps:
1. Write `extract_frames.py` using `opencv-python-headless`:
   - Input: path to a timelapse, label, print_id
   - Output: `frames/<print_id>/<print_id>_<frame_idx>.jpg` at 10s cadence
   - Resize to 256x256 (allows random crop to 224 at training time)
2. Write `build_splits.py`:
   - Group frames by `print_id`
   - Split print_ids: 70% train / 15% val / 15% test, stratified by `(printer_id, class)`
   - Copy/symlink frames into `print_data/stage1/{train,val,test}/{healthy,failure}/`
   - Same for `print_data/stage2/{train,val,test}/{spaghetti,bed_adhesion,layer_shift,other}/` (failure-only)
3. Write `curate_eval_set.py`:
   - For each class, sample 200 candidate test-split frames
   - For each frame, show it (or a thumbnail in terminal via `imgcat`/sixel, or open in default viewer)
   - You tap `k` to keep, `s` to skip
   - Save kept frames into `print_data/eval/<class>/`
   - Target: 50-100 frames per class

Acceptance:
- `print_data/stage1/train/healthy/` and `print_data/stage1/train/failure/` exist with frames
- `print_data/eval/` has hand-curated frames per class
- No `print_id` appears in more than one of `{train, val, test}`

---

## Milestone 3 — Stage 1 Training (1 day)

Goal: binary `healthy / failure` model.

Deliverables:
- `training/train_stage1.py`
- `models/stage1_v1.pt` saved
- Validation metrics + eval-set metrics written to `runs/stage1_binary/results.csv` and `EVAL_RESULTS.md`

Steps:
1. Write the training script. It is ~30 lines, mostly the `model.train()` call from the previous answer.
2. Train on your machine with a CUDA GPU (or Colab if no GPU).
   - `uv run python training/train_stage1.py`
   - Watch loss curve. Should plateau by epoch 15-20. Patience=10 stops it early.
3. After training: `uv run python training/grade_against_eval.py --model runs/stage1_binary/weights/best.pt --eval print_data/eval`
4. If eval set F1 < 0.85, iterate: check augmentations, class balance, learning rate. If still bad, you may be underdata'd on failure class — collect more or relax success threshold.

Acceptance:
- Val F1 ≥ 0.90
- Eval set F1 ≥ 0.85 (lower target because eval is harder)
- False-positive rate on healthy eval frames < 5%

---

## Milestone 4 — Stage 2 Training (1 day)

Goal: multi-class on failure-only frames.

Deliverables:
- `training/train_stage2.py` (95% the same script as stage 1, different data path)
- `models/stage2_v1.pt` saved
- Eval-set per-class F1 written

Steps:
1. Train on `print_data/stage2`
2. Grade per-class
3. If any class has F1 < 0.70, consider merging into `other` and retraining

Acceptance:
- Per-class F1 ≥ 0.70 on val and eval sets, OR small classes merged into `other`

---

## Milestone 5 — Inference Service Skeleton (3-5 days)

Goal: end-to-end pipeline running on the Jetson with no model yet (mock predictor returns "healthy"). Validates everything except inference.

Deliverables:
- `service/octoprint_client.py`
- `service/inference.py` (with a `MockPredictor` for now)
- `service/decision.py`
- `service/action.py`
- `service/db.py`
- `service/notify.py`
- `service/webhook.py`
- `service/main.py`
- `service/heartbeat.py`
- `config.example.yaml` + `config.yaml` (gitignored)
- All unit tests passing

`config.yaml` shape:
```yaml
printers:
  - id: printer_01
    octoprint_url: http://192.168.1.101
    api_key: ${OCTOPRINT_KEY_01}    # env var interpolation
    camera_snapshot_url: http://192.168.1.101:8080/?action=snapshot
  - id: printer_02
    ...
inference:
  cadence_seconds: 10
  rolling_window: 5
  vote_threshold: 3
  stage1_confidence_threshold: 0.80
  stage2_confidence_threshold: 0.70
  shadow_mode: true                  # FLIP TO false AFTER WEEK OF SHADOW
action:
  pause_retry_attempts: 3
  pause_retry_delays_seconds: [2, 5, 10]
notify:
  ntfy_url: https://ntfy.sh/your-secret-topic
  webhook_url: http://192.168.1.50:8000/ntfy
storage:
  frames_dir: /mnt/ssd/frames
  db_path: /mnt/ssd/print-vision.db
  # Tiered retention (256 GB SSD constraint)
  hot_retention_days: 7          # every frame kept this long
  warm_retention_days: 90        # only flagged/sampled frames after hot tier
  healthy_sample_rate: 0.01      # 1 in 100 healthy frames kept for retraining diversity
  min_free_gb: 30                # bail out & alert if free space drops below this
heartbeat:
  interval_minutes: 5
```

SQLite schema (`service/db.py`):
```sql
CREATE TABLE prints (
  id TEXT PRIMARY KEY,
  printer_id TEXT NOT NULL,
  started_at TIMESTAMP NOT NULL,
  ended_at TIMESTAMP,
  end_reason TEXT,
  gcode_filename TEXT
);

CREATE TABLE frames (
  id TEXT PRIMARY KEY,
  print_id TEXT REFERENCES prints(id),
  printer_id TEXT NOT NULL,
  captured_at TIMESTAMP NOT NULL,
  image_path TEXT,                -- NULL after retention deletes the file; row stays
  stage1_label TEXT,
  stage1_confidence REAL,
  stage2_label TEXT,
  stage2_confidence REAL,
  action_taken TEXT,              -- 'pause', 'notify_only', 'shadow_log', or NULL
  operator_label TEXT,            -- 'confirmed', 'false_positive', or NULL
  operator_responded_at TIMESTAMP,
  is_healthy_sample INTEGER NOT NULL DEFAULT 0   -- 1 if picked for the 1-in-100 retraining sample
);

CREATE INDEX idx_frames_print ON frames(print_id);
CREATE INDEX idx_frames_captured ON frames(captured_at);
CREATE INDEX idx_frames_retention ON frames(captured_at, operator_label, stage1_label, is_healthy_sample);

-- Retention queries (run nightly via retention_cleanup.py):
--
-- Hot tier expiry (>7 days, not flagged, not labeled, not sampled): delete file, NULL the path
-- DELETE FROM disk WHERE id IN (
--   SELECT id FROM frames
--   WHERE captured_at < datetime('now', '-7 days')
--     AND image_path IS NOT NULL
--     AND operator_label IS NULL
--     AND stage1_label != 'failure'
--     AND is_healthy_sample = 0
-- );
--
-- Warm tier expiry (>90 days, no operator label): delete file, NULL the path
-- DELETE FROM disk WHERE id IN (
--   SELECT id FROM frames
--   WHERE captured_at < datetime('now', '-90 days')
--     AND image_path IS NOT NULL
--     AND operator_label IS NULL
-- );
```

`service/main.py` orchestration (pseudo):
```
async def monitor_printer(printer_config):
    while True:
        job = await octoprint.get_job(printer_config)
        if job.state == "Printing" and not currently_monitoring:
            print_id = open_window(printer_config, job)
        elif currently_monitoring and job.state != "Printing":
            close_window(print_id, end_reason=job.state)
            currently_monitoring = False
        if currently_monitoring:
            frame = await octoprint.get_snapshot(printer_config)
            db.save_frame(frame, print_id, printer_config.id)
            stage1 = predictor.predict_stage1(frame)
            if stage1.label == "failure" and stage1.confidence > threshold:
                stage2 = predictor.predict_stage2(frame)
                decision.record(printer_config.id, stage1, stage2)
                if decision.should_pause(printer_config.id):
                    await action.pause_and_notify(printer_config, stage1, stage2, frame)
        await asyncio.sleep(10)

async def main():
    config = load_config()
    await asyncio.gather(*[monitor_printer(p) for p in config.printers], heartbeat_loop())
    # FastAPI webhook server runs alongside via uvicorn
```

Tests (`tests/`):
- `test_octoprint_client.py` — mock OctoPrint to 200/500/timeout, verify retry behavior
- `test_decision.py` — rolling window vote with synthetic predictions
- `test_action.py` — mock pause + notify, verify retry-with-backoff, verify SHADOW_MODE bypasses pause

Acceptance:
- Service starts and runs against one real OctoPrint with `MockPredictor` returning "healthy"
- A print starts → window opens → frames pulled every 10s → all logged to SQLite
- Print ends → window closes cleanly
- Heartbeat fires every 5 min to ntfy
- Killing the service and restarting recovers gracefully (reads SQLite for in-flight prints)

---

## Milestone 6 — Wire Real Models In (1-2 days)

Goal: replace `MockPredictor` with real stage 1 + stage 2 Hailo HEF engines.

Note: Hailo compilation can happen on the training machine (x86 Linux) OR on the Pi. Training-machine compilation is faster; Pi-side compilation works but is slower. Either way the resulting `.hef` is the same.

Steps:
1. **Build the calibration set.** Pick ~500 representative frames covering healthy + all failure classes. Mix per-printer lighting variation. Copy them into `calib_frames/`. The compiler needs these to tune INT8 quantization scales.
2. **Install Hailo Dataflow Compiler** on the training machine (x86 Linux required — not macOS):
   ```bash
   # Download from https://hailo.ai/developer-zone/ (free account needed)
   pip install hailo_dataflow_compiler-*.whl
   pip install hailo_model_zoo
   ```
3. **Export stage 1 from Ultralytics to ONNX:**
   ```bash
   uv run yolo export model=runs/stage1_binary/weights/best.pt format=onnx imgsz=224 opset=11
   # produces best.onnx
   ```
4. **Compile ONNX to HEF:**
   ```bash
   hailo parser onnx best.onnx --hw-arch hailo8
   hailo optimize best.har --calib-set-path calib_frames/ --hw-arch hailo8
   hailo compile best_optimized.har --hw-arch hailo8
   # produces best.hef
   mv best.hef models/stage1_v1.hef
   ```
5. **If compilation fails** with "unsupported layer" errors, you have three options in order of preference:
   - a. Re-export from Ultralytics with a different opset (try 11, 12, 13)
   - b. Use Hailo Model Zoo's pre-compiled MobileNetV2 / EfficientNet-Lite0, fine-tune the classifier head, recompile only the head (the backbone stays as the pre-compiled `.hef`)
   - c. Fall back to YOLOv5n-cls (older but maximum Hailo compatibility)
6. **Repeat steps 3-4 for stage 2** producing `models/stage2_v1.hef`.
7. **Install HailoRT on the Pi 5:**
   ```bash
   # Follow Hailo's Pi 5 install guide. Roughly:
   sudo apt install hailo-all
   # Verify the AI HAT is detected:
   hailortcli fw-control identify
   # should print the device info
   ```
8. **Copy HEFs to the Pi:** `scp models/*.hef pi@inference-host:/opt/print-vision/models/`
9. **Update `service/inference.py`** to load HEFs via HailoRT Python bindings:
   ```python
   from hailo_platform import HEF, VDevice, ConfigureParams, HailoStreamInterface
   hef = HEF('/opt/print-vision/models/stage1_v1.hef')
   target = VDevice()
   network_group = target.configure(hef, ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe))[0]
   # ... run inference per frame
   ```
   (Or, simpler: use the `hailo_apps_infra` Python package which wraps this with a clean classification helper.)
10. **Verify predictions** on 20-30 test frames match what Ultralytics returned during training validation. If accuracy drops noticeably (more than ~2-3 percentage points), your calibration set is too narrow — expand it and recompile.

Acceptance:
- `service/inference.py` returns real predictions on the Pi
- Inference latency < 100ms per frame on Hailo (you have 10s of budget, this is huge headroom)
- Per-class accuracy on a held-out test set matches training-time accuracy within 3%

---

## Milestone 7 — Notification + Labeling Loop (2 days)

Goal: ntfy notifications with action buttons that feed labels back into SQLite.

Steps:
1. Set up ntfy: either `ntfy.sh/your-secret-topic` (hosted, free, easy) or self-host
2. Implement `service/notify.py`:
   - Send notification with `Title: "PRINTER 03 — bed_adhesion (87% conf)"`, attached snapshot, two action buttons
   - Button format (HTTP POST action):
     ```
     Actions: http, Real failure, http://192.168.1.50:8000/ntfy?frame_id=<id>&label=confirmed, method=POST;
              http, False alarm, http://192.168.1.50:8000/ntfy?frame_id=<id>&label=false_positive, method=POST
     ```
3. Implement `service/webhook.py` (FastAPI route):
   - `POST /ntfy` with `frame_id` and `label` query params
   - Update SQLite: `UPDATE frames SET operator_label = ?, operator_responded_at = NOW() WHERE id = ?`
   - Return 200
4. Wire `service/action.py` to call notify on every would-pause / actual pause
5. Test end-to-end on your phone: simulate a failure, get notification, tap button, see SQLite row update

Acceptance:
- Notifications arrive on phone with snapshot + buttons
- Tapping either button updates the SQLite row correctly
- A stale button tap (after print already ended) is handled gracefully

---

## Milestone 8 — Shadow Mode Deployment (1 week observation, ~0.5 day setup)

Goal: deploy to all 10 printers, no auto-pause, measure false-positive rate.

Steps:
1. Mount **256 GB external USB 3.0 SSD** on Pi 5 at `/mnt/ssd`. Format as ext4 (avoid exFAT — slow for many-small-file workloads).
2. Copy code: `git clone` on Pi, `uv sync`
3. Place `config.yaml` with all 10 printer entries, `shadow_mode: true`
4. Install systemd unit (`deploy/print-vision.service`):
   ```
   [Service]
   ExecStart=/opt/print-vision/.venv/bin/python -m service.main
   Restart=always
   RestartSec=10
   ```
5. `systemctl enable --now print-vision`
6. Set up tiered retention cron (run daily at 03:00):
   ```
   0 3 * * * /opt/print-vision/.venv/bin/python scripts/retention_cleanup.py
   ```
   The script deletes hot-tier files >7 days old (unless flagged/labeled/sampled) and warm-tier files >90 days old (unless operator-labeled). Always check free disk space first and log per-tier counts removed.
7. Watch for one full week. Track:
   - How many "would-pause" decisions fired
   - How many were false positives (you decide by reviewing the snapshot)
   - Latency from visual onset of real failures to detection

Acceptance:
- 1 week of clean run, no crashes
- False-positive rate measured; if > 1/printer/week, tune `stage1_confidence_threshold` upward
- True-positive recall measured; if any real failure was missed, investigate

**Decision gate:** If FP rate is acceptable, proceed to Milestone 9. If not, retrain v2 with newly-labeled frames before proceeding.

---

## Milestone 9 — Live on One Printer (1 week observation)

Goal: prove auto-pause works in production on a single printer before fleet rollout.

Steps:
1. In `config.yaml`, change `shadow_mode: true` to per-printer config:
   ```yaml
   inference:
     shadow_mode_default: true
   per_printer_overrides:
     printer_01:
       shadow_mode: false       # ONLY printer_01 goes live
   ```
2. `systemctl restart print-vision`
3. Watch printer_01 closely for 1 week
4. If a pause fires, investigate within 1 hour. Tap label button. If false positive, raise threshold.

Acceptance:
- 1 week, no catastrophic false pauses (printer_01's pause didn't damage anything)
- At least 1 real failure caught + auto-paused successfully
- Operator trusts the system enough to roll out

---

## Milestone 10 — Rollout to All 10 (0.5 day)

Goal: full production.

Steps:
1. Flip `shadow_mode_default: false` in config
2. Remove per-printer overrides
3. `systemctl restart print-vision`
4. Watch dashboard / notifications for 24 hours
5. Document the runbook (`RUNBOOK.md`) — how to silence a printer, how to investigate a false positive, how to retrain

Acceptance:
- All 10 printers under auto-pause
- Heartbeat watchdog active
- Retention cron running
- You sleep through the night without checking on prints

---

## Ongoing — v2 and beyond

- Track FP/FN rate weekly via a small SQL query
- When you have ~500 new operator-labeled frames (mix of `confirmed` and `false_positive`), retrain v2
- Always score v2 against the curated eval set before deploying
- v2 must beat v1 on every class to ship
- Keep v1 weights — if v2 underperforms in production, roll back

---

## Risk register

| Risk | Mitigation |
|---|---|
| Stage 2 has too few samples per class | Decided at Milestone 1; merge classes if needed |
| Eval set F1 < 0.85 even after tuning | Collect more data, or accept v1 ships as alert-only (no auto-pause) |
| OctoPrint API key rotation breaks service | Store keys in env vars referenced by config; document rotation |
| Pi 5 dies | Order a spare (~$80 + ~$249 = ~$330 to keep on hand); Jetson Orin Nano is the documented fallback |
| Hailo compilation fails on YOLOv8n-cls | Fallback chain: try different ONNX opset → use Hailo Model Zoo MobileNetV2 with custom head → fall back to YOLOv5n-cls |
| INT8 quantization causes accuracy drop > 3% | Expand calibration set with more failure examples; re-run optimize step |
| Hosted ntfy goes down | Self-host ntfy on the Pi alongside the service |
| Operator stops tapping buttons | Plan v2 retraining cadence based on calendar (quarterly), not just label count |
| 256 GB SSD fills up unexpectedly | `min_free_gb: 30` config bail-out + retention script logs free space daily. If hit, raise sampling rate threshold or shorten hot retention. Spare is a $20 swap. |
| Retention script bug deletes too much | All retention queries use `image_path IS NOT NULL` predicate AND skip operator-labeled frames. SQLite rows preserved even after disk delete — full audit trail. Test on a dev SQLite before running on prod. |

---

## What this plan does NOT include (deferred)

- Operator dashboard (stretch, post-v1)
- Multi-printer-firmware support (Klipper, Bambu) — OctoPrint only for v1
- Slicer-render-vs-camera comparison
- Failure recovery automation (cancel, restart, queue)
- Public release / open source
