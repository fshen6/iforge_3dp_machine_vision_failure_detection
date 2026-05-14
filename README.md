# iForge 3D Printer Failure Detection

Computer vision system for detecting in-progress failures across a 10-printer 3D print farm. Runs locally on a Raspberry Pi 5 with a Hailo-8 AI HAT, integrates with OctoPrint, auto-pauses on detected failure, and notifies the operator via ntfy with action buttons that feed labels back into the training set.

**Status:** planning complete, implementation not started.

## Quick links

| Document | What it is |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | Problem statement, premises, approaches considered, recommended approach, failure modes, what's NOT in scope. |
| [docs/ENGINEERING_PLAN.md](docs/ENGINEERING_PLAN.md) | Concrete 10-milestone build plan with file structure, SQLite schema, config shape, acceptance criteria per milestone. |
| [docs/TEST_PLAN.md](docs/TEST_PLAN.md) | Test plan — interactions to verify, edge cases, critical end-to-end paths, eval methodology. |

## At a glance

- **Hardware:** Raspberry Pi 5 8GB + Hailo-8 AI HAT (26 TOPS) + 1 TB external USB SSD
- **Model:** YOLOv8n-cls (stage 1 binary healthy/failure), YOLOv8s-cls (stage 2 multi-class failure type)
- **Inference cadence:** one frame per printer every 10 seconds
- **Detection coverage:** full print duration (not just first 10 minutes)
- **Failure response:** auto-pause via OctoPrint API + ntfy notification with `Real failure` / `False alarm` action buttons
- **Total cost:** ~$330 in hardware (Pi 5 + AI HAT) + a 1 TB SSD

## Approach (short version)

Two-stage classification:

```
Frame  →  Stage 1 (binary)  →  healthy?     →  log, continue
                            →  failure?     →  Stage 2 (multi-class)  →  spaghetti / bed_adhesion / layer_shift / other
                                                                       →  rolling-window vote (3 of 5)
                                                                       →  pause + notify with snapshot
```

Two-stage architecture is the answer to class imbalance: the binary stage uses the abundant healthy data, and the multi-class stage only sees failures so it never gets overwhelmed by the majority class.

## Build status

Tracked per milestone in [ENGINEERING_PLAN.md](docs/ENGINEERING_PLAN.md):

- [ ] M0 — Project setup
- [ ] M1 — Dataset audit
- [ ] M2 — Dataset prep
- [ ] M3 — Stage 1 training
- [ ] M4 — Stage 2 training
- [ ] M5 — Inference service skeleton
- [ ] M6 — Wire real models in (Hailo HEF compilation)
- [ ] M7 — Notification + labeling loop
- [ ] M8 — Shadow mode deployment (1 week observation)
- [ ] M9 — Live on one printer (1 week observation)
- [ ] M10 — Rollout to all 10

## Planning history

These docs were produced through a structured planning workflow:

1. **Office hours** (2026-05-11) — problem framing, premise challenge, approach selection. Output: `docs/DESIGN.md`.
2. **Engineering review** (2026-05-11) — architecture, code quality, test coverage, performance review. Six issues raised, six decisions made. Output: updates to `docs/DESIGN.md` + `docs/TEST_PLAN.md`.
3. **Engineering plan** (2026-05-14) — concrete 10-milestone build plan. Updated for Pi 5 + Hailo hardware choice. Output: `docs/ENGINEERING_PLAN.md`.

## License

TBD.
