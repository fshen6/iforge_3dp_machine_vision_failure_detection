# Training Hardware — RunPod

Decision: RunPod cloud GPU. Pay per hour, no long-term commitment, modern stack out of the box.

## Decision summary

| | Choice | Why |
|---|---|---|
| Provider | RunPod | Cheapest sane option, real SSH access, GPU images pre-built |
| GPU | RTX 3090 24 GB | Sweet spot. Plenty of VRAM, tensor cores for AMP, ~$0.34/hr Community Cloud |
| Pod type | On-Demand (not Spot) | Training runs are short (~2 hr). Spot's ~50% savings not worth the interrupt risk. |
| Image | PyTorch 2.1 or 2.4 with CUDA 12.x | Pre-installed Python, PyTorch, CUDA |
| Disk | 30 GB container + 30 GB Network Volume for dataset | Volume persists between sessions; only $0.05/GB/month while idle |
| Region | Closest to your physical location | Lower SSH/SCP latency |

**Cost per full v1 training run (both stages):** ~$0.70–1.50.

## Workflow at a glance

```
┌─────────────────────────────────────────────────────────────┐
│  YOUR LAPTOP/DESKTOP (CPU only, free)                       │
│  1. Decode 80 GB raw timelapses → ~15 GB JPEG frames         │
│  2. Build labels.csv, splits.json, yolo/ symlink view        │
│  3. Compress: tar -cf print_data.tar yolo/ frames/           │
│  4. Upload to RunPod Network Volume (~25 min, one-time)      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  RUNPOD POD (RTX 3090, $0.34/hr, ~2 hours total)             │
│  5. ssh in, cd to volume                                     │
│  6. uv sync, install ultralytics                             │
│  7. yolo classify train data=yolo/stage1 ... (~45 min)       │
│  8. yolo classify train data=yolo/stage2 ... (~45 min)       │
│  9. Grade against eval set                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  DOWNLOAD                                                   │
│  10. scp best.pt files back to laptop                        │
│  11. Terminate pod (Network Volume keeps dataset for next run)│
└─────────────────────────────────────────────────────────────┘
```

## Step-by-step

### 0. One-time RunPod setup

1. Sign up at runpod.io, add a payment method, add ~$10 of credit.
2. Generate an SSH key locally (`ssh-keygen -t ed25519 -C "runpod"`) and paste the public key into RunPod → Settings → SSH Keys.
3. Create a **Network Volume**:
   - Storage > Network Volumes > New
   - Size: 50 GB (gives headroom for raw + decoded + checkpoints + models output)
   - Pick the same region you'll launch pods in
   - Cost: $0.05/GB/month = $2.50/month while idle. Trivial.

### 1. Decode and prep dataset locally

This is the slow CPU-bound step. Do it on your laptop/desktop **before** spinning up GPU time so you don't pay for idle GPU.

Your local scripts from Milestone 2 produce `print_data/` (see [`DATASET_STRUCTURE.md`](DATASET_STRUCTURE.md)). After that finishes:

```bash
# Pack just what RunPod needs
cd print_data
tar -cf yolo_view.tar yolo/
tar -cf frames.tar frames/
ls -lh *.tar
# yolo_view.tar:  ~50 MB  (all symlinks)
# frames.tar:    ~15 GB
```

### 2. Upload to the Network Volume

The volume is accessible by mounting it to a pod. Spin up the cheapest pod available (e.g. a CPU-only pod or a $0.15/hr GPU pod) with the volume mounted, then upload over SSH. Don't use a 3090 for this — wasting GPU time.

```bash
# From your laptop, after the pod is running with the volume at /workspace
rsync -avzP yolo_view.tar frames.tar labels.csv splits.json \
  root@<runpod-ssh-host>:/workspace/

# On the pod
cd /workspace
tar -xf yolo_view.tar
tar -xf frames.tar
# Rebuild symlinks if tar didn't preserve them
ls yolo/stage1/train/healthy/ | head -3
```

Once the volume has the data, terminate that cheap pod. The data stays on the volume.

### 3. Launch the training pod

In RunPod web UI:
- GPU: **RTX 3090** (Community Cloud is cheapest, ~$0.34/hr; Secure Cloud is ~$0.44/hr with better SLAs)
- Pod template: **PyTorch 2.4** or similar (any recent PyTorch image is fine)
- Attach the Network Volume at `/workspace`
- Set container disk to 30 GB (for caches, model checkpoints, etc.)
- Click Deploy

Wait ~60 seconds for the pod to boot. Connect via SSH (RunPod gives you a port).

### 4. Environment setup on the pod

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Project setup
cd /workspace
git clone https://github.com/fshen6/iforge_3dp_machine_vision_failure_detection.git
cd iforge_3dp_machine_vision_failure_detection
uv sync                                # installs deps from pyproject.toml

# Sanity check GPU
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA GeForce RTX 3090
```

### 5. Train both stages

```bash
# Stage 1 — binary
uv run python training/train_stage1.py \
  --data /workspace/yolo/stage1 \
  --epochs 30 \
  --batch 64 \
  --imgsz 224 \
  --device 0

# ~45 min on a 3090. Saves to runs/stage1_binary/weights/best.pt

# Stage 2 — multi-class on failure-only frames
uv run python training/train_stage2.py \
  --data /workspace/yolo/stage2 \
  --epochs 30 \
  --batch 64 \
  --imgsz 224 \
  --device 0

# ~45 min on a 3090. Saves to runs/stage2_multiclass/weights/best.pt
```

**Pro tip: run inside `tmux`.** Otherwise an SSH disconnect kills training.

```bash
tmux new -s train
# ... run training commands ...
# Detach: Ctrl-b then d
# Reattach later: tmux attach -t train
```

### 6. Grade against the eval set

```bash
uv run python training/grade_against_eval.py \
  --model runs/stage1_binary/weights/best.pt \
  --eval /workspace/yolo/eval/stage1

uv run python training/grade_against_eval.py \
  --model runs/stage2_multiclass/weights/best.pt \
  --eval /workspace/yolo/eval/stage2
```

If eval F1 is below the threshold defined in `ENGINEERING_PLAN.md` Milestone 3/4, iterate on the pod (cheap) before you tear down.

### 7. Pull artifacts down and terminate

```bash
# From your laptop
mkdir -p ~/print-vision-runs/$(date +%Y%m%d)
cd ~/print-vision-runs/$(date +%Y%m%d)
scp -r root@<runpod-ssh-host>:/workspace/iforge_3dp_machine_vision_failure_detection/runs/stage1_binary/weights/best.pt stage1_v1.pt
scp -r root@<runpod-ssh-host>:/workspace/iforge_3dp_machine_vision_failure_detection/runs/stage2_multiclass/weights/best.pt stage2_v1.pt
```

Then in RunPod web UI: terminate the pod. **The Network Volume keeps your dataset** so the next training run skips the upload step entirely.

## Hailo compilation (Milestone 6)

The Hailo Dataflow Compiler needs x86 Linux. Two options:

**Option A — Use a fresh RunPod CPU pod.** Spin up a cheap CPU-only pod ($0.10-0.20/hr), install the Hailo Dataflow Compiler, mount the same Network Volume, run the ONNX → HEF pipeline. ~30 min, total cost ~$0.10.

**Option B — Use the same training pod.** While the 3090 pod is still alive after training, just install the Hailo compiler there and compile. The 3090 is overkill for compilation (it's CPU-side work) but saves the overhead of spinning a new pod.

Either way the output is a `.hef` file you `scp` to the Pi 5.

## Cost summary

| Item | Cost |
|---|---|
| RTX 3090 Community Cloud | $0.34/hr |
| Network Volume (50 GB, idle storage) | $2.50/month |
| One-time upload pod (cheap CPU pod, ~30 min) | $0.05 |
| Stage 1 training (~45 min) | $0.26 |
| Stage 2 training (~45 min) | $0.26 |
| Eval + buffer (~15 min) | $0.09 |
| Hailo compilation (CPU pod, ~30 min) | $0.05 |
| **Per training cycle total** | **~$0.70–1.00** |

A year of retraining quarterly = **~$4** in GPU time + $30 in volume storage = $34. Less than one decent meal.

## Gotchas

- **Use Community Cloud for cost, Secure Cloud for reliability.** If you can't tolerate a rare host-side restart, pay the extra $0.10/hr for Secure Cloud.
- **Pods auto-stop after the time you set, but they keep billing while "stopped" if container disk persists.** Make sure to *terminate* (not just stop) when done. Only the Network Volume keeps billing while idle.
- **Network Volume is region-locked.** If you want to pick a pod in a different region later, you'll have to copy the data to a new volume there.
- **Save checkpoints every epoch.** Ultralytics does this by default. If the pod dies mid-training, you restart from `last.pt` not from scratch.
- **Don't pay for GPU during dataset upload.** Use the cheap CPU pod or spot-pricing GPU pod for upload, only spin the 3090 for actual training.
- **Public Network Volume disks are NOT encrypted at rest by default.** If your timelapses contain anything sensitive (probably not, but worth knowing), use Secure Cloud which is encrypted.

## When to revisit this choice

- **Cost > $20/month consistently:** Buy a used RTX 3090 (~$700-900) — payback in ~3 years of monthly retraining. Or build a dedicated training box.
- **iForge gets HPC access at Sheffield:** Stanage cluster has A100s, free for staff/students. Switch to HPC and keep RunPod as backup.
- **You start training models bigger than YOLOv8m-cls:** May need A100 (~$1.50/hr) or H100 (~$2-3/hr) tier.
