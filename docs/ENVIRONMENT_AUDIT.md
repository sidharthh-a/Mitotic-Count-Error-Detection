# Environment Audit — 5 November 2026

This audit was performed directly on the machine hosting the project. It replaces the
historical MacBook Air M4 description in the research blueprint, which describes the
**original pilot machine only** and must not be assumed valid here.

## Measured facts

| Item | Value | How measured |
|---|---|---|
| Operating system | Windows 11 (build 10.0.26300), x86_64 | `uname -a` under MSYS2/Git-Bash |
| Shell | Git Bash (MINGW64_NT-10.0) | `uname -a` |
| CPU | 11th Gen Intel Core i5-1155G7 @ 2.50 GHz | `Get-CimInstance Win32_Processor` |
| CPU cores / threads | 4 physical / 8 logical | same |
| RAM | 7.7 GiB total | `Win32_ComputerSystem.TotalPhysicalMemory` |
| GPU | Intel Iris Xe Graphics (integrated), 1 GiB shared | `Win32_VideoController` |
| CUDA | Not available | `torch.cuda.is_available() == False` |
| Python | 3.13.15 (`C:\Users\sidha\AppData\Local\Programs\Python\Python313`) | `python --version` |
| PyTorch | 2.7.0+cpu (CPU-only wheel) | `torch.__version__` |
| Torchvision | 0.22.0+cpu | `torchvision.__version__` |
| torch.get_num_threads() | 4 | runtime |
| Git | 2.56.0.windows.2 | `git --version` |
| Conda | not installed | `conda --version` |
| Disk C: | 403.0 GB total, **158 GB free** | `Win32_LogicalDisk` |
| Disk D: (project location) | 72.8 GB total, **3.8 GB free** | `Win32_LogicalDisk` |

## Installed Python packages relevant to the project

Present: `torch`, `torchvision`, `numpy`, `Pillow`, `opencv-python (cv2)`, `pandas`,
`matplotlib`, `scipy`, `scikit-learn`, `streamlit`, `tqdm`, `PyYAML`, `pytest`, `requests`.

Missing at audit time: `tifffile`, `SimpleITK`, `albumentations`, `pycocotools`,
`torchmetrics`, `evalutils`.

## Consequences for execution strategy

1. **Local GPU training is impossible.** There is no CUDA device and no Apple MPS.
   The blueprint's historical MPS-based pilot timings do not transfer.
2. **CPU-only RetinaNet training is not viable at study scale.** A ResNet-50-FPN
   RetinaNet at 512x512 on 4 cores / 7.7 GiB RAM is roughly two orders of magnitude
   slower than a GPU. Full source-baseline training to a usable checkpoint within the
   seven-day window is not credible on this machine.
3. **Decision: train on Google Colab (CUDA), develop and evaluate locally.**
   - Local machine: dataset audit, annotation parsing, region generation, split
     freezing, selection-strategy implementation, evaluator, tests, Streamlit demo,
     result tables and figures.
   - Colab: source detector training, error-predictor fitting, adaptation fine-tunes,
     sliding-window inference over full specimen images.
   - The codebase must therefore be Colab-compatible: a single installable package,
     config-driven entry points, no absolute local paths, and resumable checkpoints.
4. **Disk is the binding constraint, not RAM.** The project lives on `D:` which has only
   ~3.8 GB free. A single MIDOG++ specimen TIFF is ~145-160 MiB, so `D:` cannot hold
   even the 55-specimen development cohort (~8 GB) plus checkpoints. Image storage must
   move to `C:` (158 GB free) via a configurable data root, or be handled on Colab only.
5. **Memory discipline matters locally.** A 7215x5412 RGB uint8 TIFF is ~117 Mpx
   (~352 MB as a NumPy array). Local code must load one image at a time, use crop-level
   reads where possible, and never hold many full images in memory.
6. **torchvision 0.22 RetinaNet API.** `torchvision.models.detection.retinanet_resnet50_fpn_v2`
   accepts `num_classes`; its classification head has `num_classes` foreground outputs and
   torchvision's detection loss expects *background removed* targets. On CPU a
   `weights=None` / `weights="DEFAULT"` mismatch and the head-size mapping must be
   checked explicitly rather than assumed (the historical pilot had a category-1 vs
   category-2 confusion documented in the blueprint).

## Not verified

- No throughput benchmark of the local CPU trainer has been run yet. It will be measured
  and recorded in `PROJECT_STATUS.md` before any training schedule is promised.
- Colab availability, GPU type and session limits belong to the user's Google account and
  cannot be audited from this machine.
