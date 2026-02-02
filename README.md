# PctoDL: Adaptive GPU Throughput Optimization for Deep Learning Inference with Power Constraints

Reference implementation and benchmarking utilities for **PctoDL**, a GPU inference throughput optimizer under **power constraints** using **NVIDIA MPS + DVFS** and lightweight performance/power prediction models.
---

## Overview

PctoDL targets **multi-tenant inference on a single GPU** under a hard **power cap**, and coordinates:

- **MPS partitioning** (per-process GPU resource slices)
- **Batch sizing** (per-tenant load shaping)
- **DVFS / power control** (frequency selection to respect the cap)

The repo includes both an online scheduler (`src/main.py`) and the offline tooling needed to profile, fit, and validate the predictor used by PctoDL.

---

## What's in This Repo

- **Benchmark matrix runner (paper tasks):** `run_benchmark.py`
- **Online scheduler (MPS isolation + power control):** `src/main.py`
- **Predictor (SM/MEM/throughput/power):** `predict.py` + `fitting_results/`
- **Frequency-aware greedy partitioning:** `greedy_partitioning.py`
- **Profiling toolkit:** `src/profile_mps.py` + `run_sweep.sh`
- **Model fitting:** `optimize_model_params.py`
- **Thermodynamic controller (power-capped DVFS):** `thermo_control.py`

High-level workflow:

```
profile (src/profile_mps.py / run_sweep.sh)
  -> traces (mps_profile/, scale_profile/)
  -> fit (optimize_model_params.py)
  -> params (fitting_results/)
  -> predict (predict.py)
  -> optimize & run (greedy_partitioning.py + src/main.py)
```

---

## System Requirements

- Linux + NVIDIA GPU + working `nvidia-smi`
- NVIDIA MPS support (Multi-Process Service)
- Python 3.9+ (tested with PyTorch + CUDA)
- Out-of-the-box benchmark tasks target `3080Ti` and `A100` (other GPUs require re-profiling + `src/config/platform.json` updates)

Note: some experiments require privileged GPU controls (power limits / clocks / compute mode). See **Permissions & Security Notes**.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## Datasets (Optional but Recommended)

The system can fall back to synthetic tensors if datasets are missing. For paper-like runs with real data, place datasets under:

- ImageNet proxy (Tiny ImageNet): `data/imagenet/tiny-imagenet-200/test/images`
- Caltech256: `data/Caltech256/`

Example download:

```bash
# Tiny ImageNet
mkdir -p data/imagenet
cd data/imagenet
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
cd ../..

# Caltech-256
mkdir -p data/Caltech256
cd data/Caltech256
wget https://sagemaker-sample-files.s3.amazonaws.com/datasets/image/caltech-256/256_ObjectCategories.tar
tar -xvf 256_ObjectCategories.tar
cd ../..
```

---

## Quick Start: Reproduce the Benchmark Matrix

1) Detect GPU:

```bash
python run_benchmark.py --detect-gpu
```

2) Run benchmarks (auto-detect GPU + interactive algorithm selection):

```bash
python run_benchmark.py
```

3) If anything gets stuck (dangling MPS / zombie python):

```bash
bash src/cleanup_processes.sh
```

### Common `run_benchmark.py` Commands

| Scenario | Command |
|---|---|
| Run (auto-detect GPU) | `python run_benchmark.py` |
| Run a specific GPU | `python run_benchmark.py --gpu 3080Ti` |
| Run one algorithm | `python run_benchmark.py --gpu 3080Ti --algorithm pctodl` |
| Run multiple algorithms | `python run_benchmark.py --gpu 3080Ti --algorithm pctodl,morak` |
| Run all (batch) | `python run_benchmark.py --gpu 3080Ti --algorithm all` |
| Run selected task IDs | `python run_benchmark.py --gpu A100 --task-id 20 21 22` |
| Resume from task ID | `python run_benchmark.py --gpu 3080Ti --start-from 5` |
| Dry run (print commands only) | `python run_benchmark.py --gpu 3080Ti --algorithm pctodl --dry-run` |
| Save a summary report | `python run_benchmark.py --save-summary` |
| Interleave algorithms per task | `python run_benchmark.py --interleave` |
| Enable shadow optimizer | `python run_benchmark.py --enable-shadow` |

### Benchmark Notes

- Supported task sets:
  - `3080Ti`: IDs **1-14** (fixed power), **29-30** (dynamic power schedule)
  - `A100`: IDs **15-28** (fixed power), **29-30** (dynamic power schedule)
- Models in the task matrix: `mnasnet`, `densenet201`, `efficientnet_v2_m`, `maxvit_t`, `mobilenet_v2`, `resnet50`, `vgg19`
- Datasets in the task matrix: `imagenet`, `caltech256`
- Output:
  - `benchmark_results/<GPU>_<algorithm>/`
  - Fixed-power CSV: `<task_id>_<model>_<dataset>_<power_cap>W.csv`
  - Dynamic-power CSV: `_power_down.csv`, `_power_up.csv`, or `_dynamic.csv`

---

## Run a Single Experiment (Direct)

If you don't need the benchmark matrix, you can call the scheduler directly:

```bash
python src/main.py \
  --algorithm pctodl \
  --model resnet50 \
  --dataset imagenet \
  --power-cap 130 \
  --iterations 10
```

Algorithms implemented in `src/main.py`: `pctodl`, `morak`, `batchdvfs`, `directcap`, `powercap`.

---

## Predictor (Offline What-if Analysis)

`predict.py` loads trained parameters from `fitting_results/` and predicts:
SM util, MEM util, throughput, and power for a given `(MPS%, batch sizes, SM freq)`.

```bash
python predict.py --model mobilenet_v2 --p 50 25 --b 16 16 --f 1950
```

Useful options:

- `--dataset` (default: `imagenet`)
- `--platform` (e.g., `3080ti`, `a100`)
- `--mem-freq` or `--gpu-mem-freq` (pick models trained for a specific memory clock)

Python API:

```python
from predict import PctoDLPredictor

predictor = PctoDLPredictor("mobilenet_v2", platform="3080ti", dataset="imagenet")
u_sm, u_mem, tput, pwr = predictor.predict_all([50, 25], [16, 16], 1950)
```

---

## Greedy Partitioning (Find Best MPS/Batch Under a Power Cap)

```bash
python greedy_partitioning.py --model mobilenet_v2 --p_cap 150 --mps_step 5
```

Useful options:

- `--dataset` (default: `imagenet`)
- `--platform` (e.g., `3080ti`, `a100`)
- `--mem-freq` (fix a memory clock; otherwise it searches)
- `--vectorized` (faster search; experimental)

This implements a frequency-aware greedy search (Master Partition + FindBestNext) using the predictor, and prints a verification command you can run via `src/profile_mps.py`.

---

## Profiling Toolkit (`src/profile_mps.py`)

GPU profiling tool for MPS (Multi-Process Service) stress testing and data collection.

Common commands:

| Test type | Command |
|---|---|
| Stress test (defaults) | `python src/profile_mps.py` |
| Single config | `python src/profile_mps.py --single --models mobilenet_v2 --p 25 25 25 --b 8 8 8` |
| Freq scaling | `python src/profile_mps.py --profile-freq-scaling --models mobilenet_v2` |
| Static power (idle) | `python src/profile_mps.py --static --f 1500` |
| Skip privileged controls | `python src/profile_mps.py --no-sudo` |

Key options (single test):

- `--p`: MPS % per task (e.g., `25 25 25`)
- `--b`: batch size per task (e.g., `8 8 8`)
- `--f`: target SM frequency (MHz)
- `--mem-freq`: target memory frequency (MHz)
- `--dataset`: `Caltech256` or `ImageNet` (optional)

---

## Profiling & Fitting (Porting to a New GPU)

`fitting_results/` contains reference parameters, but they may not match your GPU/driver/CUDA stack. For best accuracy, re-profile and re-fit:

1) Frequency sweep (vertical): edit and run `run_sweep.sh`
```bash
./run_sweep.sh
```

2) MPS/batch sweep (horizontal):
```bash
python src/profile_mps.py
```

3) Fit model parameters:
```bash
# Fit all supported models
python optimize_model_params.py

# Fit one model / one step
python optimize_model_params.py --model mobilenet_v2 --step sm
```

Tip: if your GPU has different supported clock states, update `src/config/platform.json` so greedy search can use the real discrete frequencies.

---

## Permissions & Security Notes

- Some scripts call `sudo` to control GPU clocks/power and to manage MPS.
- The code reads `SUDO_PASSWORD` from the environment in several places (e.g., `src/main.py`, `src/profile_mps.py`). **Do not hardcode or commit secrets.**
- If you prefer not to provide a password, you can:
  - run only predictor/greedy (offline), or
  - use profiling modes that skip privileged controls (`src/profile_mps.py --no-sudo`), or
  - configure your system's sudoers for the required `nvidia-smi` commands.

---

## Repository Layout

```
.
├── src/                      # scheduler, profiler, configs
├── fitting_results/          # trained predictor parameters (reference)
├── mps_profile/              # profiling traces (MPS/batch sweep)
├── scale_profile/            # frequency scaling traces
├── benchmark_results/        # benchmark outputs
├── run_benchmark.py          # reproducing benchmark matrix
├── predict.py                # predictor CLI + API
├── greedy_partitioning.py    # greedy partitioning search
└── optimize_model_params.py  # fit parameters from profiling traces
```

---

## Citation

If you use this repository in your research, please cite the paper:

```bibtex
@inproceedings{pctodl,
  title     = {PctoDL: Adaptive GPU Throughput Optimization for Deep Learning Inference with Power Constraints},
  author    = {TBD},
  booktitle = {TBD},
  year      = {TBD}
}
```

---

## Contact / Issues

If you hit reproducibility issues (driver/CUDA/clock states), please include:

- GPU model + driver version (`nvidia-smi`)
- Python + PyTorch versions
- the command you ran and the generated CSV/log file
