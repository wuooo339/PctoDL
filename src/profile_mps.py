#!/usr/bin/env python3
"""
Automated MPS Stress Test Profile (3-Layer Matrix)
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import multiprocessing as mp

# =========================================================================
# Test matrix configuration
# =========================================================================

MPS_TEST_MATRIX = {
    5: {
        "task_counts": [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "batches": [4, 8, 16, 24]
    },

    10: {
        "task_counts": [1, 2, 4, 6, 8, 10],
        "batches": [4, 8, 16, 24, 32, 48]
    },
    15: {
        "task_counts": [1, 2, 3, 4, 5, 6],
        "batches": [4, 8, 16, 24, 32, 64]
    },

    20: {
        "task_counts": [1, 2, 3, 4, 5],
        "batches": [4, 8, 16, 24, 32, 64, 96, 128]
    },
    25: {
        "task_counts": [1, 2, 3, 4],
        "batches": [4, 8, 16, 24, 32, 64, 96, 128, 160]
    },

    30: {
        "task_counts": [1, 2, 3],
        "batches": [4, 8, 16, 24, 32, 64, 96, 128, 160]
    },
    35: {
        "task_counts": [1, 2],
        "batches": [4, 8, 16, 24, 32, 64, 96, 128, 160, 192, 256]
    },
    40: {
        "task_counts": [1, 2],
        "batches": [4, 8, 16, 24, 32, 64, 96, 128, 160, 192, 256]
    }
}

# Global settings
ITERATIONS = 1
DEFAULT_WARMUP_SECONDS = 8
DEFAULT_MEASURE_SECONDS = 7
DEFAULT_SUDO_PASSWORD = "wzk123456"
OUTPUT_DIR = "mps_profile"

# =========================================================================

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _ensure_src_on_path() -> None:
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

def _set_performance_mode(*, sudo_password: Optional[str], gpu_index: int, mem_freq: Optional[int] = None) -> None:
    """Lock clocks to max; optionally set a custom memory clock."""
    if sudo_password is None: return
    try:
        subprocess.run(["bash", "-lc", f'echo "{sudo_password}" | sudo -S nvidia-smi -i {gpu_index} -pm 1'], capture_output=True, check=False)
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        max_sm = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
        target_mem = mem_freq if mem_freq is not None else pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)
        pynvml.nvmlShutdown()

        mem_str = f"{target_mem} MHz (custom)" if mem_freq is not None else f"{target_mem} MHz (max)"
        print(f"🚀 [Setup] Locking GPU Clocks -> SM: {max_sm} MHz, Mem: {mem_str}")

        cmds = [
            f'echo "{sudo_password}" | sudo -S nvidia-smi -i {gpu_index} -lmc {target_mem}',
            f'echo "{sudo_password}" | sudo -S nvidia-smi -i {gpu_index} -lgc {max_sm}'
        ]
        for c in cmds:
            subprocess.run(["bash", "-lc", c], capture_output=True, check=False)
    except Exception as e:
        print(f"⚠️ Failed to lock clocks: {e}")

def _reset_gpu_defaults(*, sudo_password: Optional[str], gpu_index: int) -> None:
    if sudo_password is None: return
    cmds = [
        f'echo "{sudo_password}" | sudo -S nvidia-smi -i {gpu_index} -rmc',
        f'echo "{sudo_password}" | sudo -S nvidia-smi -i {gpu_index} -rgc',
        f'echo "{sudo_password}" | sudo -S nvidia-smi -i {gpu_index} -rac',
    ]
    for c in cmds:
        subprocess.run(["bash", "-lc", c], capture_output=True, check=False)

def _best_effort_start_mps(*, sudo_password: Optional[str], repo_root: Path) -> None:
    if sudo_password is None: return
    script = repo_root / "start_mps.sh"
    
    pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    log_dir = os.environ.get("CUDA_MPS_LOG_DIRECTORY")
    
    # Auto-create log directory
    if log_dir:
        subprocess.run(["mkdir", "-p", log_dir], check=False)
        # Ensure log dir is writable before the daemon starts
        subprocess.run(f'echo "{sudo_password}" | sudo -S chown -R {os.getuid()} "{log_dir}"', shell=True, check=False)

    # Explicit UID (avoid sudo auto-detection)
    target_uid = os.getuid()
    
    if script.exists():
        print(f"🚀 [Setup] Starting MPS Service on {pipe_dir} (User: {target_uid})...")
        
        # Key: pass MPS_UID through the environment
        cmd = f'echo "{sudo_password}" | sudo -S env MPS_UID="{target_uid}" CUDA_MPS_PIPE_DIRECTORY="{pipe_dir}" CUDA_MPS_LOG_DIRECTORY="{log_dir}" "{script}"'
        
        # Capture output for debugging
        process = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
        
        if process.returncode != 0:
            print(f"❌ [Setup Error] Script failed:\n{process.stderr}")
        else:
            print(f"✅ [Setup] MPS Script executed.")
            # Python-side wait is a safety net; main logic is in the script
            import time
            for _ in range(50):
                if os.path.exists(pipe_dir):
                    break
                time.sleep(0.1)
def _best_effort_shutdown_mps(*, sudo_password: Optional[str], repo_root: Path) -> None:
    if sudo_password is None: return
    script = repo_root / "shutdown_mps.sh"

    pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    log_dir = os.environ.get("CUDA_MPS_LOG_DIRECTORY")

    if script.exists():
        print(f"🛑 [Cleanup] Stopping MPS Service...")
        cmd = f'echo "{sudo_password}" | sudo -S env CUDA_MPS_PIPE_DIRECTORY="{pipe_dir}" CUDA_MPS_LOG_DIRECTORY="{log_dir}" "{script}"'
        result = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, check=False, cwd=str(repo_root), timeout=30)

        if result.returncode == 0:
            print(f"✅ [Cleanup] MPS Service stopped.")
        else:
            print(f"⚠️ [Cleanup] MPS shutdown returned non-zero: {result.returncode}")
            if result.stderr:
                print(f"   stderr: {result.stderr.strip()}")
    else:
        print(f"⚠️ [Cleanup] shutdown_mps.sh not found, MPS may still be running.")
def _static_worker(model_name, batch_size, mps_percent, stop_event, ready_event, samples_counter, dataset_name=None, repo_root=None, save_output=False):
    """
    Static power worker: load data only, no model inference.
    Used to measure GPU power when data loading has no compute workload.
    """
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(mps_percent))

    try:
        _ensure_src_on_path()
        import torch
        import doInference
        import dataLoader
        import numpy as np

        dev = doInference.get_device()
        dtype = torch.float16 if dev.type == "cuda" else torch.float32

        # No model loading; device only

        # --- Data loading ---
        input_images = []
        if dataset_name and repo_root:
            if dataset_name.lower() == 'caltech256':
                data_path = repo_root / "data" / "Caltech256"
            elif dataset_name.lower() == 'imagenet':
                data_path = repo_root / "data" / "imagenet" / "tiny-imagenet-200" / "test" / "images"
            else:
                data_path = None

            if data_path and data_path.exists():
                print(f"[{os.getpid()}] Loading images from {data_path}...")

                output_dir_path = None
                if save_output:
                    output_dir = repo_root / "output"
                    output_dir.mkdir(exist_ok=True)
                    output_dir_path = str(output_dir)

                input_images = dataLoader.load_and_preprocess_images(
                    str(data_path), output_dir=output_dir_path
                )

                if not input_images:
                     print(f"[{os.getpid()}] Failed to load images, using synthetic data.")
            else:
                print(f"[{os.getpid()}] Dataset path {data_path} not found, using synthetic data.")

        use_synthetic = not input_images
        if use_synthetic:
            # Create tensor without inference
            inp = torch.zeros((int(batch_size), 3, 224, 224), device=dev, dtype=dtype)

        # Simulated warmup (tensor ops only)
        if use_synthetic:
            _ = inp.sum()  # Lightweight op to ensure tensor is usable
        else:
            sample_img = torch.from_numpy(input_images[0]).unsqueeze(0).to(dev).to(dtype)
            _ = sample_img.sum()

        if dev.type == "cuda":
            torch.cuda.synchronize()

        ready_event.set()

        image_idx = 0
        while not stop_event.is_set():
            try:
                if not use_synthetic:
                    # Build batches without inference (access only, no tensor compute)
                    for _ in range(int(batch_size)):
                        _ = input_images[image_idx % len(input_images)]
                        image_idx += 1

                # Truly static: idle without GPU compute
                time.sleep(0.05)

            except Exception as e:
                print(f"[{os.getpid()}] Error: {e}")
                break
    except Exception as e:
        print(f"[{os.getpid()}] Worker error: {e}")


def _child_worker(model_name, batch_size, mps_percent, stop_event, ready_event, samples_counter, dataset_name=None, repo_root=None, save_output=False):
    # --- Debug logging ---
    print(f"[{os.getpid()}] Debug Worker: UID={os.getuid()}")
    print(f"[{os.getpid()}] Debug Worker: MPS_PIPE={os.environ.get('CUDA_MPS_PIPE_DIRECTORY', 'NOT_SET')}")
    print(f"[{os.getpid()}] Debug Worker: THREAD_PERCENT={os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', 'NOT_SET')}")
    # -------------------

    # CRITICAL: Set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE BEFORE any CUDA initialization
    # This must be set before importing torch or any CUDA-related libraries
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(mps_percent))

    try:
        _ensure_src_on_path()
        import torch
        import doInference
        import dataLoader
        import numpy as np

        dev = doInference.get_device()
        dtype = torch.float16 if dev.type == "cuda" else torch.float32

        model = doInference.load_model(model_name, use_cache=True)

        # --- Data loading ---
        input_images = []
        if dataset_name and repo_root:
            if dataset_name.lower() == 'caltech256':
                data_path = repo_root / "data" / "Caltech256"
            elif dataset_name.lower() == 'imagenet':
                data_path = repo_root / "data" / "imagenet" / "tiny-imagenet-200" / "test" / "images"
            else:
                data_path = None

            if data_path and data_path.exists():
                print(f"[{os.getpid()}] Loading images from {data_path}...")

                output_dir_path = None
                if save_output:
                    output_dir = repo_root / "output"
                    output_dir.mkdir(exist_ok=True)
                    output_dir_path = str(output_dir)

                input_images = dataLoader.load_and_preprocess_images(
                    str(data_path), output_dir=output_dir_path
                )

                if not input_images:
                     print(f"[{os.getpid()}] Failed to load images, falling back to synthetic data.")
            else:
                print(f"[{os.getpid()}] Dataset path {data_path} not found, falling back to synthetic data.")

        use_synthetic = not input_images
        if use_synthetic:
            inp = torch.zeros((int(batch_size), 3, 224, 224), device=dev, dtype=dtype)

        # Warmup
        with torch.no_grad():
            if use_synthetic:
                _ = model(inp)
            else:
                sample_img = torch.from_numpy(input_images[0]).unsqueeze(0).to(dev).to(dtype)
                _ = model(sample_img)
            if dev.type == "cuda": torch.cuda.synchronize()

        ready_event.set()

        image_idx = 0
        while not stop_event.is_set():
            try:
                if use_synthetic:
                    batch_data = inp
                else:
                    # --- Build a real data batch ---
                    batch_imgs = []
                    for _ in range(int(batch_size)):
                        batch_imgs.append(input_images[image_idx % len(input_images)])
                        image_idx += 1
                    batch_data = torch.from_numpy(np.stack(batch_imgs)).to(dev).to(dtype)

                with torch.no_grad():
                    _ = model(batch_data)
                    if dev.type == "cuda": torch.cuda.synchronize()

                with samples_counter.get_lock():
                    samples_counter.value += int(batch_size)
            except torch.OutOfMemoryError:
                print(f"[{os.getpid()}] OOM detected, exiting gracefully")
                break
            except Exception as e:
                print(f"[{os.getpid()}] Error: {e}")
                break
    except torch.OutOfMemoryError:
        print(f"[{os.getpid()}] OOM during initialization, exiting")
    except Exception as e:
        print(f"[{os.getpid()}] Worker error: {e}")

def _measure(handle, seconds):
    """
    Measure GPU metrics via pynvml.

    Returns:
        (elapsed, power_avg, power_max, clock, sm_util, mem_util, pcie_mb)
    """
    import pynvml

    power_samples, clock_samples, sm_util_samples, mem_util_samples, pcie_mb_samples = [], [], [], [], []

    start_t = time.perf_counter()
    last_sample_t = start_t

    while time.perf_counter() - start_t < seconds:
        try:
            # Detect stalls (no samples for > 2s)
            current_t = time.perf_counter()
            if current_t - last_sample_t > 2.0:
                print(f"    [Warning] Measurement stuck, breaking...")
                break
            last_sample_t = current_t

            power_samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            clock_samples.append(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))

            utils = pynvml.nvmlDeviceGetUtilizationRates(handle)
            sm_util_samples.append(utils.gpu)
            mem_util_samples.append(utils.memory)

            try:
                tx_bytes = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                rx_bytes = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                pcie_mb_samples.append((tx_bytes + rx_bytes) / 1000.0)
            except Exception:
                pcie_mb_samples.append(0.0)

            time.sleep(0.05)
        except Exception:
            break

    elapsed = time.perf_counter() - start_t

    def _avg(samples):
        return sum(samples) / len(samples) if samples else 0.0

    return (
        elapsed,
        _avg(power_samples),
        max(power_samples) if power_samples else 0.0,
        _avg(clock_samples),
        _avg(sm_util_samples),
        _avg(mem_util_samples),
        _avg(pcie_mb_samples)
    )


def _measure_with_timeout(handle, measure_seconds, timeout_seconds=60):
    """Measurement with timeout."""
    import pynvml
    import threading

    result = {"data": None, "error": None, "done": False}

    def measure_thread():
        try:
            result["data"] = _measure(handle, measure_seconds)
        except Exception as e:
            result["error"] = str(e)
        finally:
            result["done"] = True

    t = threading.Thread(target=measure_thread, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if not result["done"]:
        print(f"    [Error] Measurement timed out after {timeout_seconds}s")
        return None

    return result["data"]


def save_result_to_csv(result_dict, filepath):
    """Save results to CSV."""
    file_exists = os.path.isfile(filepath)
    fieldnames = [
        "mps_percent", "num_tasks", "total_mps_load",
        "batch_size", "iteration", "throughput_total", "throughput_per_task",
        "power_avg", "power_max", "clock",
        "sm_util", "mem_util", "pcie_bw_mb"
    ]

    with open(filepath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_dict)


def _set_specific_clock(*, sudo_password: Optional[str], gpu_index: int, clock: int, mem_clock: Optional[int] = None) -> None:
    """Set GPU SM clock; optionally set memory clock."""
    if sudo_password is None: return
    try:
        # Set SM clock
        cmd = f'echo "{sudo_password}" | sudo -S nvidia-smi -i {gpu_index} -lgc {clock}'
        subprocess.run(["bash", "-lc", cmd], capture_output=True, check=True)
        print(f"✅ [Setup] Set GPU SM Clock to -> {clock} MHz")

        # Set memory clock (if specified)
        if mem_clock is not None:
            cmd = f'echo "{sudo_password}" | sudo -S nvidia-smi -i {gpu_index} -lmc {mem_clock}'
            subprocess.run(["bash", "-lc", cmd], capture_output=True, check=True)
            print(f"✅ [Setup] Set GPU Mem Clock to -> {mem_clock} MHz")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to set clock. Error: {e.stderr.decode()}")
    except Exception as e:
        print(f"⚠️ Failed to set clock: {e}")


def _get_supported_clocks(*, sudo_password: Optional[str], gpu_index: int, repo_root: Path) -> List[int]:
    """Get supported SM clock list."""
    clocks = []
    if sudo_password:
        try:
            print("INFO: Attempting to get supported clocks via nvidia-smi...")
            cmd = f'echo "{sudo_password}" | sudo -S nvidia-smi -i {gpu_index} --query-supported-clocks=gr --format=csv,noheader,nounits'
            result = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, check=True, timeout=10)
            found_clocks = [int(c.strip()) for c in result.stdout.split(',') if c.strip().isdigit()]
            if found_clocks:
                clocks = found_clocks
                print(f"INFO: Found {len(clocks)} supported clocks via nvidia-smi.")
        except Exception as e:
            print(f"⚠️ nvidia-smi failed to get clocks: {e}. Falling back to JSON config.")
            clocks = []

    if not clocks:
        platform_json_path = repo_root / "src" / "config" / "platform.json"
        if platform_json_path.exists():
            try:
                print(f"INFO: Reading clocks from {platform_json_path}...")
                import json
                with open(platform_json_path, 'r') as f:
                    data = json.load(f)
                # New format: data["platforms"][0]["clocks"]["SM_Clocks"]
                if "platforms" in data and len(data["platforms"]) > 0:
                    first_platform = data["platforms"][0]
                    clocks = first_platform.get("clocks", {}).get("SM_Clocks", [])
                    if clocks:
                        print(f"INFO: Loaded {len(clocks)} clocks from {platform_json_path} (platform: {first_platform['name']}).")
                else:
                    # Legacy format compatibility
                    clocks = data.get("SM Clocks", [])
                    if clocks:
                        print(f"INFO: Loaded {len(clocks)} clocks from {platform_json_path}.")
            except Exception as e:
                print(f"⚠️ Failed to read {platform_json_path}: {e}")
                clocks = []

    if not clocks:
        return []

    min_freq, max_freq = min(clocks), max(clocks)
    freq_steps = list(range(min_freq, max_freq + 150, 150))
    if max_freq not in freq_steps:
        freq_steps.append(max_freq)

    return sorted(list(set(freq_steps)))


def save_freq_scaling_result_to_csv(result_dict, filepath):
    """Save frequency scaling results."""
    file_exists = os.path.isfile(filepath)
    fieldnames = [
        "model_name", "batch_size", "target_freq", "measured_freq",
        "throughput", "power_avg", "power_max", "sm_util", "mem_util",
        "throughput_normalized"
    ]
    with open(filepath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_dict)


def run_freq_scaling_test(args, repo_root, sudo_pass):
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(args.gpu))

    supported_clocks = _get_supported_clocks(sudo_password=sudo_pass, gpu_index=args.gpu, repo_root=repo_root)
    if not supported_clocks:
        print("❌ Cannot get supported clocks. Aborting frequency scaling test.")
        return

    print(f"🔬 Found {len(supported_clocks)} supported clock frequencies to test.")

    for model_name in args.models:
        out_dir = Path("scale_profile")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"freq_scaling_{model_name}.csv"

        if csv_path.exists():
            csv_path.unlink()

        print(f"\n⚡️ STARTING FREQUENCY SCALING TEST ⚡️")
        print(f"Model: {model_name}")
        print(f"Output: {csv_path}")
        print(f"="*60)

        mps_percent = 100
        batch_size = 128
        num_tasks = 1

        base_throughput = None

        for freq in supported_clocks:
            _set_specific_clock(sudo_password=sudo_pass, gpu_index=args.gpu, clock=freq)

            print(f"\n>>> [Test] Freq={freq}MHz | Model={model_name} | Batch={batch_size}")

            # [Critical fix] Set env var before starting child process
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(mps_percent))

            stop_event, ready_event, counter = mp.Event(), mp.Event(), mp.Value("Q", 0)
            worker_func = _static_worker if model_name == "static" else _child_worker
            p = mp.Process(target=worker_func, args=(
                model_name, batch_size, mps_percent, stop_event, ready_event, counter, args.dataset, repo_root, args.save_output
            ), daemon=True)
            p.start()

            if not ready_event.wait(timeout=180) or not p.is_alive():
                print("❌ Worker failed to start. Skipping this frequency.")
                if p.is_alive(): p.terminate()
                continue

            time.sleep(args.warmup_seconds)

            base_count = counter.value
            elapsed, power_avg, power_max, clock, sm_u, mem_u, _ = _measure(handle, args.measure_seconds)
            curr_count = counter.value

            delta = max(0, curr_count - base_count)
            throughput = delta / elapsed if elapsed > 0 else 0

            if base_throughput is None:
                base_throughput = throughput

            print(f"  Result: Tput={throughput:.1f}, Power={power_avg:.1f}W (max={power_max:.1f}W), SM Util={sm_u:.1f}%")

            result_data = {
                "model_name": model_name,
                "batch_size": batch_size,
                "target_freq": freq,
                "measured_freq": round(clock, 1),
                "throughput": round(throughput, 2),
                "power_avg": round(power_avg, 2),
                "power_max": round(power_max, 2),
                "sm_util": round(sm_u, 1),
                "mem_util": round(mem_u, 1),
                "throughput_normalized": round(throughput / base_throughput, 4) if base_throughput else 1.0
            }
            save_freq_scaling_result_to_csv(result_data, csv_path)

            stop_event.set()
            p.join(timeout=2)
            if p.is_alive(): p.kill()
            time.sleep(1)

        _reset_gpu_defaults(sudo_password=sudo_pass, gpu_index=args.gpu)
        print("\n✅ Frequency scaling test finished.")


def save_single_result_to_csv(result_dict, filepath="profile_results.csv"):
    """Save single-test results to CSV."""
    file_exists = os.path.isfile(filepath)
    fieldnames = [
        "model_name", "num_tasks", "p_list", "b_list", "total_mps_load",
        "target_freq", "target_mem_freq", "measured_freq", "sm_util", "mem_util",
        "throughput_total", "throughput_per_task", "power_avg", "power_max", "elapsed"
    ]
    with open(filepath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_dict)


def save_sweep_result_to_csv(result_dict, filepath="sweep_results.csv"):
    """Save sweep results to CSV (streamed)."""
    file_exists = os.path.isfile(filepath)
    fieldnames = [
        "model_name", "p_list", "b_list", "target_freq", "mem_freq", "sm_freq",
        "sm_util", "mem_util", "throughput_total", "power_avg", "power_max"
    ]
    # Serialize lists as strings; None to empty string
    row = {k: (str(v) if isinstance(v, list) else ("" if v is None else v)) for k, v in result_dict.items()}
    with open(filepath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            f.flush()  # Ensure header is written immediately
        writer.writerow(row)
        f.flush()  # Flush to avoid data loss on interrupt


def run_single_test(model_name, p_list, b_list, f, mem_freq=None, measure_seconds=DEFAULT_MEASURE_SECONDS, warmup_seconds=DEFAULT_WARMUP_SECONDS, sudo_password=None, dataset_name=None, repo_root=None, save_output=False):
    """
    Single test: input p, b, f; output SM, MEM, Throughput, Power.

    Args:
        model_name: Model name
        p_list: MPS percentage list (per task)
        b_list: Batch size list (per task)
        f: Target SM clock (MHz); None uses current clock
        mem_freq: Target memory clock (MHz); None uses max clock
        measure_seconds: Measurement duration
        warmup_seconds: Warmup duration
        sudo_password: sudo password (for clock locking)
        dataset_name: Dataset name (Caltech256, ImageNet)
        repo_root: Repo root
        save_output: Whether to save output images

    Returns:
        dict: {sm_util, mem_util, throughput, power, clock, ...}
    """
    import pynvml

    num_tasks = len(p_list)
    total_load = sum(p_list)

    if total_load > 100:
        print(f"⚠️ Total MPS load ({total_load}%) exceeds 100%")
        return None

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # Lock clocks (SM + Memory)
    if f:
        # If mem_freq not specified, use max memory clock
        if mem_freq is None:
            try:
                max_mem = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                mem_freq = max_mem
                print(f"    [Info] Using max Memory clock: {max_mem} MHz")
            except Exception as e:
                print(f"    [Warn] Failed to get max memory clock: {e}")

        if sudo_password:
            # Lock SM and memory clocks
            _set_specific_clock(sudo_password=sudo_password, gpu_index=0, clock=f, mem_clock=mem_freq)
        # Wait for clocks to stabilize
        time.sleep(1)
        # Verify clocks
        actual_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        actual_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        print(f"    [Info] SM freq: {actual_sm} MHz, Mem freq: {actual_mem} MHz")

    # Launch worker processes
    stop_events, ready_events, counters, procs = [], [], [], []
    worker_func = _static_worker if model_name == "static" else _child_worker

    # [Critical fix] Set env vars before starting each child process
    # With spawn, child processes inherit the parent's environment
    for i in range(num_tasks):
        # Set env var before launch so children see the correct value
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(p_list[i]))

        s_evt = mp.Event()
        r_evt = mp.Event()
        cnt = mp.Value("Q", 0)
        p = mp.Process(target=worker_func, args=(
            model_name, b_list[i], p_list[i], s_evt, r_evt, cnt, dataset_name, repo_root, save_output
        ), daemon=True)
        p.start()
        stop_events.append(s_evt)
        ready_events.append(r_evt)
        counters.append(cnt)
        procs.append(p)
        if i % 4 == 0:
            time.sleep(0.2)

    # Wait for all workers to be ready
    all_ready = all(r.wait(timeout=180) for r in ready_events)
    if not all_ready or any(not proc.is_alive() for proc in procs):
        print("❌ One or more workers failed to start")
        for s in stop_events: s.set()
        for proc in procs: proc.terminate()
        return None

    time.sleep(warmup_seconds)

    # Measurement
    base_counts = [c.value for c in counters]
    print(f"    [Debug] Base counts: {base_counts}")

    elapsed, power_avg, power_max, clock, sm_u, mem_u, _ = _measure(handle, measure_seconds)
    curr_counts = [c.value for c in counters]
    print(f"    [Debug] Curr counts: {curr_counts}, elapsed={elapsed:.2f}s, sm_u={sm_u}%")

    deltas = [max(0, c - b) for c, b in zip(curr_counts, base_counts)]
    # static model has no compute; throughput = 0
    if model_name == "static":
        throughput_total = 0.0
    else:
        throughput_total = sum(deltas) / elapsed if elapsed > 0 else 0
    throughput_per_task = throughput_total / num_tasks if num_tasks > 0 else 0

    # Cleanup
    for s in stop_events: s.set()
    for proc in procs:
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()

    # Restore default clocks
    if f and sudo_password:
        _reset_gpu_defaults(sudo_password=sudo_password, gpu_index=0)

    pynvml.nvmlShutdown()

    return {
        "model_name": model_name,
        "num_tasks": num_tasks,
        "p_list": p_list,
        "b_list": b_list,
        "total_mps_load": total_load,
        "target_freq": f,
        "target_mem_freq": mem_freq,
        "measured_freq": round(clock, 1),
        "sm_util": round(sm_u, 1),
        "mem_util": round(mem_u, 1),
        "throughput_total": round(throughput_total, 2),
        "throughput_per_task": round(throughput_per_task, 2),
        "power_avg": round(power_avg, 2),
        "power_max": round(power_max, 2),
        "elapsed": round(elapsed, 2)
    }


def run_stress_test(args, repo_root, sudo_pass, mem_freq=None):
    """Run stress test and save a unified CSV format."""
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(args.gpu))
    max_mem = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    # Use specified mem_freq; fall back to max if None
    target_mem_freq = mem_freq if mem_freq is not None else max_mem

    out_dir = Path("mps_profile")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_suffix = f"_{args.dataset.lower()}" if args.dataset else ""
    csv_path = out_dir / f"mps_results{dataset_suffix}.csv"

    print(f"\n🔥🔥 STARTING MATRIX STRESS TEST 🔥🔥")
    print(f"Output: {csv_path}")
    print(f"="*60)

    for model_name in args.models:
        print(f"\n--- Model: {model_name} ---")

        for mps_percent, config in MPS_TEST_MATRIX.items():
            task_counts = config["task_counts"]
            batch_list = config["batches"]

            for num_tasks in task_counts:
                if mps_percent * num_tasks > 100:
                    print(f"⚠️ Skipping: {num_tasks}x{mps_percent}% = {mps_percent*num_tasks}% > 100%")
                    continue

                total_load = mps_percent * num_tasks
                # Build p_list and b_list (same value per task)
                p_list = [mps_percent] * num_tasks
                b_list = [batch_list[0]] * num_tasks  # Start with first batch; overwritten later

                for batch in batch_list:
                    b_list = [batch] * num_tasks  # Update batch size for all tasks

                    print(f"\n>>> [Test] MPS={mps_percent}% | Tasks={num_tasks} (Load={total_load}%) | Batch={batch}")

                    stop_events, ready_events, counters, procs = [], [], [], []
                    worker_func = _static_worker if model_name == "static" else _child_worker

                    # [Critical fix] Set env var before starting all child processes
                    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(mps_percent))

                    for i in range(num_tasks):
                        s_evt, r_evt, cnt = mp.Event(), mp.Event(), mp.Value("Q", 0)
                        p = mp.Process(target=worker_func, args=(
                            model_name, batch, mps_percent, s_evt, r_evt, cnt, args.dataset, repo_root, args.save_output
                        ), daemon=True)
                        p.start()
                        stop_events.append(s_evt); ready_events.append(r_evt); counters.append(cnt); procs.append(p)
                        if i % 4 == 0: time.sleep(0.2)

                    print(f"Waiting for {len(procs)} workers...")
                    # Wait with timeout while checking process status
                    start_wait = time.time()
                    timeout = 60  # 60s timeout
                    ready_idx = set()
                    while time.time() - start_wait < timeout:
                        # Check which processes are ready
                        for idx, (p, r) in enumerate(zip(procs, ready_events)):
                            if idx not in ready_idx and r.is_set():
                                ready_idx.add(idx)
                        # Check which processes have crashed
                        alive_procs = [p for p in procs if p.is_alive()]
                        if len(ready_idx) + len(alive_procs) < len(procs):
                            # Some exited before ready (likely OOM)
                            print(f"  ⚠️ {len(procs) - len(alive_procs)} workers died during init (likely OOM)")
                            break
                        if len(ready_idx) == len(procs):
                            break
                        time.sleep(0.5)

                    all_ready = len(ready_idx) == len(procs)

                    if not all_ready or any(not p.is_alive() for p in procs):
                        print("❌ One or more workers failed to start. Skipping...")
                        for s in stop_events: s.set()
                        for p in procs: p.terminate()
                        continue

                    print(f"Ready. Warming up {args.warmup_seconds}s...")
                    time.sleep(args.warmup_seconds)

                    oom_skipped = False
                    for it in range(ITERATIONS):
                        # Check if processes are still alive
                        if not all(p.is_alive() for p in procs):
                            print(f"  ⚠️ Worker进程已崩溃 (可能是OOM), 跳过此配置")
                            oom_skipped = True
                            break

                        base_counts = [c.value for c in counters]
                        # Measurement with timeout
                        measure_result = _measure_with_timeout(handle, args.measure_seconds, timeout_seconds=60)

                        if measure_result is None:
                            print(f"  ⚠️ 测量超时, 跳过此配置")
                            oom_skipped = True
                            break

                        elapsed, power_avg, power_max, clock, sm_u, mem_u, pcie_mb = measure_result
                        curr_counts = [c.value for c in counters]
                        print(f"    [Debug] Base: {base_counts}, Curr: {curr_counts}, elapsed={elapsed:.2f}s, sm_u={sm_u:.1f}%")

                        deltas = [max(0, c - b) for c, b in zip(curr_counts, base_counts)]
                        tput_total = sum(deltas) / elapsed if elapsed > 0 else 0

                        print(f"  Iter {it+1}: Tput={tput_total:.1f}, Power={power_avg:.1f}W (max={power_max:.1f}W), SM Util={sm_u:.1f}%, Mem={mem_u:.1f}% | Writing...")

                        # Unified output format
                        result_data = {
                            "model_name": model_name,
                            "p_list": p_list,
                            "b_list": b_list,
                            "target_freq": None,  # Stress test does not set clocks
                            "mem_freq": target_mem_freq,
                            "sm_freq": round(clock, 1),
                            "sm_util": round(sm_u, 1),
                            "mem_util": round(mem_u, 1),
                            "throughput_total": round(tput_total, 2),
                            "power_avg": round(power_avg, 2),
                            "power_max": round(power_max, 2)
                        }
                        save_sweep_result_to_csv(result_data, csv_path)

                    # If OOM skipped, write a failed record
                    if oom_skipped:
                        failed_data = {
                            "model_name": model_name,
                            "p_list": p_list,
                            "b_list": b_list,
                            "target_freq": None,
                            "mem_freq": target_mem_freq,
                            "sm_freq": 0,
                            "sm_util": 0,
                            "mem_util": 0,
                            "throughput_total": 0,
                            "power_avg": 0,
                            "power_max": 0
                        }
                        save_sweep_result_to_csv(failed_data, csv_path)
                        print(f"  ❌ OOM: 记录失败配置后继续")

                    # Cleanup
                    for s in stop_events: s.set()
                    for p in procs: p.join(timeout=2)
                    for p in procs:
                        if p.is_alive(): p.kill()

                    # Clear GPU memory
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                    time.sleep(2)  # Wait for memory release


def main():
    parser = argparse.ArgumentParser(description="MPS Profile Tool")
    parser.add_argument("--models", nargs="+", default=['mnasnet', 'densenet201', 'efficientnet_v2_m', 'maxvit_t', 'mobilenet_v2', 'resnet50', 'vgg19'])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to use for inference (e.g., Caltech256, ImageNet)")
    parser.add_argument("--save-output", action="store_true", help="Save processed images to the output folder.")
    parser.add_argument("--profile-freq-scaling", action="store_true", help="Run the frequency scaling test instead of the stress test.")

    # Static power test options
    parser.add_argument("--static", action="store_true", help="Run static power test (idle at specified frequency)")

    # Single-test options
    parser.add_argument("--single", action="store_true", help="Run single test with given p, b, f")
    parser.add_argument("--p", nargs="+", type=float, default=[100], help="MPS percent(s) (e.g., 20 20 20 or 50,30,20)")
    parser.add_argument("--b", nargs="+", type=int, default=[8], help="Batch size(s) (e.g., 8 16 32)")
    parser.add_argument("--f", type=int, default=None, help="Target SM frequency in MHz (default: current freq)")
    parser.add_argument("--mem-freq", type=int, default=None, help="Target memory frequency in MHz (default: max)")

    # Frequency sweep options
    parser.add_argument("--sweep", action="store_true", help="Sweep frequency range")
    parser.add_argument("--f-start", type=int, default=1200, help="Start frequency for sweep (default: 1200)")
    parser.add_argument("--f-end", type=int, default=2100, help="End frequency for sweep (default: 2100)")
    parser.add_argument("--f-step", type=int, default=150, help="Frequency step for sweep (default: 150)")

    parser.add_argument("--measure-seconds", type=float, default=DEFAULT_MEASURE_SECONDS)
    parser.add_argument("--warmup-seconds", type=float, default=DEFAULT_WARMUP_SECONDS)
    parser.add_argument("--no-sudo", action="store_true")
    parser.add_argument("--sudo-password", type=str, default=DEFAULT_SUDO_PASSWORD)
    args = parser.parse_args()

    # Set MPS environment variables for all processes (inherited by child workers)
    import pwd
    actual_user = os.environ.get("SUDO_USER", os.environ.get("USER", "wzk"))
    uid = pwd.getpwnam(actual_user).pw_uid
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    repo_root = _repo_root()
    passw = None if args.no_sudo else (os.environ.get("SUDO_PASSWORD") or args.sudo_password)

    # [Critical fix] Set env vars before calling _best_effort_start_mps
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = f"/tmp/nvidia-mps-{uid}"
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = f"/tmp/nvidia-log-{uid}"

    # Single-test mode
    if args.single:
        # Start MPS service if partitioning is needed
        _best_effort_start_mps(sudo_password=passw, repo_root=repo_root)

        model_name = args.models[0] if args.models else 'mobilenet_v2'

        # args.p and args.b are now lists
        p_list = list(args.p)
        b_list = list(args.b)

        # Expand to the same length
        max_len = max(len(p_list), len(b_list))
        if len(p_list) == 1:
            p_list = p_list * max_len
        if len(b_list) == 1:
            b_list = b_list * max_len

        if len(p_list) != len(b_list):
            print("Error: --p and --b must have the same number of values")
            return

        # Resolve mem_freq; if None, use max memory clock
        disp_mem_freq = args.mem_freq
        if disp_mem_freq is None:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                disp_mem_freq = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                disp_mem_freq = 'max'

        print(f"\n{'='*60}")
        print(f"  Single Test: {model_name}")
        print(f"  MPS:         {p_list}")
        print(f"  Batch:       {b_list}")
        print(f"  SM Freq:     {args.f or 'current'} MHz")
        print(f"  Mem Freq:    {disp_mem_freq} MHz")
        print(f"{'='*60}\n")

        repo_root = _repo_root()

        result = run_single_test(
            model_name=model_name,
            p_list=p_list,
            b_list=b_list,
            f=args.f,
            mem_freq=args.mem_freq,
            measure_seconds=args.measure_seconds,
            warmup_seconds=args.warmup_seconds,
            sudo_password=passw,
            dataset_name=args.dataset,
            repo_root=repo_root,
            save_output=args.save_output
        )

        if result:
            print(f"\n{'='*60}")
            print(f"  Results:")
            print(f"{'='*60}")
            print(f"  Model:            {result['model_name']}")
            print(f"  Tasks:            {result['num_tasks']}")
            print(f"  MPS per task:     {result['p_list']}")
            print(f"  Total MPS load:   {result['total_mps_load']:.1f}%")
            print(f"  Batch per task:   {result['b_list']}")
            print(f"  Target Freq:      {result['target_freq']} MHz")
            print(f"  Mem Freq:         {result.get('target_mem_freq') or args.mem_freq} MHz")
            print(f"  Measured Freq:    {result['measured_freq']} MHz")
            print(f"  SM Util:          {result['sm_util']:.1f}%")
            print(f"  MEM Util:         {result['mem_util']:.1f}%")
            print(f"  Throughput (tot): {result['throughput_total']:.2f} images/sec")
            print(f"  Throughput/task:  {result['throughput_per_task']:.2f} images/sec")
            print(f"  Power (avg):      {result['power_avg']:.2f} W")
            print(f"  Power (max):      {result['power_max']:.2f} W")
            print(f"{'='*60}")

            # Save to CSV
            csv_path = "profile_results.csv"
            save_single_result_to_csv(result, csv_path)
            print(f"\n💾 Result saved to: {csv_path}")

        # Stop MPS service
        _best_effort_shutdown_mps(sudo_password=passw, repo_root=repo_root)
        return

    # Static power test mode
    if args.static:
        repo_root = _repo_root()
        _best_effort_shutdown_mps(sudo_password=passw, repo_root=repo_root)

        # Get max memory clock for display
        disp_mem_freq = args.mem_freq
        if disp_mem_freq is None:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(args.gpu))
                disp_mem_freq = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                disp_mem_freq = 'max'

        print(f"\n{'='*60}")
        print(f"  Static Power Test (Idle)")
        print(f"  SM Freq:     {args.f if args.f else 'current'} MHz")
        print(f"  Mem Freq:    {disp_mem_freq} MHz")
        print(f"  Duration:    {args.measure_seconds} seconds")
        print(f"{'='*60}\n")

        try:
            import pynvml
            if 'handle' not in dir():
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(args.gpu))

            # Set clocks
            if args.f and passw:
                _set_specific_clock(sudo_password=passw, gpu_index=args.gpu, clock=args.f, mem_clock=args.mem_freq)
                if args.mem_freq is None:
                    max_mem = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    cmd = f'echo "{passw}" | sudo -S nvidia-smi -i {args.gpu} -lmc {max_mem}'
                    subprocess.run(["bash", "-lc", cmd], capture_output=True, check=False)
                time.sleep(2.0)  # Wait for clocks to stabilize

                # Verify clocks
                actual_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                actual_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                print(f"    Actual SM freq: {actual_sm} MHz, Mem freq: {actual_mem} MHz")

            print(f"\n🔇 Measuring idle power for {args.measure_seconds} seconds...\n")

            # Measure power
            power_samples, clock_samples, sm_util_samples, mem_util_samples = [], [], [], []
            start_t = time.perf_counter()
            while time.perf_counter() - start_t < args.measure_seconds:
                try:
                    power_samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
                    clock_samples.append(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
                    utils = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    sm_util_samples.append(utils.gpu)
                    mem_util_samples.append(utils.memory)
                    time.sleep(0.1)
                except Exception:
                    break

            elapsed = time.perf_counter() - start_t
            power_avg = sum(power_samples) / len(power_samples) if power_samples else 0
            power_max = max(power_samples) if power_samples else 0
            clock_avg = sum(clock_samples) / len(clock_samples) if clock_samples else 0
            sm_util_avg = sum(sm_util_samples) / len(sm_util_samples) if sm_util_samples else 0
            mem_util_avg = sum(mem_util_samples) / len(mem_util_samples) if mem_util_samples else 0

            print(f"{'='*60}")
            print(f"  Static Power Results:")
            print(f"{'='*60}")
            print(f"  SM Freq:      {args.f if args.f else 'N/A'} MHz (measured: {clock_avg:.0f} MHz)")
            print(f"  Mem Freq:     {args.mem_freq if args.mem_freq else 'N/A'} MHz")
            print(f"  Duration:     {elapsed:.1f} seconds")
            print(f"  SM Util:      {sm_util_avg:.1f}%")
            print(f"  MEM Util:     {mem_util_avg:.1f}%")
            print(f"  Power (avg):  {power_avg:.2f} W")
            print(f"  Power (max):  {power_max:.2f} W")
            print(f"{'='*60}")

            # Save to CSV
            csv_path = "static_power_results.csv"
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode='a', newline='') as f:
                fieldnames = ["sm_freq", "mem_freq", "measured_sm_freq", "duration", "sm_util", "mem_util", "power_avg", "power_max"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "sm_freq": args.f,
                    "mem_freq": args.mem_freq,
                    "measured_sm_freq": round(clock_avg, 1),
                    "duration": round(elapsed, 1),
                    "sm_util": round(sm_util_avg, 1),
                    "mem_util": round(mem_util_avg, 1),
                    "power_avg": round(power_avg, 2),
                    "power_max": round(power_max, 2)
                })
            print(f"\n💾 Result saved to: {csv_path}")

            pynvml.nvmlShutdown()
        finally:
            # Restore default clocks
            if args.f and passw:
                _reset_gpu_defaults(sudo_password=passw, gpu_index=args.gpu)
            print(f"\n✅ Static power test completed. Frequencies reset.")
        return

    # Frequency sweep mode
    if args.sweep:
        # Start MPS service if partitioning is needed
        _best_effort_start_mps(sudo_password=passw, repo_root=repo_root)

        model_name = args.models[0] if args.models else 'mobilenet_v2'
        repo_root = _repo_root()

        # args.p and args.b are now lists
        p_list = list(args.p)
        b_list = list(args.b)

        max_len = max(len(p_list), len(b_list))
        if len(p_list) == 1:
            p_list = p_list * max_len
        if len(b_list) == 1:
            b_list = b_list * max_len

        # Get max memory clock for display
        disp_mem_freq = args.mem_freq
        if disp_mem_freq is None:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                disp_mem_freq = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                disp_mem_freq = 'max'

        print(f"\n{'='*60}")
        print(f"  Frequency Sweep: {model_name}")
        print(f"  MPS:  {p_list}")
        print(f"  Batch: {b_list}")
        print(f"  SM Freq Range: {args.f_start}-{args.f_end} MHz, step={args.f_step}")
        print(f"  Mem Freq: {disp_mem_freq} MHz")
        if args.dataset:
            print(f"  Dataset: {args.dataset}")
        print(f"{'='*60}\n")

        freqs = list(range(args.f_start, args.f_end + 1, args.f_step))
        if args.f_end not in freqs:
            freqs.append(args.f_end)

        print(f"{'Freq(MHz)':>10} | {'SM(%)':>6} | {'MEM(%)':>6} | {'TP(img/s)':>10} | {'Power(W)':>12}")
        print("-" * 65)

        out_dir = Path("scale_profile")
        out_dir.mkdir(parents=True, exist_ok=True)
        sweep_csv_path = out_dir / f"{model_name}_sweep_results.csv"

        for freq in freqs:
            result = run_single_test(
                model_name=model_name,
                p_list=p_list,
                b_list=b_list,
                f=freq,
                mem_freq=args.mem_freq,
                measure_seconds=args.measure_seconds,
                warmup_seconds=args.warmup_seconds,
                sudo_password=passw,
                dataset_name=args.dataset,
                repo_root=repo_root,
                save_output=args.save_output
            )
            if result:
                print(f"{freq:>10} | {result['sm_util']:>6.1f} | {result['mem_util']:>6.1f} | {result['throughput_total']:>10.2f} | {result['power_avg']:>8.2f} ({result['power_max']:.2f})")
                # Stream results to CSV
                save_sweep_result_to_csv({
                    "model_name": result["model_name"],
                    "p_list": result["p_list"],
                    "b_list": result["b_list"],
                    "target_freq": result["target_freq"],
                    "mem_freq": result.get("target_mem_freq"),
                    "sm_freq": result["measured_freq"],
                    "sm_util": result["sm_util"],
                    "mem_util": result["mem_util"],
                    "throughput_total": result["throughput_total"],
                    "power_avg": result["power_avg"],
                    "power_max": result["power_max"]
                }, sweep_csv_path)

        print(f"\n💾 Sweep results saved to: {sweep_csv_path}")

        # Stop MPS service
        _best_effort_shutdown_mps(sudo_password=passw, repo_root=repo_root)
        return

    _best_effort_start_mps(sudo_password=passw, repo_root=repo_root)

    if args.profile_freq_scaling:
        _reset_gpu_defaults(sudo_password=passw, gpu_index=args.gpu)
        try:
            run_freq_scaling_test(args, repo_root, passw)
        finally:
            _reset_gpu_defaults(sudo_password=passw, gpu_index=args.gpu)
            _best_effort_shutdown_mps(sudo_password=passw, repo_root=repo_root)
    else:
        _set_performance_mode(sudo_password=passw, gpu_index=args.gpu, mem_freq=args.mem_freq)
        try:
            run_stress_test(args, repo_root, passw, mem_freq=args.mem_freq)
        finally:
            _reset_gpu_defaults(sudo_password=passw, gpu_index=args.gpu)
            _best_effort_shutdown_mps(sudo_password=passw, repo_root=repo_root)


if __name__ == "__main__":
    main()
