import sys
import os
import argparse
import time
import copy
import threading
from datetime import datetime
from pathlib import Path
import pynvml
import json
import torch
import numpy as np
import multiprocessing as mp
from queue import Empty
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from torchvision import transforms, datasets

# System configuration constants
SUDO_PASSWORD = os.environ.get('SUDO_PASSWORD', 'wzk123456')
_NVML_SM_CLOCKS = None
_NVML_MEM_CLOCKS = None
_NVML_SM_BY_MEM = None

# Add src directory to Python path
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import globalConfig
import dataLoader
import doProfile
import doInference
import scheduler
from thermo_control import ThermodynamicModelController
from shadow_optimizer import ShadowOptimizer, ShadowConfig
from predict import PctoDLPredictor

# PctoDL greedy algorithm integrated into PID scheduler
PCTODL_AVAILABLE = False

# ========== Runtime Configuration Parameters ==========
# Number of test iterations
NUM_ITERATIONS = 150

# Power cap list (unit: W)
POWER_CAP_LIST = [130, 180, 230]

# Power cap switch points (iteration count, starting from 0)
POWER_CAP_SWITCH_POINTS = [0, 50, 100]

# ====================================

# Default memory frequency per platform (MHz)
DEFAULT_MEM_FREQ = {
    '3080ti': 5001,
    'a100': 1215,
    'a100-sxm': 1215,
    'a100-pcie': 1215,
}

def get_default_mem_freq(platform_name=None):
    """Get default memory frequency based on platform name"""
    if platform_name:
        platform_lower = platform_name.lower()
        for key, value in DEFAULT_MEM_FREQ.items():
            if key in platform_lower:
                return value
    # Default to 3080Ti frequency
    return 5001

def set_gpu_clocks(gpu_freq_min, gpu_freq_max, mem_freq_min=None, mem_freq_max=None):
    """Set GPU clocks directly using sudo

    Args:
        gpu_freq_min: GPU minimum frequency (MHz)
        gpu_freq_max: GPU maximum frequency (MHz)
        mem_freq_min: Memory minimum frequency (MHz), None means not set
        mem_freq_max: Memory maximum frequency (MHz), None means not set
    """
    gpu_min = int(gpu_freq_min)
    gpu_max = int(gpu_freq_max)

    gpu_cmd = ['sudo', '-S', 'nvidia-smi', '-lgc', f'{gpu_min},{gpu_max}']

    gpu_result = subprocess.run(
        gpu_cmd,
        input=SUDO_PASSWORD + '\n',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def _report(cmd, result, label):
        if result.returncode != 0:
            out = result.stdout.strip()
            err = result.stderr.strip()
            msg = err or out or "unknown error"
            print(f"Warning: {label} setting failed: {' '.join(cmd)} -> {msg}")

    _report(gpu_cmd, gpu_result, "GPU frequency")

    # Only set memory frequency when explicitly specified
    if mem_freq_min is not None and mem_freq_max is not None:
        mem_min = int(mem_freq_min)
        mem_max = int(mem_freq_max)
        mem_cmd = ['sudo', '-S', 'nvidia-smi', '-lmc', f'{mem_min},{mem_max}']
        mem_result = subprocess.run(
            mem_cmd,
            input=SUDO_PASSWORD + '\n',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _report(mem_cmd, mem_result, "Memory frequency")

def set_mem_clock_only(mem_freq):
    """Set memory frequency only, without locking SM frequency (for directcap algorithm)"""
    mem_val = int(mem_freq)
    mem_cmd = ['sudo', '-S', 'nvidia-smi', '-lmc', f'{mem_val},{mem_val}']
    result = subprocess.run(
        mem_cmd,
        input=SUDO_PASSWORD + '\n',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        out = result.stdout.strip()
        err = result.stderr.strip()
        msg = err or out or "unknown error"
        print(f"Warning: Memory frequency setting failed: {' '.join(mem_cmd)} -> {msg}")

def reset_mem_clock():
    """Reset memory frequency lock, let system manage automatically"""
    reset_cmd = ['sudo', '-S', 'nvidia-smi', '-rmc']
    result = subprocess.run(
        reset_cmd,
        input=SUDO_PASSWORD + '\n',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        out = result.stdout.strip()
        err = result.stderr.strip()
        msg = err or out or "unknown error"
        print(f"Warning: Reset memory frequency failed: {' '.join(reset_cmd)} -> {msg}")
    else:
        print("Memory frequency lock reset, system will manage memory frequency automatically")

def get_current_sm_clock(handle):
    """Read current SM running frequency (MHz)"""
    try:
        return float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
    except Exception:
        return None

def snap_to_supported_clock(target_mhz, supported_clocks):
    """Align target frequency to nearest value in supported list"""
    if not supported_clocks:
        return float(target_mhz)
    return float(min(supported_clocks, key=lambda c: abs(c - target_mhz)))

def _load_nvml_supported_clocks(handle):
    """Read supported memory/graphics frequency lists from NVML and cache"""
    global _NVML_SM_CLOCKS, _NVML_MEM_CLOCKS, _NVML_SM_BY_MEM
    if _NVML_SM_CLOCKS is not None and _NVML_MEM_CLOCKS is not None:
        return
    try:
        mem_list = list(pynvml.nvmlDeviceGetSupportedMemoryClocks(handle))
        mem_list.sort()
        sm_set = set()
        sm_by_mem = {}
        for m in mem_list:
            sm_list = list(pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, int(m)))
            sm_list.sort()
            sm_set.update(sm_list)
            sm_by_mem[int(m)] = sm_list
        _NVML_MEM_CLOCKS = mem_list
        _NVML_SM_CLOCKS = sorted(list(sm_set))
        _NVML_SM_BY_MEM = sm_by_mem
    except Exception:
        _NVML_SM_CLOCKS = []
        _NVML_MEM_CLOCKS = []
        _NVML_SM_BY_MEM = {}

def snap_freq_pair(freq, mem_freq, platform_cfg, handle=None):
    """
    Align SM/MEM frequency to supported list:
    1) Prefer NVML supported list; fall back to platform.json list if unavailable.
    """
    supported_mem = getattr(platform_cfg, "mem_clocks", []) or []
    supported_sm = getattr(platform_cfg, "sm_clocks", []) or []

    if handle:
        _load_nvml_supported_clocks(handle)
        if _NVML_MEM_CLOCKS:
            supported_mem = _NVML_MEM_CLOCKS
        if _NVML_SM_CLOCKS:
            supported_sm = _NVML_SM_CLOCKS

    mem = snap_to_supported_clock(mem_freq, supported_mem)

    # If NVML provides SM list for specific mem, prefer using it
    if handle and _NVML_SM_BY_MEM and int(mem) in _NVML_SM_BY_MEM and _NVML_SM_BY_MEM[int(mem)]:
        sm = snap_to_supported_clock(freq, _NVML_SM_BY_MEM[int(mem)])
    else:
        sm = snap_to_supported_clock(freq, supported_sm)

    return sm, mem

def reset_power_limit_to_default():
    """Restore GPU default power limit to avoid residual -pl settings affecting scheduling"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.default_limit', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"Warning: Unable to read default power limit: {result.stderr.strip()}")
            return
        default_limit = result.stdout.strip().splitlines()[0].strip()
        if not default_limit:
            print("Warning: Default power limit not obtained, skipping restore.")
            return
        subprocess.run(
            ['sudo', '-S', 'nvidia-smi', '-pl', default_limit],
            input=SUDO_PASSWORD + '\n',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        print(f"Default power limit restored: {default_limit}W")
    except Exception as e:
        print(f"Warning: Failed to restore default power limit: {e}")

def find_bs(batch_size_list, tmp_bs):
    batch_size_list = batch_size_list[::-1]
    for bs in batch_size_list:
        if bs < tmp_bs:
            return bs
    return batch_size_list[0]

# =========================================================================
#  MPS Multi-Process Isolation Architecture
# =========================================================================

def _mps_worker_process(
    worker_id,
    mps_cap,
    model_name,
    input_queue,
    output_queue,
    status_queue,
    stop_event,
    dataset_name=None,
    repo_root=None,
    warmup_seconds=2.0,
    measure_seconds=3.0,
    oom_guard=False,
):
    """
    Independent worker process: set MPS environment variables and execute inference
    """
    import sys

    try:
        # Reduce fragmentation; safe no-op on older torch.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # [Key] Set environment variables before loading any CUDA library
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(mps_cap))

        # Send status to main process via queue
        status_queue.put(("starting", worker_id, mps_cap))

        # Must import torch inside subprocess to ensure environment takes effect
        import torch
        import doInference as worker_inference
        import dataLoader as worker_loader
        import numpy as np

        # Get device
        device = worker_inference.get_device()

        status_queue.put(("loading", worker_id, model_name))

        # Load model
        model = worker_inference.load_model(model_name, use_cache=True)

        status_queue.put(("ready", worker_id, str(device)))

        dtype = torch.float16 if device.type == "cuda" else torch.float32

        # Data loading (prefer real data, fall back to synthetic data on failure)
        input_images = []
        if dataset_name and repo_root:
            if dataset_name.lower() == 'caltech256':
                data_path = os.path.join(repo_root, "data", "Caltech256")
            elif dataset_name.lower() == 'imagenet':
                data_path = os.path.join(repo_root, "data", "imagenet", "tiny-imagenet-200", "test", "images")
            else:
                data_path = None

            if data_path and os.path.exists(data_path):
                input_images = worker_loader.load_and_preprocess_images(data_path)

        use_synthetic = not input_images
        image_idx = 0
        
        while not stop_event.is_set():
            try:
                req = input_queue.get(timeout=0.1)
            except Empty:
                continue

            requested_bs = int(req.get('batch_size', 1))
            if not oom_guard:
                batch_size = max(1, requested_bs)

                if use_synthetic:
                    input_tensor = torch.zeros((batch_size, 3, 224, 224), device=device, dtype=dtype)
                else:
                    batch_imgs = []
                    for _ in range(batch_size):
                        batch_imgs.append(input_images[image_idx % len(input_images)])
                        image_idx += 1
                    input_tensor = torch.from_numpy(np.stack(batch_imgs)).to(device).to(dtype)

                # Warmup
                warmup_end = time.time() + warmup_seconds
                while time.time() < warmup_end and not stop_event.is_set():
                    with torch.no_grad():
                        _ = model(input_tensor)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()

                # Measure window
                total_samples = 0
                start_time = time.time()
                while time.time() - start_time < measure_seconds and not stop_event.is_set():
                    with torch.no_grad():
                        _ = model(input_tensor)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                    total_samples += batch_size

                elapsed = max(1e-6, time.time() - start_time)
                throughput = total_samples / elapsed
                latency = (batch_size / throughput * 1000.0) if throughput > 0 else 0.0
                output_queue.put((latency, throughput))
                continue

            # Adaptive batch fallback: if the requested batch OOMs (due to multi-worker memory pressure),
            # keep halving until it fits. Report the actual batch used so the controller can learn.
            batch_used = max(1, requested_bs)
            oom_retries = 0

            while batch_used >= 1 and not stop_event.is_set():
                try:
                    # Build one input tensor per request and reuse it across warmup/measure windows.
                    if use_synthetic:
                        input_tensor = torch.zeros((batch_used, 3, 224, 224), device=device, dtype=dtype)
                    else:
                        batch_imgs = []
                        for _ in range(batch_used):
                            batch_imgs.append(input_images[image_idx % len(input_images)])
                            image_idx += 1
                        input_tensor = torch.from_numpy(np.stack(batch_imgs)).to(device).to(dtype)

                    # Probe once to ensure this batch is feasible before spending time in warmup/measure.
                    with torch.inference_mode():
                        _ = model(input_tensor)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()

                    # Warmup
                    warmup_end = time.time() + warmup_seconds
                    while time.time() < warmup_end and not stop_event.is_set():
                        with torch.inference_mode():
                            _ = model(input_tensor)
                            if device.type == 'cuda':
                                torch.cuda.synchronize()

                    # Measure window
                    total_samples = 0
                    start_time = time.time()
                    while time.time() - start_time < measure_seconds and not stop_event.is_set():
                        with torch.inference_mode():
                            _ = model(input_tensor)
                            if device.type == 'cuda':
                                torch.cuda.synchronize()
                        total_samples += batch_used

                    elapsed = max(1e-6, time.time() - start_time)
                    throughput = total_samples / elapsed
                    latency = (batch_used / throughput * 1000.0) if throughput > 0 else 0.0
                    output_queue.put((latency, throughput, requested_bs, batch_used, oom_retries))
                    break
                except torch.cuda.OutOfMemoryError:
                    oom_retries += 1
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if batch_used <= 1:
                        output_queue.put((0.0, 0.0, requested_bs, 0, oom_retries))
                        break
                    batch_used = max(1, int(batch_used // 2))

    except Exception as e:
        # Send error status via queue
        if 'status_queue' in dir():
            try:
                status_queue.put(("error", worker_id, str(e)))
            except:
                pass
        import traceback
        traceback.print_exc()

class MPSCluster:
    def __init__(self, model_name, gpu_resources_list, dataset_name=None, repo_root=None,
                 warmup_seconds=2.0, measure_seconds=3.0, oom_guard=True):
        self.workers = []
        self.input_queues = []
        self.output_queues = []
        self.stop_events = []
        self.gpu_resources = gpu_resources_list
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.repo_root = repo_root
        self.warmup_seconds = warmup_seconds
        self.measure_seconds = measure_seconds
        # Enable per-worker OOM backoff by default; under multi-process MPS, memory pressure
        # varies over time and fixed per-task batch sizes can OOM intermittently.
        self.oom_guard = bool(oom_guard)
        self.status_queue = mp.Queue()

        # Use spawn to avoid CUDA initialization issues
        self.ctx = mp.get_context('spawn')

    def start(self):
        print(f"Starting MPS Cluster: {len(self.gpu_resources)} Worker processes...")
        for i, mps_cap in enumerate(self.gpu_resources):
            iq = self.ctx.Queue()
            oq = self.ctx.Queue()
            stop_evt = self.ctx.Event()

            p = self.ctx.Process(
                target=_mps_worker_process,
                args=(
                    i,
                    mps_cap,
                    self.model_name,
                    iq,
                    oq,
                    self.status_queue,
                    stop_evt,
                    self.dataset_name,
                    self.repo_root,
                    self.warmup_seconds,
                    self.measure_seconds,
                    self.oom_guard,
                ),
                daemon=True
            )
            p.start()

            self.workers.append(p)
            self.input_queues.append(iq)
            self.output_queues.append(oq)
            self.stop_events.append(stop_evt)

        # Wait for worker processes to start and load models
        print(f"Initializing Worker processes...")
        ready_count = 0
        total_workers = len(self.workers)
        worker_status = {}

        while ready_count < total_workers:
            try:
                status, worker_id, info = self.status_queue.get(timeout=30)
                if status == "starting":
                    worker_status[worker_id] = f"MPS={info}%"
                    print(f"   [{worker_id}] Starting ({worker_status[worker_id]})", flush=True)
                elif status == "loading":
                    worker_status[worker_id] = f"Loading {info}..."
                    print(f"   [{worker_id}] {worker_status[worker_id]}", flush=True)
                elif status == "ready":
                    worker_status[worker_id] = "Ready"
                    ready_count += 1
                    print(f"   [{worker_id}] Ready (device: {info})", flush=True)
                elif status == "error":
                    worker_status[worker_id] = "Error"
                    print(f"   [{worker_id}] Error: {info}", flush=True)
            except:
                break

        print(f"All Workers ready ({ready_count}/{total_workers})")

    def submit_all(self, batch_size_list):
        """Submit tasks to all Workers"""
        active_workers = min(len(self.workers), len(batch_size_list))
        for i in range(active_workers):
            self.input_queues[i].put({'batch_size': batch_size_list[i]})
            
    def wait_all_results(self):
        """Wait and collect all results.

        Returns:
            (latencies, throughputs, batch_used_list)
        """
        latencies = []
        throughputs = []
        batch_used = []

        for i in range(len(self.workers)):
            try:
                msg = self.output_queues[i].get(timeout=20.0)
                if isinstance(msg, tuple) and len(msg) >= 2:
                    lat = float(msg[0])
                    tput = float(msg[1])
                    used = None
                    if len(msg) >= 4:
                        try:
                            used = int(msg[3])
                        except Exception:
                            used = None
                    latencies.append(lat)
                    throughputs.append(tput)
                    batch_used.append(used)
                else:
                    latencies.append(0.0)
                    throughputs.append(0.0)
                    batch_used.append(None)
            except Empty:
                latencies.append(0.0)
                throughputs.append(0.0)
                batch_used.append(None)

        return latencies, throughputs, batch_used

    def stop(self, timeout=0.5):
        print("Stopping MPS Cluster...")
        for evt in self.stop_events:
            evt.set()
        deadline = time.time() + max(0.0, float(timeout))
        for p in self.workers:
            remaining = max(0.0, deadline - time.time())
            p.join(timeout=remaining)
        for p in self.workers:
            if p.is_alive():
                p.kill()

# =========================================================================

def start_server(model):
    """Start MPS server"""
    print("\nStarting MPS service...")
    start_time = time.time()
    subprocess.run(['sudo', '-S', './start_mps.sh'], input=SUDO_PASSWORD + '\n', stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(2)
    end_time = time.time()
    print(f"Startup completed, elapsed: {(end_time-start_time)*1000:.3f} ms")
    return True

def batchdvfs_start(configs, log_txt, args):
    """BatchDVFS algorithm - MPS multi-process version"""
    gpu_resource_list = configs.model.resources
    monitoer_executor = configs.monitor_executor

    mps_cluster = MPSCluster(configs.model.model_name, gpu_resource_list, repo_root=repo_root)
    mps_cluster.start()

    try:
        power_cap = float(args.power_cap if args.power_cap is not None else configs.model.power_cap)
        use_power_schedule = args.power_cap is None

        print("Initializing BatchDVFS scheduler...", flush=True)
        batchdvfs = scheduler.BatchDVFS(power_cap, configs.model.alpha, configs.platform.sm_clocks, configs.model.batch_size_list, log_txt)
        print("BatchDVFS scheduler initialization complete", flush=True)

        # BatchDVFS does not adjust memory frequency, reset memory frequency lock to let system manage automatically
        reset_mem_clock()

        batch_size_list = [256] * len(gpu_resource_list)
        if configs.platform.sm_clocks:
            frequency = configs.platform.sm_clocks[len(configs.platform.sm_clocks) // 2]
        else:
            frequency = 1155
        total_throught = 0
        number = 0

        powerCapList = POWER_CAP_LIST if use_power_schedule else [power_cap]
        switch_points = POWER_CAP_SWITCH_POINTS if use_power_schedule else []
        num_iterations = args.iterations
        current_bs = 256

        print("Test config: total_iterations={}, power_caps={}, switch_points={}".format(num_iterations, powerCapList, switch_points))

        # When appending to an existing CSV, avoid duplicating the header row.
        if log_txt.tell() == 0:
            log_txt.write("iter,mps_resources,batch_sizes,sm_freq_mhz,mem_freq_mhz,sm_util,mem_util,throughput_total,power_avg,power_max,power_cap_w,timestamp\n")

        for i in range(num_iterations):
            if i in switch_points:
                switch_idx = switch_points.index(i)
                if switch_idx < len(powerCapList):
                    batchdvfs.power_cap = powerCapList[switch_idx]
                    if i == switch_points[0]:
                        batchdvfs.min_bs = 0
                        batchdvfs.max_bs = len(configs.model.batch_size_list)
                    else:
                        batchdvfs.min_bs = configs.model.batch_size_list.index(current_bs) if current_bs in configs.model.batch_size_list else 0
                        batchdvfs.max_bs = len(configs.model.batch_size_list)

            readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            timestamp = readable_time
            print("\n[Iteration {}/{}] Starting test - {}".format(i+1, num_iterations, readable_time))
            print("  Params: batch_size={}, frequency={}MHz, power_cap={}W".format(
                batch_size_list[0], frequency, batchdvfs.power_cap))

            stop_event = threading.Event()
            sampler_thread = monitoer_executor.submit(
                doProfile.sample_gpu_info,
                stop_event,
                configs.handle,
                mps_cluster.warmup_seconds,
            )

            # Align to supported frequency to avoid nvidia-smi rejection
            target_freq = frequency
            frequency = snap_to_supported_clock(frequency, configs.platform.sm_clocks)
            # BatchDVFS only sets SM frequency, not memory frequency
            set_gpu_clocks(frequency, frequency)
            actual_sm = get_current_sm_clock(configs.handle)
            if actual_sm is not None:
                print(f"  Actual SM freq: {actual_sm:.0f}MHz (target {target_freq:.0f}MHz, locked {frequency:.0f}MHz)")
            else:
                print("  Actual SM freq: N/A")

            mps_resources_str = "\"[" + ",".join(f"{x:g}" for x in gpu_resource_list) + "]\""
            batch_sizes_str = "\"[" + ",".join(str(x) for x in batch_size_list) + "]\""

            mps_cluster.submit_all(batch_size_list)
            latencies, throughputs, _batch_used = mps_cluster.wait_all_results()

            stop_event.set()

            iter_throughput = sum(throughputs)
            (
                max_power,
                avg_power,
                max_sm_util,
                avg_sm_util,
                max_mem_used,
                avg_mem_used,
                max_pcie,
                avg_pcie,
                max_mem_bw_util,
                avg_mem_bw_util,
                *_,
            ) = sampler_thread.result()
            sm_util = avg_sm_util
            mem_util = avg_mem_bw_util

            # Get current actual memory frequency for logging
            actual_mem_freq = 0
            try:
                actual_mem_freq = pynvml.nvmlDeviceGetClockInfo(configs.handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                pass

            log_txt.write(
                f"{i+1},{mps_resources_str},{batch_sizes_str},{frequency:.0f},{actual_mem_freq:.0f},{sm_util:.2f},{mem_util:.2f},{iter_throughput:.2f},{avg_power:.2f},{max_power:.2f},{batchdvfs.power_cap:.0f},{timestamp}\n"
            )
            log_txt.flush()

            print("  Throughput: {:.2f} samples/s | Power: {:.2f}W".format(iter_throughput, avg_power))

            start_counting_from = max(3, num_iterations // 2)
            if i >= start_counting_from:
                total_throught += iter_throughput
                number += 1

            prev_frequency = frequency
            # First snap to supported frequency value
            frequency = snap_to_supported_clock(frequency, configs.platform.sm_clocks)
            scheduler_input_freq = frequency
            frequency, next_bs = batchdvfs.scheduler(max_power, frequency, batch_size_list[0])
            current_bs = next_bs
            batch_size_list = [next_bs] * len(gpu_resource_list)

            # [Key fix] Frequency overrun protection: only force frequency down when scheduler didn't reduce but power still exceeds
            if max_power > batchdvfs.power_cap:
                # Snap again to ensure frequency is in the list
                frequency = snap_to_supported_clock(frequency, configs.platform.sm_clocks)
                # Only force frequency down when scheduler didn't change frequency and power still exceeds
                if abs(frequency - scheduler_input_freq) < 1.0 and frequency > configs.platform.sm_clocks[0]:
                    freq_idx = configs.platform.sm_clocks.index(frequency)
                    if freq_idx > 0:
                        frequency = configs.platform.sm_clocks[freq_idx - 1]
                        print(f"  [FreqGuard] Scheduler didn't reduce freq but power exceeded, forcing: {prev_frequency:.0f}MHz -> {frequency}MHz")

        if number > 0:
            print(f"\nAverage throughput: {total_throught/number:.2f}")

    finally:
        mps_cluster.stop()

def morak_start(configs, log_txt, args):
    """Morak algorithm - MPS multi-process version"""
    gpu_resource_list = configs.model.resources
    monitoer_executor = configs.monitor_executor

    mps_cluster = MPSCluster(configs.model.model_name, gpu_resource_list, repo_root=repo_root)
    mps_cluster.start()

    try:
        power_cap = float(args.power_cap if args.power_cap is not None else configs.model.power_cap)
        use_power_schedule = args.power_cap is None

        morak = scheduler.Morak(power_cap, configs.model.alpha, configs.model.slo, configs.model.belta, configs.platform.sm_clocks, configs.model.batch_size_list, log_txt)

        # Morak does not adjust memory frequency, reset memory frequency lock to let system manage automatically
        reset_mem_clock()

        batch_size_list = [96] * len(gpu_resource_list)
        # Start from middle frequency to accelerate convergence (reference BatchDVFS)
        if configs.platform.sm_clocks:
            frequency = configs.platform.sm_clocks[len(configs.platform.sm_clocks) // 2]
        else:
            frequency = 810  # Default middle frequency
        total_throught = 0
        number = 0

        powerCapList = POWER_CAP_LIST if use_power_schedule else [power_cap]
        switch_points = POWER_CAP_SWITCH_POINTS if use_power_schedule else []
        num_iterations = args.iterations

        # When appending to an existing CSV, avoid duplicating the header row.
        if log_txt.tell() == 0:
            log_txt.write("iter,mps_resources,batch_sizes,sm_freq_mhz,mem_freq_mhz,sm_util,mem_util,throughput_total,power_avg,power_max,power_cap_w,timestamp\n")

        for i in range(num_iterations):
            if i in switch_points:
                switch_idx = switch_points.index(i)
                if switch_idx < len(powerCapList):
                    morak.power_cap = powerCapList[switch_idx]

            readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            timestamp = readable_time
            print("\n[Iteration {}/{}] Starting test - {}".format(i+1, num_iterations, readable_time))
            print("  Params: batch_size={}, frequency={}MHz, power_cap={}W".format(
                batch_size_list[0], frequency, morak.power_cap))

            mps_resources_str = "\"[" + ",".join(f"{x:g}" for x in gpu_resource_list) + "]\""
            batch_sizes_str = "\"[" + ",".join(str(x) for x in batch_size_list) + "]\""

            stop_event = threading.Event()
            sampler_thread = monitoer_executor.submit(
                doProfile.sample_gpu_info,
                stop_event,
                configs.handle,
                mps_cluster.warmup_seconds,
            )

            target_freq = frequency
            frequency = snap_to_supported_clock(frequency, configs.platform.sm_clocks)
            # Morak only sets SM frequency, not memory frequency
            set_gpu_clocks(frequency, frequency)
            actual_sm = get_current_sm_clock(configs.handle)
            if actual_sm is not None:
                print(f"  Actual SM freq: {actual_sm:.0f}MHz (target {target_freq:.0f}MHz, locked {frequency:.0f}MHz)")
            else:
                print("  Actual SM freq: N/A")

            mps_cluster.submit_all(batch_size_list)
            latencies, throughputs, _batch_used = mps_cluster.wait_all_results()

            stop_event.set()

            (
                max_power,
                avg_power,
                max_sm_util,
                avg_sm_util,
                max_mem_used,
                avg_mem_used,
                max_pcie,
                avg_pcie,
                max_mem_bw_util,
                avg_mem_bw_util,
                *_,
            ) = sampler_thread.result()
            sm_util = avg_sm_util
            mem_util = avg_mem_bw_util
            iter_throughput = sum(throughputs)
            max_latency = max(latencies) if latencies else 0

            # Get current actual memory frequency for logging
            actual_mem_freq = 0
            try:
                actual_mem_freq = pynvml.nvmlDeviceGetClockInfo(configs.handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                pass

            log_txt.write(
                f"{i+1},{mps_resources_str},{batch_sizes_str},{frequency:.0f},{actual_mem_freq:.0f},{sm_util:.2f},{mem_util:.2f},{iter_throughput:.2f},{avg_power:.2f},{max_power:.2f},{morak.power_cap:.0f},{timestamp}\n"
            )
            log_txt.flush()

            print("  Throughput: {:.2f} samples/s | Power: {:.2f}W | Max latency: {:.2f}ms".format(iter_throughput, avg_power, max_latency))

            start_counting_from = max(3, num_iterations // 2)
            if i >= start_counting_from:
                total_throught += iter_throughput
                number += 1

            prev_frequency = frequency
            # First snap to supported frequency value
            frequency = snap_to_supported_clock(frequency, configs.platform.sm_clocks)
            scheduler_input_freq = frequency
            frequency, next_bs = morak.scheduler(max_latency, max_power, frequency, batch_size_list[0])
            batch_size_list = [next_bs] * len(gpu_resource_list)

            # [Key fix] Frequency overrun protection: only force frequency down when scheduler didn't reduce but power still exceeds
            if max_power > morak.power_cap:
                # Snap again to ensure frequency is in the list
                frequency = snap_to_supported_clock(frequency, configs.platform.sm_clocks)
                # Only force frequency down when scheduler didn't change frequency and power still exceeds
                if abs(frequency - scheduler_input_freq) < 1.0 and frequency > configs.platform.sm_clocks[0]:
                    freq_idx = configs.platform.sm_clocks.index(frequency)
                    if freq_idx > 0:
                        frequency = configs.platform.sm_clocks[freq_idx - 1]
                        print(f"  [FreqGuard] Scheduler didn't reduce freq but power exceeded, forcing: {prev_frequency:.0f}MHz -> {frequency}MHz")

            log_txt.flush()

        if number > 0:
            print(f"Average throughput: {total_throught/number:.2f}")

    finally:
        mps_cluster.stop()

def _apply_mps_config(mps_cluster, gpu_resources_list, model_name, dataset_name=None):
    if mps_cluster:
        mps_cluster.stop()
    mps_cluster = MPSCluster(
        model_name,
        gpu_resources_list,
        dataset_name=dataset_name,
        repo_root=repo_root,
        oom_guard=True,
    )
    mps_cluster.start()
    return mps_cluster


def pctodl_start(configs, log_txt, args):
    """
    PctoDL end-to-end flow:
    1) Greedy partitioning for initial config.
    2) Run MPS inference.
    3) Thermodynamic model-based control for power.
    4) Shadow optimizer for joint refinement.
    """
    model_name = configs.model.model_name
    use_power_schedule = args.power_cap is None
    if use_power_schedule:
        # Power scheduling mode: use first value in schedule list as initial power cap
        power_cap = float(POWER_CAP_LIST[0])
    else:
        power_cap = float(args.power_cap if args.power_cap is not None else configs.model.power_cap)
    mps_step = int(args.mps_step)
    dataset_name = args.dataset
    if dataset_name and dataset_name.lower() != "imagenet":
        dataset_name = "imagenet"
    if dataset_name is None:
        data_path = getattr(configs.model, "data_path", None)
        if data_path and os.path.exists(data_path):
            if "imagenet" in data_path.lower():
                dataset_name = "imagenet"

    if args.no_greedy:
        gpu_resources_list = configs.model.resources
        batch_size_list = configs.model.batch_size_list or [32] * len(gpu_resources_list)
        frequency = configs.platform.sm_clocks[-1]
        mem_freq = configs.platform.mem_clocks[-1] if configs.platform.mem_clocks else get_default_mem_freq(args.platform)
    else:
        from greedy_partitioning import GreedyPartitioner
        # Apply power margin: greedy algorithm uses (power_cap - power_margin)
        greedy_power_cap = power_cap - args.power_margin
        partitioner = GreedyPartitioner(
            model_name,
            greedy_power_cap,
            mps_step=mps_step,
            mem_freq=args.mem_freq,
            platform=args.platform,
            dataset=dataset_name,
            vectorized=getattr(args, "greedy_vectorized", False),
        )
        greedy_result = partitioner.run()
        # greedy_partitioning.GreedyPartitioner.run() returns:
        #   (tasks, sm_freq, mem_freq, partition_time_ms, total_configs_searched)
        # Keep backward-compat if the return signature changes.
        if isinstance(greedy_result, (list, tuple)) and len(greedy_result) >= 3:
            tasks, frequency, mem_freq = greedy_result[:3]
        else:
            raise ValueError(f"Unexpected greedy partitioner return: {greedy_result!r}")
        if not tasks:
            print("WARNING: Greedy partitioner returned empty config, falling back to defaults.")
            gpu_resources_list = configs.model.resources
            batch_size_list = configs.model.batch_size_list or [32] * len(gpu_resources_list)
            frequency = configs.platform.sm_clocks[-1]
            mem_freq = configs.platform.mem_clocks[-1] if configs.platform.mem_clocks else get_default_mem_freq(args.platform)
        else:
            gpu_resources_list = [float(t.mps) for t in tasks]
            batch_size_list = [int(t.batch) for t in tasks]

    gpu_resources_list = [float(x) for x in gpu_resources_list]
    batch_size_list = [int(x) for x in batch_size_list]

    if len(batch_size_list) < len(gpu_resources_list):
        default_bs = batch_size_list[0] if batch_size_list else 32
        batch_size_list = batch_size_list + [default_bs] * (len(gpu_resources_list) - len(batch_size_list))
    elif len(batch_size_list) > len(gpu_resources_list):
        batch_size_list = batch_size_list[:len(gpu_resources_list)]
    frequency = float(frequency)
    mem_freq = float(mem_freq)

    # Predictor for control + shadow optimizer
    predictor = PctoDLPredictor(
        model_name,
        mem_freq=int(mem_freq),
        platform=args.platform,
        dataset=dataset_name,
    )
    controller = ThermodynamicModelController(power_cap, predictor=predictor)
    thermo_scheduler = scheduler.ThermoPowerScheduler(controller, log_txt)

    shadow_optimizer = None
    shadow_event_log = None
    if not args.disable_shadow:
        def _shadow_batch_pool():
            # Finer batch ladder helps validate shadow optimizer behavior (e.g., step=4).
            # Kept as a static pool; feasibility is still filtered by MPS share in shadow_optimizer.
            pool = [1, 2]
            pool.extend(range(4, 257, 4))
            return sorted(set(int(b) for b in pool if int(b) > 0))

        shadow_optimizer = ShadowOptimizer(
            p_cap=power_cap,
            predictor=predictor,
            sm_freqs=configs.platform.sm_clocks,
            mem_freqs=configs.platform.mem_clocks,
            batch_pool=_shadow_batch_pool(),
            max_batch_size=None,
            max_mps_per_task=None,
            mps_step=mps_step,
            trust_radius=args.shadow_radius,
            history_window=args.shadow_history,
            gp_refit_interval=args.shadow_gp_refit_interval,
            gp_max_points=args.shadow_gp_max_points,
            reconfig_threshold=args.shadow_threshold,
            allow_mem_adjustment=args.shadow_adjust_mem,
        )
        # Sidecar log for shadow decisions; keep CSV strictly numeric for downstream parsers.
        try:
            shadow_event_path = f"{log_txt.name}.shadow.log"
            shadow_event_log = open(shadow_event_path, "a", encoding="utf-8")
            if shadow_event_log.tell() == 0:
                shadow_event_log.write(
                    "timestamp,iter,event,power_avg_w,power_max_w,power_cap_w,shadow_interval,detail\n"
                )
            shadow_event_log.write(
                f"{datetime.now().isoformat()},{0},enabled,,,,{max(1, int(args.shadow_interval))},shadow optimizer enabled\n"
            )
            shadow_event_log.flush()
        except Exception:
            shadow_event_log = None

    def _run_greedy_for_cap(target_power_cap):
        from greedy_partitioning import GreedyPartitioner
        # Apply power margin: greedy uses (target_power_cap - power_margin)
        greedy_power_cap = target_power_cap - args.power_margin
        partitioner = GreedyPartitioner(
            model_name,
            greedy_power_cap,
            mps_step=mps_step,
            mem_freq=args.mem_freq,
            platform=args.platform,
            dataset=dataset_name,
            vectorized=getattr(args, "greedy_vectorized", False),
        )
        greedy_result = partitioner.run()
        if isinstance(greedy_result, (list, tuple)) and len(greedy_result) >= 3:
            tasks, next_freq, next_mem = greedy_result[:3]
        else:
            raise ValueError(f"Unexpected greedy partitioner return: {greedy_result!r}")
        if not tasks:
            return None
        next_gpu_resources = [float(t.mps) for t in tasks]
        next_batch_sizes = [int(t.batch) for t in tasks]
        if len(next_batch_sizes) < len(next_gpu_resources):
            default_bs = next_batch_sizes[0] if next_batch_sizes else 32
            next_batch_sizes = next_batch_sizes + [default_bs] * (len(next_gpu_resources) - len(next_batch_sizes))
        elif len(next_batch_sizes) > len(next_gpu_resources):
            next_batch_sizes = next_batch_sizes[:len(next_gpu_resources)]
        return next_gpu_resources, next_batch_sizes, float(next_freq), float(next_mem)

    mps_cluster = None
    mps_cluster = _apply_mps_config(mps_cluster, gpu_resources_list, model_name, dataset_name=dataset_name)

    try:
        total_throughput = 0.0
        number = 0
        num_iterations = args.iterations
        shadow_interval = max(1, args.shadow_interval)
        power_caps = POWER_CAP_LIST if use_power_schedule else [power_cap]
        switch_points = POWER_CAP_SWITCH_POINTS if use_power_schedule else []
        schedule_len = min(len(power_caps), len(switch_points))
        switch_idx = 0
        greedy_executor = ThreadPoolExecutor(max_workers=1)
        greedy_future = None
        greedy_target_cap = None
        queued_power_cap = None
        pending_config = None
        iterations_since_shadow_change = 0  # Track iterations since last shadow optimizer change
        shadow_pending = False  # Interval reached but waiting for power to be under power_cap

        print("\n========== PctoDL Scheduler Started ==========")
        print(f"Initial config: MPS={gpu_resources_list}, Batch={batch_size_list}, SM={frequency:.0f}, Mem={mem_freq:.0f}")

        # When appending to an existing CSV, avoid duplicating the header row.
        if log_txt.tell() == 0:
            log_txt.write("iter,mps_resources,batch_sizes,sm_freq_mhz,mem_freq_mhz,sm_util,mem_util,throughput_total,power_avg,power_max,power_cap_w,timestamp\n")

        for i in range(num_iterations):
            if pending_config:
                next_gpu_resources, next_batch_sizes, next_freq, next_mem = pending_config
                pending_config = None

                mps_changed = next_gpu_resources != gpu_resources_list
                mem_changed = int(next_mem) != int(mem_freq)

                gpu_resources_list = [float(x) for x in next_gpu_resources]
                batch_size_list = [int(x) for x in next_batch_sizes]
                frequency = float(next_freq)
                mem_freq = float(next_mem)

                if mps_changed:
                    mps_cluster = _apply_mps_config(mps_cluster, gpu_resources_list, model_name, dataset_name=dataset_name)

                if mem_changed:
                    predictor = PctoDLPredictor(
                        model_name,
                        mem_freq=int(mem_freq),
                        platform=args.platform,
                        dataset=dataset_name,
                    )
                    controller = ThermodynamicModelController(power_cap, predictor=predictor)
                    thermo_scheduler = scheduler.ThermoPowerScheduler(controller, log_txt)
                    if shadow_optimizer:
                        shadow_optimizer.predictor = predictor

            readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("\n[Iteration {}/{}] Start test - {}".format(i + 1, num_iterations, readable_time))
            print("  Params: batch_size={}, frequency={}MHz, memFreq={}MHz, power_cap={}W".format(
                batch_size_list[0], frequency, mem_freq, power_cap))

            target_freq, target_mem = frequency, mem_freq
            frequency, mem_freq = snap_freq_pair(frequency, mem_freq, configs.platform, configs.handle)
            set_gpu_clocks(frequency, frequency, mem_freq, mem_freq)
            actual_sm = get_current_sm_clock(configs.handle)
            if actual_sm is not None:
                print(f"  Actual SM frequency: {actual_sm:.0f}MHz (target {target_freq:.0f}MHz, locked {frequency:.0f}MHz)")
            else:
                print("  Actual SM frequency: N/A")

            # Start power sampling thread (consistent with BatchDVFS/Morak)
            stop_event = threading.Event()
            sampler_thread = configs.monitor_executor.submit(
                doProfile.sample_gpu_info,
                stop_event,
                configs.handle,
                mps_cluster.warmup_seconds,
            )

            # Extra warmup for cold start (first few iterations)
            if i < 3:
                print(f"  [ColdStart] Extra warmup for iteration {i+1} (GPU stabilizing from cold start)...")
                for warmup_round in range(1):  # 1 extra warmup round (using max_power so less needed)
                    mps_cluster.submit_all(batch_size_list)
                    mps_cluster.wait_all_results()
                print(f"  Stabilization complete, starting measurement...")

            batch_size_list_req = list(batch_size_list)
            mps_cluster.submit_all(batch_size_list_req)
            latencies, throughputs, batch_used = mps_cluster.wait_all_results()

            # Workers may auto-reduce batch on OOM; treat that as the actual batch for modeling/logging.
            batch_size_list_used = []
            for j in range(len(batch_size_list_req)):
                u = None
                if j < len(batch_used):
                    u = batch_used[j]
                if u is None or int(u) <= 0:
                    u = batch_size_list_req[j]
                batch_size_list_used.append(int(u))

            # Stop sampling and collect power data
            stop_event.set()
            (
                max_power,
                avg_power,
                max_sm_util,
                avg_sm_util,
                max_mem_used,
                avg_mem_used,
                max_pcie,
                avg_pcie,
                max_mem_bw_util,
                avg_mem_bw_util,
                *_,
            ) = sampler_thread.result()
            sm_util = avg_sm_util
            mem_util = avg_mem_bw_util
            iter_throughput = sum(throughputs)

            print("  Throughput: {:.2f} samples/s | Power: {:.2f}W".format(iter_throughput, avg_power))

            timestamp = readable_time
            mps_str = "\"[" + ",".join(f"{x:g}" for x in gpu_resources_list) + "]\""
            bs_str = "\"[" + ",".join(str(x) for x in batch_size_list_used) + "]\""
            log_txt.write(
                f"{i+1},{mps_str},{bs_str},{frequency:.0f},{mem_freq:.0f},{sm_util:.2f},{mem_util:.2f},{iter_throughput:.2f},{avg_power:.2f},{max_power:.2f},{power_cap:.0f},{timestamp}\n"
            )
            log_txt.flush()

            # Feed back the feasible batches (and back off further if a worker still failed).
            next_batches = list(batch_size_list_used)
            for idx, tput in enumerate(throughputs):
                if tput <= 0 and idx < len(next_batches):
                    next_batches[idx] = max(1, int(next_batches[idx] // 2))
            if next_batches != batch_size_list:
                if next_batches != batch_size_list_used:
                    print(f"  [OOM] Backing off batch sizes (next iter): {batch_size_list_used} -> {next_batches}")
                batch_size_list = next_batches

            start_counting_from = max(3, num_iterations // 2)
            if i >= start_counting_from:
                total_throughput += iter_throughput
                number += 1

            # Thermodynamic controller update (SM frequency only)
            # Always use max_power as the control metric
            power_for_control = max_power

            prev_frequency = frequency
            frequency = thermo_scheduler.scheduler(
                current_power=power_for_control,
                current_sm=frequency,
                p_list=gpu_resources_list,
                b_list=batch_size_list,
                dt=1.0,
            )

            # [Key fix] Frequency guard: only force one-step down when scheduler did not reduce freq but power is still over the cap
            if max_power > power_cap:
                # Snap controller output frequency to supported list
                frequency = snap_to_supported_clock(frequency, configs.platform.sm_clocks)
                # Only force a downstep when scheduler kept freq unchanged and power is still over the cap
                if abs(frequency - prev_frequency) < 1.0 and frequency > configs.platform.sm_clocks[0]:
                    freq_idx = configs.platform.sm_clocks.index(frequency)
                    if freq_idx > 0:
                        frequency = configs.platform.sm_clocks[freq_idx - 1]
                        print(f"  [FreqGuard] Scheduler did not reduce freq but power exceeded cap; forcing downclock: {prev_frequency:.0f}MHz -> {frequency}MHz")

            # Shadow optimizer (periodic) - only optimizes MPS and batch sizes, NOT frequency
            if shadow_optimizer:
                current_cfg = ShadowConfig(
                    p_list=gpu_resources_list,
                    b_list=batch_size_list,
                    sm_freq=frequency,
                    mem_freq=mem_freq,
                    throughput=iter_throughput,
                    power=max_power,  # Use max_power for consistency with controller
                )
                shadow_optimizer.update(current_cfg)
                should_try_shadow = shadow_pending or ((i + 1) % shadow_interval == 0)
                if should_try_shadow:
                    # Do not switch configs while power is still above the cap; wait until stable.
                    # Use avg_power for stability gating; max_power can be spiky and blocks shadow forever.
                    if avg_power > power_cap:
                        shadow_pending = True
                        if (i + 1) % shadow_interval == 0:
                            print(f"  [Shadow] Skip (power unstable): avg={avg_power:.2f}W (max={max_power:.2f}W) > cap={power_cap:.0f}W")
                        if shadow_event_log:
                            shadow_event_log.write(
                                f"{datetime.now().isoformat()},{i+1},skip_power,{avg_power:.2f},{max_power:.2f},{power_cap:.0f},{shadow_interval},avg_power>cap\n"
                            )
                            shadow_event_log.flush()
                    else:
                        # Pass current frequency to shadow optimizer - it will keep these values
                        candidate = shadow_optimizer.suggest(current_cfg, current_sm_freq=frequency, current_mem_freq=mem_freq)
                        shadow_pending = False
                        if candidate:
                            print("  [Shadow] Applying candidate config (MPS + batch only)")
                            candidate_desc = (
                                f"    candidate: p_list={candidate.p_list}, b_list={candidate.b_list}"
                            )
                            print(candidate_desc)
                            # Note: do not write to CSV to avoid breaking the format

                            # Apply candidate changes (MPS and batch only - frequency controlled by PID)
                            mps_changed = candidate.p_list != gpu_resources_list

                            gpu_resources_list = [float(x) for x in candidate.p_list]
                            batch_size_list = [int(x) for x in candidate.b_list]
                            # Note: frequency and mem_freq are NOT updated from shadow optimizer
                            # They are controlled by the PID/thermo controller

                            if mps_changed:
                                mps_cluster = _apply_mps_config(mps_cluster, gpu_resources_list, model_name, dataset_name=dataset_name)
                            iterations_since_shadow_change = 0  # Reset counter after shadow optimizer applies changes
                            print(f"  [Shadow] Config applied - will use max_power for next 3 iterations to stabilize")
                            if shadow_event_log:
                                shadow_event_log.write(
                                    f"{datetime.now().isoformat()},{i+1},applied,{avg_power:.2f},{max_power:.2f},{power_cap:.0f},{shadow_interval},p_list={candidate.p_list} b_list={candidate.b_list}\n"
                                )
                                shadow_event_log.flush()
                        else:
                            if shadow_event_log:
                                shadow_event_log.write(
                                    f"{datetime.now().isoformat()},{i+1},no_candidate,{avg_power:.2f},{max_power:.2f},{power_cap:.0f},{shadow_interval},below_threshold_or_not_improving\n"
                                )
                                shadow_event_log.flush()

            if use_power_schedule and switch_idx < schedule_len:
                if i == switch_points[switch_idx]:
                    new_power_cap = float(power_caps[switch_idx])
                    if new_power_cap != power_cap:
                        power_cap = new_power_cap
                        controller.p_cap = power_cap
                        if shadow_optimizer:
                            shadow_optimizer.p_cap = power_cap
                        print(f"  [PowerCap] Updated to {power_cap:.0f}W (schedule)")
                        # Note: do not write to CSV to avoid breaking the format
                        if not args.no_greedy and greedy_future is None:
                            greedy_target_cap = power_cap
                            greedy_future = greedy_executor.submit(_run_greedy_for_cap, power_cap)
                        elif not args.no_greedy and greedy_future is not None:
                            queued_power_cap = power_cap
                    switch_idx += 1

            if greedy_future and greedy_future.done():
                result = greedy_future.result()
                greedy_future = None
                if result and greedy_target_cap == power_cap:
                    pending_config = result
                if queued_power_cap and queued_power_cap == power_cap:
                    greedy_target_cap = queued_power_cap
                    queued_power_cap = None
                    greedy_future = greedy_executor.submit(_run_greedy_for_cap, power_cap)

            iterations_since_shadow_change += 1  # Increment counter each iteration
            log_txt.flush()

        if number > 0:
            print(f"\nAverage throughput: {total_throughput/number:.2f}")
    finally:
        if 'greedy_executor' in locals():
            greedy_executor.shutdown(wait=False)
        if shadow_event_log:
            try:
                shadow_event_log.close()
            except Exception:
                pass
        if mps_cluster:
            mps_cluster.stop()

def set_nvidia_power_limit(power_cap_w):
    """Set GPU power cap via nvidia-smi -pl (auto clamps to hardware minimum)."""
    try:
        # Get minimum supported power limit
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.min_limit', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            min_limit = float(result.stdout.strip())
            if power_cap_w < min_limit:
                print(f"WARNING: Requested power {power_cap_w}W below hardware minimum {min_limit}W, clamping to {min_limit}W")
                power_cap_w = min_limit
        else:
            print(f"WARNING: Failed to read minimum power limit, using requested value: {result.stderr.strip()}")

        result = subprocess.run(
            ['sudo', '-S', 'nvidia-smi', '-pl', str(int(power_cap_w))],
            input=SUDO_PASSWORD + '\n',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"WARNING: Failed to set power cap: {result.stderr.strip()}")
            return False
        print(f"Power cap set: {power_cap_w}W")
        return True
    except Exception as e:
        print(f"WARNING: Exception while setting power cap: {e}")
        return False

def directcap_start(configs, log_txt, args):
    """
    NVIDIA Direct PowerCap algorithm:
    - Use greedy partitioning to get initial MPS and batch_size config
    - Use nvidia-smi -pl to set GPU power cap directly
    - Do not control frequency (let GPU auto-boost)
    - Auto lower memory frequency by one step after consecutive over-cap events
    - After meeting power cap, explore lower memory frequencies for best config
    - Log to the same CSV throughout
    """
    model_name = configs.model.model_name
    use_power_schedule = args.power_cap is None
    if use_power_schedule:
        # Power scheduling mode: use first value in schedule list as initial power cap
        power_cap = float(POWER_CAP_LIST[0])
    else:
        power_cap = float(args.power_cap if args.power_cap is not None else configs.model.power_cap)
    mps_step = int(args.mps_step)
    dataset_name = args.dataset
    if dataset_name and dataset_name.lower() != "imagenet":
        dataset_name = "imagenet"
    if dataset_name is None:
        data_path = getattr(configs.model, "data_path", None)
        if data_path and os.path.exists(data_path):
            if "imagenet" in data_path.lower():
                dataset_name = "imagenet"

    # Use greedy partitioning to determine initial config
    # Prefer CLI-provided mps-resources and batch-sizes
    if args.mps_resources and args.batch_sizes:
        gpu_resources_list = json.loads(args.mps_resources)
        batch_size_list = json.loads(args.batch_sizes)
        frequency = configs.platform.sm_clocks[-1]
        mem_freq = configs.platform.mem_clocks[-1] if configs.platform.mem_clocks else get_default_mem_freq(args.platform)
        print(f"[DirectCap] Using preset config: mps_resources={gpu_resources_list}, batch_sizes={batch_size_list}")
    elif args.no_greedy:
        gpu_resources_list = configs.model.resources
        batch_size_list = configs.model.batch_size_list or [32] * len(gpu_resources_list)
        frequency = configs.platform.sm_clocks[-1]
        mem_freq = configs.platform.mem_clocks[-1] if configs.platform.mem_clocks else get_default_mem_freq(args.platform)
    else:
        from greedy_partitioning import GreedyPartitioner
        # Apply power margin: greedy algorithm uses (power_cap - power_margin)
        greedy_power_cap = power_cap - args.power_margin
        partitioner = GreedyPartitioner(
            model_name,
            greedy_power_cap,
            mps_step=mps_step,
            mem_freq=args.mem_freq,
            platform=args.platform,
            dataset=dataset_name,
            vectorized=getattr(args, "greedy_vectorized", False),
        )
        greedy_result = partitioner.run()
        if isinstance(greedy_result, (list, tuple)) and len(greedy_result) >= 3:
            tasks, frequency, mem_freq = greedy_result[:3]
        else:
            raise ValueError(f"Unexpected greedy partitioner return: {greedy_result!r}")
        if not tasks:
            print("WARNING: Greedy partitioner returned empty config, falling back to defaults.")
            gpu_resources_list = configs.model.resources
            batch_size_list = configs.model.batch_size_list or [32] * len(gpu_resources_list)
            frequency = configs.platform.sm_clocks[-1]
            mem_freq = configs.platform.mem_clocks[-1] if configs.platform.mem_clocks else get_default_mem_freq(args.platform)
        else:
            gpu_resources_list = [float(t.mps) for t in tasks]
            batch_size_list = [int(t.batch) for t in tasks]

    gpu_resources_list = [float(x) for x in gpu_resources_list]
    batch_size_list = [int(x) for x in batch_size_list]

    # Memory frequency auto-adjust variables
    supported_mem_freqs = configs.platform.mem_clocks if configs.platform.mem_clocks else []
    # Initialize memory frequency to highest tier
    mem_freq = supported_mem_freqs[-1] if supported_mem_freqs else mem_freq
    current_mem_freq_idx = len(supported_mem_freqs) - 1 if supported_mem_freqs else 0
    power_exceed_count = 0  # Counter for consecutive power-cap exceedances
    POWER_EXCEED_THRESHOLD = 5  # Lower memory frequency after this many consecutive exceeds
    POWER_TOLERANCE = 1.02  # Power tolerance (102% counts as exceed)

    # Memory frequency exploration variables
    EXPLORE_ITERATIONS = 5  # Iterations per frequency during exploration
    in_exploration_mode = False  # Whether in exploration mode
    exploration_start_iter = 0  # Iteration index when exploration starts
    exploration_candidate_freqs = []  # Candidate memory frequency list
    exploration_current_freq_idx = 0  # Current exploration frequency index
    exploration_results = {}  # {mem_freq: avg_throughput}

    if len(batch_size_list) < len(gpu_resources_list):
        default_bs = batch_size_list[0] if batch_size_list else 32
        batch_size_list = batch_size_list + [default_bs] * (len(gpu_resources_list) - len(batch_size_list))
    elif len(batch_size_list) > len(gpu_resources_list):
        batch_size_list = batch_size_list[:len(gpu_resources_list)]

    def _run_greedy_for_cap(target_power_cap):
        from greedy_partitioning import GreedyPartitioner
        # Apply power margin: greedy uses (target_power_cap - power_margin)
        greedy_power_cap = target_power_cap - args.power_margin
        partitioner = GreedyPartitioner(
            model_name,
            greedy_power_cap,
            mps_step=mps_step,
            mem_freq=args.mem_freq,
            platform=args.platform,
            dataset=dataset_name,
            vectorized=getattr(args, "greedy_vectorized", False),
        )
        greedy_result = partitioner.run()
        if isinstance(greedy_result, (list, tuple)) and len(greedy_result) >= 3:
            tasks, next_freq, next_mem = greedy_result[:3]
        else:
            raise ValueError(f"Unexpected greedy partitioner return: {greedy_result!r}")
        if not tasks:
            return None
        next_gpu_resources = [float(t.mps) for t in tasks]
        next_batch_sizes = [int(t.batch) for t in tasks]
        if len(next_batch_sizes) < len(next_gpu_resources):
            default_bs = next_batch_sizes[0] if next_batch_sizes else 32
            next_batch_sizes = next_batch_sizes + [default_bs] * (len(next_gpu_resources) - len(next_batch_sizes))
        elif len(next_batch_sizes) > len(next_gpu_resources):
            next_batch_sizes = next_batch_sizes[:len(next_gpu_resources)]
        return next_gpu_resources, next_batch_sizes, float(next_freq), float(next_mem)

    mps_cluster = None
    mps_cluster = _apply_mps_config(mps_cluster, gpu_resources_list, model_name, dataset_name=dataset_name)

    def _run_exploration_iteration(batch_size_list, mem_freq):
        """Run one exploration iteration and return throughput."""
        stop_event = threading.Event()
        sampler_thread = configs.monitor_executor.submit(
            doProfile.sample_gpu_info,
            stop_event,
            configs.handle,
            mps_cluster.warmup_seconds,
        )
        mps_cluster.submit_all(batch_size_list)
        latencies, throughputs, batch_used = mps_cluster.wait_all_results()
        stop_event.set()
        _, avg_power, *_ = sampler_thread.result()
        return sum(throughputs), avg_power

    try:
        total_throughput = 0.0
        number = 0
        num_iterations = args.iterations
        power_caps = POWER_CAP_LIST if use_power_schedule else [power_cap]
        switch_points = POWER_CAP_SWITCH_POINTS if use_power_schedule else []
        schedule_len = min(len(power_caps), len(switch_points))
        switch_idx = 0
        greedy_executor = ThreadPoolExecutor(max_workers=1)
        greedy_future = None
        greedy_target_cap = None
        queued_power_cap = None
        pending_config = None

        print("\n========== DirectCap Scheduler Started (NVIDIA direct power control) ==========")
        print(f"Initial config: MPS={gpu_resources_list}, Batch={batch_size_list}")
        print(f"Initial memory frequency: {mem_freq}MHz (tier {current_mem_freq_idx + 1}/{len(supported_mem_freqs)})")
        print(f"Note: power cap set via nvidia-smi -pl; GPU auto-boosts")
        print(f"Auto-adjust memory frequency: after {POWER_EXCEED_THRESHOLD} consecutive exceeds of power cap ({POWER_TOLERANCE*100:.0f}%), step down one tier")
        print(f"After power stabilizes, explore lower memory frequencies for best config")

        # When appending to an existing CSV, avoid duplicating the header row.
        if log_txt.tell() == 0:
            log_txt.write("iter,mps_resources,batch_sizes,sm_freq_mhz,mem_freq_mhz,sm_util,mem_util,throughput_total,power_avg,power_max,power_cap_w,timestamp\n")

        if supported_mem_freqs:
            try:
                set_mem_clock_only(mem_freq)
            except Exception as e:
                print(f"  WARNING: Failed to set memory frequency: {e}")

        if not use_power_schedule:
            set_nvidia_power_limit(power_cap)

        for i in range(num_iterations):
            if pending_config:
                next_gpu_resources, next_batch_sizes, next_freq, next_mem = pending_config
                pending_config = None

                mps_changed = next_gpu_resources != gpu_resources_list

                gpu_resources_list = [float(x) for x in next_gpu_resources]
                batch_size_list = [int(x) for x in next_batch_sizes]
                if mps_changed:
                    mps_cluster = _apply_mps_config(mps_cluster, gpu_resources_list, model_name, dataset_name=dataset_name)

            readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            timestamp = readable_time
            stop_event = threading.Event()
            sampler_thread = configs.monitor_executor.submit(
                doProfile.sample_gpu_info,
                stop_event,
                configs.handle,
                mps_cluster.warmup_seconds,
            )
            if i < 3:
                for warmup_round in range(1):
                    mps_cluster.submit_all(batch_size_list)
                    mps_cluster.wait_all_results()

            batch_size_list_req = list(batch_size_list)
            mps_cluster.submit_all(batch_size_list_req)
            latencies, throughputs, batch_used = mps_cluster.wait_all_results()

            batch_size_list_used = []
            for j in range(len(batch_size_list_req)):
                u = None
                if j < len(batch_used):
                    u = batch_used[j]
                if u is None or int(u) <= 0:
                    u = batch_size_list_req[j]
                batch_size_list_used.append(int(u))

            stop_event.set()
            (
                max_power,
                avg_power,
                max_sm_util,
                avg_sm_util,
                max_mem_used,
                avg_mem_used,
                max_pcie,
                avg_pcie,
                max_mem_bw_util,
                avg_mem_bw_util,
                *_,
            ) = sampler_thread.result()
            sm_util = avg_sm_util
            mem_util = avg_mem_bw_util
            iter_throughput = sum(throughputs)

            try:
                current_sm = get_current_sm_clock(configs.handle)
            except:
                current_sm = 0

            print("\n[Iteration {}/{}] {} | Throughput: {:.2f} samples/s | Power: {:.2f}W | MemFreq: {}MHz".format(
                i + 1, num_iterations, readable_time, iter_throughput, avg_power, int(mem_freq)))

            mps_str = "\"[" + ",".join(f"{x:g}" for x in gpu_resources_list) + "]\""
            bs_str = "\"[" + ",".join(str(x) for x in batch_size_list_used) + "]\""
            log_txt.write(
                f"{i+1},{mps_str},{bs_str},{current_sm:.0f},{mem_freq:.0f},{sm_util:.2f},{mem_util:.2f},{iter_throughput:.2f},{avg_power:.2f},{max_power:.2f},{power_cap:.0f},{timestamp}\n"
            )
            log_txt.flush()

            # Feed back the feasible batches
            next_batches = list(batch_size_list_used)
            for idx, tput in enumerate(throughputs):
                if tput <= 0 and idx < len(next_batches):
                    next_batches[idx] = max(1, int(next_batches[idx] // 2))
            if next_batches != batch_size_list:
                batch_size_list = next_batches

            start_counting_from = max(3, num_iterations // 2)
            if i >= start_counting_from:
                total_throughput += iter_throughput
                number += 1

            # ============ Memory frequency exploration mode ============
            if in_exploration_mode:
                # Run in exploration mode
                if exploration_current_freq_idx < len(exploration_candidate_freqs):
                    current_explore_freq = exploration_candidate_freqs[exploration_current_freq_idx]
                    if i - exploration_start_iter < EXPLORE_ITERATIONS:
                        # Continue exploring current frequency
                        pass
                    else:
                        # Finish current frequency exploration, record result, and switch to next frequency
                        recent_throughputs = []
                        for j in range(exploration_start_iter + EXPLORE_ITERATIONS, i + 1):
                            # Simplified: use current mem_freq iteration result
                            pass
                        exploration_results[current_explore_freq] = iter_throughput  # Simplified single-iter result
                        exploration_current_freq_idx += 1
                        exploration_start_iter = i + 1
                        if exploration_current_freq_idx < len(exploration_candidate_freqs):
                            # Switch to next frequency
                            mem_freq = exploration_candidate_freqs[exploration_current_freq_idx]
                            mem_freq = float(mem_freq)
                            try:
                                set_mem_clock_only(mem_freq)
                                print(f"\n  [Exploration] Switch to memory frequency: {mem_freq}MHz")
                            except Exception as e:
                                print(f"  WARNING: Failed to set memory frequency: {e}")
                        else:
                            # Exploration complete; select best frequency
                            in_exploration_mode = False
                            if exploration_results:
                                best_mem_freq = max(exploration_results, key=exploration_results.get)
                                best_tp = exploration_results[best_mem_freq]
                                mem_freq = best_mem_freq
                                try:
                                    set_mem_clock_only(mem_freq)
                                    print(f"\n  [Exploration] Complete; select best memory frequency: {mem_freq}MHz (throughput: {best_tp:.2f})")
                                except Exception as e:
                                    print(f"  WARNING: Failed to set memory frequency: {e}")
                else:
                    in_exploration_mode = False
            # ============ Normal mode: power cap check and auto lower memory frequency ============
            elif not use_power_schedule and supported_mem_freqs:
                if avg_power > power_cap * POWER_TOLERANCE:
                    power_exceed_count += 1
                    if power_exceed_count >= POWER_EXCEED_THRESHOLD:
                        if current_mem_freq_idx > 0:
                            current_mem_freq_idx -= 1
                            old_mem_freq = mem_freq
                            mem_freq = supported_mem_freqs[current_mem_freq_idx]
                            mem_freq = float(mem_freq)
                            try:
                                set_mem_clock_only(mem_freq)
                                print(f"\n  [DirectCap] Lowering memory frequency: {old_mem_freq}MHz -> {mem_freq}MHz")
                            except Exception as e:
                                print(f"  WARNING: Failed to set memory frequency: {e}")
                        power_exceed_count = 0
                else:
                    # Power stable; trigger exploration mode (only if current freq >= 6000)
                    if power_exceed_count > 0 and mem_freq >= 6000 and len(supported_mem_freqs) > 1:
                        # Enter exploration mode
                        in_exploration_mode = True
                        exploration_start_iter = i + 1
                        exploration_current_freq_idx = 0
                        # Filter candidate frequencies: >=5001 and <= current frequency
                        current_freq_val = int(mem_freq)
                        exploration_candidate_freqs = sorted([f for f in supported_mem_freqs if 5001 <= f <= current_freq_val])
                        exploration_candidate_freqs = list(dict.fromkeys(exploration_candidate_freqs))  # De-duplicate while preserving order
                        print(f"\n  [DirectCap] Power stable; trigger memory-frequency exploration")
                        print(f"  [Exploration] Candidate frequencies: {exploration_candidate_freqs}")
                    power_exceed_count = 0
            # ============ Power-cap check end ============

            # Dynamic power-cap schedule switch (sync config update)
            if use_power_schedule and switch_idx < schedule_len:
                if i == switch_points[switch_idx]:
                    new_power_cap = float(power_caps[switch_idx])

                    # Recompute config when power cap changes
                    if new_power_cap != power_cap:
                        print(f"\n  [PowerCap] Preparing to switch power cap: {power_cap:.0f}W -> {new_power_cap:.0f}W")

                        # If greedy enabled, compute new config synchronously first
                        if not args.no_greedy:
                            print(f"  [PowerCap] Computing optimal config for {new_power_cap:.0f}W...")
                            try:
                                result = _run_greedy_for_cap(new_power_cap)
                                if result:
                                    next_gpu_resources, next_batch_sizes, next_freq, next_mem = result

                                    # Apply new config immediately
                                    mps_changed = next_gpu_resources != gpu_resources_list
                                    gpu_resources_list = [float(x) for x in next_gpu_resources]
                                    batch_size_list = [int(x) for x in next_batch_sizes]

                                    if mps_changed:
                                        mps_cluster = _apply_mps_config(mps_cluster, gpu_resources_list, model_name, dataset_name=dataset_name)

                                    print(f"  [PowerCap] New config applied: MPS={gpu_resources_list}, Batch={batch_size_list}")
                                else:
                                    print(f"  WARNING: Greedy algorithm returned no valid config; keeping current config")
                            except Exception as e:
                                print(f"  WARNING: Greedy algorithm failed: {e}; keeping current config")

                    # Always set power cap at switch points (including i=0)
                    power_cap = new_power_cap
                    set_nvidia_power_limit(power_cap)
                    print(f"  [PowerCap] Power cap set to {power_cap:.0f}W")
                    switch_idx += 1

            # Handle async greedy results (optimize outside schedule switch)
            if greedy_future and greedy_future.done():
                result = greedy_future.result()
                greedy_future = None
                if result and greedy_target_cap == power_cap:
                    pending_config = result
                if queued_power_cap and queued_power_cap == power_cap:
                    greedy_target_cap = queued_power_cap
                    queued_power_cap = None
                    greedy_future = greedy_executor.submit(_run_greedy_for_cap, power_cap)

            log_txt.flush()

        if number > 0:
            print(f"\nAverage throughput: {total_throughput/number:.2f}")
    finally:
        if 'greedy_executor' in locals():
            greedy_executor.shutdown(wait=False)
        if mps_cluster:
            mps_cluster.stop()
        # Restore default power cap
        reset_power_limit_to_default()

def start(configs, log_txt, args, iterations=None, use_power_schedule=False, use_greedy=True):
    """
    PID Scheduler - MPS multi-process version
    """
    gpu_resource_list = configs.model.resources
    monitoer_executor = configs.monitor_executor
    
    log_txt.write("--------------CLIENT FINISHED------------------\n")
    print("\n========== PID Scheduler Started ==========")
    log_txt.write("\n========== PID Scheduler Started ==========\n")

    batch_size_list = []
    
    # Try greedy algorithm (if cold start and allowed)
    if use_greedy:
        print("\nRunning greedy algorithm to determine initial MPS config...")
        log_txt.write("Running greedy algorithm to determine initial MPS config...\n")
        
        try:
            from mps_optimizer import SystemModel, GreedyPartitioner

            params_file = os.path.join(os.path.dirname(__file__), '..', 'physics_params.json')
            profile_dir = os.path.join(os.path.dirname(__file__), "profile")
            sys_model = SystemModel(params_file=params_file)
            
            supported_freqs = configs.platform.sm_clocks if hasattr(configs.platform, 'sm_clocks') else [210, 360, 510, 660, 810, 960, 1110, 1260, 1410, 1560, 1710, 1860, 2010, 2100]
            profile_dir = os.path.join(os.path.dirname(__file__), "profile")
            print("\nInitializing greedy partitioner...")
            partitioner = GreedyPartitioner(sys_model, supported_freqs, profile_dir)
            power_cap = configs.model.power_cap if hasattr(configs.model, 'power_cap') else 230
            model_name = configs.model.model_name
            
            print(f"\nStart optimization: model={model_name}, power cap={power_cap}W")
            log_file = os.path.join(os.path.dirname(__file__), '..', 'logs', f'{model_name}_partition_log.txt')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            task_configs, optimal_freq = partitioner.optimize(power_cap, [model_name], log_file)
            if task_configs and len(task_configs) > 0:
                gpu_resources_list = [task.mps_ratio for task in task_configs]
                batch_size_list = [task.batch_size for task in task_configs]
                
                print(f"\nGreedy optimization complete:")
                print(f"  - Task count: {len(task_configs)}")
                print(f"  - MPS allocation: {gpu_resources_list}")
                print(f"  - Batch Size: {batch_size_list}")
                print(f"  - Optimal frequency: {optimal_freq} MHz")
                
                log_txt.write(f"Greedy result: MPS={gpu_resources_list}, Batch={batch_size_list}, Freq={optimal_freq}MHz\n")
            else:
                print("  WARNING: Greedy algorithm returned no valid config, using defaults")
                log_txt.write("Greedy algorithm returned no valid config, using defaults\n")
                gpu_resources_list = configs.model.resources
                if hasattr(configs.model, 'batch_size_list') and configs.model.batch_size_list:
                    batch_size_list = configs.model.batch_size_list
                else:
                    batch_size_list = [32] * len(gpu_resources_list)
                    
        except Exception as e:
            print(f"  ERROR: Greedy algorithm failed: {e}")
            import traceback
            traceback.print_exc()
            log_txt.write(f"Greedy algorithm failed: {e}\n")
            
            # Fallback to default config
            gpu_resources_list = configs.model.resources
            if hasattr(configs.model, 'batch_size_list') and configs.model.batch_size_list:
                batch_size_list = configs.model.batch_size_list
            else:
                batch_size_list = [32] * len(gpu_resources_list)
        
    else:
        print("Skipping greedy algorithm, using config file/defaults...")
        log_txt.write("Skipping greedy algorithm, using config file/defaults...\n")
        gpu_resources_list = configs.model.resources
        # FIX: check batch_size_list attribute directly
        if hasattr(configs.model, 'batch_size_list') and configs.model.batch_size_list:
             batch_size_list = configs.model.batch_size_list
        else:
             batch_size_list = [32] * len(gpu_resources_list)

    # Ensure gpu_resources_list and batch_size_list lengths match
    if not gpu_resources_list:
        gpu_resources_list = configs.model.resources
    
    # Truncate or pad
    if len(batch_size_list) < len(gpu_resources_list):
        default_bs = batch_size_list[0] if batch_size_list else 32
        batch_size_list = batch_size_list + [default_bs] * (len(gpu_resources_list) - len(batch_size_list))
    elif len(batch_size_list) > len(gpu_resources_list):
        batch_size_list = batch_size_list[:len(gpu_resources_list)]

    print(f"Final Config: MPS={gpu_resources_list}, Batch={batch_size_list}")
    log_txt.write(f"Final Config: MPS={gpu_resources_list}, Batch={batch_size_list}\n")
    print("\n========== MPS Cluster initialization ==========")
    mps_cluster = MPSCluster(configs.model.model_name, gpu_resources_list, repo_root=repo_root)
    mps_cluster.start()

    try:
        def _load_other_max_bs(config_model_name, suffix):
            config_path = os.path.join(
                os.path.dirname(__file__),
                "config",
                config_model_name,
                f"{config_model_name}{suffix}.json",
            )
            if not os.path.exists(config_path):
                return None
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return None
            candidates = []
            batch_list = data.get("batch size list") or data.get("batch_size_list")
            if isinstance(batch_list, list) and batch_list:
                candidates.append(max(int(x) for x in batch_list))
            max_bs = data.get("max batchsize") or data.get("max_bs")
            if max_bs:
                try:
                    candidates.append(int(max_bs))
                except ValueError:
                    pass
            return min(candidates) if candidates else None

        def _build_batch_candidates(max_cap):
            candidates = []
            for b in (1, 2, 4, 8, 16, 32):
                if b <= max_cap:
                    candidates.append(b)
            for b in range(48, max_cap + 1, 16):
                candidates.append(b)
            return sorted(set(candidates))

        model_name_mapping = {"mobilenet_v2": "mobilenetv2"}
        config_model_name = model_name_mapping.get(args.model, args.model)
        max_bs = getattr(configs.model, "max_bs", None)
        max_bs = int(max_bs) if max_bs else None
        if len(set(batch_size_list)) > 1:
            dynamic_batch_list = sorted(list(set(batch_size_list)))
            if max_bs:
                dynamic_batch_list = [b for b in dynamic_batch_list if b <= max_bs]
        else:
            max_cap = max_bs or (batch_size_list[0] if batch_size_list else 256)
            dynamic_batch_list = _build_batch_candidates(max_cap)

        sm_pid = scheduler.PID(configs.model.power_cap, configs.model.alpha, 7e-4, 4e-5, 5e-6, configs.platform.sm_clocks, log_txt)
        mem_pid = scheduler.PID(configs.model.power_cap, configs.model.alpha, 1e-4, 1e-6, 1e-5, configs.platform.mem_clocks, log_txt)
        bs_pid = scheduler.PID(configs.model.power_cap, configs.model.alpha, 9e-5, 7e-5, 7e-5, dynamic_batch_list, log_txt)
        pid_scheduler = scheduler.PIDScheduler(sm_pid, mem_pid, bs_pid, log_txt)

        frequency = 810
        memFreqList = [5001, 9251, 9501]
        memFreq = memFreqList[1]
        
        powerCapList = POWER_CAP_LIST
        switch_points = POWER_CAP_SWITCH_POINTS
        num_iterations = iterations if iterations is not None else NUM_ITERATIONS
        
        total_throught = 0
        number = 0
        
        if use_power_schedule:
            pid_scheduler.sm_pid.power_cap = powerCapList[0]
        else:
            pid_scheduler.sm_pid.power_cap = configs.model.power_cap

        for i in range(num_iterations):
            if use_power_schedule and i in switch_points:
                switch_idx = switch_points.index(i)
                if switch_idx < len(powerCapList):
                    pid_scheduler.sm_pid.power_cap = powerCapList[switch_idx]
                    print(f" [Iteration {i}] Switching power cap to: {powerCapList[switch_idx]}W")

            readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("\n[Iteration {}/{}] Start test - {}".format(i+1, num_iterations, readable_time))
            target_freq, target_mem = frequency, memFreq
            frequency, memFreq = snap_freq_pair(frequency, memFreq, configs.platform, configs.handle)
            print("  Params: batch_size={}, frequency={}MHz (target {}MHz), memFreq={}MHz (target {}MHz), power_cap={}W".format(
                batch_size_list[0], frequency, target_freq, memFreq, target_mem, pid_scheduler.sm_pid.power_cap))

            stop_event = threading.Event()
            sampler_thread = monitoer_executor.submit(
                doProfile.sample_gpu_info,
                stop_event,
                configs.handle,
                mps_cluster.warmup_seconds,
            )

            set_gpu_clocks(frequency, frequency, memFreq, memFreq)
            actual_sm = get_current_sm_clock(configs.handle)
            if actual_sm is not None:
                print(f"  Actual SM frequency: {actual_sm:.0f}MHz (target {target_freq:.0f}MHz, locked {frequency:.0f}MHz)")
            else:
                print("  Actual SM frequency: N/A")

            mps_cluster.submit_all(batch_size_list)
            latencies, throughputs, _batch_used = mps_cluster.wait_all_results()

            stop_event.set()

            iter_throughput = sum(throughputs)
            max_power, avg_power, *_ = sampler_thread.result()

            print("  Throughput: {:.2f} samples/s | Power: {:.2f}W".format(iter_throughput, avg_power))

            start_counting_from = max(3, num_iterations // 2)
            if i >= start_counting_from:
                total_throught += iter_throughput
                number += 1

            frequency, memFreq, next_bs = pid_scheduler.scheduler(max_power, frequency, memFreq, batch_size_list[0])

            # Frequency boundary guard
            if frequency < 660 and memFreq != 5001:
                memFreq = 5001
                frequency = 810

            batch_size_list = [next_bs] * len(gpu_resources_list)
            log_txt.flush()

        if number > 0:
            print(f"\nAverage throughput: {total_throught/number:.2f}")

    finally:
        mps_cluster.stop()

if __name__=="__main__":
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='GPU Inference Optimization Framework (MPS Isolation Mode)')
    parser.add_argument('--algorithm', type=str, default='pctodl', choices=['pctodl', 'morak', 'batchdvfs', 'directcap', 'powercap'],
                        help="Scheduling algorithm: pctodl (DVFS control), morak, batchdvfs, directcap/powercap (nvidia-smi -pl power cap)")
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--power-cap', type=int, default=None)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--config-path', type=str, default=None)
    parser.add_argument('--no-greedy', action='store_true', help='Disable greedy algorithm; force resource allocation from config file')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name for predictor/greedy')
    parser.add_argument('--platform', type=str, default=None, help='Platform name for predictor/greedy')
    parser.add_argument('--power-margin', type=float, default=0.0, help='Power margin (W) for greedy partitioning only. Greedy uses (power_cap - margin)')
    parser.add_argument('--mem-freq', type=int, default=None, help='Fix memory frequency (MHz) for greedy/control')
    parser.add_argument('--mps-step', type=int, default=5, help='MPS step size for greedy/shadow')
    parser.add_argument(
        '--greedy-vectorized',
        action='store_true',
        help='Use greedy_partitioning.py vectorized search (faster; experimental). Only affects greedy partitioning.',
    )
    parser.add_argument('--disable-shadow', action='store_true', help='Disable shadow optimizer')
    parser.add_argument('--shadow-interval', type=int, default=3, help='Shadow optimizer interval (iterations)')
    parser.add_argument('--shadow-radius', type=int, default=3, help='Shadow optimizer trust-region radius')
    parser.add_argument('--shadow-history', type=int, default=300, help='Shadow optimizer history window')
    parser.add_argument('--shadow-gp-refit-interval', type=int, default=3, help='Shadow GP refit interval (updates)')
    parser.add_argument('--shadow-gp-max-points', type=int, default=300, help='Max points used to fit shadow GP (subsample if needed)')
    parser.add_argument('--shadow-threshold', type=float, default=0.003, help='Shadow optimizer gain threshold')
    parser.add_argument('--shadow-adjust-mem', action='store_true', help='Allow shadow optimizer to adjust memory frequency')
    parser.add_argument('--log-file', type=str, default=None, help='Override log file path')
    parser.add_argument('--power-schedule', type=str, default=None, help='Power schedule: "up" (130->180->230W) or "down" (200->160->120W) for dynamic power cap testing')
    parser.add_argument('--mps-resources', type=str, default=None, help='Preset MPS resources list (JSON format, e.g., "[15,15,15,10,10,10,25]")')
    parser.add_argument('--batch-sizes', type=str, default=None, help='Preset batch sizes list (JSON format, e.g., "[32,32,16,32,32,16,16]")')
    args = parser.parse_args()
    if args.power_schedule == "up":
        #  130W -> 180W -> 230W
        POWER_CAP_LIST = [130, 180, 230]
        POWER_CAP_SWITCH_POINTS = [0, 50, 100]
    elif args.power_schedule == "down":
        #  200W -> 160W -> 120W
        POWER_CAP_LIST = [200, 160, 120]
        POWER_CAP_SWITCH_POINTS = [0, 50, 100]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_base_dir = os.path.join(script_dir, 'config')
    platform_config_path = os.path.join(config_base_dir, 'platform.json')
    with open(platform_config_path, 'r', encoding='utf-8') as file:
        platform_data = json.load(file)

    model_name_mapping = {'mobilenet_v2': 'mobilenetv2'}
    config_model_name = model_name_mapping.get(args.model, args.model)

    if args.config_path:
        config_file_path = args.config_path
    else:
        if args.algorithm == "pctodl":
            config_file_path = os.path.join(config_base_dir, config_model_name, f'{config_model_name}.json')
        elif args.algorithm == "morak":
            config_file_path = os.path.join(config_base_dir, config_model_name, f'{config_model_name}_morak.json')
        elif args.algorithm == "batchdvfs":
            config_file_path = os.path.join(config_base_dir, config_model_name, f'{config_model_name}_batchdvfs.json')
        elif args.algorithm == "directcap":
            config_file_path = os.path.join(config_base_dir, config_model_name, f'{config_model_name}.json')
            
    with open(config_file_path, 'r', encoding='utf-8') as file:
        model_data = json.load(file)
    
    if args.power_cap:
        model_data["Power Cap"] = args.power_cap
    
    configs = globalConfig.GlobalConfig(platform_data, model_data, platform_hint=args.platform)

    log_path = args.log_file if args.log_file else configs.model.log_path
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Benchmark runs often reuse the same `--log-file` path (e.g., under benchmark_results/).
    # If the file already exists, append instead of overwriting so repeated runs accumulate.
    file_exists = os.path.exists(log_path)
    file_nonempty = file_exists and os.path.getsize(log_path) > 0
    is_benchmark_output = "benchmark_results" in Path(log_path).parts
    open_mode = "a" if (is_benchmark_output and file_nonempty) else "w"

    with open(log_path, open_mode) as log_txt:
        reset_power_limit_to_default()
        # Reset GPU
        os.system(f"echo '{SUDO_PASSWORD}' | sudo -S nvidia-smi -rmc > /dev/null 2>&1")
        os.system(f"echo '{SUDO_PASSWORD}' | sudo -S nvidia-smi -rgc > /dev/null 2>&1")
        shutdown_mps = os.path.join(script_dir, 'shutdown_mps.sh')
        start_mps = os.path.join(script_dir, 'start_mps.sh')
        subprocess.run(['sudo', '-S', shutdown_mps], input=SUDO_PASSWORD + '\n', stdout=subprocess.PIPE, text=True)
        time.sleep(1)
        subprocess.run(['sudo', '-S', start_mps], input=SUDO_PASSWORD + '\n', stdout=subprocess.PIPE, text=True)
        time.sleep(2)
        
        print("MPS Server Started.")
        
        use_power_schedule = args.power_cap is None
        
        if args.algorithm == "pctodl":
            pctodl_start(configs, log_txt, args)
        elif args.algorithm == "morak":
            morak_start(configs, log_txt, args)
        elif args.algorithm == "batchdvfs":
            batchdvfs_start(configs, log_txt, args)
        elif args.algorithm == "directcap":
            directcap_start(configs, log_txt, args)
        elif args.algorithm == "powercap":
            # using nvidia-smi -pl to control power cap
            directcap_start(configs, log_txt, args)
        elif args.algorithm == "powercap":
            powercap_start(configs, log_txt, args)

    cleanup_script = os.path.join(script_dir, 'cleanup_processes.sh')
    if os.path.exists(cleanup_script):
        subprocess.run(
            ['bash', '-lc', f'echo "{SUDO_PASSWORD}" | sudo -S "{cleanup_script}"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    else:
        shutdown_mps = os.path.join(script_dir, 'shutdown_mps.sh')
        subprocess.run(f'echo "{SUDO_PASSWORD}" | sudo -S {shutdown_mps}', shell=True)
    print("Done.")
