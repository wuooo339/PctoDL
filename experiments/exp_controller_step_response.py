#!/usr/bin/env python3
"""
Experiment 2: Controller Stability and Step Response

Implements the step-response experiment described in experiments/experiment_spec.md.

Output: experiments/controller_stability.csv

By default, this script runs on a real GPU (NVML + MPS + sudo clock lock). For
development/CI without a GPU, use --mode simulate to generate a deterministic
closed-loop trace driven by the PctoDL predictor.

Quick start:
  # Power setpoints: 230W -> 130W -> 180W, 50 iterations per phase
  python experiments/exp_controller_step_response.py

  # Custom power sequence: 250W -> 150W -> 200W, 30 iterations per phase
  python experiments/exp_controller_step_response.py --caps "[250,150,200]" --iters-per-phase 30

  # Simulation mode (No GPU required)
  python experiments/exp_controller_step_response.py --mode simulate --caps "[250,150,200]"

Key args:
  --caps            List of power setpoints (W), e.g., "[230,130,180]"
  --iters-per-phase Number of iterations per power phase
  --mode            'gpu' or 'simulate'
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import multiprocessing as mp

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from predict import PctoDLPredictor
import scheduler
import doProfile
from thermo_control import ThermodynamicModelController
import main as pctodl_main


CSV_COLUMNS = [
    "time_ms", "iter", "method", "model", "p_list", "b_list", 
    "mem_freq", "sm_freq", "sm_util", "mem_util", 
    "p_cap", "meas_power", "max_power", "meas_throughput"
]

def _ensure_csv_header(path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=list(columns)).to_csv(path, index=False)

def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def _load_platform_clocks(platform_key: Optional[str]) -> Tuple[List[float], List[float]]:
    cfg = _REPO_ROOT / "src" / "config" / "platform.json"
    with open(cfg, "r", encoding="utf-8") as f:
        data = json.load(f)

    platforms = data.get("platforms", [])
    chosen = None
    key = _norm(platform_key) if platform_key else ""
    if key:
        for p in platforms:
            name = _norm(p.get("name", ""))
            if key in name:
                chosen = p
                break
    if chosen is None and platforms:
        chosen = platforms[0]

    clocks = (chosen or {}).get("clocks", {})
    sm = clocks.get("SM_Clocks") or clocks.get("Graphics_Clocks") or []
    mem = clocks.get("Memory_Clocks")
    if mem is None:
        mem = clocks.get("Memory_Clock")
    if isinstance(mem, (int, float)):
        mem = [mem]

    return sorted(float(x) for x in sm), sorted(float(x) for x in (mem or []))


def _short_platform_key_from_gpu_name(gpu_name: str) -> Optional[str]:
    n = str(gpu_name).lower()
    if "a100" in n and "pcie" in n: return "a100-pcie"
    if "a100" in n and "sxm" in n: return "a100-sxm"
    if "a100" in n: return "a100"
    if "3080" in n and "ti" in n: return "3080ti"
    if "3090" in n: return "3090"
    return None


def _detect_gpu_name_nvml() -> Optional[str]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        return str(name)
    except Exception:
        return None


def _parse_list(arg: str, elem_type=float) -> list:
    s = (arg or "").strip()
    if not s: return []
    try:
        v = json.loads(s)
        if isinstance(v, list): return [elem_type(x) for x in v]
    except Exception:
        pass
    parts = [p for p in s.replace(",", " ").split() if p]
    return [elem_type(p) for p in parts]


def _nearest_supported(target_mhz: float, supported: Sequence[float]) -> float:
    if not supported: return float(target_mhz)
    return float(min(supported, key=lambda x: abs(float(x) - float(target_mhz))))


def _sudo_run(cmd: List[str], sudo_password: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sudo", "-S"] + cmd,
        input=(sudo_password + "\n"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _run_script_as_user(script_path: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(script_path)],
        cwd=str(_REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

def _ensure_mps_env(pipe_dir: str, log_dir: str) -> Tuple[str, str]:
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_dir)
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(log_dir)
    return os.environ["CUDA_MPS_PIPE_DIRECTORY"], os.environ["CUDA_MPS_LOG_DIRECTORY"]

def _maybe_fix_mps_permissions(pipe_dir: str, log_dir: str, sudo_password: str) -> None:
    needs_fix = False
    for d in (pipe_dir, log_dir):
        if os.path.exists(d) and not os.access(d, os.W_OK):
            needs_fix = True
            break
    
    # Check PID file ownership
    pid_file = os.path.join(pipe_dir, "nvidia-cuda-mps-control.pid")
    if os.path.exists(pid_file):
        try:
            pid_uid = os.stat(pid_file).st_uid
            if pid_uid != os.getuid():
                needs_fix = True
        except Exception:
            needs_fix = True

    if needs_fix:
        print(f"⚠️  Cleaning up MPS directories via sudo: {pipe_dir}")
        _sudo_run(["pkill", "-9", "nvidia-cuda-mps-control"], sudo_password)
        _sudo_run(["pkill", "-9", "nvidia-cuda-mps-server"], sudo_password)
        _sudo_run(["rm", "-rf", pipe_dir], sudo_password)
        _sudo_run(["rm", "-rf", log_dir], sudo_password)

    os.makedirs(pipe_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


def _lock_clocks(sm_freq: float, mem_freq: float, sudo_password: str) -> None:
    sm = int(round(float(sm_freq)))
    mem = int(round(float(mem_freq)))
    _sudo_run(["nvidia-smi", "-lgc", f"{sm},{sm}"], sudo_password)
    _sudo_run(["nvidia-smi", "-lmc", f"{mem},{mem}"], sudo_password)


def _reset_clocks(sudo_password: str) -> None:
    _sudo_run(["nvidia-smi", "-rgc"], sudo_password)
    _sudo_run(["nvidia-smi", "-rmc"], sudo_password)


def _start_mps(sudo_password: str) -> None:
    script = _REPO_ROOT / "start_mps.sh"
    if not script.exists():
        raise FileNotFoundError(f"Missing {script}")
    pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    log_dir = os.environ.get("CUDA_MPS_LOG_DIRECTORY", "/tmp/nvidia-log")
    _maybe_fix_mps_permissions(pipe_dir, log_dir, sudo_password)
    env = os.environ.copy()
    env["SUDO_PASSWORD"] = sudo_password
    res = _run_script_as_user(script, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"start_mps.sh failed: {res.stderr.strip() or res.stdout.strip()}")
    time.sleep(2.0)


def _stop_mps(sudo_password: str) -> None:
    script = _REPO_ROOT / "shutdown_mps.sh"
    if not script.exists(): return
    env = os.environ.copy()
    env["SUDO_PASSWORD"] = sudo_password
    _run_script_as_user(script, env=env)
    time.sleep(1.0)


def _mps_client_sanity_check(mps_pct: int = 25) -> Tuple[bool, str]:
    env = os.environ.copy()
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(mps_pct))
    code = (
        "import torch\n"
        "print('cuda_available', torch.cuda.is_available())\n"
    )
    try:
        res = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception as e:
        return False, repr(e)

    if res.returncode == 0:
        return True, res.stdout.strip()
    return False, (res.stderr.strip() or "").strip()


@dataclass
class StepwiseController:
    step_mhz: float = 50.0
    deadband_w: float = 2.0
    f_min: float = 210.0
    f_max: float = 2100.0
    supported: Sequence[float] = ()

    def __post_init__(self):
        """Initialize dynamic step-size configurations."""
        # [3080Ti Adaptation] Dynamic step-size configuration
        # The 3080Ti has ~127 frequency points with a 15MHz step size.
        self.steps = {
            "huge": 10,   # ~150 MHz (For severe deviations)
            "large": 5,   # ~75 MHz
            "medium": 2,  # ~30 MHz
            "small": 1    # ~15 MHz (Fine-tuning)
        }

    def _get_dynamic_step_mhz(self, current_power: float, p_cap: float) -> float:
        """
        [Core Logic] Calculates the frequency adjustment step size (MHz) 
        based on the power gap. Implements the Coarse-to-Fine strategy.
        """
        gap_ratio = abs(current_power - p_cap) / max(1.0, p_cap)

        if gap_ratio > 0.20:    # Gap > 20%
            step_idx = self.steps["huge"]
        elif gap_ratio > 0.10:  # Gap > 10%
            step_idx = self.steps["large"]
        elif gap_ratio > 0.05:  # Gap > 5%
            step_idx = self.steps["medium"]
        else:                   # Gap < 5%
            step_idx = self.steps["small"]

        # Convert index steps to MHz.
        # Assuming each index step corresponds to 15MHz (the 3080Ti frequency granularity).
        return float(step_idx) * 15.0

    def step(self, f_curr: float, p_meas: float, p_cap: float) -> float:
        err = float(p_cap) - float(p_meas)
        if abs(err) <= float(self.deadband_w):
            return float(f_curr)

        # 使用动态步长而不是固定步长
        dynamic_step = self._get_dynamic_step_mhz(float(p_meas), float(p_cap))
        cand = float(f_curr) + (dynamic_step if err > 0 else -dynamic_step)
        cand = float(np.clip(cand, self.f_min, self.f_max))
        return _nearest_supported(cand, self.supported)

class SimulatedPlant:
    def __init__(self, predictor, p_list, b_list, mem_freq, tau_s=2.0, noise_power_w=2.0, noise_tput=5.0, seed=0):
        self.predictor = predictor
        self.p_list = list(p_list)
        self.b_list = list(b_list)
        self.mem_freq = float(mem_freq)
        self.tau_s = float(tau_s)
        self.noise_power_w = float(noise_power_w)
        self.noise_tput = float(noise_tput)
        self.rng = np.random.default_rng(int(seed))
        self.p_meas = 0.0
        self.tput = 0.0

    def reset(self, f_init: float) -> None:
        _, _, tput_ss, p_ss = self.predictor.predict_all(self.p_list, self.b_list, float(f_init), mem_freq=self.mem_freq)
        self.p_meas = float(p_ss) * 0.6
        self.tput = float(tput_ss) * 0.6

    def step(self, f_curr: float, dt: float) -> Tuple[float, float, float]:
        _, _, tput_ss, p_ss = self.predictor.predict_all(self.p_list, self.b_list, float(f_curr), mem_freq=self.mem_freq)
        a = 1.0 - np.exp(-float(dt) / max(1e-6, self.tau_s))
        self.p_meas = float(self.p_meas + a * (float(p_ss) - self.p_meas))
        self.tput = float(self.tput + a * (float(tput_ss) - self.tput))
        p_noise = float(self.rng.normal(0.0, self.noise_power_w))
        meas_power = max(0.0, self.p_meas + p_noise)
        max_power = meas_power + abs(p_noise)
        return meas_power, max_power, max(0.0, self.tput + self.rng.normal(0.0, self.noise_tput))


def _caps_for_iter(caps: Sequence[float], iters_per_phase: int, i: int) -> float:
    if not caps: return 0.0
    idx = min(int(i // max(1, int(iters_per_phase))), len(caps) - 1)
    return float(caps[idx])


def _parse_methods(arg: str) -> List[str]:
    if not arg: return ["Stepwise", "Standard PID", "Thermo-MPC"]
    raw = [x.strip() for x in str(arg).replace(";", ",").split(",") if x.strip()]
    canon = []
    for m in raw:
        k = _norm(m)
        if k in {"stepwise", "step"}: canon.append("Stepwise")
        elif k in {"standardpid", "pid"}: canon.append("Standard PID")
        elif k in {"thermompc", "thermo"}: canon.append("Thermo-MPC")
    return list(dict.fromkeys(canon)) 


def _step_down_one(freq_mhz: float, supported: Sequence[float]) -> float:
    if not supported: return float(freq_mhz)
    idx = int(np.argmin([abs(float(x) - float(freq_mhz)) for x in supported]))
    if idx <= 0: return float(supported[0])
    return float(supported[idx - 1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Experiment 2: controller step response")
    ap.add_argument("--output", type=str, default=str(_REPO_ROOT / "experiments" / "controller_stability.csv"))
    ap.add_argument("--mode", type=str, default="gpu", choices=["gpu", "simulate"])
    ap.add_argument("--platform", type=str, default=None)
    ap.add_argument("--dataset", type=str, default="imagenet")
    ap.add_argument("--model", type=str, default="efficientnet_v2_m")
    ap.add_argument("--p-list", type=str, default="[25,25,25,25]")
    ap.add_argument("--b-list", type=str, default="[16,16,16,16]")
    ap.add_argument("--mem-freq", type=float, default=5001.0)
    ap.add_argument("--initial-sm-freq", type=float, default=None)
    ap.add_argument("--caps", type=str, default="[230,130,180]")
    ap.add_argument("--iters-per-phase", type=int, default=50)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--methods", type=str, default="Stepwise,Standard PID,Thermo-MPC")
    ap.add_argument("--control-metric", type=str, default="max", choices=["avg", "max"])
    ap.add_argument("--control-dt", type=float, default=1.0)
    ap.add_argument("--warmup-seconds", type=float, default=2.0)
    ap.add_argument("--measure-seconds", type=float, default=3.0)
    ap.add_argument("--step-mhz", type=float, default=30.0)
    ap.add_argument("--pid-kp", type=float, default=0.05, help="Updated default for stability")
    ap.add_argument("--pid-ki", type=float, default=0.005, help="Updated default for stability")
    ap.add_argument("--pid-kd", type=float, default=0.0)
    ap.add_argument("--pid-alpha", type=float, default=0.97)
    ap.add_argument("--thermo-eta", type=float, default=0.8)
    ap.add_argument("--thermo-max-delta-f", type=float, default=300.0)
    ap.add_argument("--no-manage-mps", action="store_true")
    ap.add_argument("--no-reset-clocks", action="store_true")
    ap.add_argument("--no-freq-guard", type=str, default="False")
    ap.add_argument("--sudo-password", type=str, default=os.environ.get("SUDO_PASSWORD", "wzk123456"))
    ap.add_argument("--mps-pipe-dir", type=str, default="/tmp/nvidia-mps")
    ap.add_argument("--mps-log-dir", type=str, default="/tmp/nvidia-log")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.output)
    _ensure_csv_header(out_path, CSV_COLUMNS)

    p_list = [float(x) for x in _parse_list(args.p_list, float)]
    b_list = [int(x) for x in _parse_list(args.b_list, int)]
    caps = [float(x) for x in _parse_list(args.caps, float)]
    
    platform_for_clocks = args.platform
    if args.mode == "gpu" and not platform_for_clocks:
        detected = _detect_gpu_name_nvml()
        if detected:
            platform_for_clocks = detected
            args.platform = _short_platform_key_from_gpu_name(detected) or args.platform

    sm_clocks, _ = _load_platform_clocks(platform_for_clocks)
    f_min, f_max = float(sm_clocks[0]), float(sm_clocks[-1])
    f_init = _nearest_supported(args.initial_sm_freq or f_min, sm_clocks)

    total_iters = int(args.iters_per_phase) * len(caps)
    sudo_password = str(args.sudo_password or "")
    methods = _parse_methods(args.methods)
    monitor_executor = ThreadPoolExecutor(max_workers=1)
    mps_cluster = None

    try:
        if args.mode == "gpu":
            if not sudo_password and not args.no_manage_mps:
                print("Warning: No sudo password provided, MPS management might fail.")

            _ensure_mps_env(args.mps_pipe_dir, args.mps_log_dir)
            
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            if not args.no_manage_mps:
                _stop_mps(sudo_password)
                _start_mps(sudo_password)

            # Sanity check
            ok, msg = _mps_client_sanity_check(mps_pct=int(round(p_list[0])))
            if not ok:
                raise SystemExit(f"MPS sanity check failed: {msg}")

            # Predictor init
            predictor = PctoDLPredictor(args.model, mem_freq=int(args.mem_freq), platform=args.platform, dataset=args.dataset)
            _lock_clocks(f_init, float(args.mem_freq), sudo_password)

            mps_cluster = pctodl_main.MPSCluster(
                args.model, [int(x) for x in p_list], 
                dataset_name=args.dataset, 
                repo_root=str(_REPO_ROOT),
                warmup_seconds=float(args.warmup_seconds),
                measure_seconds=float(args.measure_seconds),
                oom_guard=True
            )
            mps_cluster.start()
        else:
            predictor = PctoDLPredictor(args.model, mem_freq=int(args.mem_freq), platform=args.platform, dataset=args.dataset)

        # Main Loop
        for rep in range(int(args.repeat)):
            for method in methods:
                f_curr = float(f_init)
                t0 = time.perf_counter()
                
                # Controller Initialization
                step_ctrl = StepwiseController(step_mhz=float(args.step_mhz), supported=sm_clocks)
                pid_ctrl = scheduler.PID(caps[0], float(args.pid_alpha), float(args.pid_kp), float(args.pid_ki), float(args.pid_kd), sm_clocks, io.StringIO())
                thermo = ThermodynamicModelController(caps[0], predictor, float(args.thermo_eta), max_delta_f=float(args.thermo_max_delta_f), f_min=f_min, f_max=f_max)
                thermo.reset()

                sim = None
                if args.mode == "simulate":
                    sim = SimulatedPlant(predictor, p_list, b_list, float(args.mem_freq), seed=int(args.seed)+rep)
                    sim.reset(f_curr)
                elif args.mode == "gpu":
                    _lock_clocks(f_curr, float(args.mem_freq), sudo_password)
                    time.sleep(0.2)

                last_cap = None
                for i in range(total_iters):
                    p_cap = _caps_for_iter(caps, int(args.iters_per_phase), i)
                    
                    if last_cap is None or abs(p_cap - last_cap) > 1e-3:
                        pid_ctrl.power_cap = float(p_cap)
                        thermo.p_cap = float(p_cap)
                        # [FIX] Important reset for Thermo-MPC
                        thermo.reset() 
                        last_cap = float(p_cap)

                    if args.mode == "gpu":
                        stop_evt = threading.Event()
                        sampler_fut = monitor_executor.submit(doProfile.sample_gpu_info, stop_evt, handle, float(args.warmup_seconds))
                        
                        mps_cluster.submit_all([int(x) for x in b_list])
                        _, throughputs, _ = mps_cluster.wait_all_results()
                        
                        stop_evt.set() # Stop profiler
                        res = sampler_fut.result()
                        
                        meas_power = float(res[1]) # avg_power
                        max_power = float(res[0])
                        meas_tput = float(sum(throughputs))
                        
                        sm_util = float(res[3]) # avg_sm_util
                        mem_util = float(res[9]) # avg_mem_bw_util
                    else:
                        meas_power, max_power, meas_tput = sim.step(f_curr, float(args.measure_seconds))
                        sm_util, mem_util = 0.0, 0.0

                    # Control logic
                    p_ctrl = max_power if args.control_metric == "max" else meas_power
                    
                    # Log
                    with open(out_path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow({
                            "time_ms": int((time.perf_counter()-t0)*1000), "iter": i, "method": method,
                            "model": args.model, "p_list": json.dumps(p_list), "b_list": json.dumps(b_list),
                            "mem_freq": args.mem_freq, "sm_freq": f_curr, "sm_util": sm_util, "mem_util": mem_util,
                            "p_cap": p_cap, "meas_power": meas_power, "max_power": max_power, "meas_throughput": meas_tput
                        })

                    if args.verbose:
                        print(f"[{method}] iter={i+1}/{total_iters} p_cap={p_cap:.0f}W p_meas={meas_power:.1f}W f={f_curr:.0f}MHz")

                    # Step
                    if method == "Stepwise":
                        f_next = step_ctrl.step(f_curr, p_ctrl, p_cap)
                    elif method == "Standard PID":
                        f_next = pid_ctrl.scheduler(p_ctrl, f_curr)
                    else:
                        # Thermo-MPC
                        state = thermo.step(f_curr, p_ctrl, float(args.control_dt), p_list=p_list, b_list=b_list)
                        f_next_raw = state.f_next
                        f_next = _nearest_supported(f_next_raw, sm_clocks)
                        
                        if str(args.no_freq_guard).lower() != "true" and p_ctrl > p_cap:
                            if f_next > f_curr: f_next = _step_down_one(f_curr, sm_clocks)
                            elif abs(f_next - f_curr) < 1.0 and f_next > f_min: f_next = _step_down_one(f_next, sm_clocks)
                        f_curr = float(f_next) # State update is done in step()
                    
                    if args.mode == "gpu":
                        _lock_clocks(f_next, float(args.mem_freq), sudo_password)
                    f_curr = float(f_next)

    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, exiting...")
    except Exception as e:
        print(f"\nCaught exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # [CRITICAL FIX] Cleanup order to prevent hangs
        print("\n[Cleanup] 1. Shutting down Profiler executor...")
        monitor_executor.shutdown(wait=False, cancel_futures=True)

        if args.mode == "gpu" and not args.no_manage_mps and sudo_password:
            print("[Cleanup] 2. Force-stopping MPS Daemon (to unblock workers)...")
            # Stopping the daemon first kills the backend, forcing workers to error out and exit
            _stop_mps(sudo_password)
        
        if mps_cluster is not None:
            print("[Cleanup] 3. Stopping MPS Cluster python objects...")
            try:
                mps_cluster.stop()
            except Exception as e:
                print(f"Warning: Error during mps_cluster.stop(): {e}")

        if args.mode == "gpu" and not args.no_reset_clocks and sudo_password:
            print("[Cleanup] 4. Resetting GPU clocks...")
            _reset_clocks(sudo_password)
        
        print("[Cleanup] Done.")

    return 0

if __name__ == "__main__":
    sys.exit(main())