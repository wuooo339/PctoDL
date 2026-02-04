#!/usr/bin/env python3
"""
Experiment 3: Shadow Optimizer Robustness & Ablation (Grey-box vs Black-box)

Per experiments/experiment_spec.md:
  - Add Gaussian perturbation (noise_level=sigma) to the GBR predictor outputs to emulate an inaccurate prior.
  - Start from a "noisy-greedy" cold start.
  - Compare:
      * Grey-box BO: GBR prior + GP residuals
      * Black-box BO: Prior=0, GP learns the full function
  - Log to experiments/shadow_ablation.csv

Quick start:
  # GPU Mode: Power cap 250W, noise levels 0.5, 0.25, 0
  python experiments/exp_shadow_optimizer_ablation.py --p-cap 250 --noise-levels 0.5,0.25,0

  # Simulation Mode (No GPU required)
  python experiments/exp_shadow_optimizer_ablation.py --mode simulate --p-cap 250 --noise-levels 0.5,0.25,0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# src/profile_mps uses multiprocessing with CUDA workers; on Linux we must use spawn
try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from greedy_partitioning import GreedyPartitioner  # noqa: E402
from predict import MODEL_ALIASES, PctoDLPredictor  # noqa: E402
from src.shadow_optimizer import ShadowConfig, ShadowOptimizer  # noqa: E402


DEFAULT_SUDO_PASSWORD = "wzk123456"
# Ensure downstream helpers that read env (e.g., profile_mps) see a valid default.
os.environ.setdefault("SUDO_PASSWORD", DEFAULT_SUDO_PASSWORD)


def _load_platform_clocks(platform_key: Optional[str]) -> Tuple[List[float], List[float]]:
    """
    Load SM + MEM supported clocks from src/config/platform.json.
    platform_key: e.g., "3080ti", "a100".
    """
    cfg = _REPO_ROOT / "src" / "config" / "platform.json"
    with open(cfg, "r", encoding="utf-8") as f:
        data = json.load(f)
    platforms = data.get("platforms", [])

    def norm(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    key = norm(platform_key) if platform_key else ""
    chosen = None
    if key:
        for p in platforms:
            name = norm(p.get("name", ""))
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

    sm_freqs = [float(x) for x in sm]
    mem_freqs = [float(x) for x in (mem or [])]

    # Keep experiment space consistent with other scripts: use 5001/9501 on 3080Ti.
    if platform_key and "3080" in str(platform_key).lower():
        mem_freqs = [5001.0, 9501.0]

    return sm_freqs, mem_freqs


def _auto_detect_platform() -> Optional[str]:
    """
    Best-effort platform auto-detection using `nvidia-smi -L`.
    Returns a normalized key compatible with src/config/platform.json lookups.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        normalized = "".join(ch for ch in line.lower() if ch.isalnum())
        if "3080ti" in normalized:
            return "3080ti"
        if "a100" in normalized:
            return "a100"
    except Exception:
        return None
    return None


def _default_mem_freq(platform_key: str) -> float:
    k = (platform_key or "").lower()
    if "a100" in k:
        return 1215.0
    # Default to the common 3080Ti "efficiency" state (matches README).
    return 5001.0


def _batch_pool() -> List[int]:
    # Match src/main.py's shadow optimizer pool (dense ladder for local search).
    pool = [1, 2]
    pool.extend(range(4, 257, 4))
    return sorted(set(int(b) for b in pool if int(b) > 0))


def _snap_to_supported_clock(target_mhz: float, supported: Sequence[float]) -> float:
    if target_mhz is None:
        return 0.0
    if not supported:
        return float(target_mhz)
    return float(min((float(x) for x in supported), key=lambda c: abs(c - float(target_mhz))))


def _parse_noise_levels(s: str) -> List[float]:
    out: List[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        out = [0.5, 0.25, 0.0]
    return out


class PerturbedPredictor:
    """
    Deterministic multiplicative Gaussian perturbation on (throughput, power).

    For a given configuration x, draw eps(x) ~ N(0, sigma) deterministically using
    a stable hash of (seed, model, x). This emulates a fixed-but-wrong offline model.
    """

    def __init__(self, base: PctoDLPredictor, *, sigma: float, seed: int) -> None:
        self._base = base
        self.sigma = float(sigma)
        self.seed = int(seed)

    def __getattr__(self, name: str):
        # Delegate to underlying predictor for all other attributes.
        return getattr(self._base, name)

    def _hash_seed(self, *, p_list: Sequence[float], b_list: Sequence[int], sm_freq: float, mem_freq: float) -> int:
        h = hashlib.sha256()
        h.update(str(self.seed).encode("utf-8"))
        h.update(b"|")
        h.update(str(getattr(self._base, "actual_name", getattr(self._base, "model_name", ""))).encode("utf-8"))
        h.update(b"|p:")
        h.update(",".join(f"{float(x):g}" for x in p_list).encode("utf-8"))
        h.update(b"|b:")
        h.update(",".join(str(int(x)) for x in b_list).encode("utf-8"))
        h.update(b"|sm:")
        h.update(str(int(round(float(sm_freq)))).encode("utf-8"))
        h.update(b"|mem:")
        h.update(str(int(round(float(mem_freq)))).encode("utf-8"))
        # Use 64-bit seed for NumPy RNG.
        return int.from_bytes(h.digest()[:8], byteorder="little", signed=False)

    def predict_all(self, p_list, b_list, f, mem_freq=None):
        sm_u, mem_u, t, p = self._base.predict_all(p_list, b_list, f, mem_freq=mem_freq)
        if self.sigma <= 0.0:
            return sm_u, mem_u, t, p

        mem_f = float(mem_freq) if mem_freq is not None else float(getattr(self._base, "mem_freq", 0.0) or 0.0)
        seed64 = self._hash_seed(p_list=p_list, b_list=b_list, sm_freq=float(f), mem_freq=mem_f)
        rng = np.random.default_rng(seed64)
        eps_t, eps_p = rng.normal(0.0, self.sigma, size=2)
        t_noisy = max(0.0, float(t) * (1.0 + float(eps_t)))
        p_noisy = max(0.0, float(p) * (1.0 + float(eps_p)))
        return sm_u, mem_u, t_noisy, p_noisy


def _ensure_csv_header(path: Path, columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def _as_json_config(p_list: Sequence[float], b_list: Sequence[int], sm_freq: float, mem_freq: float) -> str:
    return json.dumps(
        {
            "p_list": [float(x) for x in p_list],
            "b_list": [int(x) for x in b_list],
            "sm_freq": float(sm_freq),
            "mem_freq": float(mem_freq),
        },
        separators=(",", ":"),
    )


def _measure_online(
    *,
    model_name: str,
    dataset_name: str,
    p_list: Sequence[float],
    b_list: Sequence[int],
    sm_freq: float,
    mem_freq: float,
    supported_sm_freqs: Sequence[float],
    supported_mem_freqs: Sequence[float],
    warmup_seconds: float,
    measure_seconds: float,
    sudo_password: Optional[str],
) -> Tuple[float, float, float]:
    from src import profile_mps  # lazy import: NVML/MPS deps

    # profile_mps locks clocks via `nvidia-smi -lgc/-lmc`, which requires supported states.
    sm_freq = _snap_to_supported_clock(float(sm_freq), supported=supported_sm_freqs)
    mem_freq = _snap_to_supported_clock(float(mem_freq), supported=supported_mem_freqs)

    result = profile_mps.run_single_test(
        model_name,
        list(p_list),
        list(b_list),
        int(round(float(sm_freq))) if sm_freq is not None else None,
        mem_freq=int(round(float(mem_freq))) if mem_freq is not None else None,
        measure_seconds=float(measure_seconds),
        warmup_seconds=float(warmup_seconds),
        sudo_password=sudo_password,
        dataset_name=dataset_name,
        repo_root=_REPO_ROOT,
        save_output=False,
    )
    if not result:
        raise RuntimeError("profile_mps.run_single_test returned None")
    return float(result["throughput_total"]), float(result["power_avg"]), float(result["power_max"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Exp3: Shadow optimizer robustness & ablation (GBR-BO vs BO)")
    ap.add_argument("--model", type=str, default="mobilenet_v2")
    ap.add_argument("--dataset", type=str, default="imagenet")
    ap.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Platform tag for model selection (e.g., 3080ti, a100). If omitted, auto-detect via nvidia-smi.",
    )
    ap.add_argument("--p-cap", type=float, default=300.0, help="Power cap (W)")
    ap.add_argument("--mem-freq", type=float, default=None, help="Fix memory frequency (MHz) for greedy + runtime")
    ap.add_argument("--mps-step", type=int, default=5)
    ap.add_argument("--noise-levels", type=str, default="0.1,0.2,0.3", help="Comma-separated sigma list")
    ap.add_argument("--methods", type=str, default="grey-box,black-box", help="Comma-separated: grey-box,black-box")
    ap.add_argument("--rounds", type=int, default=25, help="Shadow optimizer iterations to log (includes iteration 0)")
    ap.add_argument(
        "--shadow-interval",
        type=int,
        default=3,
        help="System tests per shadow iteration (total tests = rounds*interval)",
    )
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--warmup-seconds", type=float, default=6.0)
    ap.add_argument("--measure-seconds", type=float, default=8.0)
    ap.add_argument("--output", type=Path, default=Path("experiments/shadow_ablation.csv"))

    ap.add_argument("--no-sudo", action="store_true", help="Do not lock clocks (online mode)")
    ap.add_argument(
        "--sudo-password",
        type=str,
        default=DEFAULT_SUDO_PASSWORD,
        help="Password used for sudo (clock lock + MPS scripts). Ignored with --no-sudo.",
    )

    ap.add_argument("--trust-radius", type=int, default=3)
    ap.add_argument("--history-window", type=int, default=200)
    ap.add_argument("--gp-refit-interval", type=int, default=3)
    ap.add_argument("--gp-max-points", type=int, default=200)
    ap.add_argument("--gp-min-points", type=int, default=None, help="Override GP min points (default: grey=5, black=1)")
    ap.add_argument("--conservative", action="store_true", help="Only accept candidates that clear the gain threshold")
    ap.add_argument("--gain-threshold", type=float, default=0.0, help="Relative gain threshold for conservative mode")

    ap.add_argument("--allow-sm-adjust", action="store_true", help="Allow searching neighboring SM frequency states")
    ap.add_argument("--allow-mem-adjust", action="store_true", help="Allow searching neighboring MEM frequency states")
    ap.add_argument("--max-tasks", type=int, default=None, help="Limit number of co-located tasks (greedy + shadow)")
    ap.add_argument("--max-mps-per-task", type=float, default=None, help="Upper bound for a single task's MPS share")
    ap.add_argument("--max-batch-size", type=int, default=None, help="Upper bound for total batch size (heuristic)")

    args = ap.parse_args()

    model_name = str(args.model)
    # Match predict.py naming with profile_mps naming when aliases exist.
    profile_model_name = MODEL_ALIASES.get(model_name, model_name)
    dataset = str(args.dataset)
    platform = args.platform or _auto_detect_platform() or "3080ti"
    platform = str(platform)
    p_cap = float(args.p_cap)
    mps_step = int(args.mps_step)
    rounds = int(args.rounds)
    shadow_interval = max(1, int(args.shadow_interval))

    if rounds <= 0:
        raise SystemExit("--rounds must be > 0")

    mem_freq = float(args.mem_freq) if args.mem_freq is not None else _default_mem_freq(platform)

    # Cache platform clock lists (and use them to snap greedy/online outputs).
    sm_freqs, mem_freqs = _load_platform_clocks(platform)
    mem_freq = _snap_to_supported_clock(mem_freq, supported=mem_freqs) if mem_freqs else float(mem_freq)

    noise_levels = _parse_noise_levels(args.noise_levels)
    methods = [m.strip().lower() for m in (args.methods or "").split(",") if m.strip()]
    if not methods:
        methods = ["grey-box", "black-box"]

    # Output schema (matches experiments/experiment_spec.md)
    cols = [
        "iteration",
        "method",
        "noise_level",
        "meas_throughput",
        "meas_power",
        "p_violation",
        "residual_t",
        "residual_p",
        "current_config",
    ]
    _ensure_csv_header(args.output, cols)

    sudo_password: Optional[str] = None
    if not args.no_sudo:
        sudo_password = str(args.sudo_password) if args.sudo_password else None
    if sudo_password:
        os.environ["SUDO_PASSWORD"] = sudo_password

    from src import profile_mps

    # Set per-user MPS dirs (same logic as src/profile_mps.py main()).
    try:
        import pwd

        actual_user = os.environ.get("SUDO_USER", os.environ.get("USER", "wzk"))
        uid = pwd.getpwnam(actual_user).pw_uid
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = f"/tmp/nvidia-mps-{uid}"
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = f"/tmp/nvidia-log-{uid}"
    except Exception:
        pass

    profile_mps._best_effort_start_mps(sudo_password=sudo_password, repo_root=_REPO_ROOT)

    try:
        with open(args.output, "a", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=cols)

            # Run heavy noise -> no noise (as recommended by the spec)
            for sigma in noise_levels:
                # Cold start via greedy partitioning using a perturbed predictor.
                greedy = GreedyPartitioner(
                    model_name,
                    p_cap,
                    mps_step=mps_step,
                    mem_freq=int(round(mem_freq)),
                    platform=platform,
                    dataset=dataset,
                    max_tasks=args.max_tasks,
                    max_mps_per_task=args.max_mps_per_task,
                    max_batchsize=args.max_batch_size,
                )
                true_predictor = greedy.predictor
                noisy_predictor = PerturbedPredictor(true_predictor, sigma=float(sigma), seed=int(args.seed))
                greedy.predictor = noisy_predictor
                greedy._predictors[float(int(round(mem_freq)))] = noisy_predictor

                tasks, init_sm, init_mem = greedy.run()
                if not tasks:
                    raise RuntimeError("GreedyPartitioner returned empty task list")
                p_list0 = [float(t.mps) for t in tasks]
                b_list0 = [int(t.batch) for t in tasks]
                init_sm = _snap_to_supported_clock(float(init_sm), supported=sm_freqs)
                init_mem = _snap_to_supported_clock(float(init_mem), supported=mem_freqs) if mem_freqs else float(init_mem)

                for method in methods:
                    method_tag = "Grey-box" if method in {"grey", "grey-box", "gbr-bo", "pctodl"} else "Black-box"

                    use_prior = method_tag == "Grey-box"
                    predictor = noisy_predictor if use_prior else None

                    gp_min = int(args.gp_min_points) if args.gp_min_points is not None else (5 if use_prior else 1)

                    opt = ShadowOptimizer(
                        p_cap=p_cap,
                        predictor=predictor,
                        sm_freqs=sm_freqs,
                        mem_freqs=mem_freqs,
                        batch_pool=_batch_pool(),
                        use_prior=use_prior,
                        max_batch_size=args.max_batch_size,
                        max_mps_per_task=args.max_mps_per_task,
                        mps_step=mps_step,
                        trust_radius=int(args.trust_radius),
                        history_window=int(args.history_window),
                        gp_refit_interval=int(args.gp_refit_interval),
                        gp_max_points=int(args.gp_max_points),
                        gp_min_points=gp_min,
                        reconfig_threshold=float(args.gain_threshold),
                        allow_sm_adjustment=bool(args.allow_sm_adjust),
                        allow_mem_adjustment=bool(args.allow_mem_adjust),
                    )
                    if opt.gp_t is None or opt.gp_p is None:
                        raise SystemExit(
                            "Exp3 requires scikit-learn (GaussianProcessRegressor). "
                            "Install it (e.g., `pip install scikit-learn`) and rerun."
                        )

                    # Current applied configuration for the next block.
                    cur_p, cur_b, cur_sm, cur_mem = list(p_list0), list(b_list0), init_sm, init_mem

                    for it in range(rounds):
                        t_samples: List[float] = []
                        p_avg_samples: List[float] = []
                        p_max_samples: List[float] = []

                        for _ in range(shadow_interval):
                            t, p_avg, p_max = _measure_online(
                                model_name=profile_model_name,
                                dataset_name=dataset,
                                p_list=cur_p,
                                b_list=cur_b,
                                sm_freq=cur_sm,
                                mem_freq=cur_mem,
                                supported_sm_freqs=sm_freqs,
                                supported_mem_freqs=mem_freqs,
                                warmup_seconds=float(args.warmup_seconds),
                                measure_seconds=float(args.measure_seconds),
                                sudo_password=sudo_password,
                            )

                            t_samples.append(float(t))
                            p_avg_samples.append(float(p_avg))
                            p_max_samples.append(float(p_max))

                            opt.update(
                                ShadowConfig(
                                    p_list=list(cur_p),
                                    b_list=list(cur_b),
                                    sm_freq=float(cur_sm),
                                    mem_freq=float(cur_mem),
                                    throughput=float(t),
                                    power=float(p_max),  # safety: use max power for constraint learning
                                )
                            )

                        meas_t = float(np.mean(t_samples)) if t_samples else 0.0
                        meas_p_avg = float(np.mean(p_avg_samples)) if p_avg_samples else 0.0
                        meas_p_max = float(max(p_max_samples)) if p_max_samples else 0.0

                        res_t, res_p = opt.residual_means()
                        row = {
                            "iteration": int(it),
                            "method": method_tag,
                            "noise_level": float(sigma),
                            "meas_throughput": meas_t,
                            "meas_power": meas_p_avg,
                            "p_violation": int(meas_p_max > p_cap + 1e-9),
                            "residual_t": float(res_t),
                            "residual_p": float(res_p),
                            "current_config": _as_json_config(cur_p, cur_b, cur_sm, cur_mem),
                        }
                        writer.writerow(row)
                        f_out.flush()

                        # Compute next config for the next iteration.
                        if it >= rounds - 1:
                            continue

                        cfg_for_suggest = ShadowConfig(
                            p_list=list(cur_p),
                            b_list=list(cur_b),
                            sm_freq=float(cur_sm),
                            mem_freq=float(cur_mem),
                            throughput=float(meas_t),
                            power=float(meas_p_max),
                        )
                        best_t = opt.best_feasible_throughput(fallback_current=meas_t)
                        cand = opt.suggest(
                            cfg_for_suggest,
                            best_throughput=best_t,
                            conservative=bool(args.conservative),
                        )
                        if cand is not None:
                            cur_p, cur_b, cur_sm, cur_mem = list(cand.p_list), list(cand.b_list), float(cand.sm_freq), float(cand.mem_freq)

    finally:
        try:
            profile_mps._best_effort_shutdown_mps(sudo_password=sudo_password, repo_root=_REPO_ROOT)
        except Exception:
            pass

    print(f"Wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
