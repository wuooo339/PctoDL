#!/usr/bin/env python3
"""
Experiment 1: Prediction Accuracy Validation

Generates (predicted vs measured) pairs for throughput and power under multi-tenant MPS.

Two modes:
  1) --offline: use existing profiling CSVs in ./mps_profile as "measured" points
  2) (default): run on hardware via src/profile_mps.run_single_test (requires GPU + MPS + sudo for clock lock)

Output: experiments/accuracy_results.csv (append-safe; can resume).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
# When invoked as `python experiments/exp_accuracy_validation.py`, Python's
# sys.path[0] is `experiments/`, so we must add repo root to import `predict.py`.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# src/profile_mps uses multiprocessing with CUDA workers; on Linux we must use spawn
# (forked subprocesses cannot re-initialize CUDA).
try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    # Start method may already be fixed by the parent process; ignore.
    pass

from predict import PctoDLPredictor  # noqa: E402


def _repo_root() -> Path:
    return _REPO_ROOT


def _load_platform_clocks(platform_key: Optional[str]) -> Tuple[List[float], List[float]]:
    """
    Load SM + MEM supported clocks from src/config/platform.json.
    platform_key: e.g., "3080ti", "a100".
    """
    cfg = _repo_root() / "src" / "config" / "platform.json"
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
    # Paper experiment constraint: only use the two representative memory states for 3080Ti.
    # (Avoid intermediate states like 9251 that may not have trained model variants.)
    if platform_key and "3080" in str(platform_key).lower():
        mem_freqs = [5001.0, 9501.0]
    return sm_freqs, mem_freqs


def _random_partition_sum(total: int, parts: int, rng: random.Random) -> List[int]:
    """
    Sample a random composition of 'total' into 'parts' positive integers.
    """
    if parts <= 0:
        return []
    if parts == 1:
        return [total]
    # Choose (parts-1) cut points in [1, total-1]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    segs = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))] + [total - cuts[-1]]
    return segs


def _random_bounded_composition(
    total: int,
    parts: int,
    min_v: int,
    max_v: int,
    rng: random.Random,
) -> List[int]:
    """
    Sample a composition of `total` into `parts` integers, each in [min_v, max_v].

    This is used to enforce per-task MPS bounds (in units of `mps_step`).
    """
    if parts <= 0:
        return []
    if parts * min_v > total or parts * max_v < total:
        raise ValueError("No feasible bounded composition for given total/parts/min/max")

    xs = [min_v] * parts
    remaining = total - parts * min_v
    caps = [max_v - min_v] * parts
    while remaining > 0:
        candidates = [i for i, cap in enumerate(caps) if cap > 0]
        if not candidates:
            break
        i = rng.choice(candidates)
        xs[i] += 1
        caps[i] -= 1
        remaining -= 1
    rng.shuffle(xs)
    return xs


def _model_max_batch_limit(model_name: str) -> Optional[int]:
    # Keep consistent with greedy_partitioning.MODEL_MAX_BATCH_LIMITS
    model_key = model_name.lower().replace("-", "_").replace(".", "")
    if "vgg19" in model_key:
        return 160
    return None


def _valid_batches_for_mps(mps: float, n_tasks: int, model_name: str, max_batchsize: Optional[int] = None) -> List[int]:
    # Match greedy_partitioning.get_batches_for_mps (coarse but safe)
    all_batches = [4, 8, 16, 32, 64, 96, 128, 192, 256]
    min_bound = max(4, int(float(mps) * 0.8))
    max_bound = int(float(mps) * 6.4)

    model_limit = _model_max_batch_limit(model_name)
    if model_limit is not None:
        max_bound = min(max_bound, model_limit)
    if max_batchsize is not None:
        per_task_cap = max(4, int(int(max_batchsize) / max(1, int(n_tasks))))
        max_bound = min(max_bound, per_task_cap)

    valid = [b for b in all_batches if min_bound <= b <= max_bound]
    if valid:
        return valid
    return [min(all_batches, key=lambda x: abs(x - min_bound))]


def _sample_config(
    rng: random.Random,
    model: str,
    n_tasks_range: Tuple[int, int],
    mps_step: int,
    sm_freqs: Sequence[float],
    mem_freqs: Sequence[float],
) -> Tuple[List[float], List[int], float, float]:
    min_tasks, max_tasks = n_tasks_range
    if mps_step <= 0:
        raise ValueError("mps_step must be > 0")

    # Enforce per-task MPS range: 5% ~ 40% (in units of mps_step).
    # With step=5, this becomes 1~8 chunks.
    if 5 % mps_step != 0 or 40 % mps_step != 0:
        raise ValueError("mps_step must divide both 5 and 40 to enforce 5~40% bound")
    min_chunk = int(5 // mps_step)
    max_chunk = int(40 // mps_step)
    total_chunks = int(100 // mps_step)

    feasible_tasks = [
        n
        for n in range(int(min_tasks), int(max_tasks) + 1)
        if n * min_chunk <= total_chunks <= n * max_chunk
    ]
    if not feasible_tasks:
        raise ValueError("No feasible num_tasks for 5~40% MPS bound; increase --min-tasks/--max-tasks")
    n_tasks = int(rng.choice(feasible_tasks))

    chunks = _random_bounded_composition(total_chunks, n_tasks, min_chunk, max_chunk, rng)
    p_list = [float(c * mps_step) for c in chunks]

    # Choose per-task batch using the same heuristic bounds used by greedy partitioning.
    b_list: List[int] = []
    for p in p_list:
        cand = _valid_batches_for_mps(p, n_tasks=n_tasks, model_name=model)
        b_list.append(int(rng.choice(cand)))

    sm_freq = float(rng.choice(list(sm_freqs))) if sm_freqs else 0.0
    mem_freq = float(rng.choice(list(mem_freqs))) if mem_freqs else 0.0
    return p_list, b_list, sm_freq, mem_freq


def _discover_trained_mem_freqs(model_name: str) -> List[int]:
    """
    Infer available mem_freq variants from filenames under fitting_results/<model>/.
    We look for the canonical suffix pattern `_memXXXX` used by optimize_model_params.py outputs.
    """
    model_dir = _repo_root() / "fitting_results" / model_name
    if not model_dir.exists():
        return []
    freqs: set[int] = set()
    for p in model_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        # Fast parse without regex imports.
        idx = name.rfind("_mem")
        if idx == -1:
            continue
        tail = name[idx + 4 :]
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            freqs.add(int(digits))
    return sorted(freqs)


def _ensure_csv_header(path: Path, columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return

    # If the file exists but lacks newly-added columns, upgrade it in-place (preserving data).
    try:
        existing = pd.read_csv(path)
    except Exception:
        return
    missing = [c for c in columns if c not in existing.columns]
    if not missing:
        return
    for c in missing:
        existing[c] = np.nan
    bak = path.with_suffix(path.suffix + ".bak")
    try:
        if not bak.exists():
            path.replace(bak)
        else:
            return
    except Exception:
        return
    existing.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Exp1: Predictor accuracy validation (pred vs meas)")
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help='Model name. If omitted or set to "all", run all models (from fitting_results/) and split samples across them.',
    )
    ap.add_argument("--dataset", type=str, default="imagenet", help="Predictor dataset tag (e.g., imagenet)")
    ap.add_argument("--platform", type=str, default="3080ti", help="Predictor platform tag (e.g., 3080ti, a100)")
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help="Total number of configurations to evaluate. If --model is set, runs that model for N. "
        "If --model is omitted or 'all', N is split across models. Defaults: all=140 (20/model), single=20.",
    )
    ap.add_argument("--seed", type=int, default=1)
    # With per-task MPS bound 5~40%, at least 3 tasks are needed to sum to 100%.
    ap.add_argument("--min-tasks", type=int, default=3)
    ap.add_argument("--max-tasks", type=int, default=8)
    ap.add_argument("--mps-step", type=int, default=5)
    ap.add_argument("--warmup-seconds", type=float, default=6.0)
    ap.add_argument("--measure-seconds", type=float, default=8.0)
    ap.add_argument("--output", type=Path, default=Path("experiments/accuracy_results.csv"))
    ap.add_argument("--verbose", action="store_true", help="Print per-iteration predict/measure progress")

    ap.add_argument("--offline", action="store_true", help="Use ./mps_profile CSVs as measurement source")
    ap.add_argument("--offline-csv", type=Path, default=None, help="Override profiling CSV for --offline")
    ap.add_argument("--no-sudo", action="store_true", help="Do not lock clocks (online mode)")
    ap.add_argument(
        "--sudo-password",
        type=str,
        default="wzk123456",
        help="Password used for sudo (clock lock + MPS scripts). Ignored with --no-sudo.",
    )
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed))

    sm_freqs, mem_freqs = _load_platform_clocks(args.platform)
    if not sm_freqs:
        raise SystemExit("No SM frequency list found in src/config/platform.json")
    if not mem_freqs:
        # Still allow: predictor supports mem_freq=None, but spec wants mem_freq recorded.
        mem_freqs = [0.0]

    # Output schema (matches experiments/experiment_spec.md)
    cols = [
        "iter",
        "model",
        "p_list",
        "b_list",
        "mem_freq",
        "sm_freq",
        "sm_util",
        "mem_util",
        "pred_throughput",
        "meas_throughput",
        "pred_power",
        "meas_power",
        "meas_power_max",
        "error_t",
        "error_p",
    ]
    _ensure_csv_header(args.output, cols)

    def discover_models() -> List[str]:
        # Prefer fitting_results/ as the source of truth (matches what's actually trained).
        fr = _repo_root() / "fitting_results"
        if fr.exists():
            models = sorted([p.name for p in fr.iterdir() if p.is_dir() and not p.name.startswith(".")])
            if models:
                return models
        # Fallback to the typical 7-model profile set.
        return ["mnasnet", "densenet201", "efficientnet_v2_m", "maxvit_t", "mobilenet_v2", "resnet50", "vgg19"]

    requested = (args.model or "all").strip().lower()
    if requested == "all":
        models_to_run = discover_models()
    else:
        models_to_run = [args.model]

    if args.n is None:
        # User requested defaults:
        # - all models: 20 per model (7 models => 140)
        # - single model: 20
        args.n = 20 * len(models_to_run) if requested == "all" else 20

    total_target = int(args.n)
    if total_target <= 0:
        raise SystemExit("--n must be > 0")

    # Split total_target across models (balanced, deterministic order).
    base = total_target // len(models_to_run)
    extra = total_target % len(models_to_run)
    per_model_counts: Dict[str, int] = {}
    for i, m in enumerate(models_to_run):
        per_model_counts[m] = base + (1 if i < extra else 0)

    # For predictor fidelity, only sample mem_freq values that have trained model variants.
    # Otherwise predict.py falls back to simplistic priors (e.g., U_sm=sum(p)*0.8 => 80%).
    platform_mem_freqs = [int(x) for x in mem_freqs if float(x) > 0]
    per_model_mem_freqs: Dict[str, List[float]] = {}
    for m in models_to_run:
        trained = _discover_trained_mem_freqs(m)
        if trained:
            inter = [f for f in trained if f in platform_mem_freqs] if platform_mem_freqs else trained
            if inter:
                per_model_mem_freqs[m] = [float(x) for x in inter]
                continue
        # Fallback: use platform clocks (may trigger predictor fallback if unsupported).
        per_model_mem_freqs[m] = [float(x) for x in mem_freqs]

    # Cache predictors by (model, memory frequency) to avoid re-loading per iteration.
    predictor_cache: Dict[Tuple[str, int], PctoDLPredictor] = {}

    def get_predictor(model_name: str, mem_freq: float) -> PctoDLPredictor:
        key = int(mem_freq) if mem_freq else 0
        cache_key = (str(model_name).lower(), key)
        pred = predictor_cache.get(cache_key)
        if pred is None:
            pred = PctoDLPredictor(
                model_name,
                mem_freq=int(mem_freq) if mem_freq else None,
                platform=args.platform,
                dataset=args.dataset,
            )
            predictor_cache[cache_key] = pred
        return pred

    if args.offline:
        # Offline measurement source: mps_profile/mps_results*.csv (same format used by predict.py --test)
        if args.offline_csv is not None:
            csvs = [args.offline_csv]
        else:
            csvs = sorted((_repo_root() / "mps_profile").glob("mps_results*.csv"))
        if not csvs:
            raise SystemExit("No profiling CSV found under ./mps_profile. Provide --offline-csv.")

        frames = []
        for p in csvs:
            df = pd.read_csv(p)
            frames.append(df)
        df_all = pd.concat(frames, ignore_index=True)
        if df_all.empty:
            raise SystemExit("No rows found in profiling CSV(s).")

        if "model_name" not in df_all.columns:
            raise SystemExit("Offline CSV(s) must contain column 'model_name'.")

        # Build per-model samples and append.
        out_rows = []
        global_iter = 0
        for model_name in models_to_run:
            n_take = int(per_model_counts.get(model_name, 0))
            if n_take <= 0:
                continue

            name_set = {str(model_name).lower()}
            # Apply known aliases (e.g., maxvit -> maxvit_t)
            try:
                from predict import MODEL_ALIASES

                alias = MODEL_ALIASES.get(str(model_name).lower())
                if alias:
                    name_set.add(str(alias).lower())
            except Exception:
                pass

            df_m = df_all[df_all["model_name"].astype(str).str.lower().isin(name_set)]
            if df_m.empty:
                raise SystemExit(f"Offline CSV(s) have no rows for model={model_name}")

            # Sample with replacement if needed.
            replace = len(df_m) < n_take
            df_m = df_m.sample(n=n_take, replace=replace, random_state=int(args.seed))

            for row in df_m.itertuples(index=False):
                global_iter += 1
                p_list = json.loads(getattr(row, "p_list")) if isinstance(getattr(row, "p_list"), str) else getattr(row, "p_list")
                b_list = json.loads(getattr(row, "b_list")) if isinstance(getattr(row, "b_list"), str) else getattr(row, "b_list")
                if not isinstance(p_list, list):
                    p_list = [float(p_list)]
                if not isinstance(b_list, list):
                    b_list = [int(b_list)]
                sm_freq = float(getattr(row, "sm_freq"))
                mem_freq = float(getattr(row, "mem_freq")) if hasattr(row, "mem_freq") else 0.0
                meas_t = float(getattr(row, "throughput_total"))
                meas_p = float(getattr(row, "power_avg"))
                meas_p_max = float(getattr(row, "power_max")) if hasattr(row, "power_max") else float("nan")

                predictor = get_predictor(model_name, mem_freq)
                pred_sm, pred_mem, pred_t, pred_p = predictor.predict_all(p_list, b_list, sm_freq)

                err_t = float(abs(1.0 - (pred_t / meas_t))) if meas_t > 1e-9 else float("nan")
                err_p = float(abs(1.0 - (pred_p / meas_p))) if meas_p > 1e-9 else float("nan")
                out_rows.append(
                    {
                        "iter": global_iter,
                        "model": model_name,
                        "p_list": json.dumps(list(map(float, p_list))),
                        "b_list": json.dumps(list(map(int, b_list))),
                        "mem_freq": float(mem_freq),
                        "sm_freq": float(sm_freq),
                        "sm_util": float(pred_sm),
                        "mem_util": float(pred_mem),
                        "pred_throughput": float(pred_t),
                        "meas_throughput": float(meas_t),
                        "pred_power": float(pred_p),
                        "meas_power": float(meas_p),
                        "meas_power_max": float(meas_p_max),
                        "error_t": err_t,
                        "error_p": err_p,
                    }
                )

        pd.DataFrame(out_rows, columns=cols).to_csv(args.output, mode="a", header=False, index=False)
        print(f"Wrote {len(out_rows)} rows to {args.output}")
        return 0

    # Online measurement mode: call src/profile_mps.run_single_test.
    # Import lazily so offline mode doesn't require NVML/MPS deps at import time.
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

    # profile_mps dataset naming is typically "ImageNet"/"Caltech256".
    dataset_profile = args.dataset
    if isinstance(dataset_profile, str):
        d = dataset_profile.strip().lower()
        if d == "imagenet":
            dataset_profile = "ImageNet"
        elif d == "caltech256":
            dataset_profile = "Caltech256"

    sudo_password = None
    if not args.no_sudo:
        # Prefer explicit CLI password; also export it so profile_mps helpers that read env keep working.
        sudo_password = os.environ.get("SUDO_PASSWORD") or args.sudo_password or getattr(profile_mps, "DEFAULT_SUDO_PASSWORD", None)
        if sudo_password:
            os.environ["SUDO_PASSWORD"] = str(sudo_password)

    # Best-effort MPS lifecycle: keep it up for the whole experiment for speed.
    profile_mps._best_effort_start_mps(sudo_password=sudo_password, repo_root=_repo_root())
    try:
        out_rows = []
        target = int(total_target)
        n_written = 0
        retries_per_iter = 3

        global_iter = 0
        for model_name in models_to_run:
            n_take = int(per_model_counts.get(model_name, 0))
            if n_take <= 0:
                continue

            for _local in range(n_take):
                global_iter += 1
                ok = False
                # Define fallbacks for the "failed row" case.
                p_list = []
                b_list = []
                sm_freq = 0.0
                mem_freq = 0.0
                pred_sm = pred_mem = pred_t = pred_p = 0.0

                for _ in range(retries_per_iter):
                    p_list, b_list, sm_freq, mem_freq = _sample_config(
                        rng,
                        model=model_name,
                        n_tasks_range=(int(args.min_tasks), int(args.max_tasks)),
                        mps_step=int(args.mps_step),
                        sm_freqs=sm_freqs,
                        mem_freqs=per_model_mem_freqs.get(model_name, mem_freqs),
                    )
                    predictor = get_predictor(model_name, mem_freq)
                    pred_sm, pred_mem, pred_t, pred_p = predictor.predict_all(p_list, b_list, sm_freq)
                    if args.verbose:
                        print(
                            f"\n[iter {global_iter}/{target}] model={model_name} "
                            f"p_list={p_list} b_list={b_list} f_sm={int(sm_freq)} mem={int(mem_freq)}"
                        )
                        print(
                            f"  predict: throughput={pred_t:.2f} img/s, power={pred_p:.2f} W "
                            f"(U_sm={pred_sm:.1f}%, U_mem={pred_mem:.1f}%)"
                        )
                        print("  measure: starting run_single_test ...")

                    meas = profile_mps.run_single_test(
                        model_name=model_name,
                        p_list=p_list,
                        b_list=b_list,
                        f=int(sm_freq),
                        mem_freq=int(mem_freq) if mem_freq else None,
                        measure_seconds=float(args.measure_seconds),
                        warmup_seconds=float(args.warmup_seconds),
                        sudo_password=sudo_password,
                        dataset_name=dataset_profile,
                        repo_root=_repo_root(),
                        save_output=False,
                    )
                    if meas is None:
                        continue

                    meas_t = float(meas.get("throughput_total", 0.0))
                    meas_p = float(meas.get("power_avg", 0.0))
                    meas_p_max = float(meas.get("power_max", float("nan")))
                    if args.verbose:
                        print(
                            f"  measure: done throughput={meas_t:.2f} img/s, "
                            f"power_avg={meas_p:.2f} W, power_max={meas_p_max:.2f} W"
                        )
                    err_t = float(abs(1.0 - (pred_t / meas_t))) if meas_t > 1e-9 else float("nan")
                    err_p = float(abs(1.0 - (pred_p / meas_p))) if meas_p > 1e-9 else float("nan")

                    out_rows.append(
                        {
                            "iter": global_iter,
                            "model": model_name,
                            "p_list": json.dumps(list(map(float, p_list))),
                            "b_list": json.dumps(list(map(int, b_list))),
                            "mem_freq": float(mem_freq),
                            "sm_freq": float(sm_freq),
                            "sm_util": float(pred_sm),
                            "mem_util": float(pred_mem),
                            "pred_throughput": float(pred_t),
                            "meas_throughput": float(meas_t),
                            "pred_power": float(pred_p),
                            "meas_power": float(meas_p),
                            "meas_power_max": float(meas_p_max),
                            "error_t": err_t,
                            "error_p": err_p,
                        }
                    )
                    ok = True
                    break

                if not ok:
                    # Record a failed row (measurement missing) and continue.
                    out_rows.append(
                        {
                            "iter": global_iter,
                            "model": model_name,
                            "p_list": json.dumps(list(map(float, p_list))) if p_list else "[]",
                            "b_list": json.dumps(list(map(int, b_list))) if b_list else "[]",
                            "mem_freq": float(mem_freq),
                            "sm_freq": float(sm_freq),
                            "sm_util": float(pred_sm),
                            "mem_util": float(pred_mem),
                            "pred_throughput": float(pred_t),
                            "meas_throughput": float("nan"),
                            "pred_power": float(pred_p),
                            "meas_power": float("nan"),
                            "meas_power_max": float("nan"),
                            "error_t": float("nan"),
                            "error_p": float("nan"),
                        }
                    )

                # Flush periodically (makes long experiments resumable)
                if len(out_rows) >= 1:
                    pd.DataFrame(out_rows, columns=cols).to_csv(args.output, mode="a", header=False, index=False)
                    n_written += len(out_rows)
                    out_rows = []
                    print(f"[progress] wrote {n_written}/{target} rows to {args.output}")

        if out_rows:
            pd.DataFrame(out_rows, columns=cols).to_csv(args.output, mode="a", header=False, index=False)
            n_written += len(out_rows)
        print(f"Done. Wrote {n_written} rows to {args.output}")
        return 0
    finally:
        profile_mps._best_effort_shutdown_mps(sudo_password=sudo_password, repo_root=_repo_root())


if __name__ == "__main__":
    raise SystemExit(main())
