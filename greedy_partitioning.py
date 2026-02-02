#!/usr/bin/env python3
"""
Iterative Frequency-Aware Greedy Partitioning for GPU Resources.
Implements the Master Partition and FindBestNext algorithms using PctoDL models.

Updates:
1. MPS Step default = 5%
2. Adaptive Batch Size search based on MPS percentage.
3. Joint Frequency-Memory Sweep (Spec Section 4.2)
   - Memory Loop (Outer): Iterate through memory states m <= F_curr_mem
   - SM Loop (Inner): Sweep SM frequencies for each memory state

Usage:
    python greedy_partitioning.py --model mobilenet_v2 --p_cap 190 --mps_step 5
    python greedy_partitioning.py --model mobilenet_v2 --p_cap 150 --mem-freq 9501
"""

import os
# Set MKL threading layer to avoid conflicts with libgomp.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import argparse
import sys
import time
import numpy as np
import re
import subprocess
import json

# Import PctoDLPredictor from predict.py
# This follows the Resource Orchestration spec (Section 4)
try:
    from predict import PctoDLPredictor
except ImportError:
    print("Error: Could not import PctoDLPredictor from predict.py.")
    print("Please ensure predict.py is in the current directory.")
    sys.exit(1)

# --- Configuration Constants ---
MIN_FREQ = 500.0
MAX_FREQ = 2100.0
FREQ_STEP = 50.0  # Resolution for SM frequency sweeping

# Memory frequency states (VDDQ steps)
# Common GPU memory frequencies (MHz)
MIN_MEM_FREQ = 500.0
MAX_MEM_FREQ = 9501.0
MEM_FREQ_STEP = 500.0  # Resolution for memory frequency sweeping

# Platform-specific memory clock states (per system spec)
PLATFORM_MEM_FREQS = {
    "3080ti": [9501.0, 5001.0],
    "a100": [1215.0],
}

# Global pool of allowed batch sizes
ALL_CANDIDATE_BATCHES = [4, 8, 16, 32, 64, 96, 128, 192, 256]

# Model-specific batch size limits (for memory-intensive models)
MODEL_MAX_BATCH_LIMITS = {
    "vgg19": 160,
}


class TaskConfig:
    """Represents a single task allocation."""
    def __init__(self, mps, batch):
        self.mps = mps
        self.batch = batch

    def __repr__(self):
        return f"(MPS={self.mps}%, BS={self.batch})"


class GreedyPartitioner:
    def __init__(
        self,
        model_name,
        p_cap,
        mps_step=5,
        mem_freq=None,
        platform=None,
        dataset=None,
        max_tasks=None,
        max_mps_per_task=None,
        max_batchsize=None,
        vectorized=False,
    ):
        self.model_name = model_name
        self.p_cap = p_cap
        self.mps_step = mps_step
        self.fixed_mem_freq = mem_freq
        self.platform = platform or self._auto_detect_platform()
        self.max_tasks = int(max_tasks) if max_tasks is not None else None
        self.max_mps_per_task = float(max_mps_per_task) if max_mps_per_task is not None else None
        self.max_batchsize = int(max_batchsize) if max_batchsize is not None else None
        self.vectorized = vectorized  
        self._metric_cache = {}
        self._predictors = {}
        self._sm_freqs = self._load_platform_sm_freqs()

        # Load the predictive models (Resource Orchestration: Section 4)
        print(f"Loading models for {model_name}...")
        self.predictor = PctoDLPredictor(model_name, mem_freq=mem_freq,
                                         platform=self.platform, dataset=dataset)
        if mem_freq is not None:
            self._predictors[float(mem_freq)] = self.predictor

        # Define SM frequency search space (High to low)
        if self._sm_freqs:
            self.freq_space = np.array(sorted(set(self._sm_freqs), reverse=True))
            freq_info = f"{min(self._sm_freqs):.0f}-{max(self._sm_freqs):.0f} ({len(self._sm_freqs)} states)"
        else:
            self.freq_space = np.arange(MAX_FREQ, MIN_FREQ - 1, -FREQ_STEP)
            freq_info = f"[{MIN_FREQ:.0f}, {MAX_FREQ:.0f}] MHz (step: {FREQ_STEP:.0f})"
        mps_upper = 100.0
        if self.max_mps_per_task is not None:
            mps_upper = min(mps_upper, float(self.max_mps_per_task))
        # Inclusive upper bound with step alignment (e.g., 5..40)
        self.mps_space = np.arange(mps_step, mps_upper + 1e-6, mps_step)

        # Define Memory frequency search space
        # All available memory frequencies (High to low)
        if self.fixed_mem_freq is not None:
            self.mem_freq_space = np.array([float(self.fixed_mem_freq)])
        else:
            self.mem_freq_space = np.array(self._get_platform_mem_freqs())

        print(f"  SM Frequency range: {freq_info}")
        print(f"  Memory Frequency states: {self._format_mem_freqs()} MHz")
        if self._sm_freqs:
            print(f"  SM Frequency states: {len(self._sm_freqs)} values")

    def _get_platform_mem_freqs(self):
        raw = str(self.platform).strip().lower() if self.platform else ""
        key = "".join(ch for ch in raw if ch.isalnum())

        # Prefer predefined high-performance memory frequencies (only reasonable high-frequency states).
        if key in PLATFORM_MEM_FREQS:
            return PLATFORM_MEM_FREQS[key]
        if "3080ti" in key:
            return PLATFORM_MEM_FREQS["3080ti"]
        if key.startswith("a100"):
            return PLATFORM_MEM_FREQS["a100"]

        # Fallback to platform.json (if not in predefined list).
        mem_clocks = self._load_platform_mem_freqs()
        if mem_clocks:
            return sorted([float(x) for x in mem_clocks], reverse=True)

        return list(np.arange(MAX_MEM_FREQ, MIN_MEM_FREQ - 1, -MEM_FREQ_STEP))

    def _format_mem_freqs(self):
        freqs = [float(x) for x in self.mem_freq_space]
        if not freqs:
            return "N/A"
        if len(freqs) <= 3:
            return ", ".join(f"{f:.0f}" for f in freqs)
        return f"{min(freqs):.0f}-{max(freqs):.0f} ({len(freqs)} states)"

    def _auto_detect_platform(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return None
            match = re.search(r"GPU \d+:\s*(.+?)\s*\(", result.stdout)
            if not match:
                return None
            gpu_name = match.group(1).strip().lower()
            normalized = "".join(ch for ch in gpu_name if ch.isalnum())
            if "3080ti" in normalized:
                print("  Auto-detected platform: 3080ti")
                return "3080ti"
            if "a100" in normalized:
                print("  Auto-detected platform: a100")
                return "a100"
        except Exception:
            return None
        return None

    def _load_platform_sm_freqs(self):
        config_path = os.path.join(os.path.dirname(__file__), "src", "config", "platform.json")
        if not os.path.exists(config_path):
            return None
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
        except Exception:
            return None
        platforms = data.get("platforms", [])
        if not platforms:
            return None
        key = str(self.platform).strip().lower() if self.platform else ""
        for platform in platforms:
            name = platform.get("name", "")
            norm = "".join(ch for ch in name.lower() if ch.isalnum())
            if key and (key == norm or key in norm):
                clocks = platform.get("clocks", {})
                sm_clocks = clocks.get("SM_Clocks") or clocks.get("Graphics_Clocks", [])
                return sorted([float(x) for x in sm_clocks])
        return None

    def _load_platform_mem_freqs(self):
        config_path = os.path.join(os.path.dirname(__file__), "src", "config", "platform.json")
        if not os.path.exists(config_path):
            return None
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
        except Exception:
            return None
        platforms = data.get("platforms", [])
        if not platforms:
            return None
        key = str(self.platform).strip().lower() if self.platform else ""
        for platform in platforms:
            name = platform.get("name", "")
            norm = "".join(ch for ch in name.lower() if ch.isalnum())
            if key and (key == norm or key in norm):
                clocks = platform.get("clocks", {})
                mem_clocks = clocks.get("Memory_Clocks") or clocks.get("Memory_Clock", [])
                if isinstance(mem_clocks, (int, float)):
                    mem_clocks = [mem_clocks]
                return sorted([float(x) for x in mem_clocks], reverse=True)
        return None

    def _snap_sm_freq(self, freq):
        if not self._sm_freqs:
            return freq
        candidates = [f for f in self._sm_freqs if f <= freq]
        if not candidates:
            return None
        return max(candidates)

    def get_batches_for_mps(self, mps, n_tasks=1):
        """
        Returns a filtered list of batch sizes appropriate for the given MPS.
        Logic based on user constraints:
          - MPS 5%  -> [4, 32]
          - MPS 40% -> [32, 256]

        Scaling factors derived:
          Min Batch approx 0.8 * MPS
          Max Batch approx 6.4 * MPS

        Also applies model-specific batch size limits for memory-intensive models.
        """
        # Calculate dynamic bounds
        min_bound = max(4, int(mps * 0.8))
        max_bound = int(mps * 6.4)

        # Apply model-specific batch size limits
        model_key = self.model_name.lower().replace("-", "_").replace(".", "")
        for model_pattern, limit in MODEL_MAX_BATCH_LIMITS.items():
            if model_pattern in model_key:
                max_bound = min(max_bound, limit)
                break

        if self.max_batchsize is not None:
            # Heuristic safety: when running multiple MPS workers, per-worker batch needs to be lower.
            per_task_cap = max(4, int(self.max_batchsize / max(1, int(n_tasks))))
            max_bound = min(max_bound, per_task_cap)

        # Filter the global list based on these bounds
        valid_batches = [b for b in ALL_CANDIDATE_BATCHES if min_bound <= b <= max_bound]

        # Fallback: if range is too strict and array is empty, return the closest single value
        if not valid_batches:
            valid_batches = [min(ALL_CANDIDATE_BATCHES, key=lambda x: abs(x - min_bound))]

        return valid_batches

    def predict_system_metrics(self, tasks, freq, mem_freq=None):
        """
        Wrapper to call predictor.predict_all for a list of TaskConfig.

        Args:
            tasks: List of TaskConfig objects
            freq: SM frequency in MHz
            mem_freq: Memory frequency in MHz. If None, uses default models.
        """
        if not tasks:
            return 0.0, 0.0, 0.0, 0.0

        task_key = tuple((float(t.mps), int(t.batch)) for t in tasks)
        cache_key = (task_key, float(freq), float(mem_freq) if mem_freq is not None else None)
        cached = self._metric_cache.get(cache_key)
        if cached is not None:
            return cached

        p_list = [t.mps for t in tasks]
        b_list = [t.batch for t in tasks]

        predictor = self.predictor
        if mem_freq is not None:
            mem_key = float(mem_freq)
            predictor = self._predictors.get(mem_key)
            if predictor is None:
                predictor = PctoDLPredictor(
                    self.model_name,
                    mem_freq=mem_key,
                    platform=self.platform,
                    dataset=self.predictor.dataset,
                )
                self._predictors[mem_key] = predictor

        # Call the prediction API with memory frequency support
        U_sm, U_mem, Tp, Pow = predictor.predict_all(p_list, b_list, freq, mem_freq=mem_freq)
        self._metric_cache[cache_key] = (U_sm, U_mem, Tp, Pow)
        return U_sm, U_mem, Tp, Pow

    def get_pruned_mem_freq_space(self, current_mem_freq):
        """
        Get pruned memory frequency search space.
        Per Spec 4.2: Once F_mem is throttled, search space restricted to m <= F_curr_mem
        """
        return [m for m in self.mem_freq_space if m <= current_mem_freq]

    def solve_freq_for_power(self, tasks, mem_freq, f_high=None, allow_raise=False):
        """
        Solve for SM frequency f* such that P_total(f*) = P_cap using bisection.
        If no solution within [MIN_FREQ, MAX_FREQ], return None.
        """
        if not tasks:
            return None

        f_low = MIN_FREQ
        if f_high is None or allow_raise:
            f_high = MAX_FREQ
        f_high = min(f_high, MAX_FREQ)
        if f_high < f_low:
            return None

        _, _, _, p_low = self.predict_system_metrics(tasks, f_low, mem_freq=mem_freq)
        _, _, _, p_high = self.predict_system_metrics(tasks, f_high, mem_freq=mem_freq)

        # 1. Lower-bound check: if even the minimum frequency exceeds the power cap, no solution.
        if (p_low - self.p_cap) > 0:
            return None

        # 2. [Critical fix] Upper-bound check: if max frequency is still under cap, full speed is safe.
        # The previous logic incorrectly returned None here.
        if (p_high - self.p_cap) < 0:
            return self._snap_sm_freq(f_high)

        # Narrow the bracket around a linearized root estimate to reduce iterations
        denom = (p_high - p_low)
        if denom > 1e-6:
            f_guess = f_low + (f_high - f_low) * (self.p_cap - p_low) / denom
            window = max(100.0, 0.2 * (f_high - f_low))
            f_guess_low = max(f_low, f_guess - window)
            f_guess_high = min(f_high, f_guess + window)
            _, _, _, p_guess_low = self.predict_system_metrics(tasks, f_guess_low, mem_freq=mem_freq)
            _, _, _, p_guess_high = self.predict_system_metrics(tasks, f_guess_high, mem_freq=mem_freq)
            if (p_guess_low - self.p_cap) <= 0 <= (p_guess_high - self.p_cap):
                f_low, f_high = f_guess_low, f_guess_high

        max_iter = 12
        eps = 1.0
        for _ in range(max_iter):
            f_mid = 0.5 * (f_low + f_high)
            _, _, _, p_mid = self.predict_system_metrics(tasks, f_mid, mem_freq=mem_freq)
            if abs(p_mid - self.p_cap) <= eps:
                return f_mid
            if p_mid > self.p_cap:
                f_high = f_mid
            else:
                f_low = f_mid

        snapped = self._snap_sm_freq(f_low)
        return snapped

    def find_best_next(self, current_tasks, r_free, current_freq, current_mem_freq):
        """
        Alg 2: FindBestNext with Joint Frequency-Memory Sweep (Spec Section 4.2)

        The Pruned Sweep Strategy:
        1. Memory Loop (Outer): Iterate through memory states m <= F_curr_mem
        2. SM Loop (Inner): For each memory state, sweep SM frequencies

        Args:
            current_tasks: List of current TaskConfig objects
            r_free: Available MPS percentage
            current_freq: Current SM frequency constraint
            current_mem_freq: Current memory frequency constraint

        Returns:
            best_candidate: TaskConfig or None
            best_freq: Optimal SM frequency or None
            best_mem_freq: Optimal memory frequency or None
        """
        best_candidate = None
        best_freq = None
        best_mem_freq = None
        best_gain_density = -float('inf')

        # Calculate Base Throughput (T_base) with current operating state
        _, _, t_base, _ = self.predict_system_metrics(current_tasks, current_freq, mem_freq=current_mem_freq)

        # Iterate over all possible MPS allocations <= r_free
        valid_mps = [m for m in self.mps_space if m <= r_free + 1e-5]
        mem_freq_candidates = self.get_pruned_mem_freq_space(current_mem_freq)

        total_steps = 0
        for mps in valid_mps:
            total_steps += len(self.get_batches_for_mps(mps)) * len(mem_freq_candidates)
        processed_steps = 0
        last_report = time.monotonic()
        report_interval = 0.5
        bar_width = 28

        def maybe_report_progress():
            nonlocal last_report
            now = time.monotonic()
            if total_steps > 0 and now - last_report >= report_interval:
                pct = processed_steps / total_steps * 100.0
                filled = int(bar_width * pct / 100.0)
                bar = "=" * filled + "-" * (bar_width - filled)
                sys.stdout.write(
                    f"\r  [Greedy] [{bar}] {pct:5.1f}% ({processed_steps}/{total_steps})"
                )
                sys.stdout.flush()
                last_report = now

        for mps in valid_mps:
            # n_tasks includes the candidate itself, because memory pressure increases with number of workers.
            current_batches = self.get_batches_for_mps(mps, n_tasks=len(current_tasks) + 1)

            for batch in current_batches:
                candidate = TaskConfig(mps, batch)
                temp_task_list = current_tasks + [candidate]

                # Joint Frequency-Memory Sweep (solve f* by power equation)
                t_best_for_candidate = 0.0
                f_best_for_candidate = None
                m_best_for_candidate = None

                # Memory Loop (Outer): Pruned by current_mem_freq
                for mem_freq in mem_freq_candidates:
                    allow_raise = mem_freq < current_mem_freq
                    sm_freq = self.solve_freq_for_power(
                        temp_task_list,
                        mem_freq,
                        f_high=current_freq,
                        allow_raise=allow_raise,
                    )
                    if sm_freq is None:
                        processed_steps += 1
                        maybe_report_progress()
                        continue
                    _, _, t_sys, _ = self.predict_system_metrics(
                        temp_task_list, sm_freq, mem_freq=mem_freq
                    )
                    if t_sys > t_best_for_candidate:
                        t_best_for_candidate = t_sys
                        f_best_for_candidate = sm_freq
                        m_best_for_candidate = mem_freq
                    processed_steps += 1
                    maybe_report_progress()

                # If a valid configuration was found for this candidate
                if f_best_for_candidate is not None:
                    # Calculate Marginal Gain Density: S_gain = (T_max - T_curr) / Cost
                    gain = t_best_for_candidate - t_base
                    cost = mps if mps > 0 else 1e-5
                    gain_density = gain / cost

                    if gain_density > best_gain_density:
                        best_gain_density = gain_density
                        best_candidate = candidate
                        best_freq = f_best_for_candidate
                        best_mem_freq = m_best_for_candidate

        if total_steps > 0:
            filled = bar_width
            bar = "=" * filled + "-" * (bar_width - filled)
            sys.stdout.write(
                f"\r  [Greedy] [{bar}] 100.0% ({processed_steps}/{total_steps})\n"
            )
            sys.stdout.flush()

        return best_candidate, best_freq, best_mem_freq, total_steps

    def find_best_next_vectorized(self, current_tasks, r_free, current_freq, current_mem_freq):
        """
        Vectorized FindBestNext - uses vectorized bisection to accelerate frequency search.

        Core optimizations:
        1. Precompute U_sm and U_mem for all candidates (frequency-independent, main cost).
        2. Use batched power prediction to accelerate bisection.
        3. Use predict_system_metrics for final selection to ensure consistency.

        Args:
            current_tasks: List of current TaskConfig objects
            r_free: Available MPS percentage
            current_freq: Current SM frequency constraint
            current_mem_freq: Current memory frequency constraint

        Returns:
            best_candidate: TaskConfig or None
            best_freq: Optimal SM frequency or None
            best_mem_freq: Optimal memory frequency or None
        """
        # Step 1: Generate all candidate configurations.
        valid_mps = [m for m in self.mps_space if m <= r_free + 1e-5]
        mem_freq_candidates = self.get_pruned_mem_freq_space(current_mem_freq)

        if not valid_mps or not mem_freq_candidates:
            return None, None, None, 0

        # Compute baseline throughput.
        _, _, t_base, _ = self.predict_system_metrics(current_tasks, current_freq, mem_freq=current_mem_freq)

        # Expand all (mps, batch) combinations (excluding mem_freq).
        configs = []  # [(mps, batch, p_list, b_list), ...]
        for mps in valid_mps:
            batches = self.get_batches_for_mps(mps, n_tasks=len(current_tasks) + 1)
            for batch in batches:
                p_list = [t.mps for t in current_tasks] + [mps]
                b_list = [t.batch for t in current_tasks] + [batch]
                configs.append((mps, batch, p_list, b_list))

        n_configs = len(configs)
        if n_configs == 0:
            return None, None, None, 0

        # Step 2: Precompute U_sm and U_mem for each mem_freq.
        # Use batched prediction to accelerate.
        U_sms_by_mem = {}  # {mem_freq: np.array of U_sm}
        U_mems_by_mem = {}  # {mem_freq: np.array of U_mem}

        # Prepare batch data.
        all_p_lists = [c[2] for c in configs]
        all_b_lists = [c[3] for c in configs]

        for mem_freq in mem_freq_candidates:
            # Get predictor for this mem_freq.
            predictor = self.predictor
            if mem_freq != self.predictor.mem_freq:
                mem_key = float(mem_freq)
                if mem_key not in self._predictors:
                    self._predictors[mem_key] = PctoDLPredictor(
                        self.model_name,
                        mem_freq=mem_key,
                        platform=self.platform,
                        dataset=self.predictor.dataset,
                    )
                predictor = self._predictors[mem_key]

            # Batch-predict U_sm and U_mem.
            U_sms, U_mems = predictor.predict_metrics_batch(all_p_lists, all_b_lists)

            U_sms_by_mem[mem_freq] = U_sms
            U_mems_by_mem[mem_freq] = U_mems

        # Expand all (config_idx, mem_freq) combinations.
        candidates = []  # [(config_idx, mem_freq), ...]
        for cfg_idx in range(n_configs):
            for mem_freq in mem_freq_candidates:
                candidates.append((cfg_idx, mem_freq))

        n_candidates = len(candidates)
        print(f"  [Vectorized] Searching {n_candidates} candidates ({n_configs} unique configs)...")

        # Step 3: Initialize frequency bounds.
        f_lows = np.full(n_candidates, MIN_FREQ)
        f_highs = np.full(n_candidates, MAX_FREQ)

        # For mem_freq >= current_mem_freq, cap SM freq (no boosting).
        # For mem_freq < current_mem_freq, allow boosting up to MAX_FREQ.
        for i, (cfg_idx, mem_freq) in enumerate(candidates):
            allow_raise = mem_freq < current_mem_freq
            if not allow_raise:
                f_highs[i] = min(current_freq, MAX_FREQ)
            else:
                f_highs[i] = MAX_FREQ

        # Step 4: Batched power prediction function.
        def batch_predict_power_fast(freqs):
            """Batch-predict power using precomputed U_sm/U_mem."""
            powers = np.zeros(n_candidates)

            for mem_freq in mem_freq_candidates:
                # Get predictor for this mem_freq.
                predictor = self.predictor
                if mem_freq != self.predictor.mem_freq:
                    mem_key = float(mem_freq)
                    if mem_key not in self._predictors:
                        self._predictors[mem_key] = PctoDLPredictor(
                            self.model_name,
                            mem_freq=mem_key,
                            platform=self.platform,
                            dataset=self.predictor.dataset,
                        )
                    predictor = self._predictors[mem_key]

                # Get U_sm and U_mem for this mem_freq.
                U_sms = U_sms_by_mem[mem_freq]
                U_mems = U_mems_by_mem[mem_freq]

                # Find candidate indices for this mem_freq.
                indices = [i for i, (cfg_idx, mf) in enumerate(candidates) if mf == mem_freq]
                if not indices:
                    continue

                # Fetch power model parameters.
                static = predictor.power_params.get('static', {})
                delta_3 = static.get('delta_3', 2.31e-8)
                delta_2 = static.get('delta_2', -5.44e-5)
                delta_1 = static.get('delta_1', 0.039)
                delta_0 = static.get('delta_0', 85.6)

                gamma_p = predictor.power_params['dynamic'].get('gamma_p', 3.0)
                P_base = predictor.power_params['dynamic'].get('P_base', 10.0)
                f_ref = predictor.power_params['dynamic'].get('f_ref', 1950.0)
                f_min_p = predictor.power_params['dynamic'].get('f_min', 210.0)

                # Batch static power.
                f_arr = freqs[indices]
                P_static = delta_3 * f_arr**3 + delta_2 * f_arr**2 + delta_1 * f_arr + delta_0

                # Batch dynamic power.
                if predictor.power_dyn_gbr is not None:
                    n_batch = len(indices)
                    X_gbr = np.zeros((n_batch, 8))

                    for j, idx in enumerate(indices):
                        cfg_idx, _ = candidates[idx]
                        mps, batch, p_list, b_list = configs[cfg_idx]
                        N = len(p_list)

                        total_S = float(N) * predictor.params_M
                        p_mean = sum(p_list) / N if N > 0 else 50.0
                        p_var = np.var(p_list) if N > 1 else 0.0
                        b_mean = sum(b_list) / N if N > 0 else 1.0
                        mem_freq_norm = float(mem_freq) / 10000.0

                        X_gbr[j] = [U_sms[cfg_idx], U_mems[cfg_idx], float(N),
                                    total_S, p_mean, p_var, b_mean, mem_freq_norm]

                    P_dyn_ref = predictor.power_dyn_gbr.predict(X_gbr)
                else:
                    P_dyn_ref = np.full(len(indices), 100.0)

                freq_ratio = np.clip((f_arr - f_min_p) / (f_ref - f_min_p), 0.0, 1.0)
                P_dyn = P_base + (P_dyn_ref - P_base) * np.power(freq_ratio, gamma_p)

                for j, idx in enumerate(indices):
                    powers[idx] = P_static[j] + P_dyn[j]

            return powers

        # Step 5: Vectorized bisection.
        powers_low = batch_predict_power_fast(f_lows)
        valid_mask = powers_low <= self.p_cap

        powers_high = batch_predict_power_fast(f_highs)
        use_high_freq = (powers_high <= self.p_cap) & valid_mask
        final_freqs = np.where(use_high_freq, f_highs, np.nan)

        need_bisect = valid_mask & ~use_high_freq & (powers_high > self.p_cap)

        max_iter = 12
        eps = 1.0
        bisect_f_lows = f_lows.copy()
        bisect_f_highs = f_highs.copy()

        for iteration in range(max_iter):
            bisect_mask = need_bisect & np.isnan(final_freqs)
            if not np.any(bisect_mask):
                break

            f_mids = 0.5 * (bisect_f_lows + bisect_f_highs)
            powers_mid = batch_predict_power_fast(f_mids)

            converged = bisect_mask & (np.abs(powers_mid - self.p_cap) <= eps)
            final_freqs = np.where(converged, f_mids, final_freqs)

            too_high = bisect_mask & ~converged & (powers_mid > self.p_cap)
            too_low = bisect_mask & ~converged & (powers_mid <= self.p_cap)

            bisect_f_highs = np.where(too_high, f_mids, bisect_f_highs)
            bisect_f_lows = np.where(too_low, f_mids, bisect_f_lows)

        still_nan = np.isnan(final_freqs) & valid_mask
        final_freqs = np.where(still_nan, bisect_f_lows, final_freqs)


        # Step 6: Select the best candidate using the batched frequency results.
        # For speed, use the bisection outputs directly.
        best_candidate = None
        best_freq = None
        best_mem_freq = None
        best_gain_density = -float('inf')

        # Find the best mem_freq for each config (highest throughput).
        for cfg_idx in range(n_configs):
            mps, batch, p_list, b_list = configs[cfg_idx]
            candidate = TaskConfig(mps, batch)
            temp_task_list = current_tasks + [candidate]

            t_best_for_candidate = 0.0
            f_best_for_candidate = None
            m_best_for_candidate = None

            # Iterate all mem_freq using the batched frequency results.
            for mem_freq in mem_freq_candidates:
                # Find the corresponding candidate index.
                cand_idx = None
                for i, (c_idx, m_freq) in enumerate(candidates):
                    if c_idx == cfg_idx and m_freq == mem_freq:
                        cand_idx = i
                        break

                if cand_idx is None or not valid_mask[cand_idx] or np.isnan(final_freqs[cand_idx]):
                    continue

                sm_freq = final_freqs[cand_idx]

                # Use the standard method to compute throughput (ensure consistency).
                _, _, t_sys, _ = self.predict_system_metrics(
                    temp_task_list, sm_freq, mem_freq=mem_freq
                )

                if t_sys > t_best_for_candidate:
                    t_best_for_candidate = t_sys
                    f_best_for_candidate = sm_freq
                    m_best_for_candidate = mem_freq

            # If a valid config was found, compute gain density.
            if f_best_for_candidate is not None:
                gain = t_best_for_candidate - t_base
                cost = mps if mps > 0 else 1e-5
                gain_density = gain / cost

                if gain_density > best_gain_density:
                    best_gain_density = gain_density
                    best_candidate = candidate
                    best_freq = self._snap_sm_freq(f_best_for_candidate)
                    best_mem_freq = m_best_for_candidate

        # Total configs searched = num configs x num mem_freq candidates.
        total_configs_searched = n_configs * len(mem_freq_candidates)
        return best_candidate, best_freq, best_mem_freq, total_configs_searched

    def distribute_residuals(self, tasks, r_free):
        """
        Distribute remaining resources proportionally to admitted tasks.
        """
        if not tasks or r_free < self.mps_step:
            return tasks

        n_tasks = len(tasks)

        # How many chunks of 'mps_step' do we have left?
        chunks = int(r_free // self.mps_step)

        # Distribute chunks round-robin style to tasks
        for i in range(chunks):
            task_idx = i % n_tasks
            if self.max_mps_per_task is not None and (tasks[task_idx].mps + self.mps_step) > self.max_mps_per_task:
                continue
            tasks[task_idx].mps += self.mps_step

        return tasks

    def find_best_freq_for_config(self, tasks, current_freq, current_mem_freq):
        """
        Given a fixed task configuration, find the optimal (SM freq, Mem freq) pair.
        Uses power-equation solve with memory pruning.
        """
        t_best = 0.0
        f_best = current_freq
        m_best = current_mem_freq

        # Pruned memory space
        mem_freq_candidates = self.get_pruned_mem_freq_space(current_mem_freq)

        for mem_freq in mem_freq_candidates:
            allow_raise = mem_freq < current_mem_freq
            sm_freq = self.solve_freq_for_power(
                tasks,
                mem_freq,
                f_high=current_freq,
                allow_raise=allow_raise,
            )
            if sm_freq is None:
                continue
            _, _, t_sys, _ = self.predict_system_metrics(tasks, sm_freq, mem_freq=mem_freq)
            if t_sys > t_best:
                t_best = t_sys
                f_best = sm_freq
                m_best = mem_freq

        return f_best, m_best, t_best

    def run(self):
        """
        Alg 1: Master Iterative Partitioning with Joint Frequency-Memory Sweep
        (Spec Section 4.1)

        Returns:
            (tasks, sm_freq, mem_freq, partition_time_ms, total_configs_searched)
        """
        import time
        partition_start = time.time()

        R_curr = []
        r_free = 100.0
        total_configs_searched = 0  # Total configs searched.
        if self.freq_space is not None and len(self.freq_space) > 0:
            F_curr = float(self.freq_space[0])
        else:
            F_curr = MAX_FREQ
        F_mem_curr = float(self.fixed_mem_freq) if self.fixed_mem_freq is not None else max(self.mem_freq_space)  # Initialize at max memory frequency

        mode_str = "[Vectorized]" if self.vectorized else "[Standard]"
        print(f"\nStarting Partitioning {mode_str} | P_cap={self.p_cap}W | MPS Step={self.mps_step}%")
        print(f"Initial State: SM Freq={F_curr:.0f} MHz, Mem Freq={F_mem_curr:.0f} MHz")
        print("-" * 85)
        print(f"{'Step':<5} | {'Decision':<28} | {'r_free':<8} | {'SM Freq':<8} | {'Mem Freq':<8} | {'Sys Tput':<10}")
        print("-" * 85)

        step = 1
        while r_free >= self.mps_step:
            if self.max_tasks is not None and len(R_curr) >= self.max_tasks:
                break
            # Find best next task with Joint Frequency-Memory Sweep.
            # Choose search method based on the vectorized flag.
            if self.vectorized:
                x_next, f_next, mem_freq_next, configs_searched = self.find_best_next_vectorized(R_curr, r_free, F_curr, F_mem_curr)
            else:
                x_next, f_next, mem_freq_next, configs_searched = self.find_best_next(R_curr, r_free, F_curr, F_mem_curr)

            total_configs_searched += configs_searched

            if x_next is not None:
                # Add task and update operating state
                R_curr.append(x_next)
                r_free -= x_next.mps
                F_curr = f_next
                F_mem_curr = mem_freq_next

                # Logging
                _, _, t_sys, _ = self.predict_system_metrics(R_curr, F_curr, mem_freq=F_mem_curr)
                print(f"{step:<5} | + {str(x_next):<27} | {r_free:>6.1f}%  | {F_curr:>6.0f}   | {F_mem_curr:>6.0f}   | {t_sys:>8.2f}")
                step += 1
            else:
                print(f"{step:<5} | Saturation/No Fit           | {r_free:>6.1f}%  | {F_curr:>6.0f}   | {F_mem_curr:>6.0f}   | -")
                break

        # Distribute Residuals
        if r_free > 1e-5:
            print(f"\nDistributing residual {r_free:.1f}% resource...")
            R_curr = self.distribute_residuals(R_curr, r_free)

            # Re-evaluate best (SM freq, Mem freq) for the final configuration
            F_curr, F_mem_curr, t_final = self.find_best_freq_for_config(R_curr, F_curr, F_mem_curr)

        partition_time_ms = (time.time() - partition_start) * 1000
        return R_curr, F_curr, F_mem_curr, partition_time_ms, total_configs_searched


def main():
    parser = argparse.ArgumentParser(description='Iterative Greedy Partitioning for GPU')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., mobilenet_v2)')
    parser.add_argument('--p_cap', type=float, default=150.0, help='Power Cap (Watts)')
    parser.add_argument('--mps_step', type=int, default=5, help='Granularity of MPS allocation (default: 5)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (e.g., imagenet, coco) for model selection')
    parser.add_argument('--platform', type=str, default=None,
                        help='Platform/GPU model (e.g., a100, 3090) for model selection')
    parser.add_argument('--mem-freq', type=int, default=None,
                        help='Fix memory frequency to a specific value (MHz). If not specified, searches optimal.')
    parser.add_argument('--vectorized', action='store_true',
                        help='Use vectorized batch prediction for faster search (experimental)')

    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = "imagenet"

    # Initialize and Run
    partitioner = GreedyPartitioner(args.model, args.p_cap, args.mps_step,
                                     mem_freq=args.mem_freq, platform=args.platform,
                                     dataset=args.dataset, vectorized=args.vectorized)
    final_tasks, final_freq, final_mem_freq, partition_time_ms, total_configs = partitioner.run()

    # Final Report
    print(f"\n{'='*70}")
    print(f" FINAL PARTITIONING RESULTS")
    print(f"{'='*70}")
    print(f" Global SM Frequency:   {final_freq:.0f} MHz")
    print(f" Global Memory Frequency: {final_mem_freq:.0f} MHz")
    print(f" Power Cap:              {args.p_cap} W")
    print(f" Total Tasks:            {len(final_tasks)}")
    print(f" Partition Time:         {partition_time_ms:.2f} ms ({partition_time_ms/1000:.3f} s)")
    print(f" Configs Searched:       {total_configs}")
    print("-" * 70)

    p_list = [t.mps for t in final_tasks]
    b_list = [t.batch for t in final_tasks]

    # Get final metrics
    U_sm, U_mem, Tp, Pow = partitioner.predictor.predict_all(
        p_list, b_list, final_freq, mem_freq=final_mem_freq
    )

    print(f"{'ID':<4} | {'MPS (%)':<10} | {'Batch':<10}")
    print("-" * 30)
    for i, t in enumerate(final_tasks):
        print(f"{i:<4} | {t.mps:<10.1f} | {t.batch:<10}")
    print("-" * 70)
    print(f" System Throughput:    {Tp:.2f} img/s")
    print(f" System Power:         {Pow:.2f} W")
    print(f" SM Util:              {U_sm:.2f} %")
    print(f" Mem Util:             {U_mem:.2f} %")
    print(f"{'='*70}")

    # Generate verification command
    p_str = " ".join([str(t.mps) for t in final_tasks])
    b_str = " ".join([str(t.batch) for t in final_tasks])
    print(f"\n Verification Command:")
    print(f" python src/profile_mps.py --single --models {args.model} \\")
    print(f"     --p {p_str} --b {b_str} --f {int(final_freq)} --mem-freq {int(final_mem_freq)}")

if __name__ == "__main__":
    main()
