#!/usr/bin/env python3
"""
Shadow Optimizer (Grey-Box BO) for online joint optimization.
Uses GBR priors + GP residuals to suggest improved configurations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import erf, exp, pi, sqrt
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
except Exception:  # pragma: no cover - optional dependency
    GaussianProcessRegressor = None
    Matern = None

warnings.filterwarnings("ignore", message=".*optimal value found for dimension.*", category=UserWarning)

from predict import PctoDLPredictor


@dataclass
class ShadowConfig:
    p_list: List[float]
    b_list: List[int]
    sm_freq: float
    mem_freq: float
    throughput: float
    power: float


class ShadowOptimizer:
    def __init__(
        self,
        p_cap: float,
        predictor: Optional[PctoDLPredictor],
        sm_freqs: Sequence[float],
        mem_freqs: Sequence[float],
        batch_pool: Sequence[int],
        *,
        use_prior: bool = True,
        max_batch_size: Optional[int] = None,
        max_mps_per_task: Optional[float] = None,
        mps_step: int = 10,
        trust_radius: int = 4,
        history_window: int = 500,
        gp_refit_interval: int = 3,
        gp_max_points: int = 500,
        gp_min_points: int = 5,
        reconfig_threshold: float = 0.05,
        allow_sm_adjustment: bool = False,
        allow_mem_adjustment: bool = False,
        explore_partition_count: bool = True,
        min_partitions: int = 1,
        max_partitions: int = 10,
        total_iterations: int = 1500,
        adaptive_threshold: bool = True,
    ) -> None:
        self.p_cap = float(p_cap)
        self.predictor = predictor
        self.sm_freqs = sorted(list(sm_freqs))
        self.mem_freqs = sorted(list(mem_freqs))
        self.batch_pool = sorted(set(int(b) for b in batch_pool))
        self.use_prior = bool(use_prior)
        self.max_batch_size = int(max_batch_size) if max_batch_size is not None else None
        self.max_mps_per_task = float(max_mps_per_task) if max_mps_per_task is not None else None
        self.mps_step = int(mps_step)
        self.trust_radius = int(trust_radius)
        self.history_window = int(history_window)
        self.gp_refit_interval = max(1, int(gp_refit_interval))
        self.gp_max_points = max(5, int(gp_max_points))
        self.gp_min_points = max(1, int(gp_min_points))
        self.reconfig_threshold = float(reconfig_threshold)
        self.allow_sm_adjustment = bool(allow_sm_adjustment)
        self.allow_mem_adjustment = bool(allow_mem_adjustment)
        self.explore_partition_count = bool(explore_partition_count)
        self.min_partitions = max(1, int(min_partitions))
        self.max_partitions = max(1, int(max_partitions))
        self.total_iterations = int(total_iterations)
        self.adaptive_threshold = bool(adaptive_threshold)

        self.max_tasks = None
        self._update_count = 0
        self.history_x: List[np.ndarray] = []
        self.history_t: List[float] = []
        self.history_p: List[float] = []
        self._best_feasible_throughput: Optional[float] = None

        if self.use_prior and self.predictor is None:
            raise ValueError("ShadowOptimizer(use_prior=True) requires a predictor instance")

        if GaussianProcessRegressor and Matern:
            kernel = Matern(nu=2.5)
            self.gp_t = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=True)
            self.gp_p = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=True)
        else:
            self.gp_t = None
            self.gp_p = None

    def _encode(self, p_list: Sequence[float], b_list: Sequence[int], sm_freq: float, mem_freq: float) -> np.ndarray:
        if self.max_tasks is None:
            self.max_tasks = len(p_list)
        p_pad = list(p_list)[: self.max_tasks] + [0.0] * max(0, self.max_tasks - len(p_list))
        b_pad = list(b_list)[: self.max_tasks] + [0] * max(0, self.max_tasks - len(b_list))
        features = [
            float(sm_freq),
            float(mem_freq),
            float(len(p_list)),
            float(np.mean(b_list)) if b_list else 0.0,
            float(np.var(b_list)) if b_list else 0.0,
        ]
        features.extend(p_pad)
        features.extend(b_pad)
        return np.array(features, dtype=np.float32)

    def _as_scalar0(self, x) -> float:
        """
        Robustly convert sklearn/NumPy outputs (scalar, 1D, etc.) to a float.
        GP implementations should return shape (n_samples,), but some wrappers
        may return 0-D arrays or Python scalars.
        """
        arr = np.asarray(x)
        if arr.size == 0:
            return 0.0
        return float(arr.reshape(-1)[0])

    def _normal_pdf(self, z: float) -> float:
        return exp(-0.5 * z * z) / sqrt(2.0 * pi)

    def _normal_cdf(self, z: float) -> float:
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    def _expected_improvement(self, mu: float, sigma: float, best: float) -> float:
        if sigma <= 1e-6:
            return max(0.0, mu - best)
        z = (mu - best) / sigma
        return (mu - best) * self._normal_cdf(z) + sigma * self._normal_pdf(z)

    def _feasibility_prob(self, mu: float, sigma: float) -> float:
        if sigma <= 1e-6:
            return 1.0 if mu <= self.p_cap else 0.0
        z = (self.p_cap - mu) / sigma
        return self._normal_cdf(z)

    def _get_dynamic_threshold(self) -> float:
        """
        Calculate dynamic reconfig threshold based on iteration progress.

        Strategy (based on 1500 total iterations):
        - First 500 iterations (0-1/3): Aggressive exploration with negative threshold
        - Remaining 1000 iterations (1/3-1): Logarithmic growth converging to 2%

        Uses log function to gradually converge to final threshold of 0.02.
        If total_iterations != 1500, scales proportionally.
        """
        if not self.adaptive_threshold:
            return self.reconfig_threshold

        # Calculate progress ratio (0.0 to 1.0)
        progress = min(1.0, float(self._update_count) / float(self.total_iterations))

        # Phase boundary: first 1/3 is aggressive (negative threshold)
        aggressive_phase = 1.0 / 3.0

        if progress < aggressive_phase:
            # Phase 1: Aggressive exploration with negative threshold
            # Linear from -0.10 to 0.0 over first 1/3
            phase_progress = progress / aggressive_phase
            threshold = -0.10 * (1.0 - phase_progress)
        else:
            # Phase 2: Logarithmic convergence to 2%
            # Map progress from [1/3, 1] to [0, 1]
            phase_progress = (progress - aggressive_phase) / (1.0 - aggressive_phase)

            # Use log function: threshold = 0.02 * (1 - exp(-k*x))
            # This starts near 0 and asymptotically approaches 0.02
            # k=4 gives good convergence: at x=1, threshold ≈ 0.0196 (98% of target)
            k = 4.0
            threshold = 0.02 * (1.0 - exp(-k * phase_progress))

        return threshold

    def residual_means(self) -> Tuple[float, float]:
        """Return (mean_delta_throughput, mean_delta_power) over the retained history window."""
        if not self.history_t:
            return 0.0, 0.0
        return float(np.mean(self.history_t)), float(np.mean(self.history_p))

    def best_feasible_throughput(self, *, fallback_current: Optional[float] = None) -> float:
        """
        Best observed feasible throughput so far.

        Args:
            fallback_current: Returned if no feasible point has been observed yet.
        """
        if self._best_feasible_throughput is not None:
            return float(self._best_feasible_throughput)
        return float(fallback_current) if fallback_current is not None else 0.0

    def update(self, config: ShadowConfig) -> None:
        self._update_count += 1
        x = self._encode(config.p_list, config.b_list, config.sm_freq, config.mem_freq)
        if self.use_prior:
            assert self.predictor is not None  # for type-checkers
            _, _, base_t, base_p = self.predictor.predict_all(
                config.p_list, config.b_list, config.sm_freq, mem_freq=config.mem_freq
            )
        else:
            base_t, base_p = 0.0, 0.0
        delta_t = float(config.throughput - base_t)
        delta_p = float(config.power - base_p)

        self.history_x.append(x)
        self.history_t.append(delta_t)
        self.history_p.append(delta_p)
        if config.power <= self.p_cap:
            if self._best_feasible_throughput is None or config.throughput > self._best_feasible_throughput:
                self._best_feasible_throughput = float(config.throughput)

        if len(self.history_x) > self.history_window:
            self.history_x = self.history_x[-self.history_window :]
            self.history_t = self.history_t[-self.history_window :]
            self.history_p = self.history_p[-self.history_window :]

        # GP fitting is cubic in the number of points; for long runs we:
        # - keep a potentially long history_window (for "memory"),
        # - but refit only periodically,
        # - and subsample a bounded number of points for fitting.
        if self.gp_t and len(self.history_x) >= self.gp_min_points and (self._update_count % self.gp_refit_interval == 0):
            n = len(self.history_x)
            if n > self.gp_max_points:
                # Evenly-spaced subsample to cover the whole window.
                idx = np.linspace(0, n - 1, num=self.gp_max_points, dtype=int)
                X = np.vstack([self.history_x[i] for i in idx])
                y_t = np.array([self.history_t[i] for i in idx])
                y_p = np.array([self.history_p[i] for i in idx])
            else:
                X = np.vstack(self.history_x)
                y_t = np.array(self.history_t)
                y_p = np.array(self.history_p)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp_t.fit(X, y_t)
                self.gp_p.fit(X, y_p)

    def _neighbors(self, values: Sequence[float], current: float) -> List[float]:
        if current not in values:
            return [current]
        idx = values.index(current)
        start = max(0, idx - self.trust_radius)
        end = min(len(values), idx + self.trust_radius + 1)
        return values[start:end]

    def _mps_candidates(self, p_list: Sequence[float]) -> List[List[float]]:
        candidates = [list(p_list)]
        for i in range(len(p_list)):
            for delta in (-self.mps_step, self.mps_step):
                new = list(p_list)
                new[i] = new[i] + delta
                if new[i] < self.mps_step:
                    continue
                if self.max_mps_per_task is not None and new[i] > self.max_mps_per_task:
                    continue
                total = sum(new)
                adjust = total - 100.0
                if abs(adjust) < 1e-6:
                    if self.max_mps_per_task is None or max(new) <= self.max_mps_per_task:
                        candidates.append(new)
                    continue
                # compensate on the largest partition
                j = max(range(len(new)), key=lambda k: new[k])
                if j == i:
                    continue
                new[j] = new[j] - adjust
                if new[j] < self.mps_step:
                    continue
                if self.max_mps_per_task is not None and new[j] > self.max_mps_per_task:
                    continue
                if self.max_mps_per_task is not None and max(new) > self.max_mps_per_task:
                    continue
                candidates.append(new)
        return candidates

    def _batches_for_mps(self, mps: float, n_tasks: int) -> List[int]:
        """
        Derive feasible batch candidates for a single partition given its MPS share.

        This mirrors greedy_partitioning.get_batches_for_mps:
          - min_bound = max(4, int(mps * 0.8))
          - max_bound = int(mps * 6.4)
        Additionally, respect model max_batch_size (from config "max batchsize") if provided.
        """
        min_bound = max(4, int(float(mps) * 0.8))
        max_bound = int(float(mps) * 6.4)
        if self.max_batch_size is not None:
            # Heuristic safety: per-worker batch decreases as number of concurrent workers increases.
            per_task_cap = max(4, int(self.max_batch_size / max(1, int(n_tasks))))
            max_bound = min(max_bound, per_task_cap)

        # If bounds collapse, still keep them sane.
        max_bound = max(min_bound, max_bound)

        pool = self.batch_pool
        if self.max_batch_size is not None:
            pool = [b for b in pool if b <= self.max_batch_size]
        valid = [b for b in pool if min_bound <= b <= max_bound]

        if valid:
            return valid

        # Fallback: choose the closest value to min_bound within pool (prefer not exceeding max_bound).
        if pool:
            return [min(pool, key=lambda x: abs(x - min_bound))]
        # Last resort: return a minimal safe batch size.
        return [min_bound]

    def _batch_neighbors(self, p_list: Sequence[float], b_list: Sequence[int]) -> List[List[int]]:
        """
        Multi-step neighborhood for batches using trust_radius.

        For each partition i, allowed batches are derived from its (candidate) MPS share p_list[i]
        and the model max_batch_size. Then we move up to trust_radius steps in the allowed list.
        """
        candidates = [list(b_list)]
        if not b_list:
            return candidates

        for i in range(len(b_list)):
            mps_i = p_list[i] if i < len(p_list) else 0.0
            allowed = self._batches_for_mps(mps_i, n_tasks=len(p_list))

            # Ensure current batch is representable in the local ladder; otherwise snap to nearest.
            cur = int(b_list[i])
            ladder = sorted(set(int(x) for x in allowed + [cur]))
            if cur not in ladder:
                cur = min(ladder, key=lambda x: abs(x - cur))
            idx = ladder.index(cur)

            # Use trust_radius to control search range
            for step in range(1, self.trust_radius + 1):
                for delta in (-step, step):
                    j = idx + delta
                    if 0 <= j < len(ladder):
                        new = list(b_list)
                        new[i] = ladder[j]
                        candidates.append(new)
        return candidates

    def _partition_count_candidates(self, current_n: int) -> List[int]:
        """
        Generate candidate partition counts to explore.

        Args:
            current_n: Current number of partitions

        Returns:
            List of partition counts to explore (including current)
        """
        if not self.explore_partition_count:
            return [current_n]

        candidates = [current_n]

        # Explore N-1 (merge partitions)
        if current_n > self.min_partitions:
            candidates.append(current_n - 1)

        # Explore N+1 (split partitions)
        if current_n < self.max_partitions:
            candidates.append(current_n + 1)

        return sorted(set(candidates))

    def _generate_config_for_partition_count(
        self,
        n_partitions: int,
        current_p_list: Sequence[float],
        current_b_list: Sequence[int]
    ) -> List[Tuple[List[float], List[int]]]:
        """
        Generate reasonable MPS and batch configurations for a given partition count.

        Args:
            n_partitions: Target number of partitions
            current_p_list: Current MPS allocation
            current_b_list: Current batch sizes

        Returns:
            List of (p_list, b_list) tuples representing valid configurations
        """
        configs = []
        current_n = len(current_p_list)

        if n_partitions == current_n:
            # Same partition count - return current config
            configs.append((list(current_p_list), list(current_b_list)))

        elif n_partitions < current_n:
            # Merge partitions: combine smallest partitions
            # Strategy: merge the two smallest MPS allocations
            indices = list(range(current_n))
            indices_sorted = sorted(indices, key=lambda i: current_p_list[i])

            # Merge the two smallest
            merge_idx1, merge_idx2 = indices_sorted[0], indices_sorted[1]
            new_p_list = []
            new_b_list = []

            merged_mps = current_p_list[merge_idx1] + current_p_list[merge_idx2]
            # Use larger batch size of the two being merged
            merged_batch = max(current_b_list[merge_idx1], current_b_list[merge_idx2])

            for i in range(current_n):
                if i == merge_idx1:
                    new_p_list.append(merged_mps)
                    new_b_list.append(merged_batch)
                elif i == merge_idx2:
                    continue  # Skip, already merged
                else:
                    new_p_list.append(current_p_list[i])
                    new_b_list.append(current_b_list[i])

            configs.append((new_p_list, new_b_list))

        else:  # n_partitions > current_n
            # Split partitions: split the largest partition
            # Strategy: split the largest MPS allocation
            max_idx = max(range(current_n), key=lambda i: current_p_list[i])
            max_mps = current_p_list[max_idx]

            # Only split if the partition is large enough
            if max_mps >= 2 * self.mps_step:
                new_p_list = list(current_p_list)
                new_b_list = list(current_b_list)

                # Split roughly in half
                split_mps1 = (max_mps // 2) // self.mps_step * self.mps_step
                split_mps2 = max_mps - split_mps1

                # Ensure both splits are at least mps_step
                if split_mps1 >= self.mps_step and split_mps2 >= self.mps_step:
                    new_p_list[max_idx] = split_mps1
                    new_p_list.append(split_mps2)

                    # Use same batch size for both splits initially
                    new_b_list.append(current_b_list[max_idx])

                    configs.append((new_p_list, new_b_list))

        return configs

    def suggest(
        self,
        config: ShadowConfig,
        current_sm_freq: float = None,
        current_mem_freq: float = None,
        *,
        best_throughput: Optional[float] = None,
        conservative: bool = True,
    ) -> Optional[ShadowConfig]:
        """
        Suggest an improved configuration using EIC.

        By default this keeps SM/MEM frequency fixed (matching the main PctoDL runtime,
        where frequency is controlled by a separate power controller). If
        allow_sm_adjustment / allow_mem_adjustment is enabled, it also searches
        neighboring frequency states within the trust region.

        If explore_partition_count is enabled, also explores configurations with
        different numbers of partitions (N-1, N, N+1).

        Args:
            config: Current shadow configuration
            current_sm_freq: Override current SM frequency (anchor for trust region)
            current_mem_freq: Override current memory frequency (anchor for trust region)
            best_throughput: EI target throughput. If omitted, uses current measured throughput.
            conservative: If True, only return a candidate that clears the reconfig_threshold.
        """
        sm_overridden = current_sm_freq is not None
        mem_overridden = current_mem_freq is not None
        # Use provided frequencies or current config frequencies
        if current_sm_freq is None:
            current_sm_freq = config.sm_freq
        if current_mem_freq is None:
            current_mem_freq = config.mem_freq

        # Candidate frequency states (trust region). By default keep fixed.
        sm_freq_candidates = [float(current_sm_freq)]
        mem_freq_candidates = [float(current_mem_freq)]
        # Backwards-compatible: if the caller provides current_*_freq, treat it as fixed (external control).
        if self.allow_sm_adjustment and self.sm_freqs and not sm_overridden:
            sm_freq_candidates = [float(x) for x in self._neighbors([float(v) for v in self.sm_freqs], float(current_sm_freq))]
        if self.allow_mem_adjustment and self.mem_freqs and not mem_overridden:
            mem_freq_candidates = [float(x) for x in self._neighbors([float(v) for v in self.mem_freqs], float(current_mem_freq))]

        # Generate partition count candidates
        current_n = len(config.p_list)
        partition_count_candidates = self._partition_count_candidates(current_n)

        best_eic = -1.0
        best_candidate = None
        cur_key = (
            tuple(float(x) for x in config.p_list),
            tuple(int(x) for x in config.b_list),
            float(current_sm_freq),
            float(current_mem_freq),
        )
        t_best = float(best_throughput) if best_throughput is not None else float(config.throughput)

        # Explore different partition counts
        for n_partitions in partition_count_candidates:
            # Generate base configurations for this partition count
            base_configs = self._generate_config_for_partition_count(
                n_partitions, config.p_list, config.b_list
            )

            for base_p_list, base_b_list in base_configs:
                # For each base config, explore MPS adjustments
                if n_partitions == current_n:
                    # Same partition count: use existing MPS candidate generation
                    mps_candidates = self._mps_candidates(base_p_list)
                else:
                    # Different partition count: just use the base config
                    mps_candidates = [base_p_list]

                for sm_freq in sm_freq_candidates:
                    for mem_freq in mem_freq_candidates:
                        for p_list in mps_candidates:
                            batch_candidates = self._batch_neighbors(p_list, base_b_list)
                            for b_list in batch_candidates:
                                # Don't "suggest" the current config; noisy models can otherwise pick it repeatedly.
                                cand_key = (
                                    tuple(float(x) for x in p_list),
                                    tuple(int(x) for x in b_list),
                                    float(sm_freq),
                                    float(mem_freq),
                                )
                                if cand_key == cur_key:
                                    continue

                                if self.use_prior:
                                    assert self.predictor is not None  # for type-checkers
                                    _, _, base_t, base_p = self.predictor.predict_all(
                                        p_list, b_list, sm_freq, mem_freq=mem_freq
                                    )
                                else:
                                    base_t, base_p = 0.0, 0.0

                                mu_t, sigma_t = float(base_t), 0.0
                                mu_p, sigma_p = float(base_p), 0.0
                                # Only use GP if it has been fit at least once; otherwise sklearn raises NotFittedError.
                                gp_ready = (
                                    self.gp_t
                                    and self.gp_p
                                    and hasattr(self.gp_t, "X_train_")
                                    and hasattr(self.gp_p, "X_train_")
                                    and len(self.history_x) >= self.gp_min_points
                                )
                                if gp_ready:
                                    x = self._encode(p_list, b_list, sm_freq, mem_freq).reshape(1, -1)
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        mu_t_gp, sigma_t_gp = self.gp_t.predict(x, return_std=True)
                                        mu_p_gp, sigma_p_gp = self.gp_p.predict(x, return_std=True)
                                    mu_t = float(base_t + self._as_scalar0(mu_t_gp))
                                    mu_p = float(base_p + self._as_scalar0(mu_p_gp))
                                    sigma_t = self._as_scalar0(sigma_t_gp)
                                    sigma_p = self._as_scalar0(sigma_p_gp)

                                eic = self._expected_improvement(mu_t, sigma_t, t_best)
                                eic *= self._feasibility_prob(mu_p, sigma_p)

                                if eic > best_eic:
                                    best_eic = eic
                                    best_candidate = ShadowConfig(
                                        p_list=list(p_list),
                                        b_list=list(b_list),
                                        sm_freq=float(sm_freq),
                                        mem_freq=float(mem_freq),
                                        throughput=float(mu_t),
                                        power=float(mu_p),
                                    )

        if best_candidate is None:
            return None
        if conservative:
            # Use dynamic threshold if adaptive_threshold is enabled
            threshold = self._get_dynamic_threshold() if self.adaptive_threshold else self.reconfig_threshold
            if (best_candidate.throughput - config.throughput) > threshold * max(config.throughput, 1.0):
                return best_candidate
            return None
        return best_candidate
