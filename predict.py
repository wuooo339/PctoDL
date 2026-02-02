#!/usr/bin/env python3
"""
Prediction module for PctoDL trained models.
This module loads and uses the models trained by optimize_model_params.py.

Usage:
    from predict import PctoDLPredictor
    predictor = PctoDLPredictor('mobilenet_v2')
    U_sm, U_mem, throughput, power = predictor.predict_all([50, 25], [16, 16], 1950)
"""

import os
import json
import sys
import subprocess
import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Compatibility fix for loading old sklearn models that reference _loss module
# This module was removed/refactored in newer sklearn versions
import importlib.util
import types

def _install_sklearn_loss_stub():
    # Only used when sklearn._loss.* is missing; do not override real sklearn modules.
    _loss_package = types.ModuleType('sklearn._loss')
    sys.modules['sklearn._loss'] = _loss_package
    sys.modules['_loss'] = _loss_package

    _loss_module = types.ModuleType('sklearn._loss.loss')
    _loss_package.loss = _loss_module
    sys.modules['sklearn._loss.loss'] = _loss_module
    sys.modules['_loss.loss'] = _loss_module

    _glm_dist_module = types.ModuleType('sklearn._loss.glm_distribution')
    _loss_package.glm_distribution = _glm_dist_module
    sys.modules['sklearn._loss.glm_distribution'] = _glm_dist_module
    sys.modules['_loss.glm_distribution'] = _glm_dist_module

    class _Distribution:
        def __init__(self):
            pass

    class TweedieDistribution(_Distribution):
        """Stub for TweedieDistribution."""
        def __init__(self, power=1.0):
            self.power = power

    class NormalDistribution(_Distribution):
        """Stub for NormalDistribution."""
        pass

    class PoissonDistribution(_Distribution):
        """Stub for PoissonDistribution."""
        pass

    class GammaDistribution(_Distribution):
        """Stub for GammaDistribution."""
        pass

    class BinomialDistribution(_Distribution):
        """Stub for BinomialDistribution."""
        pass

    _glm_dist_module.TweedieDistribution = TweedieDistribution
    _glm_dist_module.NormalDistribution = NormalDistribution
    _glm_dist_module.PoissonDistribution = PoissonDistribution
    _glm_dist_module.GammaDistribution = GammaDistribution
    _glm_dist_module.BinomialDistribution = BinomialDistribution

    class HalfSquaredError:
        """Stub for HalfSquaredError loss function."""
        def __init__(self, sample_weight=None):
            self.sample_weight = sample_weight

    _loss_module.HalfSquaredError = HalfSquaredError
    _loss_module.AbsoluteError = type('AbsoluteError', (), {})
    _loss_module.SquaredError = type('SquaredError', (), {})
    _loss_module.PinholeLoss = type('PinholeLoss', (), {})
    _loss_module.HuberLoss = type('HuberLoss', (), {})
    _loss_module.PinballLoss = type('PinballLoss', (), {})
    _loss_module.HalfPoissonLoss = type('HalfPoissonLoss', (), {})
    _loss_module.HalfGammaLoss = type('HalfGammaLoss', (), {})
    _loss_module.HalfTweedieLoss = type('HalfTweedieLoss', (), {})
    _loss_module.HalfTweedieLossIdentity = type('HalfTweedieLossIdentity', (), {})
    _loss_module.HalfBinomialLoss = type('HalfBinomialLoss', (), {})
    _loss_module.HalfMultinomialLoss = type('HalfMultinomialLoss', (), {})

if importlib.util.find_spec('sklearn._loss.loss') is None:
    _install_sklearn_loss_stub()

def _install_numpy_module_aliases():
    # Legacy pickles may reference numpy._core.* modules.
    try:
        import numpy.core as _np_core
        import numpy.core.numeric as _np_core_numeric
        sys.modules.setdefault('numpy._core', _np_core)
        sys.modules.setdefault('numpy._core.numeric', _np_core_numeric)
    except Exception:
        pass

_install_numpy_module_aliases()

def _ensure_numpy_bit_generator_pickle_compat():
    # Older pickles may pass the BitGenerator class instead of its name.
    try:
        from numpy.random import _pickle as np_random_pickle
        from numpy.random import MT19937, PCG64, Philox, SFC64
        for cls in (MT19937, PCG64, Philox, SFC64):
            if cls not in np_random_pickle.BitGenerators:
                np_random_pickle.BitGenerators[cls] = cls
    except Exception:
        pass

_ensure_numpy_bit_generator_pickle_compat()

# Configuration constants
F_REF = 1950.0  # MHz (anchor frequency)
F_MIN = 210.0   # MHz (minimum frequency)
MAX_TASKS = 20

# Default memory frequency (fallback)
DEFAULT_MEM_FREQ = 9501.0  # MHz

# Model name aliases (predict.py name -> profile_mps.py / doInference.py name)
MODEL_ALIASES = {
    'maxvit': 'maxvit_t',
    'mobilenetv2': 'mobilenet_v2',
}

# All supported models for --model all
# Must match models in fitting_results/ and mps_profile/mps_results_*.csv
ALL_MODELS = [
    'densenet201',
    'efficientnet_v2_m',
    'maxvit_t',
    'mnasnet',
    'mobilenet_v2',
    'resnet50',
    'vgg19',
]


# =========================================================================
#  Auto-Detection Utilities
# =========================================================================

def _normalize_platform_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum()) if name else ""

def _load_platform_clocks(platform_name: str | None) -> tuple[list[float], list[float]]:
    if not platform_name:
        return [], []
    config_path = os.path.join(os.path.dirname(__file__), "src", "config", "platform.json")
    if not os.path.exists(config_path):
        return [], []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return [], []

    platforms = data.get("platforms", [])
    if not platforms:
        return [], []

    target_key = _normalize_platform_key(platform_name)
    for platform in platforms:
        name = platform.get("name", "")
        name_key = _normalize_platform_key(name)
        if target_key and (target_key == name_key or target_key in name_key or name_key in target_key):
            clocks = platform.get("clocks", {})
            sm_clocks = clocks.get("SM_Clocks") or clocks.get("Graphics_Clocks", [])
            mem_clocks = clocks.get("Memory_Clocks") or clocks.get("Memory_Clock", [])
            if isinstance(sm_clocks, (int, float)):
                sm_clocks = [sm_clocks]
            if isinstance(mem_clocks, (int, float)):
                mem_clocks = [mem_clocks]
            return [float(x) for x in sm_clocks], [float(x) for x in mem_clocks]
    return [], []

def get_gpu_memory_clock() -> float:
    """
    Auto-detect GPU memory clock frequency using nvidia-smi.
    Returns memory frequency in MHz, or DEFAULT_MEM_FREQ if detection fails.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=clocks.mem', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # Output format: "xxx MHz"
            output = result.stdout.strip().splitlines()[0] if result.stdout else ""
            freq_str = output.replace('MHz', '').strip()
            if freq_str:
                freq = float(freq_str)
                print(f"[Info] Auto-detected memory frequency: {freq} MHz")
                return freq
    except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e:
        pass

    print(f"[Warning] Could not auto-detect memory frequency, using default: {DEFAULT_MEM_FREQ} MHz")
    return DEFAULT_MEM_FREQ


def detect_platform_name() -> str | None:
    """
    Auto-detect platform name for model suffix matching.
    Returns lowercase platform name (e.g., "a100") or None if unknown.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() <= 0:
            return None
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode('utf-8')
        name_lower = gpu_name.lower()
        if "a100" in name_lower:
            return "a100"
        if "3080" in name_lower and "ti" in name_lower:
            return "3080ti"
        if "3090" in name_lower:
            return "3090"
        if "a10" in name_lower:
            return "a10"
    except Exception:
        return None
    return None


# =========================================================================
#  Model Definitions (must match optimize_model_params.py)
# =========================================================================

class Saturation(nn.Module):
    """Adaptive Sharpened Saturation Function"""
    def __init__(self, init_nu: float = 2.0):
        super().__init__()
        self.raw_nu = nn.Parameter(torch.tensor(init_nu))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nu = nn.functional.softplus(self.raw_nu)
        denominator = torch.pow(1.0 + torch.pow(x + 1e-8, nu), 1.0 / nu)
        return x / (denominator + 1e-8)


class SMBaseDemandModel(nn.Module):
    """
    Base SM Utilization Model (Single-Task)
    Formula: u_sm^base(r, b) = 100 * tanh((λ_sm * ln(b) + β_sm) / r^γ_sm)
    Physical Limit: 100 (tanh asymptotically approaches 1)
    Saturation: tanh((λ_sm * ln(b) + β_sm) / r^γ_sm)
    """
    def __init__(self):
        super().__init__()
        self.raw_lambda_sm = nn.Parameter(torch.tensor(0.01))
        self.raw_gamma_sm = nn.Parameter(torch.tensor(0.5))
        self.raw_beta_sm = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r, b = x[:, 0], x[:, 1]
        lambda_sm = torch.abs(self.raw_lambda_sm)
        gamma_sm = torch.abs(self.raw_gamma_sm)
        beta_sm = self.raw_beta_sm  # Can be negative

        # Saturation: tanh((λ_sm * ln(b) + β_sm) / r^γ_sm)
        r_norm = torch.clamp(r, min=1.0)
        log_b = torch.log(b + 1e-8)
        saturation = torch.tanh((lambda_sm * log_b + beta_sm) / (r_norm ** gamma_sm))

        # Physical limit: 100
        return 100.0 * saturation


class SMSystemModel(nn.Module):
    """
    Multi-Task SM Utilization with Anchor-Based Probabilistic Model
    Per Spec §3.1.1 Stage 2 - SM Manifold: Anchor-Based Probabilistic Model

    Core Logic: Two-Phase Reconstruction
    Phase 1: Determine dynamic saturation limit (The Ceiling)
    Phase 2: Model probabilistic contention path (The Path)

    Mathematical Formulation (Updated):
    - Phase 1 (Dynamic Ceiling):
      U_limit = L(u) + (100 - L(u)) * tanh(kappa * xi) / tanh(kappa)
      where xi = (u - u_ref) / (100 - u_ref)
      and u_ref = u_SM^base(r_max=40, b) for each batch

    - Phase 2 (Probabilistic Superposition):
      Û_SM^ref({u_i}) = Û_SM^limit · [1 - ∏(1 - u_i^base/Û_SM^limit)]

    Where:
    - avg_u_base = (1/N) * sum(u_sm,i^base): average baseline utilization
    - L(u) = alpha * u + beta: Linear Anchor Locus from offline profiling
    - Û_SM^limit: dynamic physical ceiling
    - kappa: curvature invariant (single global parameter)
    """
    def __init__(self, alpha_init=1.0, beta_init=0.0, kappa_init=2.0):
        super().__init__()
        # Linear Anchor Locus parameters: L(u) = alpha * u + beta
        self.register_buffer('alpha', torch.tensor(alpha_init, dtype=torch.float32))
        self.register_buffer('beta', torch.tensor(beta_init, dtype=torch.float32))
        # Kappa: curvature invariant
        self.register_buffer('kappa', torch.tensor(kappa_init, dtype=torch.float32))

    def forward(self, u_base_list: torch.Tensor, r_list: torch.Tensor, b_list: torch.Tensor,
                N: torch.Tensor, u_ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_base_list: [batch, MAX_TASKS] base utilizations
            r_list: [batch, MAX_TASKS] partition ratios (MPS percentages)
            b_list: [batch, MAX_TASKS] batch sizes
            N: [batch] number of active tasks
            u_ref: [batch] reference baseline u_SM^base(r_max=40, b) for each batch

        Returns:
            U_sm_ref: [batch] system-wide SM utilization reference
        """
        batch_size = u_base_list.shape[0]
        max_tasks = u_base_list.shape[1]

        # Create active mask: [batch, MAX_TASKS]
        task_indices = torch.arange(max_tasks, device=u_base_list.device).float()
        active_mask = (task_indices.unsqueeze(0) < N.unsqueeze(1)).float()

        # Sum of base utilizations for active tasks
        sum_u_base = torch.sum(u_base_list * active_mask, dim=1)

        # Average baseline: (1/N) * sum(u_sm,i^base)
        avg_u_base = sum_u_base / (N + 1e-6)

        # Get parameters
        alpha = self.alpha
        beta = self.beta
        kappa = self.kappa

        # Step 1: Compute xi = (u_base - u_ref) / (100 - u_ref)
        xi = (avg_u_base - u_ref) / (100.0 - u_ref + 1e-6)
        xi = torch.clamp(xi, 0.0, 1.0)

        # Step 2: Compute L(u) = alpha * u + beta
        L_u = alpha * avg_u_base + beta

        # Step 3: Compute U_limit = L(u) + (100 - L(u)) * tanh(kappa * xi) / tanh(kappa)
        U_SM_limit = L_u + (100.0 - L_u) * torch.tanh(kappa * xi) / torch.tanh(kappa)

        # Step 4: Probabilistic Superposition
        u_base_masked = u_base_list * active_mask
        p_occupy = u_base_masked / (U_SM_limit.unsqueeze(1) + 1e-6)
        p_occupy = torch.clamp(p_occupy, 0.0, 1.0)

        p_idle = 1.0 - p_occupy
        log_p_idle = torch.log(p_idle + 1e-10) * active_mask
        log_prob_idle_product = torch.sum(log_p_idle, dim=1)
        prob_idle_product = torch.exp(log_prob_idle_product)

        # System-wide utilization
        U_sm_ref = U_SM_limit * (1.0 - prob_idle_product)

        return U_sm_ref


class MEMBaseDemandModel(nn.Module):
    """
    Base MEM Utilization Model (Single-Task)
    Formula: u_mem^base(r, b) = (φ_mem * r^α_mem) * tanh(λ_mem * ln(b) + β_mem)
    Elastic Capacity: φ_mem * r^α_mem
    Kinetic Saturation: tanh(λ_mem * ln(b) + β_mem)
    """
    def __init__(self):
        super().__init__()
        self.raw_phi_mem = nn.Parameter(torch.tensor(10.0))
        self.raw_alpha_mem = nn.Parameter(torch.tensor(0.5))
        self.raw_lambda_mem = nn.Parameter(torch.tensor(0.01))
        self.raw_beta_mem = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r, b = x[:, 0], x[:, 1]
        phi_mem = nn.functional.softplus(self.raw_phi_mem)
        alpha_mem = torch.abs(self.raw_alpha_mem)
        lambda_mem = torch.abs(self.raw_lambda_mem)
        beta_mem = self.raw_beta_mem  # Can be negative

        # Elastic capacity: φ_mem * r^α_mem
        r_norm = torch.clamp(r / 100.0, 1e-6, 1.0)
        capacity = phi_mem * (r_norm ** alpha_mem)

        # Kinetic saturation: tanh(λ_mem * ln(b) + β_mem)
        log_b = torch.log(b + 1e-8)
        saturation = torch.tanh(lambda_mem * log_b + beta_mem)

        return capacity * saturation


class MEMSystemModel(nn.Module):
    """
    Multi-Task MEM Utilization with Partition-Aware Point-Slope Manifold
    Per Spec §3.1.1 Stage 2:

    U_mem^ref(N, r_bar, b_bar) = (1/N) * sum(u_mem,i^base) + K(r_bar, b_bar) * (N - 1)

    Where the slope coefficient K(r_bar, b_bar) = r_bar * k(b_bar), and k(b_bar) is
    a regime-switching function with threshold tau:

    - Linear regime (b_bar <= tau): k = k1 * b_bar + k2
    - Saturation-drift regime (b_bar > tau): k = alpha * (1 - exp(-beta * b_bar)) + mu * (b_bar - tau) + C

    With continuity constraint at tau:
    C = (k1 * tau + k2) - alpha * (1 - exp(-beta * tau))

    This captures additive interference penalty where slope depends on partition
    size (r_bar) and workload intensity (b_bar).
    """
    def __init__(self, max_tasks: int = 20):
        super().__init__()
        self.max_tasks = max_tasks

        # Linear regime parameters (k1, k2)
        self.raw_k1 = nn.Parameter(torch.tensor(0.01))
        self.raw_k2 = nn.Parameter(torch.tensor(1.0))

        # Saturation-drift regime parameters (alpha, beta, mu)
        self.raw_alpha = nn.Parameter(torch.tensor(5.0))
        self.raw_beta = nn.Parameter(torch.tensor(0.1))
        self.raw_mu = nn.Parameter(torch.tensor(0.01))

        # Threshold parameter (tau) - learnable switching point
        self.raw_tau = nn.Parameter(torch.tensor(16.0))

    def forward(self, u_base_list: torch.Tensor, r_list: torch.Tensor, b_list: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_base_list: [batch, MAX_TASKS] base utilizations
            r_list: [batch, MAX_TASKS] partition ratios (MPS percentages)
            b_list: [batch, MAX_TASKS] batch sizes
            N: [batch] number of active tasks

        Returns:
            U_mem_ref: [batch] system-wide memory utilization reference
        """
        batch_size = u_base_list.shape[0]
        max_tasks = u_base_list.shape[1]

        # Create active mask: [batch, MAX_TASKS]
        task_indices = torch.arange(max_tasks, device=u_base_list.device).float()
        active_mask = (task_indices.unsqueeze(0) < N.unsqueeze(1)).float()

        # Sum of base utilizations for active tasks
        sum_u_mem = torch.sum(u_base_list * active_mask, dim=1)

        # Average baseline: (1/N) * sum(u_mem,i^base)
        avg_u_mem = sum_u_mem / (N + 1e-6)

        # Calculate mean partition ratio r_bar for active tasks
        masked_r = r_list * active_mask
        sum_r = torch.sum(masked_r, dim=1)
        r_bar = sum_r / (N + 1e-6)

        # Calculate mean batch intensity b_bar for active tasks
        masked_b = b_list * active_mask
        sum_b = torch.sum(masked_b, dim=1)
        b_bar = sum_b / (N + 1e-6)

        # Get parameters
        k1 = torch.abs(self.raw_k1)
        k2 = torch.abs(self.raw_k2)
        alpha = torch.abs(self.raw_alpha)
        beta = torch.abs(self.raw_beta)
        mu = torch.abs(self.raw_mu)
        tau = torch.abs(self.raw_tau)

        # Compute continuity constraint C
        # C = (k1 * tau + k2) - alpha * (1 - exp(-beta * tau))
        C = (k1 * tau + k2) - alpha * (1.0 - torch.exp(-beta * tau))

        # Piecewise function k(b_bar)
        # Linear regime: b_bar <= tau
        k_linear = k1 * b_bar + k2

        # Saturation-drift regime: b_bar > tau
        # alpha * (1 - exp(-beta * b_bar)) + mu * (b_bar - tau) + C
        k_saturation = alpha * (1.0 - torch.exp(-beta * b_bar)) + mu * (b_bar - tau) + C

        # Switch between regimes based on tau
        k_b = torch.where(b_bar <= tau, k_linear, k_saturation)

        # Slope coefficient: K(r_bar, b_bar) = r_bar * k(b_bar)
        # Note: Per spec, K = r_bar * k(b_bar) without additional scaling
        K = r_bar * k_b

        # System-wide reference: avg_u_mem + K * (N - 1)
        # When N=1, interference term vanishes
        U_mem_ref = avg_u_mem + K * (N - 1.0)

        return U_mem_ref


class ThroughputKineticModel(nn.Module):
    """Throughput Model with Frequency Scaling"""
    def __init__(self, f_ref: float = F_REF, f_min: float = F_MIN):
        super().__init__()
        self.f_ref = f_ref
        self.f_min = f_min
        self.raw_kappa_bw = nn.Parameter(torch.tensor(50.0))
        self.raw_gamma_t = nn.Parameter(torch.tensor(0.8))
        self.raw_T_start = nn.Parameter(torch.tensor(100.0))

    def forward(self, U_mem: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        kappa_bw = nn.functional.softplus(self.raw_kappa_bw)
        gamma_t = torch.abs(self.raw_gamma_t)
        T_start = nn.functional.softplus(self.raw_T_start)

        T_ref = kappa_bw * U_mem
        freq_ratio = torch.clamp((f - self.f_min) / (self.f_ref - self.f_min), min=0.0, max=1.0)
        T_f = T_start + (T_ref - T_start) * torch.pow(freq_ratio, gamma_t)
        return T_f


# =========================================================================
#  Main Predictor Class
# =========================================================================

class PctoDLPredictor:
    """
    Load and use trained PctoDL models for prediction.

    Usage:
        # Default prediction (uses default models)
        predictor = PctoDLPredictor('mobilenet_v2')
        U_sm, U_mem, throughput, power = predictor.predict_all([50, 25], [16, 16], 1950)

        # Prediction with specific dataset
        predictor = PctoDLPredictor('mobilenet_v2', dataset='imagenet')
        U_sm, U_mem, throughput, power = predictor.predict_all([50, 25], [16, 16], 1950)

        # Prediction with specific platform (e.g., A100)
        predictor = PctoDLPredictor('mobilenet_v2', platform='a100')
        U_sm, U_mem, throughput, power = predictor.predict_all([50, 25], [16, 16], 1950)

        # Prediction with specific memory frequency
        predictor = PctoDLPredictor('mobilenet_v2', mem_freq=9501)
        U_sm, U_mem, throughput, power = predictor.predict_all([50, 25], [16, 16], 1950)

        # Prediction with all options
        predictor = PctoDLPredictor('mobilenet_v2', dataset='imagenet', platform='a100', mem_freq=9501)
        U_sm, U_mem, throughput, power = predictor.predict_all([50, 25], [16, 16], 1950)
    """

    def __init__(self, model_name: str, model_dir: str = "fitting_results",
                 mem_freq: float = None, platform: str = None, dataset: str = None):
        """
        Initialize predictor.

        Args:
            model_name: Name of the model (e.g., 'mobilenet_v2')
            model_dir: Directory containing trained models
            mem_freq: Memory frequency (MHz). If specified, uses models trained for this frequency.
            platform: Platform/GPU name (e.g., 'a100', '3090').
            dataset: Dataset name (e.g., 'imagenet', 'coco').
        """
        self.model_name = model_name
        self.actual_name = MODEL_ALIASES.get(model_name, model_name)
        self.model_dir = os.path.join(model_dir, self.actual_name)
        self.platform = platform.lower() if isinstance(platform, str) else detect_platform_name()
        self.dataset = dataset.lower() if isinstance(dataset, str) else dataset

        # Auto-detect default mem_freq for the specified platform if not provided
        if mem_freq is None and self.platform:
            _, mem_clocks = _load_platform_clocks(self.platform)
            if mem_clocks:
                mem_freq = max(mem_clocks)
                print(f"[Info] Using platform.json mem_freq for {self.platform}: {int(mem_freq)} MHz")
            elif self.platform == 'a100':
                mem_freq = 1215
                print(f"[Info] Auto-detected mem_freq for A100: {mem_freq} MHz")

        self.mem_freq = mem_freq
        # Build suffix: order is {dataset}_{platform}_mem{xxx}
        self.dataset_suffix = f"_{self.dataset}" if self.dataset else ""
        self.platform_suffix = f"_{self.platform}" if self.platform else ""
        self.mem_freq_suffix = f"_mem{int(mem_freq)}" if mem_freq else ""
        self.full_suffix = self.dataset_suffix + self.platform_suffix + self.mem_freq_suffix

        # Verify directory exists
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                f"Available models: {', '.join(os.listdir(model_dir)) if os.path.exists(model_dir) else 'none'}"
            )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model complexity for workload signature
        self._load_model_complexity()

        # Load all model parameters
        self._load_models()
        self._load_params()

    def _get_model_path(self, base_name: str) -> str:
        """Get path for model file, checking for full suffix (dataset + platform + mem_freq)."""
        # Try full suffix first (e.g., "_imagenet_a100_mem9501")
        if self.full_suffix:
            suffix_path = os.path.join(self.model_dir, f"{base_name}{self.full_suffix}.pth")
            if os.path.exists(suffix_path):
                return suffix_path
            suffix_pkl = os.path.join(self.model_dir, f"{base_name}{self.full_suffix}.pkl")
            if os.path.exists(suffix_pkl):
                return suffix_pkl

        # Try dataset + platform suffix
        dataset_platform = self.dataset_suffix + self.platform_suffix
        if dataset_platform:
            suffix_path = os.path.join(self.model_dir, f"{base_name}{dataset_platform}.pth")
            if os.path.exists(suffix_path):
                return suffix_path
            suffix_pkl = os.path.join(self.model_dir, f"{base_name}{dataset_platform}.pkl")
            if os.path.exists(suffix_pkl):
                return suffix_pkl

        # Try platform only suffix
        if self.platform_suffix:
            suffix_path = os.path.join(self.model_dir, f"{base_name}{self.platform_suffix}.pth")
            if os.path.exists(suffix_path):
                return suffix_path
            suffix_pkl = os.path.join(self.model_dir, f"{base_name}{self.platform_suffix}.pkl")
            if os.path.exists(suffix_pkl):
                return suffix_pkl

        # Try platform + mem_freq suffix (e.g., "_a100_mem1215")
        if self.platform_suffix and self.mem_freq_suffix:
            platform_memfreq = self.platform_suffix + self.mem_freq_suffix
            suffix_path = os.path.join(self.model_dir, f"{base_name}{platform_memfreq}.pth")
            if os.path.exists(suffix_path):
                return suffix_path
            suffix_pkl = os.path.join(self.model_dir, f"{base_name}{platform_memfreq}.pkl")
            if os.path.exists(suffix_pkl):
                return suffix_pkl

        # Try dataset + mem_freq suffix (e.g., "_imagenet_mem9501")
        if self.dataset_suffix and self.mem_freq_suffix:
            dataset_memfreq = self.dataset_suffix + self.mem_freq_suffix
            suffix_path = os.path.join(self.model_dir, f"{base_name}{dataset_memfreq}.pth")
            if os.path.exists(suffix_path):
                return suffix_path
            suffix_pkl = os.path.join(self.model_dir, f"{base_name}{dataset_memfreq}.pkl")
            if os.path.exists(suffix_pkl):
                return suffix_pkl

        # Try mem_freq only suffix
        if self.mem_freq_suffix:
            suffix_path = os.path.join(self.model_dir, f"{base_name}{self.mem_freq_suffix}.pth")
            if os.path.exists(suffix_path):
                return suffix_path
            suffix_pkl = os.path.join(self.model_dir, f"{base_name}{self.mem_freq_suffix}.pkl")
            if os.path.exists(suffix_pkl):
                return suffix_pkl

        # Fall back to default model
        default_path = os.path.join(self.model_dir, f"{base_name}.pth")
        if os.path.exists(default_path):
            return default_path
        return os.path.join(self.model_dir, f"{base_name}.pkl")

    def _get_json_path(self, base_name: str) -> str:
        """Get path for JSON params file, checking for full suffix first."""
        # Try full suffix first (e.g., "_imagenet_a100_mem9501")
        if self.full_suffix:
            suffix_path = os.path.join(self.model_dir, f"{base_name}{self.full_suffix}.json")
            if os.path.exists(suffix_path):
                return suffix_path

        # Try dataset + platform suffix
        dataset_platform = self.dataset_suffix + self.platform_suffix
        if dataset_platform:
            suffix_path = os.path.join(self.model_dir, f"{base_name}{dataset_platform}.json")
            if os.path.exists(suffix_path):
                return suffix_path

        # Try platform only suffix
        if self.platform_suffix:
            suffix_path = os.path.join(self.model_dir, f"{base_name}{self.platform_suffix}.json")
            if os.path.exists(suffix_path):
                return suffix_path

        # Try platform + mem_freq suffix (e.g., "_a100_mem1215")
        if self.platform_suffix and self.mem_freq_suffix:
            platform_memfreq = self.platform_suffix + self.mem_freq_suffix
            suffix_path = os.path.join(self.model_dir, f"{base_name}{platform_memfreq}.json")
            if os.path.exists(suffix_path):
                return suffix_path

        # Try dataset + mem_freq suffix (e.g., "_imagenet_mem9501")
        if self.dataset_suffix and self.mem_freq_suffix:
            dataset_memfreq = self.dataset_suffix + self.mem_freq_suffix
            suffix_path = os.path.join(self.model_dir, f"{base_name}{dataset_memfreq}.json")
            if os.path.exists(suffix_path):
                return suffix_path

        # Try mem_freq only suffix
        if self.mem_freq_suffix:
            suffix_path = os.path.join(self.model_dir, f"{base_name}{self.mem_freq_suffix}.json")
            if os.path.exists(suffix_path):
                return suffix_path

        # Fall back to default params
        default_path = os.path.join(self.model_dir, f"{base_name}.json")
        if os.path.exists(default_path):
            return default_path
        return None

    def _load_params(self):
        """Load JSON parameters."""
        # Load throughput parameters
        tp_path = self._get_json_path("throughput_params")
        if tp_path:
            with open(tp_path, 'r') as f:
                self.tp_params = json.load(f)
        else:
            self.tp_params = {'kappa_ideal': 50.0, 'gamma_t': 0.8, 'T_start': 100.0, 'f_ref': F_REF, 'f_min': F_MIN}

        # Load power parameters
        power_path = self._get_json_path("power_params")
        if power_path:
            with open(power_path, 'r') as f:
                self.power_params = json.load(f)
        else:
            # Default power parameters (fallback)
            self.power_params = {
                'static': {'delta_3': 2.31e-8, 'delta_2': -5.44e-5, 'delta_1': 0.039, 'delta_0': 85.6},
                'dynamic': {'gamma_p': 3.0, 'P_base': 10.0, 'f_ref': F_REF, 'f_min': F_MIN}
            }

    def _load_model_complexity(self):
        """Load model complexity for workload signature (I_workload)."""
        # Try multiple paths for model_table.csv
        paths = [
            os.path.join('profile', 'model_table.csv'),
            os.path.join('..', 'profile', 'model_table.csv'),
            os.path.join('src', 'profile', 'model_table.csv'),
        ]

        self.gflops = 1000.0  # Default ~1G FLOPs
        self.params_M = 25.0  # Default ~25M params

        for path in paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    for _, row in df.iterrows():
                        ref = row['Reference'].lower()
                        actual_name_lower = self.actual_name.lower()
                        if ref == actual_name_lower:
                            self.gflops = float(row['GFLOPS'])
                            self.params_M = float(row['Parameters'].replace('M', ''))
                            break
                    break  # Found and loaded
                except Exception:
                    continue

    def _load_models(self):
        """Load all trained models with memory frequency support."""
        # Load L-curve parameters
        l_curve_path = self._get_json_path("sm_l_curve")
        self.l_curve_params = {'L_alpha': 0.85, 'L_beta': 15.0}
        if l_curve_path:
            with open(l_curve_path, 'r') as f:
                self.l_curve_params = json.load(f)

        # Load kappa parameters
        kappa_path = self._get_json_path("sm_kappa")
        self.kappa_params = {'kappa': 2.0}
        if kappa_path:
            with open(kappa_path, 'r') as f:
                self.kappa_params = json.load(f)

        # SM Base Model
        self.sm_base = None
        sm_base_path = self._get_model_path("sm_base_model")
        if os.path.exists(sm_base_path):
            self.sm_base = SMBaseDemandModel()
            self.sm_base.load_state_dict(torch.load(sm_base_path, map_location=self.device))
            self.sm_base.eval()

        # SM System Model
        self.sm_sys = None
        sm_sys_path = self._get_model_path("sm_system_model")
        if os.path.exists(sm_sys_path):
            # Load with new parameters
            L_alpha = self.l_curve_params.get('L_alpha', 0.85)
            L_beta = self.l_curve_params.get('L_beta', 15.0)
            kappa = self.kappa_params.get('kappa', 2.0)
            self.sm_sys = SMSystemModel(alpha_init=L_alpha, beta_init=L_beta, kappa_init=kappa)
            self.sm_sys.load_state_dict(torch.load(sm_sys_path, map_location=self.device))
            self.sm_sys.eval()

        # SM GBR
        self.sm_gbr = None
        sm_gbr_path = self._get_model_path("sm_gbr_model")
        if os.path.exists(sm_gbr_path):
            self.sm_gbr = joblib.load(sm_gbr_path)

        # MEM Base Model
        self.mem_base = None
        mem_base_path = self._get_model_path("mem_base_model")
        if os.path.exists(mem_base_path):
            self.mem_base = MEMBaseDemandModel()
            self.mem_base.load_state_dict(torch.load(mem_base_path, map_location=self.device))
            self.mem_base.eval()

        # MEM System Model
        self.mem_sys = None
        mem_sys_path = self._get_model_path("mem_system_model")
        if os.path.exists(mem_sys_path):
            self.mem_sys = MEMSystemModel()
            self.mem_sys.load_state_dict(torch.load(mem_sys_path, map_location=self.device))
            self.mem_sys.eval()

        # MEM GBR
        self.mem_gbr = None
        mem_gbr_path = self._get_model_path("mem_gbr_model")
        if os.path.exists(mem_gbr_path):
            self.mem_gbr = joblib.load(mem_gbr_path)

        # Throughput Kinetic Model (base model with frequency scaling)
        self.throughput_model = None
        tp_path = self._get_model_path("throughput_kinetic_model")
        if os.path.exists(tp_path):
            self.throughput_model = ThroughputKineticModel()
            self.throughput_model.load_state_dict(torch.load(tp_path, map_location=self.device))
            self.throughput_model.eval()

        # Throughput GBR Model (residual calibration)
        self.throughput_gbr = None
        tp_gbr_path = self._get_model_path("throughput_gbr_model")
        if os.path.exists(tp_gbr_path):
            self.throughput_gbr = joblib.load(tp_gbr_path)

        # Power Dynamic Anchor GBR
        self.power_dyn_gbr = None
        power_gbr_path = self._get_model_path("power_dyn_anchor_gbr")
        if os.path.exists(power_gbr_path):
            self.power_dyn_gbr = joblib.load(power_gbr_path)

    def predict_sm(self, p_list, b_list):
        """Predict SM utilization."""
        N = len(p_list)

        # Stage 1: Base Demand
        u_bases = []
        for pi, bi in zip(p_list, b_list):
            if self.sm_base is None:
                # Fallback: simple estimate
                u_bases.append(pi * 0.8)
            else:
                with torch.no_grad():
                    xt = torch.tensor([[pi, bi]], dtype=torch.float32).to(self.device)
                    u_bases.append(float(self.sm_base(xt).item()))

        # Compute u_ref for each batch: u_ref = u_SM^base(r=40, b, N=1)
        # This is the reference baseline (single-task at partition=40%)
        u_ref_list = []
        for bi in b_list:
            if self.sm_base is None:
                u_ref_list.append(40.0)  # Fallback
            else:
                with torch.no_grad():
                    xt = torch.tensor([[40.0, bi]], dtype=torch.float32).to(self.device)
                    u_ref_list.append(float(self.sm_base(xt).item()))

        # Stage 2: Physics Prior (Dynamic Ceiling System Model)
        if self.sm_sys and N > 1:
            u_base_padded = u_bases + [0.0] * (MAX_TASKS - N)
            r_base_padded = [float(p) for p in p_list] + [0.0] * (MAX_TASKS - N)
            b_base_padded = [float(b) for b in b_list] + [0.0] * (MAX_TASKS - N)
            u_base_tensor = torch.tensor([u_base_padded], dtype=torch.float32).to(self.device)
            r_base_tensor = torch.tensor([r_base_padded], dtype=torch.float32).to(self.device)
            b_base_tensor = torch.tensor([b_base_padded], dtype=torch.float32).to(self.device)
            N_tensor = torch.tensor([N], dtype=torch.float32).to(self.device)
            u_ref_tensor = torch.tensor([u_ref_list[0]], dtype=torch.float32).to(self.device)  # Use first batch's u_ref
            with torch.no_grad():
                U_prior = float(self.sm_sys(u_base_tensor, r_base_tensor, b_base_tensor, N_tensor, u_ref_tensor).item())
        else:
            U_prior = sum(u_bases)

        # Stage 3: Residual Calibration (with extended features per Spec §3.1.1)
        if self.sm_gbr:
            linear_sum = sum(u_bases)
            var_r = np.var(p_list) if len(p_list) > 1 else 0.0
            var_b = np.var(b_list) if len(b_list) > 1 else 0.0
            b_mean = sum(b_list) / N if N > 0 else 1.0
            b_sum = sum(b_list)
            p_mean = sum(p_list) / N if N > 0 else 50.0
            p_sum = sum(p_list)

            # Memory Context: ω_mem (use predictor mem_freq when available)
            mem_freq = self.mem_freq if self.mem_freq is not None else DEFAULT_MEM_FREQ

            # Workload Signature: I_workload (arithmetic intensity profile)
            compute_memory_ratio = self.gflops / (self.params_M + 1.0)

            x_gbr = np.array([[
                U_prior,                      # Physics Prior: Û_sm^ref
                linear_sum,                   # Linear Sum: Σu_sm,i
                float(N),                     # Concurrency: N
                var_r,                        # Heterogeneity: Var(r)
                var_b,                        # Heterogeneity: Var(b)
                b_mean,                       # Batch: mean batch size
                b_sum,                        # Batch: total batch size
                p_mean,                       # Partition: mean percentage
                p_sum,                        # Partition: total percentage
                float(mem_freq) / 10000.0,    # Memory Context: ω_mem (normalized)
                self.gflops / 1000.0,         # Workload: GFLOPS (normalized)
                self.params_M / 100.0,        # Workload: Parameters (normalized)
                compute_memory_ratio,         # Workload: Compute-to-memory ratio
            ]])
            U_sm = float(self.sm_gbr.predict(x_gbr)[0])
        else:
            U_sm = U_prior

        return U_sm

    def predict_mem(self, p_list, b_list):
        """Predict Memory utilization."""
        N = len(p_list)

        # Stage 1: Base Demand
        u_bases = []
        for pi, bi in zip(p_list, b_list):
            if self.mem_base is None:
                # Fallback: simple estimate
                u_bases.append(pi * 0.3)
            else:
                with torch.no_grad():
                    xt = torch.tensor([[pi, bi]], dtype=torch.float32).to(self.device)
                    u_bases.append(float(self.mem_base(xt).item()))

        # Stage 2: Physics Prior (Partition-Aware Point-Slope Manifold)
        if self.mem_sys and N > 1:
            u_base_padded = u_bases + [0.0] * (MAX_TASKS - N)
            r_base_padded = [float(p) for p in p_list] + [0.0] * (MAX_TASKS - N)
            b_base_padded = [float(b) for b in b_list] + [0.0] * (MAX_TASKS - N)
            u_base_tensor = torch.tensor([u_base_padded], dtype=torch.float32).to(self.device)
            r_base_tensor = torch.tensor([r_base_padded], dtype=torch.float32).to(self.device)
            b_base_tensor = torch.tensor([b_base_padded], dtype=torch.float32).to(self.device)
            N_tensor = torch.tensor([N], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                U_prior = float(self.mem_sys(u_base_tensor, r_base_tensor, b_base_tensor, N_tensor).item())
        else:
            U_prior = sum(u_bases)

        # Stage 3: Residual Calibration (with extended features per Spec §3.1.1)
        if self.mem_gbr:
            linear_sum = sum(u_bases)
            var_r = np.var(p_list) if len(p_list) > 1 else 0.0
            var_b = np.var(b_list) if len(b_list) > 1 else 0.0
            b_mean = sum(b_list) / N if N > 0 else 1.0
            b_sum = sum(b_list)

            # Memory Context: ω_mem (use predictor mem_freq when available)
            mem_freq = self.mem_freq if self.mem_freq is not None else DEFAULT_MEM_FREQ

            # Workload Signature: I_workload (arithmetic intensity profile)
            compute_memory_ratio = self.gflops / (self.params_M + 1.0)

            x_gbr = np.array([[
                U_prior,                      # Physics Prior: Û_mem^ref
                linear_sum,                   # Linear Sum: Σu_mem,i
                float(N),                     # Concurrency: N
                var_r,                        # Heterogeneity: Var(r)
                var_b,                        # Heterogeneity: Var(b)
                b_mean,                       # Batch: mean batch size
                b_sum,                        # Batch: total batch size
                float(mem_freq) / 10000.0,    # Memory Context: ω_mem (normalized)
                self.gflops / 1000.0,         # Workload: GFLOPS (normalized)
                self.params_M / 100.0,        # Workload: Parameters (normalized)
                compute_memory_ratio,         # Workload: Compute-to-memory ratio
            ]])
            U_mem = float(self.mem_gbr.predict(x_gbr)[0])
        else:
            U_mem = U_prior

        return U_mem

    def predict_metrics_batch(self, p_lists, b_lists):
        """
        Batch helper for greedy_partitioning.py vectorized search.

        This is correctness-first: it reuses predict_sm/predict_mem per config.
        It enables greedy_partitioning.py --vectorized without changing model
        behavior. If you need more speed, this can be optimized later by
        vectorizing the underlying torch/GBR computations.
        """
        if len(p_lists) != len(b_lists):
            raise ValueError(f"p_lists and b_lists length mismatch: {len(p_lists)} vs {len(b_lists)}")

        n = len(p_lists)
        U_sms = np.zeros(n, dtype=float)
        U_mems = np.zeros(n, dtype=float)
        for i, (p_list, b_list) in enumerate(zip(p_lists, b_lists)):
            U_sms[i] = float(self.predict_sm(p_list, b_list))
            U_mems[i] = float(self.predict_mem(p_list, b_list))
        return U_sms, U_mems

    def predict_throughput(self, U_mem, f, p_list=None, b_list=None):
        """
        Predict throughput at given frequency.

        Uses Prior-Correction Strategy:
        1. Compute theoretical prior: T_prior = kappa_ideal(batch) * U_mem
        2. Apply GBR residual calibration
        3. Apply kinetic frequency scaling

        Args:
            U_mem: Memory utilization (%)
            f: SM frequency (MHz)
            p_list: List of MPS percentages (needed for GBR)
            b_list: List of batch sizes (needed for GBR)
        """
        # Load batch-aware kappa parameters
        kappa_a = self.tp_params.get('kappa_a', 0.0)
        kappa_b = self.tp_params.get('kappa_b', self.tp_params.get('kappa_ideal', 50.0))
        batch_aware = self.tp_params.get('batch_aware_kappa', False)
        gamma_t = self.tp_params.get('gamma_t', 0.8)
        T_start = self.tp_params.get('T_start', 100.0)
        f_ref = self.tp_params.get('f_ref', F_REF)
        f_min = self.tp_params.get('f_min', F_MIN)

        # Compute batch-aware kappa_ideal using batch mean
        if b_list is not None and len(b_list) > 0 and batch_aware:
            # Always use batch mean for kappa calculation
            batch_array = np.array(b_list, dtype=float)
            batch_mean = np.mean(batch_array)
            kappa_ideal = kappa_a * np.log(batch_mean + 1e-8) + kappa_b
        else:
            kappa_ideal = kappa_b

        # Step 1: Compute theoretical prior at reference frequency
        T_prior = kappa_ideal * U_mem

        # Step 2: Apply GBR residual calibration
        if self.throughput_gbr is not None and p_list is not None and b_list is not None:
            N = len(p_list)
            mem_freq = self.mem_freq if self.mem_freq else DEFAULT_MEM_FREQ

            # Workload signature features
            compute_memory_ratio = self.gflops / (self.params_M + 1.0)

            # Batch-related features
            batch_first = float(b_list[0]) if len(b_list) > 0 else 1.0
            batch_sum = sum(b_list)
            batch_mean = batch_sum / N if N > 0 else 1.0

            # Partition-related features
            p_first = float(p_list[0]) if len(p_list) > 0 else 50.0
            p_sum = sum(p_list)
            p_mean = p_sum / N if N > 0 else 50.0
            p_var = np.var(p_list) if len(p_list) > 1 else 0.0

            # Build GBR features: [T_prior, U_mem, N, mem_freq, gflops, params_M, compute_ratio, batch_*, partition_*]
            x_gbr = np.array([[
                T_prior,                      # Theoretical Prior: T̂_prior
                U_mem,                        # MEM Utilization: U_mem^ref
                float(N),                     # Concurrency: N
                float(mem_freq) / 10000.0,    # Memory Context: ω_mem (normalized)
                self.gflops / 1000.0,         # Workload: GFLOPS (normalized)
                self.params_M / 100.0,        # Workload: Parameters (normalized)
                compute_memory_ratio,         # Workload: Compute-to-memory ratio
                batch_first,                  # Batch: first task batch size
                batch_sum,                    # Batch: total batch size
                batch_mean,                   # Batch: mean batch size
                p_first,                      # Partition: first task percentage
                p_sum,                        # Partition: total percentage
                p_mean,                       # Partition: mean percentage
                p_var,                        # Partition: variance
            ]])
            T_ref = float(self.throughput_gbr.predict(x_gbr)[0])
        else:
            # Fallback to simple prior
            T_ref = T_prior

        # Step 3: Apply kinetic frequency scaling
        # For memory-bound workloads (like on A100), throughput is less dependent on SM frequency
        # Use formula: T(f) = T_ref * (f / f_ref)^γ_t for compute-bound, or nearly constant for memory-bound
        # We use a hybrid approach based on T_start value:
        # - If T_start is close to 0: multiplicative scaling (T = T_ref * (f/f_ref)^γ_t)
        # - If T_start is significant: additive scaling (original formula)
        freq_ratio = np.clip((f - f_min) / (f_ref - f_min), 0.0, 1.0)

        # Use multiplicative scaling for better generalization, especially for memory-bound workloads
        # This avoids the issue where T_start is too low causing excessive scaling
        if T_start < T_ref * 0.2:  # T_start is small relative to T_ref
            # Multiplicative scaling: T(f) = T_ref * (f/f_ref)^γ_t
            T = T_ref * np.power(freq_ratio, gamma_t * 0.5)  # Reduced sensitivity
        else:
            # Original additive scaling
            T = T_start + (T_ref - T_start) * np.power(freq_ratio, gamma_t)

        # Debug info
        if os.environ.get("PCTODL_DEBUG") == "1" and (T < 5000 or T > 15000):
            print(f"  [DEBUG] U_mem={U_mem:.1f}, κ_ideal={kappa_ideal:.2f}, T_prior={T_prior:.1f}, T_ref={T_ref:.1f}, T_start={T_start:.1f}, f={f}, γ_t={gamma_t:.2f}, T={T:.1f}")

        return float(T)

    def predict_power_static(self, f):
        """
        Predict static power at given frequency.

        P_static(f) = delta_3 * f^3 + delta_2 * f^2 + delta_1 * f + delta_0
        Parameters are loaded from power_params.json (platform and memory frequency specific).
        """
        static = self.power_params.get('static', {})
        delta_3 = static.get('delta_3', 2.31e-8)
        delta_2 = static.get('delta_2', -5.44e-5)
        delta_1 = static.get('delta_1', 0.039)
        delta_0 = static.get('delta_0', 85.6)
        return delta_3 * f**3 + delta_2 * f**2 + delta_1 * f + delta_0

    def predict_power_dynamic(self, U_sm, U_mem, N, f, p_list=None, b_list=None):
        """Predict dynamic power at given frequency."""
        # Get parameters
        gamma_p = self.power_params['dynamic'].get('gamma_p', 3.0)
        P_base = self.power_params['dynamic'].get('P_base', 10.0)
        f_ref = self.power_params['dynamic'].get('f_ref', F_REF)
        f_min = self.power_params['dynamic'].get('f_min', F_MIN)

        if self.power_dyn_gbr is None:
            # Fallback: simple scaling model
            P_static = self.predict_power_static(f)
            P_dyn_ref = 100.0  # Default
            freq_ratio = np.clip((f - f_min) / (f_ref - f_min), 0.0, 1.0)
            P_dyn = P_base + (P_dyn_ref - P_base) * np.power(freq_ratio, gamma_p)
            return P_static + float(P_dyn)

        # Use GBR to predict P_dyn at anchor
        # Features: [sm_util, mem_util, N, total_S, p_mean, p_var, b_mean, mem_freq]
        # total_S = N * params_M (using self.params_M from complexity table)
        total_S = float(N) * self.params_M

        # Partition and batch features for power model
        if p_list is None:
            p_list = [50.0] * N
        p_mean = sum(p_list) / N if N > 0 else 50.0
        p_var = np.var(p_list) if N > 1 else 0.0

        if b_list is None:
            b_list = [1] * N
        b_mean = sum(b_list) / N if N > 0 else 1.0

        # Memory frequency (normalized to 10000 MHz scale)
        mem_freq = float(self.mem_freq) / 10000.0 if self.mem_freq else 0.5

        x_gbr = np.array([[U_sm, U_mem, float(N), total_S, p_mean, p_var, b_mean, mem_freq]])
        P_dyn_ref = float(self.power_dyn_gbr.predict(x_gbr)[0])

        # Apply frequency scaling
        freq_ratio = np.clip((f - f_min) / (f_ref - f_min), 0.0, 1.0)
        P_dyn = P_base + (P_dyn_ref - P_base) * np.power(freq_ratio, gamma_p)

        return self.predict_power_static(f) + P_dyn

    def predict_all(self, p_list, b_list, f, mem_freq=None):
        """
        Predict all metrics: U_sm, U_mem, throughput, power.

        Args:
            p_list: List of MPS percentages (e.g., [50, 25])
            b_list: List of batch sizes (e.g., [16, 16])
            f: SM Frequency in MHz (e.g., 1950)
            mem_freq: Memory frequency in MHz. If specified, uses models trained for this frequency.
                     If None, uses the predictor's default (or platform-specific) models.

        Returns:
            (U_sm, U_mem, throughput, power)
        """
        # If mem_freq is specified and different from the predictor's setting,
        # we need to use a predictor with the correct mem_freq suffix
        if mem_freq is not None and mem_freq != self.mem_freq:
            # Create a temporary predictor with the specified mem_freq
            temp_predictor = PctoDLPredictor(
                self.model_name,
                mem_freq=mem_freq,
                platform=self.platform,
                dataset=self.dataset
            )
            return temp_predictor.predict_all(p_list, b_list, f, mem_freq=mem_freq)

        N = len(p_list)

        # Predict utilizations
        U_sm = self.predict_sm(p_list, b_list)
        U_mem = self.predict_mem(p_list, b_list)

        # Predict throughput (pass p_list and b_list for GBR residual calibration)
        throughput = self.predict_throughput(U_mem, f, p_list, b_list)

        # Predict power
        power = self.predict_power_dynamic(U_sm, U_mem, N, f, p_list, b_list)

        return U_sm, U_mem, throughput, power


# =========================================================================
#  Command-line Interface
# =========================================================================

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Predict using trained PctoDL models")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., mobilenet_v2) or 'all' to test all 7 models")
    parser.add_argument("--p", type=float, nargs="+", help="MPS percentages (e.g., 50 25)")
    parser.add_argument("--b", type=int, nargs="+", help="Batch sizes (e.g., 16 16)")
    parser.add_argument("--f", type=float, default=1950, help="SM frequency in MHz (default: 1950)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g., imagenet, coco). Uses corresponding dataset-specific models")
    parser.add_argument("--platform", type=str, default=None,
                        help="Platform/GPU model (e.g., a100, 3090, 3080ti). Uses corresponding platform-specific models")
    parser.add_argument("--mem-freq", type=int, default=None,
                        help="Memory frequency in MHz. If specified, uses models trained for this frequency")
    parser.add_argument("--gpu-mem-freq", action="store_true",
                        help="Auto-detect GPU memory frequency and use corresponding model")
    parser.add_argument("--test", action="store_true",
                        help="Run test on random data points from mps_profile files")
    parser.add_argument("--sample", type=int, default=100,
                        help="Number of random samples to test (default: 100)")
    parser.add_argument("--use-actual-util", action="store_true",
                        help="Use actual SM/MEM utilization from data for throughput/power prediction")
    args = parser.parse_args()

    # Validate arguments
    if args.model.lower() != 'all' and args.model.lower() not in [m.lower() for m in ALL_MODELS] + list(MODEL_ALIASES.keys()):
        parser.error(f"Invalid model: {args.model}. Use 'all' or one of: {', '.join(ALL_MODELS)}")

    if not args.test:
        if args.p is None or args.b is None:
            parser.error("arguments --p and --b are required when --test is not used")

    # Determine platform, dataset and memory frequency for model selection
    model_dataset = args.dataset
    if not model_dataset:
        model_dataset = "imagenet"
    model_platform = args.platform
    model_mem_freq = None

    if not args.test:
        # Prediction mode
        if args.gpu_mem_freq:
            model_mem_freq = get_gpu_memory_clock()
            print(f"[Info] Dataset: {model_dataset or 'default'}, Platform: {model_platform or 'default'}, "
                  f"Memory frequency: auto-detected = {model_mem_freq} MHz")
        elif args.mem_freq is not None:
            model_mem_freq = float(args.mem_freq)
            print(f"[Info] Dataset: {model_dataset or 'default'}, Platform: {model_platform or 'default'}, "
                  f"Memory frequency: {int(model_mem_freq)} MHz")
        else:
            print(f"[Info] Dataset: {model_dataset or 'default'}, Platform: {model_platform or 'default'}")

        # Create predictor with all options
        predictor = PctoDLPredictor(args.model, mem_freq=model_mem_freq,
                                    platform=model_platform, dataset=model_dataset)

        # Make prediction
        pred_sm, pred_mem, pred_throughput, pred_power = predictor.predict_all(args.p, args.b, args.f)

        # Print results
        print("=" * 60)
        print(f"Prediction for {args.model}")
        if model_dataset:
            print(f"Dataset: {model_dataset}")
        if model_platform:
            print(f"Platform: {model_platform}")
        if model_mem_freq:
            print(f"Model: memory frequency = {int(model_mem_freq)} MHz")
        print("=" * 60)
        print(f"Configuration: p_list={args.p}, b_list={args.b}, f={args.f} MHz")
        if args.gpu_mem_freq:
            print(f"GPU Memory Frequency: {int(get_gpu_memory_clock())} MHz")
        print("-" * 60)
        print(f"SM Utilization:  {pred_sm:.2f}%")
        print(f"MEM Utilization: {pred_mem:.2f}%")
        print(f"Throughput:      {pred_throughput:.2f} samples/s")
        print(f"Power:           {pred_power:.2f} W")
        print("=" * 60)

        # Verification command hint
        p_str = ' '.join(str(int(p)) for p in args.p)
        b_str = ' '.join(str(b) for b in args.b)
        # Use alias if exists, otherwise use original name
        profile_model = MODEL_ALIASES.get(args.model, args.model)
        print("\nTo verify, run:")
        print(f"  python src/profile_mps.py --single --models {profile_model} --dataset ImageNet \\")
        print(f"      --p {p_str} --b {b_str} --f {int(args.f)} \\")
        if args.dataset:
            print(f"      --dataset {args.dataset} \\")
        if args.platform:
            print(f"      --machine {args.platform} \\")
        if args.mem_freq:
            print(f"      --mem-freq {int(args.mem_freq)}")
        print("=" * 60)

    else:
        # =========================================================================
        #  Test Mode
        # =========================================================================
        print("=" * 70)
        print("PREDICTION TEST MODE")
        print("=" * 70)
        print(f"Models: {args.model}")
        print(f"Dataset: {model_dataset or 'default'}")
        print(f"Platform: {model_platform or 'default'}")
        print(f"Samples: {args.sample}")
        print("=" * 70)

        # Determine which models to test
        if args.model.lower() == 'all':
            models_to_test = ALL_MODELS.copy()
        else:
            models_to_test = [args.model]

        # Find mps_profile directory
        script_dir = Path(__file__).parent
        mps_dir = script_dir / 'mps_profile'

        if not mps_dir.exists():
            print(f"Error: mps_profile directory not found at {mps_dir}")
            sys.exit(1)

        # Find matching CSV files based on dataset and platform
        csv_files = []
        for f in mps_dir.glob('mps_results*.csv'):
            fname = f.name

            # Parse filename to get dataset and platform
            parts = fname.replace('mps_results', '').replace('.csv', '').strip('_').split('_')
            file_dataset = 'default'
            file_platform = 'default'

            known_machines = ['a100', 'v100', 'a40', 'a6000', '3080ti', '3090', '3080', '4090', 'a10', 'a30']
            known_datasets = ['imagenet', 'coco', 'voc', 'kitti', 'bert', 'llama', 'pubmed',
                              'cifar', 'caltech256', 'fashionmnist', 'mnist', 'imagenette']

            for part in parts:
                part_lower = part.lower()
                if part_lower in known_machines:
                    file_platform = part_lower
                elif part_lower in known_datasets:
                    file_dataset = part_lower

            # Filter by dataset and platform
            matches_dataset = model_dataset is None or file_dataset == model_dataset
            matches_platform = model_platform is None or file_platform == model_platform

            if matches_dataset and matches_platform:
                csv_files.append(f)

        if not csv_files:
            print(f"Error: No matching data files found for dataset={model_dataset}, platform={model_platform}")
            sys.exit(1)

        print(f"Found {len(csv_files)} matching data files:")
        for f in csv_files:
            print(f"  - {f.name}")

        # Load all data
        all_data = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df['_source'] = csv_file.name
            all_data.append(df)

        test_df = pd.concat(all_data, ignore_index=True)
        print(f"Total data points: {len(test_df)}")

        # Group by memory frequency
        if 'mem_freq' not in test_df.columns:
            print("Warning: mem_freq column not found, using default frequency")
            test_df['mem_freq'] = DEFAULT_MEM_FREQ

        # Model alias lookup
        model_aliases = {k.lower(): v.lower() for k, v in MODEL_ALIASES.items()}

        # Run test for each model
        all_results = []
        model_metrics = {}
        output_dir = Path('image/predict_error')
        output_dir.mkdir(parents=True, exist_ok=True)

        for current_model in models_to_test:
            print(f"\n{'#'*70}")
            print(f"# Testing model: {current_model}")
            print(f"{'#'*70}")

            # Normalize model name for comparison
            model_name = current_model.lower()
            if model_name in model_aliases:
                model_name = model_aliases[model_name]

            # Filter data for this model
            model_df = test_df[test_df['model_name'].str.lower() == model_name]

            if len(model_df) == 0:
                print(f"Warning: No data found for model '{current_model}', skipping...")
                continue

            print(f"Model data points: {len(model_df)}")

            # Get unique memory frequencies for this model
            mem_freqs = model_df['mem_freq'].unique()
            print(f"Memory frequencies: {list(mem_freqs)}")

            # Process each memory frequency group
            model_results = []
            for mem_freq in mem_freqs:
                mem_freq = float(mem_freq)
                group = model_df[model_df['mem_freq'] == mem_freq]
                group_df = group.copy()
                n_samples = len(group_df)

                # Sample if needed
                if n_samples > args.sample:
                    sample_indices = np.random.choice(n_samples, args.sample, replace=False)
                    group_df = group_df.iloc[sample_indices].reset_index(drop=True)
                    print(f"  MemFreq {int(mem_freq)} MHz: Sampled {args.sample} from {n_samples} points")
                else:
                    print(f"  MemFreq {int(mem_freq)} MHz: Using all {n_samples} points")

                # Detect platform from data source (auto-detect from filename)
                source_files = group_df['_source'].unique()
                detected_platform = model_platform  # User-specified or None
                for src in source_files:
                    if 'a100' in src.lower():
                        detected_platform = 'a100'
                        break
                    elif 'v100' in src.lower():
                        detected_platform = 'v100'
                        break

                # If still None, infer from memory frequency when possible
                if detected_platform is None:
                    if int(mem_freq) == 1215:
                        detected_platform = 'a100'
                    else:
                        detected_platform = '3080ti'

                # Create predictor
                predictor = PctoDLPredictor(current_model, mem_freq=int(mem_freq),
                                            platform=detected_platform, dataset=model_dataset)

                # Print model loading status
                print(f"    Model Status:")
                print(f"      SM:  base={'✓' if predictor.sm_base else '✗'}, sys={'✓' if predictor.sm_sys else '✗'}, gbr={'✓' if predictor.sm_gbr else '✗'}")
                print(f"      MEM: base={'✓' if predictor.mem_base else '✗'}, sys={'✓' if predictor.mem_sys else '✗'}, gbr={'✓' if predictor.mem_gbr else '✗'}")
                print(f"      TP:  gbr={'✓' if predictor.throughput_gbr else '✗'}")
                print(f"      PWR: gbr={'✓' if predictor.power_dyn_gbr else '✗'}")

                # Run predictions
                use_actual = args.use_actual_util
                for idx, row in group_df.iterrows():
                    try:
                        p_str = str(row['p_list'])
                        b_str = str(row['b_list'])
                        p_list = eval(p_str) if p_str.startswith('[') else [float(p_str)]
                        b_list = eval(b_str) if b_str.startswith('[') else [int(b_str)]

                        f = float(row['sm_freq'])

                        actual_sm = float(row['sm_util'])
                        actual_mem = float(row['mem_util'])
                        actual_throughput = float(row['throughput_total'])
                        actual_power = float(row['power_avg'])

                        if use_actual:
                            pred_sm = actual_sm
                            pred_mem = actual_mem
                            pred_throughput = predictor.predict_throughput(actual_mem, f, p_list, b_list)
                            pred_power = predictor.predict_power_dynamic(actual_sm, actual_mem, len(p_list), f, p_list, b_list)
                        else:
                            pred_sm, pred_mem, pred_throughput, pred_power = predictor.predict_all(p_list, b_list, f)

                        model_results.append({
                            'model': current_model,
                            'mem_freq': int(mem_freq),
                            'p_list': p_list,
                            'b_list': b_list,
                            'f': f,
                            'pred_sm': pred_sm,
                            'pred_mem': pred_mem,
                            'pred_throughput': pred_throughput,
                            'pred_power': pred_power,
                            'actual_sm': actual_sm,
                            'actual_mem': actual_mem,
                            'actual_throughput': actual_throughput,
                            'actual_power': actual_power,
                            'source': row['_source']
                        })
                    except Exception as e:
                        print(f"      ERROR at idx {idx}: {e}")
                        continue

            if not model_results:
                print(f"Warning: No valid predictions for model '{current_model}'")
                continue

            print(f"  Completed {len(model_results)} predictions for {current_model}")

            # Compute metrics for this model
            pred_sm_arr = np.array([r['pred_sm'] for r in model_results])
            pred_mem_arr = np.array([r['pred_mem'] for r in model_results])
            pred_tp_arr = np.array([r['pred_throughput'] for r in model_results])
            pred_power_arr = np.array([r['pred_power'] for r in model_results])
            actual_sm_arr = np.array([r['actual_sm'] for r in model_results])
            actual_mem_arr = np.array([r['actual_mem'] for r in model_results])
            actual_tp_arr = np.array([r['actual_throughput'] for r in model_results])
            actual_power_arr = np.array([r['actual_power'] for r in model_results])

            def compute_metrics(pred, actual, name):
                mae = np.mean(np.abs(pred - actual))
                rmse = np.sqrt(np.mean((pred - actual) ** 2))
                mape = np.mean(np.abs(pred - actual) / np.maximum(actual, 0.1)) * 100
                ss_res = np.sum((actual - pred) ** 2)
                ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                return {'name': name, 'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}

            model_metrics[current_model] = {
                'sm_util': compute_metrics(pred_sm_arr, actual_sm_arr, 'SM Util (%)'),
                'mem_util': compute_metrics(pred_mem_arr, actual_mem_arr, 'MEM Util (%)'),
                'throughput': compute_metrics(pred_tp_arr, actual_tp_arr, 'Throughput (img/s)'),
                'power': compute_metrics(pred_power_arr, actual_power_arr, 'Power (W)'),
                'n_samples': len(model_results)
            }

            # Save individual model results
            results_df = pd.DataFrame(model_results)
            csv_path = output_dir / f'predict_results_{current_model}.csv'
            results_df.to_csv(csv_path, index=False)

            # Save individual model metrics
            json_path = output_dir / f'predict_metrics_{current_model}.json'
            with open(json_path, 'w') as f:
                json.dump({
                    'model': current_model,
                    'dataset': model_dataset,
                    'platform': model_platform,
                    'n_samples': len(model_results),
                    **model_metrics[current_model]
                }, f, indent=2)

            all_results.extend(model_results)

        if not all_results:
            print("Error: No valid predictions made for any model")
            sys.exit(1)

        # Generate suffix from dataset and platform
        suffix = ''
        if model_dataset:
            suffix += f'_{model_dataset}'
        if model_platform:
            suffix += f'_{model_platform}'
        if not suffix:
            suffix = '_default'

        # Print summary for all models
        print("\n" + "=" * 80)
        print("ALL MODELS PREDICTION SUMMARY")
        print("=" * 80)

        # Print per-model metrics
        for model_name, metrics in model_metrics.items():
            print(f"\n{model_name} ({metrics['n_samples']} samples):")
            for key in ['sm_util', 'mem_util', 'throughput', 'power']:
                m = metrics[key]
                print(f"  {m['name']:20s}: MAE={m['mae']:8.4f}, RMSE={m['rmse']:8.4f}, MAPE={m['mape']:6.2f}%, R²={m['r2']:.4f}")

        # Aggregate all results
        pred_sm_arr = np.array([r['pred_sm'] for r in all_results])
        pred_mem_arr = np.array([r['pred_mem'] for r in all_results])
        pred_tp_arr = np.array([r['pred_throughput'] for r in all_results])
        pred_power_arr = np.array([r['pred_power'] for r in all_results])
        actual_sm_arr = np.array([r['actual_sm'] for r in all_results])
        actual_mem_arr = np.array([r['actual_mem'] for r in all_results])
        actual_tp_arr = np.array([r['actual_throughput'] for r in all_results])
        actual_power_arr = np.array([r['actual_power'] for r in all_results])

        def compute_metrics(pred, actual, name):
            mae = np.mean(np.abs(pred - actual))
            rmse = np.sqrt(np.mean((pred - actual) ** 2))
            mape = np.mean(np.abs(pred - actual) / np.maximum(actual, 0.1)) * 100
            ss_res = np.sum((actual - pred) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            return {'name': name, 'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}

        metrics_sm = compute_metrics(pred_sm_arr, actual_sm_arr, 'SM Util (%)')
        metrics_mem = compute_metrics(pred_mem_arr, actual_mem_arr, 'MEM Util (%)')
        metrics_tp = compute_metrics(pred_tp_arr, actual_tp_arr, 'Throughput (img/s)')
        metrics_power = compute_metrics(pred_power_arr, actual_power_arr, 'Power (W)')

        print(f"\n{'='*80}")
        print("AGGREGATE METRICS (All Models)")
        print(f"{'='*80}")
        for m in [metrics_sm, metrics_mem, metrics_tp, metrics_power]:
            print(f"\n{m['name']}:")
            print(f"  MAE:  {m['mae']:.4f}")
            print(f"  RMSE: {m['rmse']:.4f}")
            print(f"  MAPE: {m['mape']:.2f}%")
            print(f"  R²:   {m['r2']:.4f}")

        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        def plot_actual_pred(ax, pred, actual, title, color):
            ax.scatter(actual, pred, alpha=0.6, edgecolors='k', linewidth=0.5, s=30, c=color)
            min_val = min(actual.min(), pred.min())
            max_val = max(actual.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect fit')
            ax.set_xlabel('Actual', fontsize=10)
            ax.set_ylabel('Predicted', fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        def plot_residuals(ax, pred, actual, title, color):
            residuals = pred - actual
            ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color=color)
            ax.axvline(x=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('Residual (Pred - Actual)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3)

        plot_actual_pred(axes[0, 0], pred_sm_arr, actual_sm_arr, f'SM Utilization (R²={metrics_sm["r2"]:.3f})', 'steelblue')
        plot_residuals(axes[1, 0], pred_sm_arr, actual_sm_arr, f'Residuals (MAE={metrics_sm["mae"]:.2f}%)', 'steelblue')

        plot_actual_pred(axes[0, 1], pred_mem_arr, actual_mem_arr, f'MEM Utilization (R²={metrics_mem["r2"]:.3f})', 'coral')
        plot_residuals(axes[1, 1], pred_mem_arr, actual_mem_arr, f'Residuals (MAE={metrics_mem["mae"]:.2f}%)', 'coral')

        plot_actual_pred(axes[0, 2], pred_tp_arr, actual_tp_arr, f'Throughput (R²={metrics_tp["r2"]:.3f})', 'mediumseagreen')
        plot_residuals(axes[1, 2], pred_tp_arr, actual_tp_arr, f'Residuals (MAE={metrics_tp["mae"]:.2f})', 'mediumseagreen')

        plot_actual_pred(axes[0, 3], pred_power_arr, actual_power_arr, f'Power (R²={metrics_power["r2"]:.3f})', 'brown')
        plot_residuals(axes[1, 3], pred_power_arr, actual_power_arr, f'Residuals (MAE={metrics_power["mae"]:.2f}W)', 'brown')

        plt.suptitle(f'PctoDL Prediction Test - All Models{suffix} ({len(all_results)} samples)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save aggregate figure
        fig_path = output_dir / f'predict_error_all_models{suffix}.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved aggregate visualization to {fig_path}")

        # Save aggregate results to CSV
        results_df = pd.DataFrame(all_results)
        csv_path = output_dir / f'predict_results_all_models{suffix}.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"Saved aggregate results to {csv_path}")

        # Save aggregate metrics to JSON
        aggregate_metrics = {
            'models_tested': models_to_test,
            'dataset': model_dataset,
            'platform': model_platform,
            'n_samples': len(all_results),
            'sm_util': metrics_sm,
            'mem_util': metrics_mem,
            'throughput': metrics_tp,
            'power': metrics_power,
            'per_model': model_metrics
        }
        json_path = output_dir / f'predict_metrics_all_models{suffix}.json'
        with open(json_path, 'w') as f:
            json.dump(aggregate_metrics, f, indent=2)
        print(f"Saved aggregate metrics to {json_path}")

        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)
