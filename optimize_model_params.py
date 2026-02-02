#!/usr/bin/env python3
"""
Physics-Informed Parameter Optimization using PyTorch
Implements PctoDL_System_Spec.md §3.1 Performance Model (Anchor-Based)

Training Process (Strictly Separated Stages):
  SM Model (Fixed Freq at Anchor):
    1. L-Curve Fitting: Fit L(u) = alpha * u + beta using lower envelope points
    2. Base Demand: Fit u_base(r, b) using single-task data at anchor frequency
    3. k_b Curve Fitting: Fit k_b per batch using U_limit = 100 * tanh(k_b * u_base)
    4. Physics Prior: Train SMSystemModel with per-batch k_b lookup
    5. Residual Calibration: Fit GBR using physics prior as feature

  MEM Model (Fixed Freq at Anchor):
    1. Base Demand: Fit u_mem_base(r, b) using single-task data
    2. Physics Prior: Project to multi-tenant priors
    3. Residual Calibration: Fit GBR

  Throughput Model (Kinetic Scaling):
    1. Reference Mapping: Fit κ_bw on Anchor Data (T_ref = κ_bw * U_mem)
    2. Frequency Scaling: Fit γ_t on Scaling Data

  Power Model (Thermodynamic):
    1. Static Baseline: Fit P_static(f) on Static Data (cubic polynomial)
    2. Dynamic Anchor: Train GBR on Anchor Data
    3. Dynamic Scaling: Fit γ_p on Scaling Data
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import joblib

# =========================================================================
#  Configuration
# =========================================================================

OUTPUT_DIR = "fitting_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Anchor frequency (fixed high-performance state)
F_REF = 1950.0  # MHz
F_MIN = 210.0   # MHz (minimum frequency in scaling data)

# Fixed memory frequency
MEM_FREQ_FIXED = 9501.0  # MHz

# =========================================================================
#  Physics-Informed Neural Network Models
# =========================================================================

class Saturation(nn.Module):
    """Adaptive Sharpened Saturation Function S(x; ν) = x / (1 + x^ν)^(1/ν)"""
    def __init__(self, init_nu: float = 2.0):
        super().__init__()
        self.raw_nu = nn.Parameter(torch.tensor(init_nu))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nu = nn.functional.softplus(self.raw_nu)
        denominator = torch.pow(1.0 + torch.pow(x + 1e-8, nu), 1.0 / nu)
        return x / (denominator + 1e-8)

# =========================================================================
#  SM/MEM Models (Fixed Frequency at Anchor)
# =========================================================================

class SMBaseDemandModel(nn.Module):
    """
    Base SM Utilization Model (Single-Task, Fixed Freq)
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

    Mathematical Formulation (STRICTLY per spec):
    - Phase 1 (Dynamic Ceiling):
      Û_SM^limit = 100 * tanh(k_b * avg_u_base)

      where k_b is numerically solved to intersect the Linear Anchor Locus:
      L(u) = alpha * u + beta defines the constraint that k_b must satisfy.

      For a given (u_initial, u_final) pair where:
      - u_initial = avg_u_base at N=1
      - u_final = system utilization at N=max
      We solve for k_b such that the probabilistic formula produces u_final.

    - Phase 2 (Probabilistic Superposition):
      Û_SM^ref({u_i}) = Û_SM^limit · [1 - ∏(1 - u_i^base/Û_SM^limit)]

    Physics Consistency:
    - At N=1: Û_SM^ref = u_1^base (strictly reproduces baseline)
    - At N→∞: Û_SM^ref → Û_SM^limit (asymptotically saturates at dynamic ceiling)
    - Heterogeneity: Naturally handles different task intensities (u_i) without uniform assumptions

    Implementation Approach:
    - k_b is learned as a function of avg_u_base: k_b = f(avg_u_base)
    - During training, we pre-compute target k_b values via numerical solving
    - The model learns to predict k_b from avg_u_base using a learnable function

    L-Curve Parameters (Linear Anchor Locus):
    - Stored as learnable parameters: L(u) = alpha * u + beta
    - These define the linear constraint that k_b must intersect
    - Saved to model checkpoint for consistency with task_num_draw.py visualization
    """
    def __init__(self, L_alpha: float = 1.0, L_beta: float = 0.0):
        super().__init__()
        # L-Curve parameters: Linear Anchor Locus L(u) = alpha * u + beta
        # These define the constraint that k_b must satisfy
        self.L_alpha = nn.Parameter(torch.tensor(L_alpha))  # Slope of L-curve
        self.L_beta = nn.Parameter(torch.tensor(L_beta))    # Intercept of L-curve

        # Learnable function that maps avg_u_base to k_b
        # k_b determines the curvature of the saturation function
        # Per spec: U_SM^limit = 100 * tanh(k_b * avg_u_base)
        self.raw_k_base = nn.Parameter(torch.tensor(0.01))  # Base k value
        self.raw_k_scale = nn.Parameter(torch.tensor(0.0))  # Scaling with avg_u_base

        # Learnable adjustment factor for fine-tuning predictions
        self.raw_scaling = nn.Parameter(torch.tensor(0.0))  # Small adjustment around 1.0

    def compute_k_b(self, avg_u_base: torch.Tensor) -> torch.Tensor:
        """
        Compute k_b as a function of avg_u_base.
        Per spec: k_b is numerically solved to intersect the Linear Anchor Locus.

        This function learns to predict k_b from avg_u_base based on
        pre-computed target values during training.

        Formula: k_b = exp(k_base) * (1 + k_scale * avg_u_base / 100)
        """
        k_base = torch.exp(self.raw_k_base)  # Ensure positive
        k_scale = torch.sigmoid(self.raw_k_scale) * 0.5  # Small adjustment
        k_b = k_base * (1.0 + k_scale * avg_u_base / 100.0)
        return k_b

    def get_L_curve_params(self) -> Tuple[float, float]:
        """
        Get the L-curve parameters (Linear Anchor Locus).

        Returns:
            alpha: Slope of the linear anchor locus L(u) = alpha * u + beta
            beta: Intercept of the linear anchor locus
        """
        return self.L_alpha.item(), self.L_beta.item()

    def get_L_curve(self, u_base: float) -> float:
        """
        Compute the L-curve value for a given base utilization.

        Args:
            u_base: Base utilization value

        Returns:
            L(u_base) = alpha * u_base + beta
        """
        return self.L_alpha.item() * u_base + self.L_beta.item()

    def forward(self, u_base_list: torch.Tensor, r_list: torch.Tensor, b_list: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_base_list: [batch, MAX_TASKS] base utilizations
            r_list: [batch, MAX_TASKS] partition ratios (MPS percentages)
            b_list: [batch, MAX_TASKS] batch sizes
            N: [batch] number of active tasks

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
        # Step 1: Predict Dynamic Saturation Limit (Dynamic Ceiling)
        # Per spec: Û_SM^limit = 100 * tanh(k_b * avg_u_base)
        # where k_b is numerically solved to intersect the Linear Anchor Locus
        k_b = self.compute_k_b(avg_u_base)
        U_SM_limit = 100.0 * torch.tanh(k_b * avg_u_base / 100.0)

        # Step 3: Probabilistic Superposition (Concurrency Scaling)
        u_base_masked = u_base_list * active_mask
        p_occupy = u_base_masked / (U_SM_limit.unsqueeze(1) + 1e-6)
        p_occupy = torch.clamp(p_occupy, 0.0, 1.0)  # Ensure valid probabilities

        # Probability of each slot being idle: (1 - p_i)
        p_idle = 1.0 - p_occupy

        # Product over all tasks: ∏(1 - p_i)
        # Use log-sum-exp trick for numerical stability when dealing with many tasks
        # log(∏ p_idle) = Σ log(p_idle) for active tasks only
        log_p_idle = torch.log(p_idle + 1e-10) * active_mask
        log_prob_idle_product = torch.sum(log_p_idle, dim=1)
        prob_idle_product = torch.exp(log_prob_idle_product)
        # System-wide utilization: Û_SM^ref = Û_SM^limit · [1 - ∏(1 - p_i)]
        U_sm_ref = U_SM_limit * (1.0 - prob_idle_product)
        # Apply learnable scaling adjustment
        # scaling = 1 + sigmoid(raw_scaling) * small_range
        # This allows fine-tuning around the probabilistic prediction
        scaling = 1.0 + torch.sigmoid(self.raw_scaling) * 0.1  # ±5% adjustment
        U_sm_ref = U_sm_ref * scaling

        return U_sm_ref


class SMSystemModelWithKBLookup(nn.Module):
    """
    SMSystemModel with per-batch k_b lookup.

    Uses pre-computed k_b values for each batch size instead of learning k_b as a function.
    This implements the curve: U_limit = 100 * tanh(k_b * u_base) where k_b is per-batch.

    Args:
        kb_fit_results: Dict mapping batch_size -> {'k_b': value, 'loss': value, 'n_points': value}
        L_alpha: Slope of L-curve
        L_beta: Intercept of L-curve
    """
    def __init__(self, kb_fit_results: dict, L_alpha: float = 1.0, L_beta: float = 0.0):
        super().__init__()

        # Store k_b per batch as a buffer (not a parameter)
        batch_sizes = sorted(kb_fit_results.keys())
        k_b_values = [kb_fit_results[b]['k_b'] for b in batch_sizes]

        # Register as buffer (saved with model but not trained)
        self.register_buffer('batch_sizes', torch.tensor(batch_sizes, dtype=torch.float32))
        self.register_buffer('k_b_values', torch.tensor(k_b_values, dtype=torch.float32))

        # L-Curve parameters
        self.L_alpha = nn.Parameter(torch.tensor(L_alpha))
        self.L_beta = nn.Parameter(torch.tensor(L_beta))

        # Learnable adjustment factor for fine-tuning predictions
        self.raw_scaling = nn.Parameter(torch.tensor(0.0))

    def get_k_b_for_batch(self, batch_size: torch.Tensor) -> torch.Tensor:
        """
        Look up k_b for given batch sizes using nearest neighbor interpolation.

        Args:
            batch_size: [batch] tensor of batch sizes

        Returns:
            k_b: [batch] tensor of k_b values
        """
        # Expand for broadcasting
        batch_expanded = batch_size.unsqueeze(1)  # [batch, 1]
        # Find nearest batch size
        diffs = torch.abs(self.batch_sizes.unsqueeze(0) - batch_expanded)  # [batch, n_batches]
        nearest_idx = torch.argmin(diffs, dim=1)  # [batch]
        k_b = self.k_b_values[nearest_idx]
        return k_b

    def forward(self, u_base_list: torch.Tensor, r_list: torch.Tensor,
                b_list: torch.Tensor, N: torch.Tensor, batch_size: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_base_list: [batch, MAX_TASKS] base utilizations
            r_list: [batch, MAX_TASKS] partition ratios
            b_list: [batch, MAX_TASKS] batch sizes
            N: [batch] number of active tasks
            batch_size: [batch] batch size for k_b lookup

        Returns:
            U_sm_ref: [batch] system-wide SM utilization reference
        """
        batch_size_actual = u_base_list.shape[0]
        max_tasks = u_base_list.shape[1]

        # Create active mask: [batch, MAX_TASKS]
        task_indices = torch.arange(max_tasks, device=u_base_list.device).float()
        active_mask = (task_indices.unsqueeze(0) < N.unsqueeze(1)).float()

        # Sum of base utilizations for active tasks
        sum_u_base = torch.sum(u_base_list * active_mask, dim=1)
        # Average baseline: (1/N) * sum(u_sm,i^base)
        avg_u_base = sum_u_base / (N + 1e-6)

        # Get k_b for each sample based on batch size
        k_b = self.get_k_b_for_batch(batch_size)

        # Compute U_limit = 100 * tanh(k_b * avg_u_base / 100)
        U_SM_limit = 100.0 * torch.tanh(k_b * avg_u_base / 100.0)

        # Probabilistic Superposition
        u_base_masked = u_base_list * active_mask
        p_occupy = u_base_masked / (U_SM_limit.unsqueeze(1) + 1e-6)
        p_occupy = torch.clamp(p_occupy, 0.0, 1.0)

        # Probability of each slot being idle: (1 - p_i)
        p_idle = 1.0 - p_occupy

        # Product over all tasks
        log_p_idle = torch.log(p_idle + 1e-10) * active_mask
        log_prob_idle_product = torch.sum(log_p_idle, dim=1)
        prob_idle_product = torch.exp(log_prob_idle_product)

        # System-wide utilization
        U_sm_ref = U_SM_limit * (1.0 - prob_idle_product)

        # Apply learnable scaling adjustment
        scaling = 1.0 + torch.sigmoid(self.raw_scaling) * 0.1
        U_sm_ref = U_sm_ref * scaling

        return U_sm_ref

    def get_kb_params(self) -> dict:
        """Return k_b parameters per batch"""
        return {
            'batches': self.batch_sizes.tolist(),
            'k_b_values': self.k_b_values.tolist()
        }


def solve_for_kb_numerical(u_initial: float, u_final: float, N_max: int, tol: float = 1e-4) -> float:
    """
    Numerically solve for k_b that satisfies the constraint from Linear Anchor Locus.

    Per Spec §3.1.1:
    - Given u_initial = avg_u_base at N=1 (all tasks have same intensity)
    - Given u_final = system utilization at N=max (observed from data)
    - Find k_b such that probabilistic formula produces u_final when:
      * All N_max tasks have base utilization = u_initial
      * Using formula: Û_SM^ref = Û_SM^limit · [1 - ∏(1 - u_i^base/Û_SM^limit)]
      * Where Û_SM^limit = 100 * tanh(k_b * u_initial)

    For N_max identical tasks with base utilization u_initial:
    Û_SM^ref = Û_SM^limit · [1 - (1 - u_initial/Û_SM^limit)^N_max]

    Args:
        u_initial: Average baseline utilization at N=1
        u_final: System utilization at N=max (target from data)
        N_max: Maximum number of tasks (for this configuration)
        tol: Numerical tolerance

    Returns:
        k_b: Curvature coefficient that produces u_final
    """
    if N_max == 1:
        # Edge case: single task, k_b can be any value that allows u_initial
        # Return a default value
        return 0.01

    if u_final <= u_initial:
        # Edge case: no saturation or negative growth
        # Return small k_b (weak curvature)
        return 0.001

    # Binary search for k_b in range [0.001, 10.0]
    k_low, k_high = 0.001, 10.0

    for _ in range(50):  # Max 50 iterations
        k_mid = (k_low + k_high) / 2.0

        # Compute U_limit with current k_mid
        U_limit = 100.0 * math.tanh(k_mid * u_initial / 100.0)

        if U_limit < 1e-6:
            # Avoid division by zero
            k_low = k_mid
            continue

        # Compute predicted u_final using probabilistic formula
        # For N_max identical tasks: U_ref = U_limit * [1 - (1 - u_initial/U_limit)^N_max]
        ratio = u_initial / U_limit
        ratio = max(0.0, min(1.0, ratio))  # Clamp to [0, 1]
        u_pred = U_limit * (1.0 - (1.0 - ratio) ** N_max)

        # Adjust search range based on prediction
        if u_pred < u_final:
            k_low = k_mid
        else:
            k_high = k_mid

        # Check convergence
        if k_high - k_low < tol:
            break

    return (k_low + k_high) / 2.0


class MEMBaseDemandModel(nn.Module):
    """
    Base MEM Utilization Model (Single-Task, Fixed Freq)
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


# =========================================================================
#  Throughput Model (Kinetic Scaling)
# =========================================================================

class ThroughputKineticModel(nn.Module):
    """
    Throughput Model with Frequency Scaling
    Phase 1: T_ref = κ_bw * U_mem (at f_ref)
    Phase 2: T(f) = T_start + (T_ref - T_start) * ((f - f_min)/(f_ref - f_min))^γ_t
    """
    def __init__(self, f_ref: float = F_REF, f_min: float = F_MIN):
        super().__init__()
        self.f_ref = f_ref
        self.f_min = f_min
        # κ_bw: bandwidth coefficient (Phase 1)
        self.raw_kappa_bw = nn.Parameter(torch.tensor(50.0))
        # γ_t: throughput scaling exponent (Phase 2)
        self.raw_gamma_t = nn.Parameter(torch.tensor(0.8))
        # T_start: throughput at f_min (learned from data)
        self.raw_T_start = nn.Parameter(torch.tensor(100.0))

    def forward(self, U_mem: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            U_mem: [batch] memory utilization at anchor
            f: [batch] frequency
        """
        kappa_bw = nn.functional.softplus(self.raw_kappa_bw)
        gamma_t = torch.abs(self.raw_gamma_t)
        T_start = nn.functional.softplus(self.raw_T_start)

        # T_ref at anchor frequency
        T_ref = kappa_bw * U_mem

        # Frequency scaling
        freq_ratio = torch.clamp((f - self.f_min) / (self.f_ref - self.f_min), min=0.0, max=1.0)
        T_f = T_start + (T_ref - T_start) * torch.pow(freq_ratio, gamma_t)

        return T_f


# =========================================================================
#  Power Model (Thermodynamic)
# =========================================================================

class PowerStaticModel(nn.Module):
    """
    Static Power Baseline (Leakage)
    P_static(f) = δ_3*f^3 + δ_2*f^2 + δ_1*f + δ_0
    """
    def __init__(self):
        super().__init__()
        # Cubic polynomial coefficients
        self.raw_delta = nn.Parameter(torch.tensor([2.31e-8, -5.44e-5, 0.039, 85.6]))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        delta_3, delta_2, delta_1, delta_0 = self.raw_delta
        return delta_3 * f**3 + delta_2 * f**2 + delta_1 * f + delta_0


class PowerDynamicAnchorModel(nn.Module):
    """
    Dynamic Power Anchor (at f_ref)
    P_dyn(f_ref) = γ*S + (κ*f_ref) * S(α*U_sm + β*U_mem)
    """
    def __init__(self):
        super().__init__()
        self.raw_gamma = nn.Parameter(torch.tensor(-3.4))
        self.raw_kappa = nn.Parameter(torch.tensor(0.02))
        self.raw_alpha = nn.Parameter(torch.tensor(0.5))
        self.raw_beta = nn.Parameter(torch.tensor(0.5))
        self.saturation = Saturation(init_nu=2.0)

    def forward(self, U_sm: torch.Tensor, U_mem: torch.Tensor, S: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            U_sm: [batch] SM utilization
            U_mem: [batch] memory utilization
            S: [batch] total parameter count (millions)
            f: [batch] frequency (currently always f_ref)
        """
        gamma = nn.functional.softplus(self.raw_gamma)
        kappa = nn.functional.softplus(self.raw_kappa)
        alpha = torch.abs(self.raw_alpha)
        beta = torch.abs(self.raw_beta)

        # Standby power
        P_standby = gamma * S

        # Dynamic activation
        util = alpha * (U_sm / 100.0) + beta * (U_mem / 100.0)
        P_dyn = (kappa * f) * self.saturation(util)

        return P_standby + P_dyn


class PowerDynamicScalingModel(nn.Module):
    """
    Dynamic Power Frequency Scaling
    P_dyn(f) = P_base + (P_dyn(f_ref) - P_base) * ((f - f_min)/(f_ref - f_min))^γ_p
    """
    def __init__(self, f_ref: float = F_REF, f_min: float = F_MIN):
        super().__init__()
        self.f_ref = f_ref
        self.f_min = f_min
        # γ_p: power scaling exponent (≈3)
        self.raw_gamma_p = nn.Parameter(torch.tensor(3.0))
        # P_base: dynamic power at f_min
        self.raw_P_base = nn.Parameter(torch.tensor(10.0))

    def forward(self, P_dyn_ref: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            P_dyn_ref: [batch] dynamic power at f_ref
            f: [batch] frequency
        """
        gamma_p = torch.abs(self.raw_gamma_p)
        P_base = nn.functional.softplus(self.raw_P_base)

        freq_ratio = torch.clamp((f - self.f_min) / (self.f_ref - self.f_min), min=0.0, max=1.0)
        P_dyn_f = P_base + (P_dyn_ref - P_base) * torch.pow(freq_ratio, gamma_p)

        return P_dyn_f


# =========================================================================
#  Data Loading (Updated for Spec 3.1 with Memory Frequency Support)
# =========================================================================

def parse_data_filename(filename: str) -> dict:
    """
    Parse data filename to extract dataset, machine model, and other metadata.
    Filename pattern: mps_results[_dataset][_machine].csv
    Examples:
        mps_results.csv -> {'dataset': 'default', 'machine': 'default'}
        mps_results_imagenet.csv -> {'dataset': 'imagenet', 'machine': 'default'}
        mps_results_imagenet_a100.csv -> {'dataset': 'imagenet', 'machine': 'a100'}
        mps_results_caltech256.csv -> {'dataset': 'caltech256', 'machine': 'default'}
    """
    import re
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]  # Remove .csv extension

    # Remove common prefix "mps_results" and any leading underscores
    if name.startswith('mps_results'):
        name = name[len('mps_results'):]
    # Remove all leading underscores
    while name.startswith('_'):
        name = name[1:]

    # Parse remaining parts (underscore separated)
    parts = name.split('_')
    if not parts or (len(parts) == 1 and not parts[0]):
        return {'dataset': 'default', 'machine': 'default'}

    result = {'dataset': 'default', 'machine': 'default'}

    # Known machine models (longer names first to match '3080ti' before '3080')
    known_machines = ['a100', 'v100', 'a40', 'a6000', '3080ti', '3090', '3080', '4090', 'a10', 'a30']
    # Known datasets - will also treat unknown strings as dataset names
    known_datasets = ['imagenet', 'coco', 'voc', 'kitti', 'bert', 'llama', 'pubmed',
                      'cifar', 'caltech256', 'fashionmnist', 'mnist', 'imagenette']

    # Filter out empty strings
    parts = [p for p in parts if p]

    # Find dataset and machine from parts
    for part in parts:
        part_lower = part.lower()
        if part_lower in known_machines:
            result['machine'] = part_lower
        elif part_lower in known_datasets:
            result['dataset'] = part_lower
        else:
            # Assume any unknown string is a dataset name
            if result['dataset'] == 'default':
                result['dataset'] = part_lower

    return result


def load_anchor_data(dataset: str = None, machine: str = None) -> pd.DataFrame:
    """
    Load Anchor Data from mps_profile directory.
    Supports filtering by dataset and machine model.

    Args:
        dataset: Dataset name (e.g., 'imagenet', 'coco'). If None, loads default.
        machine: Machine model (e.g., 'a100', '3090'). If None, loads default.

    Returns:
        DataFrame with anchor data
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir

    # Check multiple possible locations
    for base in [script_dir, os.path.join(script_dir, 'src', 'profile'), os.path.join(script_dir, '..')]:
        mps_dir = os.path.join(base, 'mps_profile')
        if os.path.exists(mps_dir):
            base_dir = base
            break
    else:
        mps_dir = os.path.join(script_dir, 'mps_profile')

    if not os.path.exists(mps_dir):
        print("  ⚠ MPS profile directory not found!")
        return pd.DataFrame()

    # Find matching files
    all_files = []
    matching_files = []
    parsed_info = []  # Store parsed info for debugging

    for f in os.listdir(mps_dir):
        if not f.endswith('.csv') or not f.startswith('mps_results'):
            continue
        all_files.append(f)

        file_info = parse_data_filename(f)
        parsed_info.append((f, file_info))

        file_dataset = file_info['dataset']
        file_machine = file_info['machine']

        # Check if file matches requested dataset and machine
        # Priority: explicit filter > default file > any matching
        # If no dataset/platform specified, prefer default file first
        if dataset is None and machine is None:
            # No filters specified - only match default file (no suffixes)
            is_default_file = (file_dataset == 'default' and file_machine == 'default')
            matches_dataset = is_default_file
            matches_machine = is_default_file
        else:
            # Filters specified - match based on criteria
            matches_dataset = (dataset is None or file_dataset == dataset or
                              (dataset == 'default' and file_dataset == 'default'))
            matches_machine = (machine is None or file_machine == machine or
                              (machine == 'default' and file_machine == 'default'))

        if matches_dataset and matches_machine:
            matching_files.append((os.path.join(mps_dir, f), file_info))

    # Debug: print parsed file info
    print(f"  Parsed file info:")
    for fname, info in parsed_info:
        match = any(os.path.basename(f[0]) == fname for f in matching_files)
        marker = "✓" if match else " "
        print(f"    {marker} {fname} -> dataset={info['dataset']}, machine={info['machine']}")

    if not matching_files:
        # No matching files found
        print(f"\n⚠ Error: No matching data files found!")
        print(f"    Requested: dataset={dataset or 'any'}, machine={machine or 'any'}")
        print(f"\n    Available files in mps_profile/:")
        for f in all_files:
            info = parse_data_filename(f)
            print(f"      - {f}  (dataset={info['dataset']}, machine={info['machine']})")
        print(f"\n    To fix this:")
        print(f"      1. Check if the file exists for your dataset/platform")
        print(f"      2. Use correct naming: mps_results[_dataset][_machine].csv")
        print(f"      3. Example: mps_results_imagenet_a100.csv for imagenet on A100")
        return None  # Return None instead of empty DataFrame to trigger proper error handling

    # Load and concatenate all matching files
    dfs = []
    for filepath, file_info in matching_files:
        df = pd.read_csv(filepath)
        df['_source_file'] = os.path.basename(filepath)
        df['_dataset'] = file_info['dataset']
        df['_machine'] = file_info['machine']
        dfs.append(df)
        print(f"  ✓ Loaded {os.path.basename(filepath)} "
              f"(dataset={file_info['dataset']}, machine={file_info['machine']})")

    if not dfs:
        print(f"\n⚠ Error: No data loaded from matching files!")
        return None

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  → Total records: {len(combined_df)}")
    return combined_df


def get_available_mem_frequencies(df: pd.DataFrame) -> list:
    """Get list of unique memory frequencies from data."""
    if 'mem_freq' in df.columns:
        return sorted(df['mem_freq'].unique())
    return []


def filter_by_mem_freq(df: pd.DataFrame, mem_freq: float) -> pd.DataFrame:
    """Filter data by specific memory frequency."""
    if 'mem_freq' in df.columns:
        return df[df['mem_freq'] == mem_freq].copy()
    return df


def load_scaling_data(target_model: str = None, dataset: str = None, platform: str = None, target_mem_freq: float = None) -> Dict[str, pd.DataFrame]:
    """
    Load Scaling Data: scale_profile/sweep_results*.csv

    Priority:
    1. sweep_results_{platform}.csv (e.g., sweep_results_a100.csv for mem_freq=1215)
    2. sweep_results.csv (global fallback)

    Note: mem_freq=1215 is typically A100, so we auto-detect and use a100 scaling data
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    profiles = {}

    # Auto-detect platform from memory frequency
    # mem_freq=1215 is A100's memory frequency
    detected_platform = platform
    if target_mem_freq == 1215 and platform != 'a100':
        print(f"  [Info] mem_freq=1215 detected, auto-switching to 'a100' platform for scaling data")
        detected_platform = 'a100'

    # Build suffix for platform-specific files
    platform_suffix = f"_{detected_platform}" if detected_platform else ""

    # Try both possible locations
    paths = [
        os.path.join(script_dir, 'scale_profile'),
        os.path.join(script_dir, 'src', 'profile', 'scale_profile'),
        os.path.join(script_dir, '..', 'scale_profile'),
    ]

    base_path = None
    for path in paths:
        if os.path.exists(path):
            base_path = path
            break

    if base_path is None:
        print("  ⚠ Scale profile directory not found!")
        return profiles

    # Try platform-specific file first, then global file
    sweep_path = None
    if platform_suffix:
        platform_sweep_path = os.path.join(base_path, f'sweep_results{platform_suffix}.csv')
        if os.path.exists(platform_sweep_path):
            sweep_path = platform_sweep_path
            print(f"  ✓ Loaded Platform-Specific Scaling Data: {detected_platform}")

    if sweep_path is None:
        sweep_path = os.path.join(base_path, 'sweep_results.csv')
        if not os.path.exists(sweep_path):
            print("  ⚠ sweep_results.csv not found!")
            return profiles

    # Load the file (only if not already loaded via platform-specific path)
    if not platform_suffix or sweep_path.endswith('sweep_results.csv'):
        df = pd.read_csv(sweep_path)
        print(f"  ✓ Loaded Scaling Data: {sweep_path}")

    # Re-read from sweep_path to get the dataframe
    df = pd.read_csv(sweep_path)

    # Group by model_name
    for model_name, group in df.groupby('model_name'):
        if target_model is None or model_name == target_model:
            profiles[model_name] = group
            if model_name == 'static':
                print(f"  ✓ Static Power Data: {len(group)} rows")
            else:
                print(f"  ✓ Scaling Data: {model_name} ({len(group)} rows)")

    return profiles


def load_static_power_data(
    scaling_data: Dict[str, pd.DataFrame],
    platform: str = None,
    target_mem_freq: float = None
) -> pd.DataFrame:
    """
    Load Static Power Data.

    Priority:
    1) scaling_data entry with model_name='static'
    2) static_sweep_results_{platform}.csv
    3) static_sweep_results.csv (fallback)
    """
    if 'static' in scaling_data:
        static_df = scaling_data['static']
        print(f"  ✓ Extracted Static Power Data: {len(static_df)} rows")
        return _normalize_static_power_df(static_df, target_mem_freq)

    # Auto-detect platform from memory frequency
    detected_platform = platform
    if target_mem_freq == 1215 and platform != 'a100':
        print("  [Info] mem_freq=1215 detected, auto-switching to 'a100' platform for static data")
        detected_platform = 'a100'

    platform_suffix = f"_{detected_platform}" if detected_platform else ""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(script_dir, 'scale_profile'),
        os.path.join(script_dir, 'src', 'profile', 'scale_profile'),
        os.path.join(script_dir, '..', 'scale_profile'),
    ]

    base_path = None
    for path in paths:
        if os.path.exists(path):
            base_path = path
            break

    if base_path is None:
        print("  ⚠ Scale profile directory not found for static power data!")
        return pd.DataFrame()

    candidates = []
    if platform_suffix:
        candidates.append(os.path.join(base_path, f'static_sweep_results{platform_suffix}.csv'))
        candidates.append(os.path.join(base_path, 'old', f'static_sweep_results{platform_suffix}.csv'))
    candidates.append(os.path.join(base_path, 'static_sweep_results.csv'))
    candidates.append(os.path.join(base_path, 'old', 'static_sweep_results.csv'))

    static_path = next((p for p in candidates if os.path.exists(p)), None)
    if static_path is None:
        print("  ⚠ Static power file not found (static_sweep_results*.csv)")
        return pd.DataFrame()

    static_df = pd.read_csv(static_path)
    print(f"  ✓ Loaded Static Power Data: {static_path} ({len(static_df)} rows)")
    return _normalize_static_power_df(static_df, target_mem_freq)


def _normalize_static_power_df(df: pd.DataFrame, target_mem_freq: float = None) -> pd.DataFrame:
    """Normalize static power dataframe to ensure power_max column and mem_freq filter."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    if 'power_max' not in df.columns:
        for col in ['power', 'static_power', 'power_avg']:
            if col in df.columns:
                df['power_max'] = df[col]
                break
        if 'power_max' not in df.columns:
            print("  ⚠ Static power data missing power column (power_max/power/static_power/power_avg)")
            return pd.DataFrame()

    if 'model_name' in df.columns:
        df = df[df['model_name'] == 'static'].copy()

    if target_mem_freq is not None and 'mem_freq' in df.columns:
        df = df[df['mem_freq'] == target_mem_freq].copy()

    return df


def load_model_complexity(target_model: str = None) -> Dict[str, Dict]:
    """Load model complexity information"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, 'profile', 'model_table.csv')
    if not os.path.exists(path):
        # Try alternate location
        path = os.path.join(script_dir, '..', 'profile', 'model_table.csv')

    complexity = {}
    if os.path.exists(path):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            ref = row['Reference'].lower()
            if target_model and ref != target_model: continue
            complexity[ref] = {
                'gflops': float(row['GFLOPS']),
                'params_M': float(row['Parameters'].replace('M', ''))
            }
    return complexity


# =========================================================================
#  Model Fitting Utilities
# =========================================================================

class PhysicsModelFitter:
    def __init__(self, model, name="model"):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, inputs, targets, epochs=2000, lr=0.01, verbose=True):
        x = torch.from_numpy(inputs).float().to(self.device)
        y = torch.from_numpy(targets).float().to(self.device).flatten()
        opt = optim.Adam(self.model.parameters(), lr=lr)
        # Learning rate scheduler: reduce by factor 0.5 if no improvement for 100 steps
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=100
        )
        loss_fn = nn.MSELoss()

        self.model.train()
        best_loss = float('inf')
        for epoch in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self.model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            opt.step()

            # Step scheduler every 100 epochs
            if (epoch + 1) % 100 == 0:
                scheduler.step(loss.item())

            if verbose and ((epoch + 1) % 100 == 0 or epoch == 0 or epoch == epochs - 1):
                grad_info = []
                current_lr = opt.param_groups[0]['lr']
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            v = p.cpu().numpy()
                            grad = p.grad.cpu().numpy()
                            if v.ndim == 0:
                                grad_info.append(f"{n}={float(v.item()):.4f}(g={float(grad.item()):.4f})")
                print(f"    Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f} | lr={current_lr:.6f} | {', '.join(grad_info[:3])}")

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(inputs).float().to(self.device)
            return self.model(x).cpu().numpy()

    def get_params(self):
        self.model.eval()
        params = {}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                v = p.cpu().numpy()
                key = n.replace('.', '_')
                if v.ndim == 0:
                    params[key] = float(v.item())
                else:
                    params[key] = v.tolist()
        return params


# =========================================================================
#  SM Model Fitting (Fixed Frequency)
# =========================================================================

def fit_sm_model(anchor_df, target_model=None, mem_freq_suffix=""):
    """
    Fit SM utilization model.

    Training Process (Strictly Separated Stages):
      Stage 1: Base Demand Model
        - Fit u_base(r, b) using single-task data at anchor frequency

      Stage 2: L-Curve Fitting (Linear Anchor Locus)
        - Fit L(u) = alpha * u + beta using lower envelope points
        - Consistent with task_num_draw.py visualization

      Stage 3: Kappa Fitting (Global)
        - Formula: U_limit = L(u) + (100 - L(u)) * tanh(kappa * xi) / tanh(kappa)
        - xi = (u - u_ref) / (100 - u_ref)
        - Fit single kappa for all batches (4, 8)

      Stage 4: Physics Prior Validation
        - Validate: U_SM_ref = U_limit * [1 - prod(1 - u_base_i/U_limit)]
        - Compute RMSE, MAE, R² against actual values

      Stage 5: Residual Calibration (GBR)
        - Use physics prior as feature
        - Fit GBR residual to correct physics model

    Args:
        anchor_df: DataFrame with anchor data
        target_model: Target model name to filter
        mem_freq_suffix: Suffix for model files based on memory frequency (e.g., "_mem9501")
    """
    print("\n" + "="*70 + "\nFITTING SM MODEL (Fixed Frequency at Anchor)\n" + "="*70)
    print(f"  Memory frequency suffix: {mem_freq_suffix}" if mem_freq_suffix else "  Using default memory frequency")
    MAX_TASKS = 20

    # Build suffix string for filenames
    file_suffix = mem_freq_suffix

    # Filter by target model if specified
    if target_model:
        anchor_df = anchor_df[anchor_df['model_name'] == target_model]

    if len(anchor_df) == 0:
        print("  ⚠ No data found")
        return

    model_names = anchor_df['model_name'].unique()
    print(f"  Models: {model_names}")

    for model_name in model_names:
        if target_model and model_name != target_model:
            continue

        print(f"\n--- Model: {model_name} ---")
        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        df = anchor_df[anchor_df['model_name'] == model_name].copy()

        # Ensure we have the necessary columns
        if 'partition_size' not in df.columns:
            df['partition_size'] = df['p_list'].apply(lambda x: eval(x)[0] if isinstance(x, str) else x)
        if 'batch_size' not in df.columns:
            df['batch_size'] = df['b_list'].apply(lambda x: eval(x)[0] if isinstance(x, str) else x)
        if 'num_tasks' not in df.columns:
            df['num_tasks'] = df['p_list'].apply(lambda x: len(eval(x)) if isinstance(x, str) else 1)

        # Group data for later use
        grouped = df.groupby(['partition_size', 'batch_size'])

        # Build single-task lookup table (needed for both Stage 1 and Stage 2)
        single_task_lookup = {}
        for _, row in df.iterrows():
            p_str, b_str = str(row['p_list']), str(row['b_list'])
            try:
                p_list = eval(p_str) if isinstance(p_str, str) and p_str.startswith('[') else [float(p_str)]
                b_list = eval(b_str) if isinstance(b_str, str) and b_str.startswith('[') else [int(b_str)]
                num_tasks = len(p_list)

                if num_tasks == 1:
                    key = (p_list[0], b_list[0])
                    single_task_lookup[key] = row['sm_util']
            except:
                continue

        print(f"  Built lookup table: {len(single_task_lookup)} single-task entries")

        # ===== Stage 1: Base Demand Model =====
        print("  [Stage 1] Base Demand Model...")
        inputs_s1, targets_s1 = [], []
        for (p, b), u_sm in single_task_lookup.items():
            inputs_s1.append([p, b])
            targets_s1.append(u_sm)

        if len(inputs_s1) < 5:
            print("  ⚠ Insufficient single-task data")
            continue

        model_base = SMBaseDemandModel()
        fitter_base = PhysicsModelFitter(model_base)
        fitter_base.fit(np.array(inputs_s1, dtype=np.float32),
                       np.array(targets_s1, dtype=np.float32),
                       epochs=500, lr=0.1)

        # Save with suffix for memory frequency specific models
        sm_base_path = os.path.join(model_dir, f"sm_base_model{file_suffix}.pth")
        torch.save(model_base.state_dict(), sm_base_path)

        # ===== Stage 2: L-Curve Fitting (Linear Anchor Locus) =====
        print("  [Stage 2] L-Curve Fitting (Linear Anchor Locus)...")

        # Collect (N=1, N=max) pairs
        l_curve_data = []

        for (partition, batch), group in grouped:
            initial_data = group[group['num_tasks'] == 1][['sm_util']]
            max_tasks = int(100 / partition) if partition > 0 else 1
            final_data = group[group['num_tasks'] == max_tasks][['sm_util']]

            if not initial_data.empty and not final_data.empty:
                u_initial = initial_data['sm_util'].values[0]
                u_final = final_data['sm_util'].values[0]
                l_curve_data.append({
                    'partition_size': partition,
                    'batch_size': batch,
                    'u_initial': u_initial,
                    'u_final': u_final,
                    'N_max': max_tasks
                })

        if len(l_curve_data) < 2:
            print("    ⚠ Insufficient (N=1, N=max) data pairs for L-curve fitting")
            L_alpha, L_beta = 1.0, 0.0
        else:
            # Fit L-curve using lower envelope points (consistent with task_num_draw.py)
            l_curve_df = pd.DataFrame(l_curve_data)
            lower_envelope_points = []

            for batch in l_curve_df['batch_size'].unique():
                batch_df = l_curve_df[l_curve_df['batch_size'] == batch]
                if not batch_df.empty:
                    # Find the point with minimum final SM utilization
                    min_idx = batch_df['u_final'].idxmin()
                    lower_point = batch_df.loc[min_idx]
                    lower_envelope_points.append((lower_point['u_initial'], lower_point['u_final']))

            if len(lower_envelope_points) >= 2:
                lower_envelope_points = sorted(lower_envelope_points, key=lambda x: x[0])
                x_points = [p[0] for p in lower_envelope_points]
                y_points = [p[1] for p in lower_envelope_points]
                coeffs = np.polyfit(x_points, y_points, 1)
            else:
                # Fallback: use all points
                u_initials = [d['u_initial'] for d in l_curve_data]
                u_finals = [d['u_final'] for d in l_curve_data]
                coeffs = np.polyfit(u_initials, u_finals, 1)

            L_alpha, L_beta = coeffs[0], coeffs[1]
            print(f"    - L(u) = {L_alpha:.4f} * u + {L_beta:.4f}")
            print(f"    - R² range: [{min([d['u_initial'] for d in l_curve_data]):.1f}, {max([d['u_initial'] for d in l_curve_data]):.1f}]")

        # Save L-curve parameters
        l_curve_params = {
            'L_alpha': float(L_alpha),
            'L_beta': float(L_beta),
            'description': 'Linear Anchor Locus: L(u) = alpha * u + beta'
        }
        l_curve_path = os.path.join(model_dir, f"sm_l_curve{file_suffix}.json")
        with open(l_curve_path, 'w') as f:
            json.dump(l_curve_params, f, indent=2)
        print(f"    - Saved L-curve params to {l_curve_path}")

        # ===== Stage 3: Kappa Fitting (Global for All Batches) =====
        print("  [Stage 3] Kappa Fitting (Global)...")
        print("    - Formula: U_limit = L(u) + (100 - L(u)) * tanh(kappa * xi) / tanh(kappa)")
        print("    - xi = (u_base - u_ref) / (100 - u_ref)")
        print("    - Fitting single kappa for all batches (4, 8)")

        # Filter for target batches (4, 8)
        target_batches = [4, 8]
        kb_fit_results = {}

        # Collect all fitting data across batches
        all_fit_data = []

        for batch in target_batches:
            batch_df = df[df['batch_size'] == batch]
            if len(batch_df) == 0:
                continue

            # Get reference point (partition=40%, N=1) → u_ref = u_base
            ref_data = batch_df[
                (batch_df['partition_size'] == 40) &
                (batch_df['num_tasks'] == 1)
            ][['sm_util']]
            if ref_data.empty:
                ref_data = batch_df[batch_df['num_tasks'] == 1][['sm_util']]
            if ref_data.empty:
                continue

            u_ref = ref_data['sm_util'].values[0]  # u_ref = u_base(partition=40%, N=1)

            # Collect (u_base, U_limit) pairs
            fit_data = []
            for partition in sorted(batch_df['partition_size'].unique()):
                if partition == 0:
                    continue

                partition_df = batch_df[batch_df['partition_size'] == partition]

                # N=1 → u_base
                n1_data = partition_df[partition_df['num_tasks'] == 1][['sm_util']]
                if n1_data.empty:
                    continue
                u_base = n1_data['sm_util'].values[0]

                # N=max → U_limit
                max_tasks = int(100 / partition)
                nmax_data = batch_df[
                    (batch_df['partition_size'] == partition) &
                    (batch_df['num_tasks'] == max_tasks)
                ][['sm_util']]
                if nmax_data.empty:
                    continue
                U_limit_actual = nmax_data['sm_util'].values[0]

                fit_data.append({
                    'batch': batch,
                    'partition': partition,
                    'u_ref': u_ref,
                    'u_base': u_base,
                    'U_limit_actual': U_limit_actual,
                    'N_max': max_tasks
                })

            if len(fit_data) >= 2:
                all_fit_data.extend(fit_data)
                print(f"    Batch {batch}: {len(fit_data)} fitting points, u_ref={u_ref:.2f}")
                kb_fit_results[batch] = {'u_ref': float(u_ref)}

        if len(all_fit_data) < 2:
            print("    ⚠ Insufficient fitting data, skipping")
        else:
            # ===== Step 3b: Fit single kappa using all data =====
            # U_limit_pred = L(u) + (100 - L(u)) * tanh(kappa * xi) / tanh(kappa)
            # where xi = (u_base - u_ref) / (100 - u_ref)
            # u_ref = u_base at partition=40%, N=1

            u_values = np.array([d['u_base'] for d in all_fit_data])
            U_limit_values = np.array([d['U_limit_actual'] for d in all_fit_data])
            u_ref_values = np.array([d['u_ref'] for d in all_fit_data])

            def compute_kappa_loss(kappa):
                """Compute MSE loss for given kappa
                Formula: U_limit = L(u) + (100 - L(u)) * tanh(kappa * xi) / tanh(kappa)
                where xi = (u - u_ref) / (100 - u_ref)
                """
                # Compute xi for each point
                xi = (u_values - u_ref_values) / (100.0 - u_ref_values)
                xi = np.clip(xi, 0, 1)

                # Compute L(u) = L_alpha * u + L_beta
                L_u = L_alpha * u_values + L_beta

                # Compute U_limit prediction
                U_limit_pred = L_u + (100.0 - L_u) * np.tanh(kappa * xi) / np.tanh(kappa)
                return np.mean((U_limit_pred - U_limit_values) ** 2)

            # Grid search for kappa
            best_kappa = 2.0
            best_loss = compute_kappa_loss(best_kappa)
            for kappa_try in np.arange(0.5, 10.0, 0.1):
                loss = compute_kappa_loss(kappa_try)
                if loss < best_loss:
                    best_loss = loss
                    best_kappa = kappa_try

            # Fine-tune using scipy
            try:
                from scipy.optimize import minimize_scalar
                result = minimize_scalar(compute_kappa_loss, bounds=(0.1, 20.0), method='bounded')
                if result.success:
                    best_kappa = result.x
                    best_loss = result.fun
            except ImportError:
                pass

            print(f"    - Kappa = {best_kappa:.6f}, MSE = {best_loss:.4f}")

            # Print fitting details for target batches (4, 8)
            print("    [Fitting Results for Target Batches]")
            xi = (u_values - u_ref_values) / (100.0 - u_ref_values)
            xi = np.clip(xi, 0, 1)
            L_u = L_alpha * u_values + L_beta
            U_limit_pred = L_u + (100.0 - L_u) * np.tanh(best_kappa * xi) / np.tanh(best_kappa)

            for i, d in enumerate(all_fit_data):
                print(f"    Batch {d['batch']}, Partition {d['partition']:.0f}%: "
                      f"u_base={d['u_base']:.2f}, U_limit_actual={d['U_limit_actual']:.2f}, U_limit_pred={U_limit_pred[i]:.2f}")

            # ===== Validate on ALL batches (not just target batches) =====
            print("    [Validation on All Batches]")
            all_batches = sorted(df['batch_size'].unique())
            validation_results = {}

            for batch in all_batches:
                batch_df = df[df['batch_size'] == batch]
                if len(batch_df) == 0:
                    continue

                # Get u_ref for this batch: u_base at partition=40%, N=1
                ref_data = batch_df[
                    (batch_df['partition_size'] == 40) &
                    (batch_df['num_tasks'] == 1)
                ][['sm_util']]
                if ref_data.empty:
                    ref_data = batch_df[batch_df['num_tasks'] == 1][['sm_util']]
                if ref_data.empty:
                    continue
                u_ref = ref_data['sm_util'].values[0]

                # Collect (u_base, U_limit) pairs for this batch
                batch_preds = []
                batch_actuals = []

                for partition in sorted(batch_df['partition_size'].unique()):
                    if partition == 0:
                        continue

                    # Get u_base (N=1)
                    n1_data = batch_df[
                        (batch_df['partition_size'] == partition) &
                        (batch_df['num_tasks'] == 1)
                    ][['sm_util']]
                    if n1_data.empty:
                        continue
                    u_base = n1_data['sm_util'].values[0]

                    # Get U_limit (N=max)
                    max_tasks = int(100 / partition)
                    nmax_data = batch_df[
                        (batch_df['partition_size'] == partition) &
                        (batch_df['num_tasks'] == max_tasks)
                    ][['sm_util']]
                    if nmax_data.empty:
                        continue
                    U_limit_actual = nmax_data['sm_util'].values[0]

                    # Compute prediction (handle u_ref >= 100)
                    denom = 100.0 - u_ref
                    if denom <= 0:
                        xi = 0.0  # or 1.0 depending on desired behavior
                    else:
                        xi = np.clip((u_base - u_ref) / denom, 0, 1)
                    L_u = L_alpha * u_base + L_beta
                    U_limit_pred = L_u + (100.0 - L_u) * np.tanh(best_kappa * xi) / np.tanh(best_kappa)

                    batch_preds.append(U_limit_pred)
                    batch_actuals.append(U_limit_actual)

                if len(batch_preds) >= 2:
                    batch_preds = np.array(batch_preds)
                    batch_actuals = np.array(batch_actuals)
                    rmse = np.sqrt(np.mean((batch_preds - batch_actuals) ** 2))
                    mae = np.mean(np.abs(batch_preds - batch_actuals))
                    ss_res = np.sum((batch_actuals - batch_preds) ** 2)
                    ss_tot = np.sum((batch_actuals - np.mean(batch_actuals)) ** 2)
                    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    validation_results[batch] = {'rmse': rmse, 'r2': r2, 'n_points': len(batch_preds)}
                    print(f"    Batch {batch}: RMSE={rmse:.4f}, R²={r2:.4f} ({len(batch_preds)} points)")

            # Store kappa (no u_ref, use CSV values directly)
            kappa_params = {
                'kappa': float(best_kappa),
                'L_alpha': float(L_alpha),
                'L_beta': float(L_beta),
                'mse': float(best_loss),
                'n_points': len(all_fit_data),
                'description': 'U_limit = L(u) + (100 - L(u)) * tanh(kappa * xi) / tanh(kappa), xi = (u - u_ref) / (100 - u_ref)'
            }

            # Save kappa results
            kappa_path = os.path.join(model_dir, f"sm_kappa{file_suffix}.json")
            with open(kappa_path, 'w') as f:
                json.dump(kappa_params, f, indent=2)
            print(f"    - Saved kappa to {kappa_path}")

        # ===== Stage 4: Physics Prior Validation =====
        print("  [Stage 4] Physics Prior Validation...")
        print("    - Using: U_limit = L(u) + (100 - L(u)) * tanh(kappa * xi) / tanh(kappa)")

        if len(all_fit_data) < 2:
            print("    ⚠ No kappa fitted, skipping validation")
        else:
            # Build u_ref lookup from CSV data
            # u_ref = u_base at partition=40%, N=1
            u_ref_lookup = {}
            for batch in all_batches:
                batch_df = df[df['batch_size'] == batch]
                if len(batch_df) == 0:
                    continue
                ref_data = batch_df[
                    (batch_df['partition_size'] == 40) &
                    (batch_df['num_tasks'] == 1)
                ][['sm_util']]
                if ref_data.empty:
                    ref_data = batch_df[batch_df['num_tasks'] == 1][['sm_util']]
                if ref_data.empty:
                    continue
                u_ref_lookup[batch] = ref_data['sm_util'].values[0]

            # Prepare validation data
            validation_data = []
            for _, row in df.iterrows():
                try:
                    p_str, b_str = str(row['p_list']), str(row['b_list'])
                    p_list = eval(p_str) if isinstance(p_str, str) and p_str.startswith('[') else [float(p_str)]
                    b_list = eval(b_str) if isinstance(b_str, str) and b_str.startswith('[') else [int(b_str)]
                    N = len(p_list)
                    batch = b_list[0]
                except Exception as e:
                    continue

                # Skip single-task data (N=1 is trivial)
                if N == 1:
                    continue

                # Get u_ref for this batch from lookup
                if batch not in u_ref_lookup:
                    continue
                u_ref = u_ref_lookup[batch]

                # Get base utilizations from lookup
                u_base_list = []
                for p, b in zip(p_list, b_list):
                    key = (p, b)
                    u_base = single_task_lookup.get(key, None)
                    if u_base is None:
                        break
                    u_base_list.append(u_base)

                if len(u_base_list) != N:
                    continue

                validation_data.append({
                    'u_base_list': u_base_list,
                    'p_list': p_list,
                    'b_list': b_list,
                    'u_ref': u_ref,
                    'batch': batch,
                    'N': N,
                    'sm_util_actual': row['sm_util']
                })

            if len(validation_data) < 5:
                print("  ⚠ Insufficient multi-task data for validation")
            else:
                # Compute predictions using physics formula with kappa
                predictions = []
                actuals = []

                for data in validation_data:
                    u_base_list = data['u_base_list']
                    u_ref = data['u_ref']
                    N = data['N']
                    avg_u_base = np.mean(u_base_list)

                    # Compute xi = (u_base - u_ref) / (100 - u_ref)
                    denom = (100.0 - u_ref)
                    if denom <= 0:
                        xi = 0.0
                    else:
                        xi = np.clip((avg_u_base - u_ref) / denom, 0, 1)

                    # Compute L(u) = L_alpha * u + L_beta
                    L_u = L_alpha * avg_u_base + L_beta

                    # Compute U_limit = L(u) + (100 - L(u)) * tanh(kappa * xi) / tanh(kappa)
                    U_limit = L_u + (100.0 - L_u) * np.tanh(best_kappa * xi) / np.tanh(best_kappa)

                    # Compute p_i for each task
                    p_list = [u / U_limit for u in u_base_list]

                    # Compute P(idle) = prod(1 - p_i)
                    P_idle = 1.0
                    for p in p_list:
                        P_idle *= (1.0 - p)

                    # Compute U_SM_ref = U_limit * [1 - P(idle)]
                    U_sm_ref = U_limit * (1.0 - P_idle)

                    predictions.append(U_sm_ref)
                    actuals.append(data['sm_util_actual'])

            # Compute metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)

            mse = np.mean((predictions - actuals) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - actuals))

            # R² score
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Max absolute error
            max_ae = np.max(np.abs(predictions - actuals))

            print(f"    - Validation samples: {len(predictions)}")
            print(f"    - RMSE: {rmse:.4f}")
            print(f"    - MAE: {mae:.4f}")
            print(f"    - R²: {r2:.4f}")
            print(f"    - Max Abs Error: {max_ae:.4f}")

            # Save validation results
            validation_results = {
                'n_samples': len(predictions),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'max_abs_error': float(max_ae),
                'description': 'Physics formula validation: U_SM_ref = U_limit * [1 - prod(1 - u_base_i/U_limit)]'
            }
            val_path = os.path.join(model_dir, f"sm_physics_validation{file_suffix}.json")
            with open(val_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            print(f"    - Saved validation results to {val_path}")

            # ===== Visualization =====
            if len(predictions) >= 5:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Plot 1: Actual vs Predicted scatter
                ax1 = axes[0, 0]
                ax1.scatter(actuals, predictions, alpha=0.6, edgecolors='k', linewidth=0.5)
                min_val = min(actuals.min(), predictions.min())
                max_val = max(actuals.max(), predictions.max())
                ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect fit')
                ax1.set_xlabel('Actual SM Util (%)', fontsize=11)
                ax1.set_ylabel('Predicted SM Util (%)', fontsize=11)
                ax1.set_title(f'Actual vs Predicted (R²={r2:.3f})', fontsize=12)
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Plot 2: Residual distribution
                ax2 = axes[0, 1]
                residuals = predictions - actuals
                ax2.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
                ax2.axvline(x=0, color='r', linestyle='--', lw=2)
                ax2.set_xlabel('Prediction Error (%)', fontsize=11)
                ax2.set_ylabel('Frequency', fontsize=11)
                ax2.set_title(f'Residual Distribution (MAE={mae:.2f})', fontsize=12)
                ax2.grid(True, alpha=0.3)

                # Plot 3: By N (task count) comparison
                ax3 = axes[1, 0]
                # Group by N
                for N in sorted(set(d['N'] for d in validation_data)):
                    N_actuals = []
                    N_preds = []
                    for i, data in enumerate(validation_data):
                        if data['N'] == N:
                            N_actuals.append(actuals[i])
                            N_preds.append(predictions[i])
                    if len(N_actuals) >= 2:
                        ax3.scatter(N_actuals, N_preds, label=f'N={N}', alpha=0.7, s=50)
                ax3.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='Perfect')
                ax3.set_xlabel('Actual SM Util (%)', fontsize=11)
                ax3.set_ylabel('Predicted SM Util (%)', fontsize=11)
                ax3.set_title('Comparison by Task Count (N)', fontsize=12)
                ax3.legend()
                ax3.grid(True, alpha=0.3)

                # Plot 4: Error vs N
                ax4 = axes[1, 1]
                N_errors = {}
                for i, data in enumerate(validation_data):
                    N = data['N']
                    err = abs(predictions[i] - actuals[i])
                    if N not in N_errors:
                        N_errors[N] = []
                    N_errors[N].append(err)
                N_vals = sorted(N_errors.keys())
                mean_errors = [np.mean(N_errors[n]) for n in N_vals]
                ax4.bar(N_vals, mean_errors, color='steelblue', edgecolor='black')
                ax4.set_xlabel('Number of Tasks (N)', fontsize=11)
                ax4.set_ylabel('Mean Absolute Error (%)', fontsize=11)
                ax4.set_title('Prediction Error by Task Count', fontsize=12)
                ax4.grid(True, alpha=0.3, axis='y')

                plt.suptitle(f'Physics Model Validation - {model_name}', fontsize=14, fontweight='bold')
                plt.tight_layout()

                fig_path = os.path.join(model_dir, f"sm_physics_validation{file_suffix}.png")
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    - Saved visualization to {fig_path}")

        # ===== Stage 5: Residual Calibration (GBR) =====
        print("  [Stage 5] Residual Calibration (GBR)...")

        if len(all_fit_data) < 2:
            print("    ⚠ No kappa fitted, skipping GBR")
        else:
            # Compute physics prior using kappa formula
            # Û_SM^ref = U_limit * [1 - prod(1 - u_base_i/U_limit)]
            # where U_limit = L(u) + (100 - L(u)) * tanh(kappa * xi) / tanh(kappa)
            physics_priors = []
            for data in validation_data:
                u_base_list = data['u_base_list']
                u_ref = data['u_ref']
                N = data['N']
                avg_u_base = np.mean(u_base_list)

                # Compute xi = (u_base - u_ref) / (100 - u_ref)
                denom = (100.0 - u_ref)
                if denom <= 0:
                    xi = 0.0
                else:
                    xi = np.clip((avg_u_base - u_ref) / denom, 0, 1)

                # Compute L(u) = L_alpha * u + L_beta
                L_u = L_alpha * avg_u_base + L_beta

                # Compute U_limit
                U_limit = L_u + (100.0 - L_u) * np.tanh(best_kappa * xi) / np.tanh(best_kappa)

                # Compute p_i for each task
                p_list = [u / U_limit for u in u_base_list]

                # Compute P(idle) = prod(1 - p_i)
                P_idle = 1.0
                for p in p_list:
                    P_idle *= (1.0 - p)

                # Compute U_SM_ref
                U_sm_ref = U_limit * (1.0 - P_idle)
                physics_priors.append(U_sm_ref)

            # Build GBR features using validation_data directly
            # validation_data already has: u_base_list, u_ref, N, sm_util_actual
            gbr_feats = []
            gbr_targets = []

            # Load model complexity for workload signature
            complexity = load_model_complexity(model_name)
            model_info = complexity.get(model_name.lower(), {})
            gflops = model_info.get('gflops', 1000.0)
            params_M = model_info.get('params_M', 25.0)
            compute_memory_ratio = gflops / (params_M + 1.0)

            # Memory frequency from the physics_priors loop context
            mem_freq = MEM_FREQ_FIXED

            for data, prior in zip(validation_data, physics_priors):
                u_base_list = data['u_base_list']
                p_list = data['p_list']
                b_list = data['b_list']
                N = data['N']

                # Linear Sum: Σu_sm,i
                linear_sum = sum(u_base_list)

                # Heterogeneity: Var(r), Var(b)
                # Using p_list and b_list from validation_data
                var_r = np.var(p_list) if len(p_list) > 1 else 0.0
                var_b = np.var(b_list) if len(b_list) > 1 else 0.0
                b_mean = sum(b_list) / N if N > 0 else 1.0
                b_sum = sum(b_list)
                p_mean = sum(p_list) / N if N > 0 else 50.0
                p_sum = sum(p_list)

                # Physics Prior is already computed from the full formula
                # Û_SM^ref = U_limit * [1 - prod(1 - u_base_i/U_limit)]

                gbr_feats.append([
                    prior,                    # Physics Prior: Û_sm^ref
                    linear_sum,               # Linear Sum: Σu_sm,i
                    float(N),                 # Concurrency: N
                    var_r,                    # Heterogeneity: Var(r)
                    var_b,                    # Heterogeneity: Var(b)
                    b_mean,                   # Batch: mean batch size
                    b_sum,                    # Batch: total batch size
                    p_mean,                   # Partition: mean percentage
                    p_sum,                    # Partition: total percentage
                    float(mem_freq) / 10000.0,  # Memory Context: ω_mem (normalized)
                    gflops / 1000.0,          # Workload: GFLOPS (normalized)
                    params_M / 100.0,         # Workload: Parameters (normalized)
                    compute_memory_ratio,     # Workload: Compute-to-memory ratio
                ])
                gbr_targets.append(data['sm_util_actual'])

        gbr_targets_np = np.array(gbr_targets, dtype=np.float32)

        gbr = GradientBoostingRegressor(n_estimators=100, max_depth=5)
        gbr.fit(np.array(gbr_feats), gbr_targets_np)
        joblib.dump(gbr, os.path.join(model_dir, f"sm_gbr_model{file_suffix}.pkl"))

        # Compute metrics
        preds = gbr.predict(np.array(gbr_feats))
        mse = np.mean((preds - gbr_targets_np) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(preds - gbr_targets_np) / np.maximum(gbr_targets_np, 1.0)) * 100
        print(f"    - GBR trained on {len(gbr_feats)} samples")
        print(f"    - RMSE: {rmse:.4f}")
        print(f"    - MAPE: {mape:.2f}%")


# =========================================================================
#  MEM Model Fitting (Fixed Frequency)
# =========================================================================

def fit_mem_model(anchor_df, target_model=None, mem_freq_suffix=""):
    """
    Fit MEM utilization model.

    Args:
        anchor_df: DataFrame with anchor data
        target_model: Target model name to filter
        mem_freq_suffix: Suffix for model files based on memory frequency (e.g., "_mem9501")
    """
    print("\n" + "="*70 + "\nFITTING MEM MODEL (Fixed Frequency at Anchor)\n" + "="*70)
    print(f"  Memory frequency suffix: {mem_freq_suffix}" if mem_freq_suffix else "  Using default memory frequency")
    MAX_TASKS = 20

    # Build suffix string for filenames
    file_suffix = mem_freq_suffix

    if target_model:
        anchor_df = anchor_df[anchor_df['model_name'] == target_model]

    if len(anchor_df) == 0:
        print("  ⚠ No data found")
        return

    model_names = anchor_df['model_name'].unique()

    for model_name in model_names:
        if target_model and model_name != target_model:
            continue

        print(f"\n--- Model: {model_name} ---")
        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        df = anchor_df[anchor_df['model_name'] == model_name].copy()

        # Build single-task lookup table
        single_task_lookup = {}
        for _, row in df.iterrows():
            try:
                p_str, b_str = str(row['p_list']), str(row['b_list'])
                p_list = eval(p_str) if isinstance(p_str, str) and p_str.startswith('[') else [float(p_str)]
                b_list = eval(b_str) if isinstance(b_str, str) and b_str.startswith('[') else [int(b_str)]
                num_tasks = len(p_list)

                if num_tasks == 1:
                    key = (p_list[0], b_list[0])
                    single_task_lookup[key] = row['mem_util']
            except:
                continue

        print(f"  Built lookup table: {len(single_task_lookup)} single-task entries")

        # Stage 1: Base Demand Model
        print("  [Stage 1] Base Demand Model...")
        inputs_s1, targets_s1 = [], []
        for (p, b), u_mem in single_task_lookup.items():
            inputs_s1.append([p, b])
            targets_s1.append(u_mem)

        if len(inputs_s1) < 5:
            print("  ⚠ Insufficient single-task data")
            continue

        model_base = MEMBaseDemandModel()
        fitter_base = PhysicsModelFitter(model_base)
        fitter_base.fit(np.array(inputs_s1, dtype=np.float32),
                       np.array(targets_s1, dtype=np.float32),
                       epochs=2000, lr=0.1)

        torch.save(model_base.state_dict(), os.path.join(model_dir, f"mem_base_model{file_suffix}.pth"))

        # Stage 2: Physics Prior (Partition-Aware Point-Slope Manifold)
        print("  [Stage 2] Physics Prior (Partition-Aware Point-Slope Manifold)...")
        inputs_s2, targets_s2 = [], []

        for _, row in df.iterrows():
            try:
                p_str, b_str = str(row['p_list']), str(row['b_list'])
                p_list = eval(p_str) if isinstance(p_str, str) and p_str.startswith('[') else [float(p_str)]
                b_list = eval(b_str) if isinstance(b_str, str) and b_str.startswith('[') else [int(b_str)]
                N = len(p_list)
            except:
                continue

            u_base_list = []
            r_base_list = []
            b_base_list = []
            for p, b in zip(p_list, b_list):
                key = (p, b)
                u_base = single_task_lookup.get(key, 0.0)
                u_base_list.append(u_base)
                r_base_list.append(float(p))  # Partition ratio (MPS percentage)
                b_base_list.append(float(b))

            # Pad all arrays to MAX_TASKS
            u_base_padded = u_base_list + [0.0] * (MAX_TASKS - N)
            r_base_padded = r_base_list + [0.0] * (MAX_TASKS - N)
            b_base_padded = b_base_list + [0.0] * (MAX_TASKS - N)
            # Input format: [u_base_1..MAX, r_base_1..MAX, b_base_1..MAX, N]
            inputs_s2.append(u_base_padded + r_base_padded + b_base_padded + [float(N)])
            targets_s2.append(row['mem_util'])

        if len(inputs_s2) < 5:
            print("  ⚠ Insufficient multi-task data")
            continue

        model_sys = MEMSystemModel(max_tasks=MAX_TASKS)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_sys.to(device)
        opt = optim.Adam(model_sys.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        # Prepare inputs: includes u_base, r_base (partition ratios), b_base, and N
        inputs_s2_tensor = np.array(inputs_s2, dtype=np.float32)
        u_base_np = inputs_s2_tensor[:, :MAX_TASKS]           # Columns 0-MAX_TASKS: base utilizations
        r_base_np = inputs_s2_tensor[:, MAX_TASKS:2*MAX_TASKS]  # Columns MAX_TASKS-2*MAX: partition ratios
        b_base_np = inputs_s2_tensor[:, 2*MAX_TASKS:3*MAX_TASKS]  # Columns 2*MAX-3*MAX: batch sizes
        N_np = inputs_s2_tensor[:, -1]                         # Last column: N
        targets_s2_np = np.array(targets_s2, dtype=np.float32)

        # Print debug info
        print(f"    Data shape: inputs={inputs_s2_tensor.shape}, targets={targets_s2_np.shape}")
        print(f"    Target range: [{targets_s2_np.min():.2f}, {targets_s2_np.max():.2f}]")

        for epoch in range(500):
            opt.zero_grad()
            u_base_t = torch.from_numpy(u_base_np).float().to(device)
            r_base_t = torch.from_numpy(r_base_np).float().to(device)
            b_base_t = torch.from_numpy(b_base_np).float().to(device)
            N_t = torch.from_numpy(N_np).float().to(device)
            y_t = torch.from_numpy(targets_s2_np).float().to(device)

            pred = model_sys(u_base_t, r_base_t, b_base_t, N_t)
            loss = loss_fn(pred, y_t)
            loss.backward()

            # Print gradient stats for first epoch
            if epoch == 0:
                grad_norms = []
                for name, param in model_sys.named_parameters():
                    if param.grad is not None:
                        grad_norms.append(f"{name}={param.grad.norm().item():.4f}")
                print(f"    Initial gradients: {', '.join(grad_norms[:3])}...")

            # Use smaller gradient clipping for this model
            torch.nn.utils.clip_grad_norm_(model_sys.parameters(), max_norm=5.0)
            opt.step()

            if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == 499:
                print(f"    Epoch {epoch+1}/500: Loss={loss.item():.2f}")

        # Save with suffix
        torch.save(model_sys.state_dict(), os.path.join(model_dir, f"mem_system_model{file_suffix}.pth"))

        # Stage 3: Residual Calibration (GBR)
        print("  [Stage 3] Residual Calibration (GBR)...")
        with torch.no_grad():
            model_sys.eval()
            u_base_t = torch.from_numpy(u_base_np).float().to(device)
            r_base_t = torch.from_numpy(r_base_np).float().to(device)
            b_base_t = torch.from_numpy(b_base_np).float().to(device)
            N_t = torch.from_numpy(N_np).float().to(device)
            priors = model_sys(u_base_t, r_base_t, b_base_t, N_t).cpu().numpy()

        # Build GBR features according to Spec §3.1.1:
        # 1. Physics Prior: Û_mem^ref
        # 2. Linear Sum: Σu_mem,i (sum of base utilizations)
        # 3. Concurrency: N (number of tasks)
        # 4. Heterogeneity: Var(r), Var(b) (variance of partition ratios and batch sizes)
        # 5. Memory Context: ω_mem (memory frequency as proxy)
        # 6. Workload Signature: I_workload (GFLOPS, params_M for arithmetic intensity profile)
        gbr_feats = []
        gbr_targets = []
        N_values = []  # Track N for each included sample
        batch_mean_values = []  # Track batch_mean for each included sample
        filtered_count = 0

        # Load model complexity for workload signature
        complexity = load_model_complexity(model_name)
        model_info = complexity.get(model_name.lower(), {})
        gflops = model_info.get('gflops', 1000.0)  # Default to ~1G FLOPs if not found
        params_M = model_info.get('params_M', 25.0)  # Default to ~25M params

        for i, (_, row) in enumerate(df.iterrows()):
            try:
                p_str, b_str = str(row['p_list']), str(row['b_list'])
                p_list = eval(p_str) if isinstance(p_str, str) and p_str.startswith('[') else [float(p_str)]
                b_list = eval(b_str) if isinstance(b_str, str) and b_str.startswith('[') else [int(b_str)]
                N = len(p_list)
            except:
                N = 1
                p_list, b_list = [50], [1]

            prior = priors[i] if i < len(priors) else 0.0

            # Filter out anomalous data points
            target_mem = row['mem_util']
            if N > 1 and target_mem < 0.1:
                filtered_count += 1
                continue

            # Track N for this sample
            N_values.append(N)

            # Track batch_mean for this sample
            b_mean = sum(b_list) / N if N > 0 else 1.0
            batch_mean_values.append(b_mean)

            # Linear Sum: Σu_mem,i
            u_base_list = []
            for p, b in zip(p_list, b_list):
                key = (p, b)
                u_base = single_task_lookup.get(key, 0.0)
                u_base_list.append(u_base)
            linear_sum = sum(u_base_list)

            # Heterogeneity: Var(r), Var(b)
            var_r = np.var(p_list) if len(p_list) > 1 else 0.0
            var_b = np.var(b_list) if len(b_list) > 1 else 0.0
            b_sum = sum(b_list)

            # Memory Context: ω_mem (memory frequency as proxy)
            mem_freq = row.get('mem_freq', MEM_FREQ_FIXED)

            # Workload Signature: I_workload (arithmetic intensity profile)
            # Use GFLOPS and params_M to capture compute vs memory bound characteristics
            # Compute-to-memory ratio can indicate I/O bound tasks
            compute_memory_ratio = gflops / (params_M + 1.0)  # Higher = more compute intensive

            gbr_feats.append([
                prior,                    # Physics Prior: Û_mem^ref
                linear_sum,               # Linear Sum: Σu_mem,i
                float(N),                 # Concurrency: N
                var_r,                    # Heterogeneity: Var(r)
                var_b,                    # Heterogeneity: Var(b)
                b_mean,                   # Batch: mean batch size
                b_sum,                    # Batch: total batch size
                float(mem_freq) / 10000.0,  # Memory Context: ω_mem (normalized)
                gflops / 1000.0,          # Workload: GFLOPS (normalized)
                params_M / 100.0,         # Workload: Parameters (normalized)
                compute_memory_ratio,     # Workload: Compute-to-memory ratio
            ])
            gbr_targets.append(target_mem)

        if filtered_count > 0:
            print(f"    Filtered {filtered_count} anomalous data points")

        gbr_targets_np = np.array(gbr_targets, dtype=np.float32)

        gbr = GradientBoostingRegressor(n_estimators=100, max_depth=5)
        gbr.fit(np.array(gbr_feats), gbr_targets_np)
        joblib.dump(gbr, os.path.join(model_dir, f"mem_gbr_model{file_suffix}.pkl"))

        # Use numerically stable MAPE
        mape = np.mean(np.abs(gbr.predict(np.array(gbr_feats)) - gbr_targets_np) /
                       np.maximum(gbr_targets_np, 0.1)) * 100
        print(f"    Final MAPE: {mape:.2f}%")

        # ===== Validation Visualization =====
        predictions = gbr.predict(np.array(gbr_feats))
        actuals = gbr_targets_np

        if len(predictions) >= 5:
            # Compute metrics
            mse = np.mean((predictions - actuals) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - actuals))
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            max_ae = np.max(np.abs(predictions - actuals))

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Plot 1: Actual vs Predicted scatter
            ax1 = axes[0, 0]
            ax1.scatter(actuals, predictions, alpha=0.6, edgecolors='k', linewidth=0.5)
            min_val = min(actuals.min(), predictions.min())
            max_val = max(actuals.max(), predictions.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect fit')
            ax1.set_xlabel('Actual MEM Util (%)', fontsize=11)
            ax1.set_ylabel('Predicted MEM Util (%)', fontsize=11)
            ax1.set_title(f'Actual vs Predicted (R²={r2:.3f})', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Residual distribution
            ax2 = axes[0, 1]
            residuals = predictions - actuals
            ax2.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            ax2.axvline(x=0, color='r', linestyle='--', lw=2)
            ax2.set_xlabel('Prediction Error (%)', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title(f'Residual Distribution (MAE={mae:.2f}%)', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Plot 3: Error vs N (task count)
            ax3 = axes[1, 0]
            N_errors = {}
            for i, N in enumerate(N_values):
                if N not in N_errors:
                    N_errors[N] = []
                N_errors[N].append(abs(predictions[i] - actuals[i]))
            N_vals = sorted(N_errors.keys())
            mean_errors = [np.mean(N_errors[n]) for n in N_vals]
            ax3.bar(N_vals, mean_errors, color='coral', edgecolor='black')
            ax3.set_xlabel('Number of Tasks (N)', fontsize=11)
            ax3.set_ylabel('Mean Absolute Error (%)', fontsize=11)
            ax3.set_title('Prediction Error by Task Count', fontsize=12)
            ax3.grid(True, alpha=0.3, axis='y')

            # Plot 4: Actual vs Predicted by batch size
            ax4 = axes[1, 1]
            batch_errors = {}
            for i, batch in enumerate(batch_mean_values):
                err = abs(predictions[i] - actuals[i])
                if batch not in batch_errors:
                    batch_errors[batch] = []
                batch_errors[batch].append(err)
            batches = sorted(batch_errors.keys())
            mean_batch_errors = [np.mean(batch_errors[b]) for b in batches]
            ax4.bar(range(len(batches)), mean_batch_errors, color='mediumseagreen', edgecolor='black')
            ax4.set_xticks(range(len(batches)))
            ax4.set_xticklabels([str(b) for b in batches], rotation=45)
            ax4.set_xlabel('Batch Size', fontsize=11)
            ax4.set_ylabel('Mean Absolute Error (%)', fontsize=11)
            ax4.set_title('Prediction Error by Batch Size', fontsize=12)
            ax4.grid(True, alpha=0.3, axis='y')

            plt.suptitle(f'MEM Model Validation - {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig_path = os.path.join(model_dir, f"mem_physics_validation{file_suffix}.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    - Saved validation plot to {fig_path}")

            # Save validation results
            val_results = {
                'n_samples': len(predictions),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'max_abs_error': float(max_ae),
                'mape': float(mape)
            }
            val_path = os.path.join(model_dir, f"mem_physics_validation{file_suffix}.json")
            with open(val_path, 'w') as f:
                json.dump(val_results, f, indent=2)
            print(f"    - Saved validation results to {val_path}")


# =========================================================================
#  Throughput Model Fitting (Kinetic Scaling)
# =========================================================================

def fit_throughput_model(anchor_df, scaling_data, target_model=None, mem_freq_suffix="", target_mem_freq=None):
    """
    Fit throughput model.

    Args:
        anchor_df: DataFrame with anchor data
        scaling_data: Dictionary of scaling DataFrames by model name
        target_model: Target model name to filter
        mem_freq_suffix: Suffix for model files based on memory frequency (e.g., "_mem9501")
        target_mem_freq: Target memory frequency (MHz) for filtering scaling data
    """
    print("\n" + "="*70 + "\nFITTING THROUGHPUT MODEL (Kinetic Scaling)\n" + "="*70)
    print(f"  Memory frequency suffix: {mem_freq_suffix}" if mem_freq_suffix else "  Using default memory frequency")
    file_suffix = mem_freq_suffix

    if target_model:
        anchor_df = anchor_df[anchor_df['model_name'] == target_model]
        scaling_data = {k: v for k, v in scaling_data.items() if k == target_model}

    if len(anchor_df) == 0:
        print("  ⚠ No anchor data found")
        return

    model_names = anchor_df['model_name'].unique()

    for model_name in model_names:
        if target_model and model_name != target_model:
            continue

        print(f"\n--- Model: {model_name} ---")
        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        df_anchor = anchor_df[anchor_df['model_name'] == model_name]

        # Phase 1: Theoretical Prior & Residual Calibration
        print("  [Phase 1] Theoretical Prior & Residual Calibration...")

        # Load model complexity for workload signature (I_workload)
        complexity = load_model_complexity(model_name)
        model_info = complexity.get(model_name.lower(), {})
        gflops = model_info.get('gflops', 1000.0)
        params_M = model_info.get('params_M', 25.0)
        compute_memory_ratio = gflops / (params_M + 1.0)

        U_mem_anchor = df_anchor['mem_util'].values.astype(np.float32)
        T_anchor = df_anchor['throughput_total'].values.astype(np.float32)

        # Step 1a: Compute κ_ideal per batch size bucket
        # Problem: kappa (T/U_mem) varies significantly with batch size
        # Solution: Compute kappa_ideal for each batch bucket and fit a function
        print("    Computing κ_ideal per batch size...")

        # Parse batch sizes for all rows first
        # Use batch_mean for computing kappa_ideal (more general for mixed batch)
        batch_sizes = []
        for i, (_, row) in enumerate(df_anchor.iterrows()):
            try:
                b_str = str(row['b_list'])
                b_list = eval(b_str) if isinstance(b_str, str) and b_str.startswith('[') else [int(b_str)]
                N = len(b_list)
                batch_sum = sum(b_list)
                batch_mean = batch_sum / N if N > 0 else 1.0
                batch_sizes.append(batch_mean)
            except:
                batch_sizes.append(1.0)

        batch_sizes = np.array(batch_sizes)

        # Define batch size buckets
        batch_buckets = [8, 16, 32, 64, 128]
        kappa_per_batch = {}

        for bucket in batch_buckets:
            mask = (batch_sizes >= bucket * 0.8) & (batch_sizes < bucket * 1.2)
            if mask.sum() >= 5:  # Need at least 5 samples
                U_mem_bucket = U_mem_anchor[mask]
                T_bucket = T_anchor[mask]
                reg_bucket = LinearRegression(fit_intercept=False)
                reg_bucket.fit(U_mem_bucket.reshape(-1, 1), T_bucket)
                kappa_bucket = float(reg_bucket.coef_[0])
                kappa_per_batch[bucket] = kappa_bucket
                print(f"      Batch {bucket}: κ_ideal = {kappa_bucket:.4f} (n={mask.sum()})")

        # Fit linear function: kappa_ideal = a * log(batch) + b
        # This allows interpolation for any batch size
        if len(kappa_per_batch) >= 2:
            batch_vals = np.array(list(kappa_per_batch.keys()))
            kappa_vals = np.array(list(kappa_per_batch.values()))

            # Fit on log(batch) for better extrapolation
            log_batch = np.log(batch_vals.astype(float))
            reg_kappa = LinearRegression(fit_intercept=True)
            reg_kappa.fit(log_batch.reshape(-1, 1), kappa_vals)
            kappa_a = float(reg_kappa.coef_[0])
            kappa_b = float(reg_kappa.intercept_)

            print(f"    κ_ideal(batch) = {kappa_a:.4f} * log(batch) + {kappa_b:.4f}")
        else:
            # Fallback to single kappa_ideal
            reg_ideal = LinearRegression(fit_intercept=False)
            reg_ideal.fit(U_mem_anchor.reshape(-1, 1), T_anchor)
            kappa_a = 0.0
            kappa_b = float(reg_ideal.coef_[0])
            kappa_per_batch = {32: kappa_b}
            print(f"    ⚠ Using single κ_ideal = {kappa_b:.4f} (insufficient batch buckets)")

        # Save kappa function parameters
        kappa_ideal = kappa_b  # Default for prediction
        print(f"    κ_ideal (default) = {kappa_ideal:.4f}")

        # Step 1b: Compute theoretical prior with batch-aware kappa
        # T_prior = kappa_ideal(batch_mean) * U_mem
        T_prior = np.zeros_like(U_mem_anchor)
        for i, (u_mem, batch) in enumerate(zip(U_mem_anchor, batch_sizes)):
            if len(kappa_per_batch) >= 2:
                # Use fitted function with batch_mean
                kappa_i = kappa_a * np.log(batch + 1e-8) + kappa_b
            else:
                kappa_i = kappa_b
            T_prior[i] = kappa_i * u_mem

        # Step 1c: Parse anchor data to get N (concurrency) and workload info
        gbr_feats = []
        gbr_targets = []

        for i, (_, row) in enumerate(df_anchor.iterrows()):
            try:
                p_str, b_str = str(row['p_list']), str(row['b_list'])
                p_list = eval(p_str) if isinstance(p_str, str) and p_str.startswith('[') else [float(p_str)]
                b_list = eval(b_str) if isinstance(b_str, str) and b_str.startswith('[') else [int(b_str)]
                N = len(p_list)
            except:
                N = 1
                p_list, b_list = [50], [1]

            # Memory context: ω_mem (memory frequency as proxy)
            mem_freq = row.get('mem_freq', MEM_FREQ_FIXED)

            # Batch-related features (critical for throughput!)
            batch_first = float(b_list[0]) if len(b_list) > 0 else 1.0  # First task batch (for reference)
            batch_sum = sum(b_list)
            batch_mean = batch_sum / N if N > 0 else 1.0  # Mean batch (for kappa calculation)

            # Partition-related features
            p_first = float(p_list[0]) if len(p_list) > 0 else 50.0
            p_sum = sum(p_list)
            p_mean = p_sum / N if N > 0 else 50.0
            p_var = np.var(p_list) if len(p_list) > 1 else 0.0

            gbr_feats.append([
                T_prior[i],                      # Theoretical Prior: T̂_prior
                U_mem_anchor[i],                 # MEM Utilization: U_mem^ref
                float(N),                        # Concurrency: N
                float(mem_freq) / 10000.0,       # Memory Context: ω_mem (normalized)
                gflops / 1000.0,                 # Workload: GFLOPS (normalized)
                params_M / 100.0,                # Workload: Parameters (normalized)
                compute_memory_ratio,            # Workload: Compute-to-memory ratio
                batch_first,                     # Batch: first task batch size
                batch_sum,                       # Batch: total batch size
                batch_mean,                      # Batch: mean batch size
                p_first,                         # Partition: first task percentage
                p_sum,                           # Partition: total percentage
                p_mean,                          # Partition: mean percentage
                p_var,                           # Partition: variance
            ])
            gbr_targets.append(T_anchor[i])

        # Step 1d: Train GBR for residual calibration with hold-out validation
        gbr_targets_np = np.array(gbr_targets, dtype=np.float32)
        gbr_feats_np = np.array(gbr_feats, dtype=np.float32)

        if len(gbr_feats_np) >= 20:
            # Hold-out validation: 80% train, 20% test
            n_samples = len(gbr_feats_np)
            n_test = max(1, int(n_samples * 0.2))
            indices = np.random.permutation(n_samples)
            train_idx, test_idx = indices[n_test:], indices[:n_test]

            X_train = gbr_feats_np[train_idx]
            y_train = gbr_targets_np[train_idx]
            X_test = gbr_feats_np[test_idx]
            y_test = gbr_targets_np[test_idx]

            tp_gbr = GradientBoostingRegressor(n_estimators=100, max_depth=5)
            tp_gbr.fit(X_train, y_train)
            joblib.dump(tp_gbr, os.path.join(model_dir, f"throughput_gbr_model{file_suffix}.pkl"))

            # Compute metrics on test set (true validation)
            T_train = tp_gbr.predict(X_train)
            T_test = tp_gbr.predict(X_test)

            # Training metrics
            mape_train = np.mean(np.abs(T_train - y_train) / np.maximum(y_train, 1.0)) * 100
            ss_res_train = np.sum((y_train - T_train) ** 2)
            ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
            r2_train = 1.0 - (ss_res_train / ss_tot_train) if ss_tot_train > 0 else 0.0

            # Test metrics (true validation)
            mape_test = np.mean(np.abs(T_test - y_test) / np.maximum(y_test, 1.0)) * 100
            ss_res_test = np.sum((y_test - T_test) ** 2)
            ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
            r2_test = 1.0 - (ss_res_test / ss_tot_test) if ss_tot_test > 0 else 0.0

            print(f"    GBR trained on {len(X_train)} samples, validated on {len(X_test)} samples")
            print(f"    Train R² = {r2_train:.4f}, MAPE = {mape_train:.2f}%")
            print(f"    Test  R² = {r2_test:.4f}, MAPE = {mape_test:.2f}%")
        else:
            # Fallback to simple linear model if insufficient data
            print(f"    ⚠ Insufficient data for hold-out validation, using all data")
            tp_gbr = GradientBoostingRegressor(n_estimators=100, max_depth=5)
            tp_gbr.fit(gbr_feats_np, gbr_targets_np)
            joblib.dump(tp_gbr, os.path.join(model_dir, f"throughput_gbr_model{file_suffix}.pkl"))

            T_all = tp_gbr.predict(gbr_feats_np)
            mape_all = np.mean(np.abs(T_all - gbr_targets_np) / np.maximum(gbr_targets_np, 1.0)) * 100
            ss_res = np.sum((gbr_targets_np - T_all) ** 2)
            ss_tot = np.sum((gbr_targets_np - np.mean(gbr_targets_np)) ** 2)
            r2_all = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            print(f"    GBR R² = {r2_all:.4f}, MAPE = {mape_all:.2f}% (all data)")

        # Phase 2: Frequency Scaling (γ_t fitting)
        print("  [Phase 2] Frequency Scaling (γ_t)...")

        if model_name not in scaling_data:
            print("  ⚠ No scaling data found, using default γ_t = 0.8")
            gamma_t = 0.8
            T_start = T_anchor.min()
        else:
            df_scale = scaling_data[model_name].copy()

            # Filter to match anchor data memory frequency
            # Use target_mem_freq if provided, otherwise use available data
            if target_mem_freq is not None:
                df_scale = df_scale[df_scale['mem_freq'] == target_mem_freq].copy()
                print(f"    Filtered to mem_freq = {target_mem_freq} MHz")
            else:
                print(f"    Using all available scaling data (mem_freq filter not applied)")
            print(f"    Remaining rows: {len(df_scale)}")

            if len(df_scale) == 0:
                mem_freq_display = target_mem_freq if target_mem_freq else "any"
                print(f"  ⚠ No scaling data for mem_freq = {mem_freq_display} MHz, using default γ_t = 0.8")
                gamma_t = 0.8
                T_start = T_anchor.min()
            else:
                # Group by configuration (p_list, b_list combination)
                df_scale['config'] = df_scale['p_list'].astype(str) + '_' + df_scale['b_list'].astype(str)
                configs = df_scale['config'].unique()

                print(f"    Found {len(configs)} unique configurations")

                gamma_t_list = []
                mape_list = []

                # Fit γ_t for each configuration separately
                for config in configs:
                    df_cfg = df_scale[df_scale['config'] == config].sort_values('sm_freq')
                    f_values = df_cfg['sm_freq'].values.astype(np.float32)
                    T_values = df_cfg['throughput_total'].values.astype(np.float32)

                    # Skip if insufficient data points
                    if len(f_values) < 3:
                        continue

                    # Get T_start and T_ref for this config
                    T_start_cfg = T_values.min()
                    # Find closest to F_REF
                    idx_ref = np.argmin(np.abs(f_values - F_REF))
                    T_ref_cfg = T_values[idx_ref]

                    # Fit γ_t for this config using normalized formula
                    # T(f) = T_start + (T_ref - T_start) * ((f - f_min)/(f_ref - f_min))^γ_t
                    # Rearranged: ((T - T_start) / (T_ref - T_start)) = ((f - f_min)/(f_ref - f_min))^γ_t

                    # Prepare normalized data
                    freq_norm = np.clip((f_values - F_MIN) / (F_REF - F_MIN), 0, 1)
                    throughput_norm = np.clip((T_values - T_start_cfg) / (T_ref_cfg - T_start_cfg + 1e-6), 0, 2)

                    # Filter valid points (avoid division issues)
                    valid_mask = (freq_norm > 0.01) & (throughput_norm > 0.01) & (throughput_norm < 2.0)
                    if valid_mask.sum() < 3:
                        continue

                    freq_norm_valid = freq_norm[valid_mask]
                    throughput_norm_valid = throughput_norm[valid_mask]

                    # Fit γ_t using log-linear regression
                    # log(throughput_norm) = γ_t * log(freq_norm)
                    log_freq = np.log(freq_norm_valid + 1e-8)
                    log_throughput = np.log(throughput_norm_valid + 1e-8)

                    # Simple linear regression to find γ_t
                    gamma_t_cfg = np.sum(log_freq * log_throughput) / np.sum(log_freq ** 2)
                    gamma_t_cfg = np.clip(gamma_t_cfg, 0.1, 2.0)  # Reasonable range

                    # Calculate MAPE for this config
                    T_pred = T_start_cfg + (T_ref_cfg - T_start_cfg) * np.power(freq_norm, gamma_t_cfg)
                    mape_cfg = np.mean(np.abs(T_pred - T_values) / (T_values + 1e-6)) * 100

                    gamma_t_list.append(gamma_t_cfg)
                    mape_list.append(mape_cfg)

                if len(gamma_t_list) == 0:
                    print("  ⚠ No valid configurations, using default γ_t = 0.8")
                    gamma_t = 0.8
                    T_start = T_anchor.min()
                else:
                    # Use median γ_t (robust to outliers)
                    gamma_t = float(np.median(gamma_t_list))
                    avg_mape = float(np.mean(mape_list))

                    # Use overall T_start (minimum across all configs)
                    T_start = float(df_scale['throughput_total'].min())

                    print(f"    T_start = {T_start:.2f}")
                    print(f"    Fitted γ_t = {gamma_t:.4f} (median of {len(gamma_t_list)} configs)")
                    print(f"    Avg MAPE per config = {avg_mape:.2f}%")

        # Create model with fitted parameters for saving
        model_kinetic = ThroughputKineticModel(f_ref=F_REF, f_min=F_MIN)
        with torch.no_grad():
            model_kinetic.raw_kappa_bw.fill_(kappa_ideal)
            model_kinetic.raw_gamma_t.fill_(gamma_t)
            model_kinetic.raw_T_start.fill_(T_start)

        # Save model with suffix
        torch.save(model_kinetic.state_dict(), os.path.join(model_dir, f"throughput_kinetic_model{file_suffix}.pth"))

        # Save parameters
        params = {
            'kappa_ideal': kappa_ideal,
            'kappa_a': float(kappa_a) if 'kappa_a' in dir() else 0.0,
            'kappa_b': float(kappa_b),
            'kappa_per_batch': {str(k): float(v) for k, v in kappa_per_batch.items()} if len(kappa_per_batch) >= 2 else kappa_per_batch,
            'gamma_t': gamma_t,
            'T_start': float(T_start),
            'f_ref': F_REF,
            'f_min': F_MIN,
            'mem_freq_suffix': file_suffix,
            'batch_aware_kappa': len(kappa_per_batch) >= 2
        }
        with open(os.path.join(model_dir, f"throughput_params{file_suffix}.json"), 'w') as f:
            json.dump(params, f, indent=2)

        # ===== Validation Visualization =====
        # Use GBR-calibrated predictions
        if tp_gbr is not None:
            T_pred_anchor = tp_gbr.predict(gbr_feats_np)
            validation_type = "GBR-Calibrated"
        else:
            # Should not happen (data < 10 samples), but fallback
            T_pred_anchor = kappa_ideal * U_mem_anchor
            validation_type = "Ideal-Prior"

        T_actual_anchor = T_anchor

        if len(T_pred_anchor) >= 5:
            mse = np.mean((T_pred_anchor - T_actual_anchor) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(T_pred_anchor - T_actual_anchor))
            ss_res = np.sum((T_actual_anchor - T_pred_anchor) ** 2)
            ss_tot = np.sum((T_actual_anchor - np.mean(T_actual_anchor)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            mape = np.mean(np.abs(T_pred_anchor - T_actual_anchor) / np.maximum(T_actual_anchor, 1.0)) * 100

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Plot 1: Actual vs Predicted (throughput)
            ax1 = axes[0, 0]
            ax1.scatter(T_actual_anchor, T_pred_anchor, alpha=0.6, edgecolors='k', linewidth=0.5, color='purple')
            min_val = min(T_actual_anchor.min(), T_pred_anchor.min())
            max_val = max(T_actual_anchor.max(), T_pred_anchor.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect fit')
            ax1.set_xlabel('Actual Throughput (images/s)', fontsize=11)
            ax1.set_ylabel('Predicted Throughput (images/s)', fontsize=11)
            ax1.set_title(f'Actual vs Predicted (R²={r2:.3f})', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Residual distribution
            ax2 = axes[0, 1]
            residuals = T_pred_anchor - T_actual_anchor
            ax2.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='purple')
            ax2.axvline(x=0, color='r', linestyle='--', lw=2)
            ax2.set_xlabel('Prediction Error (images/s)', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title(f'Residual Distribution (MAE={mae:.2f})', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Plot 3: Residuals vs MEM utilization
            ax3 = axes[1, 0]
            ax3.scatter(U_mem_anchor, residuals, alpha=0.6, edgecolors='k', linewidth=0.5, color='orange')
            ax3.axhline(y=0, color='r', linestyle='--', lw=2)
            ax3.set_xlabel('MEM Utilization (%)', fontsize=11)
            ax3.set_ylabel('Residual (images/s)', fontsize=11)
            ax3.set_title('Residuals vs MEM Utilization', fontsize=12)
            ax3.grid(True, alpha=0.3)

            # Plot 4: Model parameters summary
            ax4 = axes[1, 1]
            ax4.axis('off')
            param_text = (
                f"Throughput Model ({validation_type})\n"
                f"{'='*35}\n\n"
                f"κ_ideal (ideal throughput coeff): {kappa_ideal:.4f}\n"
                f"γ_t (frequency exponent): {gamma_t:.4f}\n"
                f"T_start (idle throughput): {T_start:.2f}\n\n"
                f"Validation Metrics\n"
                f"{'='*35}\n"
                f"R² Score: {r2:.4f}\n"
                f"RMSE: {rmse:.4f}\n"
                f"MAE: {mae:.4f}\n"
                f"MAPE: {mape:.2f}%\n"
                f"Samples: {len(T_pred_anchor)}"
            )
            ax4.text(0.1, 0.5, param_text, fontsize=11, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.suptitle(f'Throughput Model Validation - {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig_path = os.path.join(model_dir, f"throughput_physics_validation{file_suffix}.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    - Saved validation plot to {fig_path}")

            # Save validation results
            val_results = {
                'n_samples': len(T_pred_anchor),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape),
                'kappa_ideal': float(kappa_ideal),
                'gamma_t': float(gamma_t),
                'validation_type': validation_type
            }
            val_path = os.path.join(model_dir, f"throughput_physics_validation{file_suffix}.json")
            with open(val_path, 'w') as f:
                json.dump(val_results, f, indent=2)
            print(f"    - Saved validation results to {val_path}")


# =========================================================================
#  Power Model Fitting (Thermodynamic)
# =========================================================================

def fit_power_model(anchor_df, scaling_data, static_power_df, complexity_table, target_model=None, mem_freq_suffix="", target_mem_freq=None):
    """
    Fit power model.

    Args:
        anchor_df: DataFrame with anchor data
        scaling_data: Dictionary of scaling DataFrames by model name
        static_power_df: DataFrame with static power data
        complexity_table: Dictionary of model complexity by model name
        target_model: Target model name to filter
        mem_freq_suffix: Suffix for model files based on memory frequency (e.g., "_mem9501")
        target_mem_freq: Target memory frequency (MHz) for filtering scaling data
    """
    print("\n" + "="*70 + "\nFITTING POWER MODEL (Thermodynamic)\n" + "="*70)
    print(f"  Memory frequency suffix: {mem_freq_suffix}" if mem_freq_suffix else "  Using default memory frequency")
    file_suffix = mem_freq_suffix

    if target_model:
        anchor_df = anchor_df[anchor_df['model_name'] == target_model]
        scaling_data = {k: v for k, v in scaling_data.items() if k == target_model}

    if len(anchor_df) == 0:
        print("  ⚠ No anchor data found")
        return

    # Phase 1: Static Power Baseline
    print("  [Phase 1] Static Power Baseline P_static(f)...")

    if len(static_power_df) > 0:
        # Fit cubic polynomial to static power data
        f_static = static_power_df['sm_freq'].values.astype(np.float32)
        P_static = static_power_df['power_max'].values.astype(np.float32)

        coeffs = np.polyfit(f_static, P_static, 3)
        print(f"    P_static(f) = {coeffs[0]:.2e}f³ + {coeffs[1]:.2e}f² + {coeffs[2]:.4f}f + {coeffs[3]:.2f}")
    else:
        # Use default coefficients from 3080Ti
        coeffs = [2.31e-8, -5.44e-5, 0.039, 85.6]
        print("    Using default static power coefficients")

    # Convert all coeffs to Python native float (not numpy float64)
    static_params = {
        'delta_3': float(np.array(coeffs[0]).item()),
        'delta_2': float(np.array(coeffs[1]).item()),
        'delta_1': float(np.array(coeffs[2]).item()),
        'delta_0': float(np.array(coeffs[3]).item())
    }

    # Phase 2 & 3: Dynamic Power for each model
    model_names = anchor_df['model_name'].unique()

    for model_name in model_names:
        if target_model and model_name != target_model:
            continue

        print(f"\n--- Model: {model_name} ---")
        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        df_anchor = anchor_df[anchor_df['model_name'] == model_name]

        # Determine frequency bounds for this model/platform
        f_ref_actual = F_REF
        f_min_actual = F_MIN
        df_scale_bounds = scaling_data.get(model_name) if isinstance(scaling_data, dict) else None
        if df_scale_bounds is not None and len(df_scale_bounds) > 0:
            df_scale_bounds = df_scale_bounds.copy()
            if target_mem_freq is not None and 'mem_freq' in df_scale_bounds.columns:
                df_scale_bounds = df_scale_bounds[df_scale_bounds['mem_freq'] == target_mem_freq]
            if len(df_scale_bounds) > 0 and 'sm_freq' in df_scale_bounds.columns:
                f_ref_actual = float(df_scale_bounds['sm_freq'].max())
                f_min_actual = float(df_scale_bounds['sm_freq'].min())
        elif len(df_anchor) > 0 and 'sm_freq' in df_anchor.columns:
            f_ref_actual = float(df_anchor['sm_freq'].max())
            f_min_actual = float(df_anchor['sm_freq'].min())
        if f_ref_actual <= f_min_actual:
            f_ref_actual = F_REF
            f_min_actual = F_MIN

        # Get model complexity
        params_M = complexity_table.get(model_name, {}).get('params_M', 25.0)

        # Phase 2: Dynamic Anchor (GBR)
        print("  [Phase 2] Dynamic Power Anchor (GBR)...")

        # Calculate P_static at f_ref
        P_static_ref = (static_params['delta_3'] * f_ref_actual**3 +
                       static_params['delta_2'] * f_ref_actual**2 +
                       static_params['delta_1'] * f_ref_actual +
                       static_params['delta_0'])

        # Calculate dynamic power target (using power_max instead of power_avg)
        P_measured = df_anchor['power_max'].values.astype(np.float32)
        P_dyn_target = P_measured - P_static_ref
        P_dyn_target = np.maximum(P_dyn_target, 0.0)  # Ensure non-negative

        # Train GBR for dynamic power at anchor
        gbr_feats = []
        for _, row in df_anchor.iterrows():
            try:
                p_str, b_str = str(row['p_list']), str(row['b_list'])
                p_list = eval(p_str) if isinstance(p_str, str) and p_str.startswith('[') else [float(p_str)]
                b_list = eval(b_str) if isinstance(b_str, str) and b_str.startswith('[') else [int(b_str)]
                N = len(p_list)
                p_mean = sum(p_list) / N
                p_var = np.var(p_list) if N > 1 else 0.0
                b_mean = sum(b_list) / N
            except:
                N = 1
                p_mean = 50.0
                p_var = 0.0
                b_mean = 1.0

            # Estimate total S based on N
            total_S = N * params_M

            # Memory frequency (normalized to 10000 MHz scale)
            mem_freq = float(row.get('mem_freq', 5001)) / 10000.0

            gbr_feats.append([
                row['sm_util'],
                row['mem_util'],
                float(N),
                total_S,
                p_mean,
                p_var,
                b_mean,
                mem_freq  # New feature: memory frequency
            ])

        gbr = GradientBoostingRegressor(n_estimators=100, max_depth=5)
        gbr.fit(np.array(gbr_feats), P_dyn_target)
        joblib.dump(gbr, os.path.join(model_dir, f"power_dyn_anchor_gbr{file_suffix}.pkl"))

        mape = np.mean(np.abs(gbr.predict(np.array(gbr_feats)) - P_dyn_target) / (P_dyn_target + 1e-6)) * 100
        print(f"    MAPE = {mape:.2f}%")

        # Phase 3: Dynamic Scaling (γ_p fitting)
        print("  [Phase 3] Dynamic Scaling (γ_p)...")

        if model_name not in scaling_data:
            print("  ⚠ No scaling data found, using default γ_p = 3.0")
            gamma_p = 3.0
            P_base = P_dyn_target.min()
        else:
            df_scale = scaling_data[model_name].copy()

            # Filter to match anchor data memory frequency
            # Use target_mem_freq if provided, otherwise use available data
            if target_mem_freq is not None:
                df_scale = df_scale[df_scale['mem_freq'] == target_mem_freq].copy()
                print(f"    Filtered to mem_freq = {target_mem_freq} MHz")
            else:
                print(f"    Using all available scaling data (mem_freq filter not applied)")
            print(f"    Remaining rows: {len(df_scale)}")

            if len(df_scale) == 0:
                mem_freq_display = target_mem_freq if target_mem_freq else "any"
                print(f"  ⚠ No scaling data for mem_freq = {mem_freq_display} MHz, using default γ_p = 3.0")
                gamma_p = 3.0
                P_base = P_dyn_target.min()
            else:
                # Aggregate power data by frequency (take median to handle outliers)
                # Group by frequency and take median of P_total (using power_max instead of power_avg)
                freq_groups = df_scale.groupby('sm_freq')['power_max'].agg(['median', 'mean', 'std', 'count']).reset_index()
                freq_groups = freq_groups.sort_values('sm_freq')

                f_values = freq_groups['sm_freq'].values.astype(np.float32)
                P_total_scale = freq_groups['median'].values.astype(np.float32)

                print(f"    Aggregated {len(df_scale)} raw points → {len(f_values)} frequency bins")

                # Calculate static and dynamic power at each frequency
                P_static_scale = (static_params['delta_3'] * f_values**3 +
                                 static_params['delta_2'] * f_values**2 +
                                 static_params['delta_1'] * f_values +
                                 static_params['delta_0'])

                P_dyn_scale = P_total_scale - P_static_scale

                # Filter out invalid P_dyn values (negative or too small)
                valid_mask = P_dyn_scale > 5.0  # Minimum 5W dynamic power
                if valid_mask.sum() < 3:
                    print("  ⚠ Too few valid P_dyn points, using default γ_p = 3.0")
                    gamma_p = 3.0
                    P_base = P_dyn_target.min()
                else:
                    f_values = f_values[valid_mask]
                    P_dyn_scale = P_dyn_scale[valid_mask]
                    P_static_scale = P_static_scale[valid_mask]

                    # Get P_dyn at f_ref (interpolate if needed)
                    if f_ref_actual in f_values:
                        idx = np.where(f_values == f_ref_actual)[0]
                        P_dyn_ref = P_dyn_scale[idx[0]]
                        f_ref_actual = float(f_ref_actual)
                    else:
                        P_dyn_ref = np.interp(f_ref_actual, np.sort(f_values), P_dyn_scale[np.argsort(f_values)])
                        f_ref_actual = float(f_ref_actual)

                    # Get P_base at f_min
                    if f_min_actual in f_values:
                        idx = np.where(f_values == f_min_actual)[0]
                        P_base = P_dyn_scale[idx[0]]
                        f_base_actual = float(f_min_actual)
                    else:
                        # Find closest to F_MIN
                        idx_min = np.argmin(np.abs(f_values - f_min_actual))
                        P_base = P_dyn_scale[idx_min]
                        f_base_actual = f_values[idx_min]

                    # Debug: print frequency range
                    print(f"    P_base from f={f_base_actual:.0f} MHz (closest to f_min={f_min_actual:.0f})")
                    print(f"    P_dyn_ref from f={f_ref_actual:.0f} MHz (interpolated={f_ref_actual in f_values})")
                    print(f"    P_base = {P_base:.2f}, P_dyn_ref = {P_dyn_ref:.2f}")
                    print(f"    Valid P_dyn range: [{P_dyn_scale.min():.1f}, {P_dyn_scale.max():.1f}] W")
                    print(f"    Frequency range: [{f_values.min():.0f}, {f_values.max():.0f}] MHz")

                    # Debug: check if P_dyn_scale exceeds P_dyn_ref significantly
                    max_excess = P_dyn_scale.max() - P_dyn_ref
                    if max_excess > 50:
                        print(f"    ⚠ Warning: Max P_dyn ({P_dyn_scale.max():.1f}W) exceeds P_dyn_ref ({P_dyn_ref:.2f}W) by {max_excess:.1f}W")
                        print(f"    This suggests high-load configs have much higher power than median at same frequency")
                        print(f"    Consider using max instead of median for conservative estimation")

                    # Fit γ_p using log-linear regression on normalized values
                    # P_dyn(f) = P_base + (P_dyn_ref - P_base) * ((f - f_min)/(f_ref - f_min))^γ_p
                    # Rearranged: ((P_dyn - P_base) / (P_dyn_ref - P_base)) = ((f - f_min)/(f_ref - f_min))^γ_p

                    freq_norm = np.clip((f_values - f_min_actual) / (f_ref_actual - f_min_actual), 0.01, 1.0)
                    power_norm = np.clip((P_dyn_scale - P_base) / (P_dyn_ref - P_base + 1e-6), 0.01, 2.0)

                    # Debug: print some normalized values
                    print(f"    freq_norm range: [{freq_norm.min():.3f}, {freq_norm.max():.3f}]")
                    print(f"    power_norm range: [{power_norm.min():.3f}, {power_norm.max():.3f}]")
                    print(f"    P_dyn_ref - P_base = {P_dyn_ref - P_base:.2f} W")

                    # Filter valid normalized values
                    valid_mask = (power_norm > 0.01) & (power_norm < 2.0)
                    if valid_mask.sum() < 3:
                        print("  ⚠ Insufficient valid normalized points, using default γ_p = 3.0")
                        gamma_p = 3.0
                    else:
                        freq_norm_valid = freq_norm[valid_mask]
                        power_norm_valid = power_norm[valid_mask]

                        # Fit γ_p using log-linear regression
                        log_freq = np.log(freq_norm_valid)
                        log_power = np.log(power_norm_valid)

                        gamma_p = np.sum(log_freq * log_power) / np.sum(log_freq ** 2)
                        gamma_p = np.clip(gamma_p, 1.0, 5.0)  # Reasonable range for power scaling

                        # Convert to Python native float
                        gamma_p = float(np.array(gamma_p).item())

                        # Calculate MAPE
                        P_pred = P_base + (P_dyn_ref - P_base) * np.power(freq_norm, gamma_p)
                        mape = np.mean(np.abs(P_pred - P_dyn_scale) / np.maximum(P_dyn_scale, 10.0)) * 100

                        # Debug: print error breakdown
                        errors = np.abs(P_pred - P_dyn_scale)
                        print(f"    Error range: [{errors.min():.1f}, {errors.max():.1f}] W")
                        print(f"    Fitted γ_p = {gamma_p:.4f}")
                        print(f"    MAPE = {mape:.2f}%")

        # Create model with fitted parameters for saving
        model_scale = PowerDynamicScalingModel(f_ref=f_ref_actual, f_min=f_min_actual)
        with torch.no_grad():
            model_scale.raw_gamma_p.fill_(gamma_p)
            model_scale.raw_P_base.fill_(P_base)

        # Save scaling model with suffix
        torch.save(model_scale.state_dict(), os.path.join(model_dir, f"power_dyn_scale_model{file_suffix}.pth"))

        # Save all parameters (ensure Python native types)
        params = {
            'static': static_params,
            'dynamic': {
                'gamma_p': float(np.array(gamma_p).item()) if not isinstance(gamma_p, float) else gamma_p,
                'P_base': float(np.array(P_base).item()) if not isinstance(P_base, float) else float(P_base),
                'f_ref': float(f_ref_actual),
                'f_min': float(f_min_actual)
            },
            'mem_freq_suffix': file_suffix
        }
        with open(os.path.join(model_dir, f"power_params{file_suffix}.json"), 'w') as f:
            json.dump(params, f, indent=2)

        # ===== Validation Visualization =====
        # Validate GBR fit on anchor data
        P_dyn_pred = gbr.predict(np.array(gbr_feats))
        P_dyn_actual = P_dyn_target

        if len(P_dyn_pred) >= 5:
            mse = np.mean((P_dyn_pred - P_dyn_actual) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(P_dyn_pred - P_dyn_actual))
            ss_res = np.sum((P_dyn_actual - P_dyn_pred) ** 2)
            ss_tot = np.sum((P_dyn_actual - np.mean(P_dyn_actual)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            mape = np.mean(np.abs(P_dyn_pred - P_dyn_actual) / np.maximum(P_dyn_actual, 1.0)) * 100

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Plot 1: Actual vs Predicted (dynamic power)
            ax1 = axes[0, 0]
            ax1.scatter(P_dyn_actual, P_dyn_pred, alpha=0.6, edgecolors='k', linewidth=0.5, color='brown')
            min_val = min(P_dyn_actual.min(), P_dyn_pred.min())
            max_val = max(P_dyn_actual.max(), P_dyn_pred.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect fit')
            ax1.set_xlabel('Actual Dynamic Power (W)', fontsize=11)
            ax1.set_ylabel('Predicted Dynamic Power (W)', fontsize=11)
            ax1.set_title(f'Actual vs Predicted (R²={r2:.3f})', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Residual distribution
            ax2 = axes[0, 1]
            residuals = P_dyn_pred - P_dyn_actual
            ax2.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='brown')
            ax2.axvline(x=0, color='r', linestyle='--', lw=2)
            ax2.set_xlabel('Prediction Error (W)', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title(f'Residual Distribution (MAE={mae:.2f}W)', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Plot 3: Residuals vs SM utilization
            ax3 = axes[1, 0]
            sm_util_values = [f[0] for f in gbr_feats]
            ax3.scatter(sm_util_values, residuals, alpha=0.6, edgecolors='k', linewidth=0.5, color='forestgreen')
            ax3.axhline(y=0, color='r', linestyle='--', lw=2)
            ax3.set_xlabel('SM Utilization (%)', fontsize=11)
            ax3.set_ylabel('Residual (W)', fontsize=11)
            ax3.set_title('Residuals vs SM Utilization', fontsize=12)
            ax3.grid(True, alpha=0.3)

            # Plot 4: Model parameters summary
            ax4 = axes[1, 1]
            ax4.axis('off')
            P_static_ref = (static_params['delta_3'] * F_REF**3 +
                           static_params['delta_2'] * F_REF**2 +
                           static_params['delta_1'] * F_REF +
                           static_params['delta_0'])
            param_text = (
                f"Power Model Parameters\n"
                f"{'='*30}\n\n"
                f"Static (P_static at {F_REF}MHz): {P_static_ref:.2f}W\n\n"
                f"Dynamic GBR:\n"
                f"  γ_p (scaling exponent): {gamma_p:.4f}\n"
                f"  P_base (idle power): {P_base:.2f}W\n\n"
                f"Validation Metrics\n"
                f"{'='*30}\n"
                f"R² Score: {r2:.4f}\n"
                f"RMSE: {rmse:.4f}\n"
                f"MAE: {mae:.4f}\n"
                f"MAPE: {mape:.2f}%\n"
                f"Samples: {len(P_dyn_pred)}"
            )
            ax4.text(0.1, 0.5, param_text, fontsize=11, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.suptitle(f'Power Model Validation - {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig_path = os.path.join(model_dir, f"power_physics_validation{file_suffix}.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    - Saved validation plot to {fig_path}")

            # Save validation results
            val_results = {
                'n_samples': len(P_dyn_pred),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape),
                'gamma_p': float(gamma_p) if isinstance(gamma_p, float) else float(np.array(gamma_p).item()),
                'P_base': float(P_base) if isinstance(P_base, float) else float(np.array(P_base).item())
            }
            val_path = os.path.join(model_dir, f"power_physics_validation{file_suffix}.json")
            with open(val_path, 'w') as f:
                json.dump(val_results, f, indent=2)
            print(f"    - Saved validation results to {val_path}")


# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Physics-Informed Parameter Optimization")
    parser.add_argument('--step', type=str, default='all',
                        choices=['sm', 'mem', 'power', 'throughput', 'all'],
                        help="Training step (default: all)")
    parser.add_argument('--model', type=str, default=None,
                        help="Target model name (e.g., mobilenet_v2)")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Dataset name for training data (e.g., imagenet, coco)")
    parser.add_argument('--platform', type=str, default=None,
                        help="Platform/GPU model for training data (e.g., a100, 3090, 3080ti)")
    parser.add_argument('--mem-freq', type=float, default=None,
                        help="Memory frequency (MHz). If specified, trains model for this specific frequency")
    parser.add_argument('--train-all-mem-freq', action='store_true',
                        help="[Deprecated] Default behavior now trains for all memory frequencies. "
                             "Use --mem-freq to train for a specific frequency only.")
    args = parser.parse_args()

    target_model = args.model.lower() if args.model else None
    # Build suffixes: dataset and platform are both for filtering data AND naming models
    dataset_suffix = f"_{args.dataset}" if args.dataset else ""
    platform_suffix = f"_{args.platform}" if args.platform else ""

    print("="*70)
    print("Physics-Informed Parameter Optimization (Anchor-Based)")
    print("Spec: PctoDL_System_Spec.md §3.1")
    print("="*70)

    # Determine which data to load based on step
    step = args.step
    needs_anchor = step in ['sm', 'mem', 'throughput', 'power', 'all']
    needs_scaling = step in ['throughput', 'power', 'all']
    needs_static = step in ['power', 'all']

    print("\n[1] Loading Data...")
    print(f"    Dataset: {args.dataset or 'default'}")
    print(f"    Platform: {args.platform or 'default'}")

    # Load anchor data with dataset/platform filters
    anchor_df = load_anchor_data(dataset=args.dataset, machine=args.platform) if needs_anchor else pd.DataFrame()

    if anchor_df is None:
        # Error already printed in load_anchor_data
        return

    if len(anchor_df) == 0:
        print("\n⚠ Error: No anchor data found!")
        print(f"    Requested: dataset={args.dataset or 'all'}, platform={args.platform or 'all'}")
        print("    Please ensure data files exist: mps_profile/mps_results*.csv")
        return

    # Get available memory frequencies
    mem_freqs = get_available_mem_frequencies(anchor_df)
    print(f"    Available memory frequencies: {mem_freqs}")

    if args.mem_freq:
        # Train for specific memory frequency only
        target_mem_freq = args.mem_freq
        if target_mem_freq not in mem_freqs:
            print(f"\n⚠ Error: Memory frequency {target_mem_freq} MHz not found!")
            print(f"    Requested: --mem-freq {target_mem_freq}")
            print(f"    Available in loaded data: {mem_freqs}")
            print(f"    Note: This platform/dataset may not have data at this frequency.")
            return
        anchor_df = filter_by_mem_freq(anchor_df, target_mem_freq)
        if len(anchor_df) == 0:
            print(f"\n⚠ Error: No data available at mem_freq = {target_mem_freq} MHz!")
            print(f"    Filtered dataset is empty.")
            return
        print(f"\n    Filtering to mem_freq = {target_mem_freq} MHz: {len(anchor_df)} rows")
        mem_freqs_to_train = [target_mem_freq]
    else:
        # Default behavior: train for ALL memory frequencies in the data
        print(f"\n    [Info] Auto-detected {len(mem_freqs)} memory frequencies: {mem_freqs}")
        print(f"    [Info] Will train models for ALL memory frequencies")
        mem_freqs_to_train = mem_freqs
        # Don't filter anchor_df yet, will filter per frequency in training loop

    scaling_data = load_scaling_data(target_model, dataset=args.dataset, platform=args.platform) if needs_scaling else {}
    static_power_df = load_static_power_data(
        scaling_data,
        platform=args.platform,
        target_mem_freq=args.mem_freq
    ) if needs_static else pd.DataFrame()
    complexity_table = load_model_complexity(target_model) if needs_static else {}

    print(f"\n    Anchor Data: {len(anchor_df)} rows")
    if needs_scaling:
        print(f"    Scaling Data: {len(scaling_data)} models")
    if needs_static:
        print(f"    Static Power: {len(static_power_df)} rows")
        print(f"    Complexity Table: {len(complexity_table)} models")

    # Function to run fitting with specified suffixes
    def run_fitting(df, mem_freq_suffix="", mem_freq_value=None):
        # Combine dataset, platform, and mem_freq suffixes
        # Order: {dataset}_{platform}_mem{xxx}
        full_suffix = dataset_suffix + platform_suffix + mem_freq_suffix
        dataset_info = f"{dataset_suffix[1:]}" if dataset_suffix else "default"
        platform_info = f"platform={platform_suffix[1:]}" if platform_suffix else "default"
        print(f"\n--- Training dataset={dataset_info}, {platform_info} "
              f"{mem_freq_suffix and mem_freq_suffix[1:] or ''} ---")

        # Reload scaling_data for this specific mem_freq (to get correct platform-specific data)
        scaling_data_local = {}
        if args.step in ['throughput', 'power', 'all'] and needs_scaling:
            scaling_data_local = load_scaling_data(target_model, dataset=args.dataset,
                                                    platform=args.platform, target_mem_freq=mem_freq_value)
        static_power_df_local = pd.DataFrame()
        if args.step in ['power', 'all'] and needs_static:
            static_power_df_local = load_static_power_data(
                scaling_data_local,
                platform=args.platform,
                target_mem_freq=mem_freq_value
            )

        if args.step in ['sm', 'all']:
            fit_sm_model(df, target_model, mem_freq_suffix=full_suffix)
        if args.step in ['mem', 'all']:
            fit_mem_model(df, target_model, mem_freq_suffix=full_suffix)
        if args.step in ['throughput', 'all']:
            fit_throughput_model(df, scaling_data_local, target_model, mem_freq_suffix=full_suffix, target_mem_freq=mem_freq_value)
        if args.step in ['power', 'all']:
            fit_power_model(
                df,
                scaling_data_local,
                static_power_df_local,
                complexity_table,
                target_model,
                mem_freq_suffix=full_suffix,
                target_mem_freq=mem_freq_value
            )

    # Train models for each memory frequency
    print(f"\n[2] Training models for {len(mem_freqs_to_train)} memory frequency/frequencies...")
    for mem_freq in mem_freqs_to_train:
        df_freq = filter_by_mem_freq(anchor_df, mem_freq)
        if len(df_freq) >= 5:  # Require minimum data points
            run_fitting(df_freq, mem_freq_suffix=f"_mem{int(mem_freq)}", mem_freq_value=mem_freq)
        else:
            print(f"  ⚠ Skipping mem_freq={mem_freq}: insufficient data ({len(df_freq)} rows)")

    print("\n" + "="*70)
    print("Fitting Complete! Results saved to:", OUTPUT_DIR)
    print("="*70)


if __name__ == "__main__":
    main()
