#!/usr/bin/env python3
"""
Thermodynamic Model-Based Control for power-capped frequency management.

Implements the Hybrid Thermodynamic Control strategy from the paper:
1. Predictive Feedforward Target: Analytical inversion of power model
2. Jacobian-Guided Feedback Correction: Adaptive sensitivity scaling

Control law:
    f_hat = f_min + (f_ref - f_min) * ((P_cap - P_base) / P_dyn_ref)^(1/gamma_p)
    f_next = Clip(f_hat + eta * integral_error / jacobian, f_min, f_max)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from predict import PctoDLPredictor


@dataclass
class ControlState:
    """State container for the thermodynamic controller."""
    f_curr: float
    p_meas: float
    error: float
    jacobian: float
    f_feedforward: float  # Predictive feedforward target
    f_next: float
    p_dyn_ref: float      # Reference dynamic power for Jacobian calculation


class ThermodynamicModelController:
    """
    Hybrid Thermodynamic Controller with Predictive Feedforward + Jacobian Feedback.

    Unlike conventional PID controllers, this controller:
    1. Computes physics-informed feedforward target by inverting the power model
    2. Uses Jacobian (dP/df) to scale feedback correction for adaptive sensitivity
    """

    def __init__(
        self,
        p_cap: float,
        predictor: Optional[PctoDLPredictor] = None,
        eta: float = 0.8,  # Damping factor for Newton step
        epsilon: float = 1e-6,
        f_min: Optional[float] = None,
        f_max: Optional[float] = None,
        max_delta_f: float = 300.0,  # Max frequency change per iteration
        min_step_f: float = 15.0,  # Minimal step size for fine-tuning near target
        proximity_threshold: float = 7.0,  # W - threshold for switching to minimal step mode
    ) -> None:
        self.p_cap = float(p_cap)
        self.predictor = predictor
        self.eta = float(eta)
        self.epsilon = float(epsilon)
        self.max_delta_f = float(max_delta_f)
        self.min_step_f = float(min_step_f)
        self.proximity_threshold = float(proximity_threshold)

        # Track p_cap changes for Stage 1 (direct jump on setpoint change)
        self._prev_p_cap = float(p_cap)
        self._cap_just_changed = True  # First iteration should always jump to feedforward target

        self.f_min, self.f_max = self._init_bounds(f_min, f_max)

    def _init_bounds(self, f_min: Optional[float], f_max: Optional[float]) -> Tuple[float, float]:
        if self.predictor:
            dyn = self.predictor.power_params.get('dynamic', {})
            model_f_min = float(dyn.get('f_min', 210.0))
            model_f_max = float(dyn.get('f_ref', 1950.0))
        else:
            model_f_min = 210.0
            model_f_max = 1950.0

        return (
            float(f_min) if f_min is not None else model_f_min,
            float(f_max) if f_max is not None else model_f_max,
        )

    def reset(self) -> None:
        """Reset controller state. Typically called when operating point changes."""
        # Trigger direct jump on next iteration (assumes reset is called due to setpoint change)
        self._cap_just_changed = True

    def _get_dynamic_params(self) -> Tuple[float, float, float, float]:
        if not self.predictor:
            return 3.0, 10.0, self.f_max, self.f_min
        dyn = self.predictor.power_params.get('dynamic', {})
        gamma_p = float(dyn.get('gamma_p', 3.0))
        p_base = float(dyn.get('P_base', 10.0))
        f_ref = float(dyn.get('f_ref', self.f_max))
        f_min = float(dyn.get('f_min', self.f_min))
        return gamma_p, p_base, f_ref, f_min

    def _estimate_util(self, p_list: Optional[Sequence[float]], b_list: Optional[Sequence[int]]) -> Tuple[float, float]:
        if not self.predictor or not p_list or not b_list:
            return 50.0, 50.0
        return self.predictor.predict_sm(p_list, b_list), self.predictor.predict_mem(p_list, b_list)

    def _estimate_p_dyn_ref(
        self,
        f_ref: float,
        u_sm: float,
        u_mem: float,
        n_tasks: int,
        p_list: Optional[Sequence[float]],
        b_list: Optional[Sequence[int]],
        p_base: float,
    ) -> float:
        """Estimate reference dynamic power at f_ref for Jacobian calculation."""
        if not self.predictor or self.predictor.power_dyn_gbr is None:
            return max(p_base, self.epsilon)
        p_total = self.predictor.predict_power_dynamic(u_sm, u_mem, n_tasks, f_ref, p_list, b_list)
        p_static = self.predictor.predict_power_static(f_ref)
        return max(float(p_total - p_static), self.epsilon)

    def _jacobian(
        self,
        f_curr: float,
        p_dyn_ref: float,
        gamma_p: float,
        p_base: float,
        f_ref: float,
        f_min: float,
        p_list: Optional[Sequence[float]] = None,
        b_list: Optional[Sequence[int]] = None,
        u_sm: Optional[float] = None,
        u_mem: Optional[float] = None,
        n_tasks: Optional[int] = None,
    ) -> float:
        """
        Compute Jacobian dP_total/df at current frequency.

        J(f) = dP_static/df + dP_dyn/df

        Static: P_static(f) = δ₁f + δ₂f² + δ₃f³
                dP_static/df = δ₁ + 2δ₂f + 3δ₃f²

        Dynamic: P_dyn(f) = P_base + (P_dyn_ref - P_base) * ((f - f_min) / (f_ref - f_min))^gamma_p
                 dP_dyn/df = (P_dyn_ref - P_base) * gamma_p * ((f - f_min) / (f_ref - f_min))^(gamma_p-1) / (f_ref - f_min)
        """
        # Static gradient: dP_static/df = δ₁ + 2δ₂f + 3δ₃f²
        static_gradient = 0.0
        if self.predictor:
            static = self.predictor.power_params.get('static', {})
            delta_3 = float(static.get('delta_3', 0.0))
            delta_2 = float(static.get('delta_2', 0.0))
            delta_1 = float(static.get('delta_1', 0.0))
            static_gradient = delta_1 + 2.0 * delta_2 * f_curr + 3.0 * delta_3 * (f_curr ** 2)

        # Dynamic gradient
        denom = max((f_ref - f_min) ** gamma_p, self.epsilon)
        base = max(f_curr - f_min, self.epsilon) ** (gamma_p - 1.0)
        dynamic_gradient = gamma_p * (p_dyn_ref - p_base) / denom * base

        # Total Jacobian (ensure minimum value for stability)
        return max(static_gradient + dynamic_gradient, self.epsilon)

    def _feedforward_target(
        self,
        p_cap: float,
        p_base: float,
        p_dyn_ref: float,
        gamma_p: float,
        f_ref: float,
        f_min: float,
        p_list: Optional[Sequence[float]] = None,
        b_list: Optional[Sequence[int]] = None,
        n_tasks: Optional[int] = None,
        u_sm: Optional[float] = None,
        u_mem: Optional[float] = None,
    ) -> float:
        """
        Compute predictive feedforward target by inverting the total power model.

        Total power: P_total(f) = P_static(f) + P_dyn(f)

        For feedforward, we use the simplified proxy inversion:
            P_total(f) ≈ P_total(f_min) + (P_total(f_ref) - P_total(f_min)) * ((f - f_min) / (f_ref - f_min))^gamma_p

        Solve for f:
            f = f_min + (f_ref - f_min) * ((P_cap - P_total(f_min)) / (P_total(f_ref) - P_total(f_min)))^(1/gamma_p)

        This accounts for both static and dynamic power components.
        """
        # Get total power at f_min and f_ref using the predictor
        if self.predictor and p_list and b_list:
            _, _, _, p_total_f_min = self.predictor.predict_all(p_list, b_list, f_min)
            _, _, _, p_total_f_ref = self.predictor.predict_all(p_list, b_list, f_ref)
        else:
            # Fallback to dynamic-only model if no predictor
            p_total_f_min = float(p_base)
            p_total_f_ref = float(p_base + p_dyn_ref)

        # If p_cap is below total power at f_min, target is f_min
        if p_cap <= p_total_f_min:
            return f_min

        # If p_cap is at or above total power at f_ref, cap at f_ref
        power_range = max(p_total_f_ref - p_total_f_min, self.epsilon)
        if p_cap >= p_total_f_ref:
            return f_ref

        # Inverse power model using total power
        power_ratio = (p_cap - p_total_f_min) / power_range
        # Clamp ratio to valid range [0, 1] to avoid numerical issues
        power_ratio = max(0.0, min(1.0, power_ratio))

        f_target = f_min + (f_ref - f_min) * (power_ratio ** (1.0 / gamma_p))
        return float(np.clip(f_target, f_min, f_ref))

    def step(
        self,
        f_curr: float,
        p_meas: float,
        dt: float = 1.0,
        u_sm: Optional[float] = None,
        u_mem: Optional[float] = None,
        p_list: Optional[Sequence[float]] = None,
        b_list: Optional[Sequence[int]] = None,
        n_tasks: Optional[int] = None,
    ) -> ControlState:
        """
        Perform one control step to compute the next SM frequency.

        The controller follows the hybrid thermodynamic control strategy:
        1. Compute feedforward target from inverted power model
        2. Accumulate integral error scaled by Jacobian
        3. Combine feedforward + feedback for final frequency
        """
        f_curr = float(f_curr)
        p_meas = float(p_meas)
        dt = float(dt)

        if u_sm is None or u_mem is None:
            u_sm, u_mem = self._estimate_util(p_list, b_list)

        if n_tasks is None:
            n_tasks = len(p_list) if p_list else 1

        gamma_p, p_base, f_ref, f_min = self._get_dynamic_params()
        p_dyn_ref = self._estimate_p_dyn_ref(f_ref, u_sm, u_mem, n_tasks, p_list, b_list, p_base)

        error = self.p_cap - p_meas

        # Detect p_cap change for Stage 1 (direct jump on setpoint change)
        # Preserve flag if it was set externally (from __init__ or reset())
        if self._cap_just_changed:
            # Flag was set externally, use it and update tracking
            self._prev_p_cap = self.p_cap
            # Keep _cap_just_changed = True for this iteration
        else:
            # Check for p_cap change
            if abs(self.p_cap - self._prev_p_cap) > 1e-3:
                self._cap_just_changed = True
                self._prev_p_cap = self.p_cap
                self.integral = 0.0
            else:
                self._cap_just_changed = False

        # Step 1: Compute predictive feedforward target
        f_feedforward = self._feedforward_target(
            p_cap=self.p_cap,
            p_base=p_base,
            p_dyn_ref=p_dyn_ref,
            gamma_p=gamma_p,
            f_ref=f_ref,
            f_min=f_min,
            p_list=p_list,
            b_list=b_list,
            n_tasks=n_tasks,
            u_sm=u_sm,
            u_mem=u_mem,
        )

        # Step 2: Compute Jacobian for feedback scaling
        jacobian = self._jacobian(
            f_curr=f_curr,
            p_dyn_ref=p_dyn_ref,
            gamma_p=gamma_p,
            p_base=p_base,
            f_ref=f_ref,
            f_min=f_min,
            p_list=p_list,
            b_list=b_list,
            u_sm=u_sm,
            u_mem=u_mem,
            n_tasks=n_tasks,
        )

        # Step 3: Newton-Raphson update using f_k as anchor
        # f_{k+1} = f_k + η * e_k / J(f_k)
        # This is a pure proportional step scaled by the physical gradient
        delta_f = self.eta * error / jacobian

        # ===== 3-Stage Control Logic =====

        # Stage 1: Power cap just changed → Direct jump to feedforward target
        if self._cap_just_changed:
            f_next = f_feedforward

        # Stage 2: Far from target (|error| > 7W) → Jacobian-based Newton step
        elif abs(error) > self.proximity_threshold:
            # Apply rate limiting for stability
            if abs(delta_f) > self.max_delta_f:
                f_next = f_curr + np.sign(delta_f) * self.max_delta_f
            elif abs(delta_f) > self.min_step_f:
                # delta_f 有符号且绝对值足够大，直接使用
                f_next = f_curr + delta_f
            else:
                # delta_f 很小但有符号，至少调整一档
                f_next = f_curr + np.sign(delta_f) * self.min_step_f

        # Stage 3: Close to target (|error| <= 7W) → Minimal step adjustments
        else:
            if error > 0:
                # Power below cap: increase frequency by minimal step
                f_next = f_curr + self.min_step_f
            elif error < 0:
                # Power above cap: decrease frequency by minimal step
                f_next = f_curr - self.min_step_f
            else:
                # Exactly at target: hold frequency
                f_next = f_curr

        # Reset the flag after using it (so next iteration doesn't incorrectly trigger Stage 1)
        if self._cap_just_changed:
            self._cap_just_changed = False

        # Final bounds check
        f_next = float(np.clip(f_next, self.f_min, self.f_max))

        return ControlState(
            f_curr=f_curr,
            p_meas=p_meas,
            error=error,
            jacobian=jacobian,
            f_feedforward=f_feedforward,
            f_next=f_next,
            p_dyn_ref=p_dyn_ref,
        )


__all__ = ["ThermodynamicModelController", "ControlState"]
