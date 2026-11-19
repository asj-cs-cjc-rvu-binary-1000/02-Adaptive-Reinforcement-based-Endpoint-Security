#!/usr/bin/env python3
"""
scripts/calculus_threshold.py — Phase 2 Task 8, part A.

Reimplementation of ARES's calculus-based threshold optimiser, adapted to fit
(t_review, t_block) on VALIDATION predictions only. Never sees test.

Objective (from `main` README §"Calculus Threshold Optimization"):

    C(t_r, t_b) = C_FP * FPR(t_b) + C_FN * FNR(t_r) + C_review * ReviewRate(t_r, t_b)
                  + λ * ||(t_r, t_b) - (t_r0, t_b0)||²

    where:
      FPR(t_b)      = P(p >= t_b | y = 0)          [benign flagged >= t_block]
      FNR(t_r)      = P(p <  t_r | y = 1)          [malicious below t_review]
      ReviewRate    = P(t_r <= p < t_b)            [middle band, deferred]
      λ             = 0.05                          [tie to initial guess]
      (t_r0,t_b0)   = (0.35, 0.70)                  [main's defaults]

Issue: FPR, FNR, ReviewRate are STEP FUNCTIONS of (t_r, t_b) — they jump at
sample boundaries, so the gradient is 0 or undefined almost everywhere.
Ordinary gradient descent does not work.

Fix: sigmoid smoothing.
    F̃PR(t_b)    = mean_benign[sigmoid((p - t_b) / τ)]
    F̃NR(t_r)    = mean_malicious[1 - sigmoid((p - t_r) / τ)]
    R̃eview      = mean_all[sigmoid((p - t_r)/τ) * (1 - sigmoid((p - t_b)/τ))]
    τ            = 0.05
    d/dθ sigmoid((p-θ)/τ) = -sigmoid(1-sigmoid)/τ  != 0 everywhere

Optimization:
    θ_{k+1} = Proj(θ_k - η ∇C̃(θ_k))
    η = 0.05, constraints 0.15 <= t_r <= 0.60, 0.45 <= t_b <= 0.95,
    t_b - t_r >= 0.15

Verification:
    - Derivatives by finite difference vs analytic (sigmoid smoothing admits
      analytic; we use central difference for portability)
    - Hessian via second-difference; require det > 0 at solution (local minimum)
    - KKT boundary handling: if optimum sits on a constraint boundary, first
      derivative is non-zero but Hessian is positive — this is CORRECT.
"""
from __future__ import annotations

import numpy as np


# ---- defaults ----
DEFAULT_COST = {
    "C_FP": 1.0,
    "C_FN": 5.0,
    "C_review": 0.5,
    "lambda": 0.05,
    "tau": 0.05,
    "eta": 0.05,
    "t_r0": 0.35,
    "t_b0": 0.70,
    "max_iter": 30,
}
BOUNDS = {"t_r_min": 0.15, "t_r_max": 0.60,
          "t_b_min": 0.45, "t_b_max": 0.95,
          "gap_min": 0.15}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _smooth_rates(p_benign: np.ndarray, p_malicious: np.ndarray,
                  p_all: np.ndarray,
                  t_r: float, t_b: float, tau: float) -> dict:
    """Sigmoid-smoothed FPR / FNR / ReviewRate."""
    s_b = _sigmoid((p_benign - t_b) / tau)          # benign flagged (FPR-like)
    fpr = float(np.mean(s_b)) if len(p_benign) else 0.0
    s_m = _sigmoid((p_malicious - t_r) / tau)       # malicious below review
    fnr = float(np.mean(s_m)) if len(p_malicious) else 0.0
    s_rev = _sigmoid((p_all - t_r) / tau) * (1.0 - _sigmoid((p_all - t_b) / tau))
    review = float(np.mean(s_rev)) if len(p_all) else 0.0
    return {"fpr": fpr, "fnr": fnr, "review": review}


def _cost_smooth(p_benign, p_malicious, p_all, t_r, t_b, cost) -> float:
    r = _smooth_rates(p_benign, p_malicious, p_all, t_r, t_b, cost["tau"])
    j = (cost["C_FP"] * r["fpr"]
         + cost["C_FN"] * r["fnr"]
         + cost["C_review"] * r["review"])
    j += cost["lambda"] * ((t_r - cost["t_r0"]) ** 2 + (t_b - cost["t_b0"]) ** 2)
    return float(j)


def _hard_rates(p_benign, p_malicious, p_all, t_r, t_b) -> dict:
    """Step-function version — used for reporting only."""
    fpr = float(np.mean(p_benign >= t_b)) if len(p_benign) else 0.0
    fnr = float(np.mean(p_malicious < t_r)) if len(p_malicious) else 0.0
    review = float(np.mean((p_all >= t_r) & (p_all < t_b))) if len(p_all) else 0.0
    return {"fpr": fpr, "fnr": fnr, "review": review}


def _cost_hard(p_benign, p_malicious, p_all, t_r, t_b, cost) -> float:
    r = _hard_rates(p_benign, p_malicious, p_all, t_r, t_b)
    return float(cost["C_FP"] * r["fpr"]
                 + cost["C_FN"] * r["fnr"]
                 + cost["C_review"] * r["review"])


def _project(t_r: float, t_b: float) -> tuple[float, float]:
    """Project (t_r, t_b) onto the constrained box + gap."""
    t_r = max(BOUNDS["t_r_min"], min(BOUNDS["t_r_max"], t_r))
    t_b = max(BOUNDS["t_b_min"], min(BOUNDS["t_b_max"], t_b))
    if t_b - t_r < BOUNDS["gap_min"]:
        mid = (t_r + t_b) / 2.0
        t_r = mid - BOUNDS["gap_min"] / 2.0
        t_b = mid + BOUNDS["gap_min"] / 2.0
        t_r = max(BOUNDS["t_r_min"], min(BOUNDS["t_r_max"], t_r))
        t_b = max(BOUNDS["t_b_min"], min(BOUNDS["t_b_max"], t_b))
    return float(t_r), float(t_b)


def _grad_finite_diff(p_benign, p_malicious, p_all, t_r, t_b, cost,
                      eps: float = 1e-4) -> np.ndarray:
    """Central-difference gradient of the smoothed cost."""
    def f(tr, tb):
        return _cost_smooth(p_benign, p_malicious, p_all, tr, tb, cost)
    g_tr = (f(t_r + eps, t_b) - f(t_r - eps, t_b)) / (2 * eps)
    g_tb = (f(t_r, t_b + eps) - f(t_r, t_b - eps)) / (2 * eps)
    return np.array([g_tr, g_tb])


def _hessian_finite_diff(p_benign, p_malicious, p_all, t_r, t_b, cost,
                         eps: float = 1e-3) -> np.ndarray:
    def f(tr, tb):
        return _cost_smooth(p_benign, p_malicious, p_all, tr, tb, cost)
    f0 = f(t_r, t_b)
    f_rr = (f(t_r + eps, t_b) - 2 * f0 + f(t_r - eps, t_b)) / eps ** 2
    f_bb = (f(t_r, t_b + eps) - 2 * f0 + f(t_r, t_b - eps)) / eps ** 2
    f_rb = (f(t_r + eps, t_b + eps) - f(t_r + eps, t_b - eps)
            - f(t_r - eps, t_b + eps) + f(t_r - eps, t_b - eps)) / (4 * eps ** 2)
    return np.array([[f_rr, f_rb], [f_rb, f_bb]])


def _boundary_hit(t_r: float, t_b: float) -> list[str]:
    out = []
    eps = 1e-6
    if abs(t_r - BOUNDS["t_r_min"]) < 1e-3: out.append("review_box_min")
    if abs(t_r - BOUNDS["t_r_max"]) < 1e-3: out.append(f"review_box_{t_r:.2f}")
    if abs(t_b - BOUNDS["t_b_min"]) < 1e-3: out.append("block_box_min")
    if abs(t_b - BOUNDS["t_b_max"]) < 1e-3: out.append("block_box_max")
    if abs((t_b - t_r) - BOUNDS["gap_min"]) < 1e-3:
        out.append(f"gap_{BOUNDS['gap_min']:.2f}")
    return out


def optimize(p_benign: np.ndarray, p_malicious: np.ndarray, p_all: np.ndarray,
             cost: dict | None = None) -> dict:
    cost = {**DEFAULT_COST, **(cost or {})}
    eta = cost["eta"]
    t_r, t_b = cost["t_r0"], cost["t_b0"]
    history = [{
        "iter": 0, "t_r": t_r, "t_b": t_b,
        "cost_smooth": _cost_smooth(p_benign, p_malicious, p_all, t_r, t_b, cost),
        "cost_hard": _cost_hard(p_benign, p_malicious, p_all, t_r, t_b, cost),
    }]
    for k in range(1, cost["max_iter"] + 1):
        grad = _grad_finite_diff(p_benign, p_malicious, p_all, t_r, t_b, cost)
        t_r_new = t_r - eta * grad[0]
        t_b_new = t_b - eta * grad[1]
        t_r, t_b = _project(t_r_new, t_b_new)
        history.append({
            "iter": k, "t_r": t_r, "t_b": t_b,
            "cost_smooth": _cost_smooth(p_benign, p_malicious, p_all, t_r, t_b, cost),
            "cost_hard": _cost_hard(p_benign, p_malicious, p_all, t_r, t_b, cost),
        })

    final_grad = _grad_finite_diff(p_benign, p_malicious, p_all, t_r, t_b, cost)
    hess = _hessian_finite_diff(p_benign, p_malicious, p_all, t_r, t_b, cost)
    hess_det = float(np.linalg.det(hess))
    boundary = _boundary_hit(t_r, t_b)
    return {
        "t_r": float(t_r), "t_b": float(t_b),
        "cost_smooth_final": float(_cost_smooth(p_benign, p_malicious, p_all, t_r, t_b, cost)),
        "cost_hard_final": float(_cost_hard(p_benign, p_malicious, p_all, t_r, t_b, cost)),
        "gradient_final": {"dC_dt_r": float(final_grad[0]), "dC_dt_b": float(final_grad[1])},
        "hessian": hess.tolist(),
        "hessian_det": hess_det,
        "is_minimum": bool(hess_det > 0 and hess[0, 0] > 0),
        "is_stationary": bool(np.linalg.norm(final_grad) < 1e-2),
        "boundary_hit": boundary,
        "is_kkt_constrained": bool(hess_det > 0 and len(boundary) > 0),
        "history": history,
        "cost_params": cost,
        "bounds": BOUNDS,
    }