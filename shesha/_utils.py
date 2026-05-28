"""
Internal utilities for bootstrap confidence intervals.
"""

import numpy as np
from typing import Optional, Union, Callable


def bootstrap_ci(
    func: Callable,
    n_bootstrap_ci: int,
    ci: float,
    bootstrap_seed: Optional[int],
    *args,
    **kwargs,
) -> dict:
    """
    Compute bootstrap confidence interval for a metric function.

    Resamples the input data (with replacement), recomputes the metric on each
    resample, and returns percentile-based confidence intervals.

    Parameters
    ----------
    func : callable
        The metric function to bootstrap. Must accept positional args
        followed by keyword args and return a float.
    n_bootstrap_ci : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95 for 95% CI).
    bootstrap_seed : int or None
        Random seed for reproducibility of the outer bootstrap.
    *args : np.ndarray
        Positional array arguments to func. Each will be resampled along axis 0
        using the same indices (paired resampling).
    **kwargs
        Additional keyword arguments passed through to func unchanged.
        If 'seed' is present, it is passed through to func for each iteration.

    Returns
    -------
    dict
        Keys: mean, ci_low, ci_high, std, n_bootstraps, ci_level
    """
    rng = np.random.default_rng(bootstrap_seed)
    n_samples = args[0].shape[0]

    scores = []
    for _ in range(n_bootstrap_ci):
        idx = rng.choice(n_samples, n_samples, replace=True)
        resampled_args = tuple(a[idx] for a in args)
        score = func(*resampled_args, **kwargs)
        if np.isfinite(score):
            scores.append(score)

    return _build_result(scores, ci)


def bootstrap_ci_two_sample(
    func: Callable,
    n_bootstrap_ci: int,
    ci: float,
    bootstrap_seed: Optional[int],
    X: np.ndarray,
    Y: np.ndarray,
    **kwargs,
) -> dict:
    """
    Bootstrap CI for two-sample metrics (e.g., rdm_similarity, CKA).

    Resamples both X and Y with the same indices (paired samples).
    """
    rng = np.random.default_rng(bootstrap_seed)
    n_samples = X.shape[0]

    scores = []
    for _ in range(n_bootstrap_ci):
        idx = rng.choice(n_samples, n_samples, replace=True)
        score = func(X[idx], Y[idx], **kwargs)
        if np.isfinite(score):
            scores.append(score)

    return _build_result(scores, ci)


def bootstrap_ci_bio(
    func: Callable,
    n_bootstrap_ci: int,
    ci: float,
    bootstrap_seed: Optional[int],
    X_control: np.ndarray,
    X_perturbed: np.ndarray,
    **kwargs,
) -> dict:
    """
    Bootstrap CI for bio perturbation metrics.

    Resamples control and perturbed populations independently.
    """
    rng = np.random.default_rng(bootstrap_seed)
    n_ctrl = X_control.shape[0]
    n_pert = X_perturbed.shape[0]

    scores = []
    for _ in range(n_bootstrap_ci):
        idx_ctrl = rng.choice(n_ctrl, n_ctrl, replace=True)
        idx_pert = rng.choice(n_pert, n_pert, replace=True)
        score = func(X_control[idx_ctrl], X_perturbed[idx_pert], **kwargs)
        if np.isfinite(score):
            scores.append(score)

    return _build_result(scores, ci)


def _build_result(scores, ci):
    """Build the CI result dictionary from a list of scores."""
    if len(scores) == 0:
        return {
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "std": np.nan,
            "n_bootstraps": 0,
            "ci_level": ci,
        }

    alpha = 1 - ci
    return {
        "mean": float(np.mean(scores)),
        "ci_low": float(np.percentile(scores, 100 * alpha / 2)),
        "ci_high": float(np.percentile(scores, 100 * (1 - alpha / 2))),
        "std": float(np.std(scores)),
        "n_bootstraps": len(scores),
        "ci_level": ci,
    }
