"""
Shared RDM computation and RDM-similarity implementation.

Public wrappers in ``shesha.core`` and ``shesha.sim`` keep their documented
parameter order and delegate here.
"""

import warnings

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

from ._validate import (
    apply_nan_policy,
    as_2d_array,
    validate_metric,
    validate_nan_policy,
    validate_paired_samples,
)

EPS = 1e-12
RDM_METRICS = ("cosine", "correlation", "euclidean")
CORR_METHODS = ("spearman", "pearson")


def validate_rdm_metric(metric: str, documented=RDM_METRICS) -> None:
    """Accept legacy SciPy distance names with a deprecation warning."""
    if metric in documented:
        return
    try:
        probe = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        pdist(probe, metric=metric)
    except (TypeError, ValueError):
        validate_metric(metric, documented)
    warnings.warn(
        f"Distance metric {metric!r} is accepted for 0.2.x compatibility but is "
        f"outside this function's documented metrics {tuple(documented)!r}. "
        "Support for undocumented SciPy metrics will be removed in 0.3.0.",
        FutureWarning,
        stacklevel=3,
    )


def compute_rdm_impl(
    X: np.ndarray,
    metric: str = "cosine",
    normalize: bool = True,
    nan_policy: str = "replace",
) -> np.ndarray:
    """Compute a condensed RDM, applying ``nan_policy`` to undefined distances."""
    X = as_2d_array(X, "X")
    validate_rdm_metric(metric)
    validate_nan_policy(nan_policy)

    if normalize and metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, EPS)

    rdm = pdist(X, metric=metric)
    if nan_policy == "propagate":
        return rdm
    if nan_policy == "omit" and not np.all(np.isfinite(rdm)):
        raise ValueError(
            "nan_policy='omit' is not supported by compute_rdm when distances are "
            "undefined because removing entries would break the condensed-RDM shape. "
            "Use nan_policy='propagate' or apply omission in a paired estimator."
        )
    handled = apply_nan_policy(rdm, nan_policy=nan_policy)
    return handled[0]


def correlate_rdms(rdm_x: np.ndarray, rdm_y: np.ndarray, method: str = "spearman") -> float:
    """Correlate two condensed RDMs. Returns 0.0 for constant RDMs."""
    if rdm_x.size < 2 or rdm_y.size < 2:
        return np.nan
    if np.std(rdm_x) < EPS or np.std(rdm_y) < EPS:
        return 0.0
    if method == "spearman":
        rho = spearmanr(rdm_x, rdm_y).correlation
    elif method == "pearson":
        rho, _ = pearsonr(rdm_x, rdm_y)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'pearson'")
    if not np.isfinite(rho):
        return np.nan
    return float(rho)


def rdm_similarity_impl(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    metric: str = "cosine",
    method: str = "spearman",
    nan_policy: str = "replace",
) -> float:
    """
    Shared RDM similarity.

    Unestimable input (fewer than 3 samples) returns NaN. Undefined distances
    are handled by ``nan_policy``.
    """
    X = as_2d_array(X, "X")
    Y = as_2d_array(Y, "Y")
    validate_paired_samples(X, Y)
    validate_rdm_metric(metric)
    validate_metric(method, CORR_METHODS, name="method")
    validate_nan_policy(nan_policy)

    if X.shape[0] < 3:
        return np.nan

    rdm_x = pdist(X, metric=metric)
    rdm_y = pdist(Y, metric=metric)
    handled = apply_nan_policy(rdm_x, rdm_y, nan_policy=nan_policy)
    if handled is None:
        return np.nan
    rdm_x, rdm_y = handled
    return correlate_rdms(rdm_x, rdm_y, method=method)
