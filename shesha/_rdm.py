"""
Shared RDM computation and RDM-similarity implementation.

Public wrappers in ``shesha.core`` and ``shesha.sim`` keep their documented
parameter order and delegate here.
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, rankdata, spearmanr

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
    """Accept only the calling function's documented distance metrics."""
    validate_metric(metric, documented)


def compute_rdm_impl(
    X: np.ndarray,
    metric: str = "cosine",
    normalize: bool = True,
    nan_policy: str = "raise",
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


def resampled_pair_index(rows: np.ndarray, n_items: int, pairs: tuple) -> np.ndarray:
    """
    Condensed-RDM indices for the pairs of a bootstrap resample.

    ``rows`` maps resample positions to item indices in ``range(n_items)`` and
    ``pairs`` is ``np.triu_indices(len(rows), k=1)``. Pairs that refer to the
    same item twice are dropped: their zero self-distance would otherwise
    appear in both RDMs and inflate agreement.
    """
    a = rows[pairs[0]]
    b = rows[pairs[1]]
    distinct = a != b
    lo = np.minimum(a[distinct], b[distinct])
    hi = np.maximum(a[distinct], b[distinct])
    return n_items * lo - lo * (lo + 1) // 2 + (hi - lo - 1)


def rowwise_correlation(A: np.ndarray, B: np.ndarray, method: str = "spearman") -> np.ndarray:
    """
    Correlate corresponding rows of ``A`` and ``B``.

    Rows with fewer than three entries or with constant values in either input
    are unestimable and return NaN.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    out = np.full(A.shape[0], np.nan)
    if A.shape[1] < 3:
        return out
    defined = (np.std(A, axis=1) >= EPS) & (np.std(B, axis=1) >= EPS)
    if method == "spearman":
        A = rankdata(A, axis=1)
        B = rankdata(B, axis=1)
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(A * A, axis=1) * np.sum(B * B, axis=1))
    defined &= denom > 0
    out[defined] = np.clip(np.sum(A * B, axis=1)[defined] / denom[defined], -1.0, 1.0)
    return out


def rdm_similarity_impl(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    metric: str = "cosine",
    method: str = "spearman",
    nan_policy: str = "raise",
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
