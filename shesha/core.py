"""
Shesha: Self-consistency Metrics for Representational Stability

Core implementations of Shesha variants for measuring geometric stability
of high-dimensional representations.
"""

import warnings
from typing import List, Optional, Union

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import rankdata, spearmanr

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ._rdm import compute_rdm_impl, rdm_similarity_impl, validate_rdm_metric
from ._utils import bootstrap_ci, bootstrap_ci_two_sample
from ._validate import (
    apply_nan_policy,
    as_2d_array,
    as_label_array,
    validate_ci,
    validate_class_counts,
    validate_fraction,
    validate_metric,
    validate_nan_policy,
    validate_paired_samples,
    validate_positive_int,
)

SAMPLE_SPLIT_WARNING = (
    "sample_split correlates unmatched RDM entries from independently drawn "
    "subsamples. Corresponding distances do not refer to the same observation "
    "pairs, so the current estimand is not scientifically identifiable. Do not "
    "use these outputs for scientific inference. shesha-geometry 0.3.0 will "
    "replace this function with a matched-replicate API (replicate_stability)."
)

ANCHOR_STABILITY_WARNING = (
    "anchor_stability correlates flattened distance profiles whose columns "
    "refer to different probe observations. Rank-normalizing does not restore "
    "correspondence, so the current estimand is not scientifically identifiable. "
    "Do not use these outputs for scientific inference. shesha-geometry 0.3.0 "
    "will replace this function with a matched-observation API "
    "(anchor_profile_stability)."
)

__all__ = [
    # Unsupervised variants
    "feature_split",
    "sample_split",
    "anchor_stability",
    # Supervised variants
    "variance_ratio",
    "supervised_alignment",
    "class_separation_ratio",
    "lda_stability",
    # Drift metrics
    "rdm_similarity",
    "rdm_drift",
    # Utilities
    "compute_rdm",
]

EPS = 1e-12


# =============================================================================
# RDM Utilities
# =============================================================================


def compute_rdm(
    X: np.ndarray,
    metric: Literal["cosine", "correlation", "euclidean"] = "cosine",
    normalize: bool = True,
    nan_policy: Literal["replace", "raise", "omit", "propagate"] = "replace",
) -> np.ndarray:
    """
    Compute Representational Dissimilarity Matrix (RDM).

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    metric : str
        Distance metric: 'cosine', 'correlation', or 'euclidean'.
    normalize : bool
        If True and metric='cosine', L2-normalize rows before computing distances.
    nan_policy : {'replace', 'raise', 'omit', 'propagate'}, default='replace'
        How to handle undefined distances (for example cosine distance of a
        zero vector). ``replace`` fills them with 1.0 for a consistent 0.2.29
        RDM API. This changes the prior undefined-distance behavior of
        ``compute_rdm``. The default will change to ``raise`` in 0.3.0.
        ``omit`` raises if omission would break condensed-RDM indexing.

    Returns
    -------
    np.ndarray
        Condensed distance vector (upper triangle of RDM).
    """
    return compute_rdm_impl(X, metric=metric, normalize=normalize, nan_policy=nan_policy)


# =============================================================================
# Unsupervised Variants
# =============================================================================


def feature_split(
    X: Union[np.ndarray, List[np.ndarray]],
    n_splits: int = 30,
    metric: Literal["cosine", "correlation"] = "cosine",
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1600,
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
    return_all_splits: bool = False,
    nan_policy: Literal["replace", "raise", "omit", "propagate"] = "replace",
) -> Union[float, dict, List]:
    """
    Feature-Split Shesha: measures coordinate-axis redundancy.

    Partitions feature dimensions into random disjoint halves, computes RDMs
    on each half, and measures their rank correlation. High values indicate
    that relational structure is distributed across the observed coordinate
    axes (redundant encoding).

    Parameters
    ----------
    X : np.ndarray or list of np.ndarray
        Data matrix of shape (n_samples, n_features), or a list of such matrices
        for batch evaluation. When a list is passed, returns a list of results
        in the same order.
    n_splits : int
        Number of random feature partitions to average over.
    metric : str
        Distance metric for RDM computation.
    seed : int, optional
        Random seed for reproducibility.
    max_samples : int, optional
        Subsample to this many samples if exceeded (for efficiency).
    n_bootstrap_ci : int, optional
        If provided, compute bootstrap confidence interval by resampling
        the input data this many times (e.g. 1000 or 10000).
    ci : float, default=0.95
        Confidence level for the interval (only used when n_bootstrap_ci is set).
    return_all_splits : bool, default=False
        If True, return a dict with the mean score and per-split correlation
        scores instead of only the mean score.
    nan_policy : {'replace', 'raise', 'omit', 'propagate'}, default='replace'
        How to handle undefined distances. ``replace`` fills them with 1.0
        (0.2.29 default). The default will change to ``raise`` in 0.3.0.

    Returns
    -------
    float or dict or list
        If X is a single array and n_bootstrap_ci is None: mean Spearman
        correlation in [-1, 1].
        If X is a single array and return_all_splits is True: dict with keys
        'mean' and 'split_scores'.
        If X is a single array and n_bootstrap_ci is set: dict with keys
        'mean', 'ci_low', 'ci_high', 'std', 'n_bootstraps', 'ci_level'.
        If X is a list: list of the above, one entry per input matrix.

    Notes
    -----
    ``feature_split`` measures how consistently pairwise relations are
    distributed across the *observed coordinate axes*. It is not a
    basis-invariant geometric robustness score.

    Orthogonal transformations preserve pairwise Euclidean geometry (and
    therefore RDM similarity) but can change this score, because they
    reallocate information across coordinates. High Shesha does not establish
    correct information, causal use, or functional preservation. Low Shesha
    does not establish information loss.

    Examples
    --------
    >>> X = np.random.randn(500, 768)  # 500 samples, 768-dim embeddings
    >>> stability = feature_split(X, n_splits=30, seed=320)
    >>> print(f"Feature-split stability: {stability:.3f}")

    >>> # Batch evaluation across multiple representations
    >>> matrices = [np.random.randn(500, 768) for _ in range(5)]
    >>> scores = feature_split(matrices, n_splits=30, seed=320)
    >>> print(scores)  # list of 5 floats

    >>> # With bootstrap confidence interval
    >>> result = feature_split(X, n_splits=30, seed=320, n_bootstrap_ci=1000)
    >>> print(f"{result['mean']:.3f} [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")

    >>> # Return per-split scores for distribution plots
    >>> result = feature_split(X, n_splits=30, seed=320, return_all_splits=True)
    >>> scores = result["split_scores"]
    """
    options = {
        "n_splits": n_splits,
        "metric": metric,
        "seed": seed,
        "max_samples": max_samples,
        "n_bootstrap_ci": n_bootstrap_ci,
        "ci": ci,
        "return_all_splits": return_all_splits,
        "nan_policy": nan_policy,
    }
    if isinstance(X, list):
        return [_feature_split_single(array, **options) for array in X]
    return _feature_split_single(X, **options)


def _feature_split_single(
    X: np.ndarray,
    n_splits: int,
    metric: str,
    seed: Optional[int],
    max_samples: Optional[int],
    n_bootstrap_ci: Optional[int],
    ci: float,
    return_all_splits: bool,
    nan_policy: str,
) -> Union[float, dict]:
    """Validate and dispatch a single feature-split input."""
    validate_positive_int(n_splits, "n_splits")
    validate_rdm_metric(metric, ("cosine", "correlation"))
    validate_positive_int(max_samples, "max_samples", allow_none=True)
    validate_positive_int(n_bootstrap_ci, "n_bootstrap_ci", allow_none=True)
    validate_ci(ci)
    validate_nan_policy(nan_policy)

    if n_bootstrap_ci is not None and return_all_splits:
        raise ValueError("return_all_splits cannot be used with n_bootstrap_ci.")

    X = as_2d_array(X, "X")

    if n_bootstrap_ci is not None:
        return bootstrap_ci(
            _feature_split_point_estimate,
            n_bootstrap_ci,
            ci,
            seed,
            X,
            n_splits=n_splits,
            metric=metric,
            seed=seed,
            max_samples=max_samples,
            nan_policy=nan_policy,
        )
    return _feature_split_point_estimate(
        X,
        n_splits=n_splits,
        metric=metric,
        seed=seed,
        max_samples=max_samples,
        nan_policy=nan_policy,
        return_all_splits=return_all_splits,
    )


def _feature_split_point_estimate(
    X: np.ndarray,
    n_splits: int,
    metric: str,
    seed: Optional[int],
    max_samples: Optional[int],
    nan_policy: str,
    return_all_splits: bool = False,
) -> Union[float, dict]:
    """Compute a feature-split point estimate on an already validated array."""
    if X.shape[0] < 4 or X.shape[1] < 4:
        return np.nan

    rng = np.random.default_rng(seed)
    X = _prepare_feature_split_data(X, metric, max_samples, rng)
    correlations = []
    for _ in range(n_splits):
        score, should_propagate = _score_feature_partition(X, metric, nan_policy, rng)
        if should_propagate:
            return _feature_split_result([], return_all_splits)
        if score is not None:
            correlations.append(score)
    return _feature_split_result(correlations, return_all_splits)


def _prepare_feature_split_data(
    X: np.ndarray,
    metric: str,
    max_samples: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply deterministic subsampling and cosine normalization."""
    if max_samples is not None and X.shape[0] > max_samples:
        X = X[rng.choice(X.shape[0], max_samples, replace=False)]
    if metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, EPS)
    return X


def _score_feature_partition(
    X: np.ndarray,
    metric: str,
    nan_policy: str,
    rng: np.random.Generator,
) -> tuple:
    """Score one random feature partition and report NaN propagation."""
    permutation = rng.permutation(X.shape[1])
    midpoint = X.shape[1] // 2
    first, second = permutation[:midpoint], permutation[midpoint : 2 * midpoint]
    rdms = apply_nan_policy(
        pdist(X[:, first], metric=metric),
        pdist(X[:, second], metric=metric),
        nan_policy=nan_policy,
    )
    if rdms is None:
        return None, True
    rdm1, rdm2 = rdms
    if rdm1.size < 2 or np.std(rdm1) < EPS or np.std(rdm2) < EPS:
        return None, False
    rho, _ = spearmanr(rdm1, rdm2)
    return (float(rho), False) if np.isfinite(rho) else (None, False)


def _feature_split_result(correlations: List[float], return_all_splits: bool):
    """Format a feature-split point estimate."""
    mean_score = float(np.mean(correlations)) if correlations else np.nan
    if return_all_splits:
        return {
            "mean": mean_score,
            "split_scores": [float(score) for score in correlations],
        }
    return mean_score


def sample_split(
    X: np.ndarray,
    n_splits: int = 30,
    subsample_fraction: float = 0.4,
    metric: Literal["cosine", "correlation"] = "cosine",
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1500,
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
    return_all_splits: bool = False,
    nan_policy: Literal["replace", "raise", "omit", "propagate"] = "replace",
) -> Union[float, dict]:
    """
    Sample-Split Shesha (invalid estimand; retained for compatibility).

    .. warning::
       This function correlates unmatched RDM entries from independently drawn
       subsamples. Corresponding distances do **not** refer to the same
       observation pairs. Current outputs must not be used for scientific
       inference. shesha-geometry 0.3.0 will replace this API with a
       matched-replicate estimator.

    The finite-distance algorithm is unchanged from earlier 0.2.x releases.
    Undefined distances now follow ``nan_policy``.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    n_splits : int
        Number of bootstrap iterations.
    subsample_fraction : float
        Fraction of samples to use in each subsample.
    metric : str
        Distance metric for RDM computation.
    seed : int, optional
        Random seed for reproducibility.
    max_samples : int, optional
        Subsample to this many samples if exceeded.
    n_bootstrap_ci : int, optional
        If provided, compute bootstrap confidence interval by resampling
        the input data this many times. Confidence intervals on this
        estimator inherit the same invalid estimand.
    ci : float, default=0.95
        Confidence level for the interval.
    return_all_splits : bool, default=False
        If True, return a dict with the mean score and per-split correlation
        scores instead of only the mean score.
    nan_policy : {'replace', 'raise', 'omit', 'propagate'}, default='replace'
        How to handle undefined distances. The default will change to
        ``raise`` in 0.3.0.

    Returns
    -------
    float or dict
        If n_bootstrap_ci is None: mean Spearman correlation. Range: [-1, 1].
        If return_all_splits is True: dict with keys 'mean' and 'split_scores'.
        If n_bootstrap_ci is set: dict with keys 'mean', 'ci_low', 'ci_high',
        'std', 'n_bootstraps', 'ci_level'.

    Examples
    --------
    >>> X = np.random.randn(1000, 384)
    >>> stability = sample_split(X, n_splits=50, seed=320)

    >>> result = sample_split(X, n_splits=50, seed=320, return_all_splits=True)
    >>> scores = result["split_scores"]
    """
    warnings.warn(SAMPLE_SPLIT_WARNING, FutureWarning, stacklevel=2)

    validate_positive_int(n_splits, "n_splits")
    validate_fraction(subsample_fraction, "subsample_fraction")
    validate_rdm_metric(metric, ("cosine", "correlation"))
    validate_positive_int(max_samples, "max_samples", allow_none=True)
    validate_positive_int(n_bootstrap_ci, "n_bootstrap_ci", allow_none=True)
    validate_ci(ci)
    validate_nan_policy(nan_policy)

    if n_bootstrap_ci is not None and return_all_splits:
        raise ValueError("return_all_splits cannot be used with n_bootstrap_ci.")

    X = as_2d_array(X, "X")

    if n_bootstrap_ci is not None:
        return bootstrap_ci(
            _sample_split_impl,
            n_bootstrap_ci,
            ci,
            seed,
            X,
            n_splits=n_splits,
            subsample_fraction=subsample_fraction,
            metric=metric,
            seed=seed,
            max_samples=max_samples,
            nan_policy=nan_policy,
        )
    return _sample_split_impl(
        X,
        n_splits=n_splits,
        subsample_fraction=subsample_fraction,
        metric=metric,
        seed=seed,
        max_samples=max_samples,
        nan_policy=nan_policy,
        return_all_splits=return_all_splits,
    )


def _sample_split_impl(
    X: np.ndarray,
    n_splits: int,
    subsample_fraction: float,
    metric: str,
    seed: Optional[int],
    max_samples: Optional[int],
    nan_policy: str,
    return_all_splits: bool = False,
) -> Union[float, dict]:
    """Compute the legacy sample-split statistic without emitting its public warning."""
    n_samples = X.shape[0]

    if n_samples < 10:
        return np.nan

    rng = np.random.default_rng(seed)

    # Subsample if needed
    if max_samples is not None and n_samples > max_samples:
        idx = rng.choice(n_samples, max_samples, replace=False)
        X = X[idx]
        n_samples = max_samples

    m = int(n_samples * subsample_fraction)
    if m < 5:
        return np.nan

    correlations = []

    for _ in range(n_splits):
        # Two independent subsamples
        idx1 = rng.choice(n_samples, m, replace=False)
        idx2 = rng.choice(n_samples, m, replace=False)

        rdm1 = pdist(X[idx1], metric=metric)
        rdm2 = pdist(X[idx2], metric=metric)
        handled = apply_nan_policy(rdm1, rdm2, nan_policy=nan_policy)
        if handled is None:
            if return_all_splits:
                return {"mean": np.nan, "split_scores": []}
            return np.nan
        rdm1, rdm2 = handled

        if rdm1.size < 2 or np.std(rdm1) < EPS or np.std(rdm2) < EPS:
            continue

        rho, _ = spearmanr(rdm1, rdm2)
        if np.isfinite(rho):
            correlations.append(rho)

    mean_score = float(np.mean(correlations)) if correlations else np.nan
    if return_all_splits:
        return {
            "mean": mean_score,
            "split_scores": [float(score) for score in correlations],
        }

    return mean_score


def anchor_stability(
    X: np.ndarray,
    n_splits: int = 30,
    n_anchors: int = 100,
    n_per_split: int = 200,
    metric: Literal["cosine", "euclidean"] = "cosine",
    rank_normalize: bool = True,
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1500,
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
    nan_policy: Literal["replace", "raise", "omit", "propagate"] = "replace",
) -> Union[float, dict]:
    """
    Anchor-based Shesha (invalid estimand; retained for compatibility).

    .. warning::
       This function correlates flattened distance profiles whose columns
       refer to different probe observations. Rank-normalizing does not
       restore correspondence. Current outputs must not be used for
       scientific inference. shesha-geometry 0.3.0 will replace this API
       with a matched-observation estimator.

    The finite-distance algorithm is unchanged from earlier 0.2.x releases.
    Undefined distances now follow ``nan_policy``.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    n_splits : int
        Number of random splits.
    n_anchors : int
        Number of fixed anchor points.
    n_per_split : int
        Number of samples per split.
    metric : str
        Distance metric.
    rank_normalize : bool
        If True, rank-normalize distances within each anchor before correlating.
    seed : int, optional
        Random seed.
    max_samples : int, optional
        Subsample to this many samples if exceeded.
    n_bootstrap_ci : int, optional
        If provided, compute bootstrap confidence interval by resampling
        the input data this many times. Confidence intervals on this
        estimator inherit the same invalid estimand.
    ci : float, default=0.95
        Confidence level for the interval.
    nan_policy : {'replace', 'raise', 'omit', 'propagate'}, default='replace'
        How to handle undefined distances. The default will change to
        ``raise`` in 0.3.0.

    Returns
    -------
    float or dict
        If n_bootstrap_ci is None: mean correlation of anchor distance profiles.
        If n_bootstrap_ci is set: dict with keys 'mean', 'ci_low', 'ci_high',
        'std', 'n_bootstraps', 'ci_level'.
    """
    warnings.warn(ANCHOR_STABILITY_WARNING, FutureWarning, stacklevel=2)

    validate_positive_int(n_splits, "n_splits")
    validate_positive_int(n_anchors, "n_anchors")
    validate_positive_int(n_per_split, "n_per_split")
    validate_rdm_metric(metric, ("cosine", "euclidean"))
    validate_positive_int(max_samples, "max_samples", allow_none=True)
    validate_positive_int(n_bootstrap_ci, "n_bootstrap_ci", allow_none=True)
    validate_ci(ci)
    validate_nan_policy(nan_policy)

    X = as_2d_array(X, "X")

    if n_bootstrap_ci is not None:
        return bootstrap_ci(
            _anchor_stability_impl,
            n_bootstrap_ci,
            ci,
            seed,
            X,
            n_splits=n_splits,
            n_anchors=n_anchors,
            n_per_split=n_per_split,
            metric=metric,
            rank_normalize=rank_normalize,
            seed=seed,
            max_samples=max_samples,
            nan_policy=nan_policy,
        )
    return _anchor_stability_impl(
        X,
        n_splits=n_splits,
        n_anchors=n_anchors,
        n_per_split=n_per_split,
        metric=metric,
        rank_normalize=rank_normalize,
        seed=seed,
        max_samples=max_samples,
        nan_policy=nan_policy,
    )


def _anchor_stability_impl(
    X: np.ndarray,
    n_splits: int,
    n_anchors: int,
    n_per_split: int,
    metric: str,
    rank_normalize: bool,
    seed: Optional[int],
    max_samples: Optional[int],
    nan_policy: str,
) -> float:
    """Compute the legacy anchor statistic without emitting its public warning."""
    n_samples = X.shape[0]

    rng = np.random.default_rng(seed)

    # Subsample if needed
    if max_samples is not None and n_samples > max_samples:
        idx = rng.choice(n_samples, max_samples, replace=False)
        X = X[idx]
        n_samples = max_samples

    # Need enough samples for anchors + two splits
    min_required = n_anchors + 2 * n_per_split
    if n_samples < min_required:
        # Reduce sizes proportionally
        scale = n_samples / min_required * 0.9
        n_anchors = max(10, int(n_anchors * scale))
        n_per_split = max(20, int(n_per_split * scale))

    if n_samples < n_anchors + 2 * n_per_split:
        return np.nan

    # Select fixed anchors
    anchor_idx = rng.choice(n_samples, n_anchors, replace=False)
    anchors = X[anchor_idx]
    remaining_idx = np.setdiff1d(np.arange(n_samples), anchor_idx)

    if len(remaining_idx) < 2 * n_per_split:
        return np.nan

    correlations = []

    for _ in range(n_splits):
        # Two disjoint splits from remaining samples
        perm = rng.permutation(remaining_idx)
        split1_idx = perm[:n_per_split]
        split2_idx = perm[n_per_split : 2 * n_per_split]

        # Distance matrices: anchors x split_samples
        D1 = cdist(anchors, X[split1_idx], metric=metric)
        D2 = cdist(anchors, X[split2_idx], metric=metric)

        if rank_normalize and nan_policy == "omit":
            ranked1 = np.full_like(D1, np.nan)
            ranked2 = np.full_like(D2, np.nan)
            for row in range(D1.shape[0]):
                valid = np.isfinite(D1[row]) & np.isfinite(D2[row])
                ranked1[row, valid] = rankdata(D1[row, valid])
                ranked2[row, valid] = rankdata(D2[row, valid])
            D1, D2 = ranked1, ranked2

        handled = apply_nan_policy(D1.ravel(), D2.ravel(), nan_policy=nan_policy)
        if handled is None:
            return np.nan
        if handled[0].shape != D1.ravel().shape:
            # Omission retains the within-anchor ranking performed above.
            rho, _ = spearmanr(handled[0], handled[1])
            if np.isfinite(rho):
                correlations.append(rho)
            continue
        D1 = handled[0].reshape(D1.shape)
        D2 = handled[1].reshape(D2.shape)

        if rank_normalize:
            D1 = np.apply_along_axis(rankdata, 1, D1)
            D2 = np.apply_along_axis(rankdata, 1, D2)

        # Flatten and correlate
        rho, _ = spearmanr(D1.ravel(), D2.ravel())
        if np.isfinite(rho):
            correlations.append(rho)

    return float(np.mean(correlations)) if correlations else np.nan


# =============================================================================
# Supervised Variants
# =============================================================================


def variance_ratio(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Union[float, dict]:
    """
    Variance Ratio Shesha: ratio of between-class to total variance.

    A simple, efficient measure of how much geometric structure is explained
    by class labels. Equivalent to the R-squared of predicting coordinates
    from class membership.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    y : np.ndarray
        Class labels of shape (n_samples,).
    n_bootstrap_ci : int, optional
        If provided, compute bootstrap confidence interval by resampling
        the input data this many times.
    ci : float, default=0.95
        Confidence level for the interval.
    seed : int, optional
        Random seed for bootstrap reproducibility.

    Returns
    -------
    float or dict
        If n_bootstrap_ci is None: between-class variance / total variance. Range: [0, 1].
        If n_bootstrap_ci is set: dict with keys 'mean', 'ci_low', 'ci_high',
        'std', 'n_bootstraps', 'ci_level'.

    Examples
    --------
    >>> X = np.random.randn(500, 768)
    >>> y = np.random.randint(0, 10, 500)
    >>> vr = variance_ratio(X, y)
    """
    validate_positive_int(n_bootstrap_ci, "n_bootstrap_ci", allow_none=True)
    validate_ci(ci)
    X = as_2d_array(X, "X")
    y = as_label_array(y, X.shape[0])

    classes = np.unique(y)
    if len(classes) < 2:
        return np.nan
    validate_class_counts(y)

    if n_bootstrap_ci is not None:
        return bootstrap_ci(
            variance_ratio,
            n_bootstrap_ci,
            ci,
            seed,
            X,
            y,
            skip_value_errors=True,
        )

    global_mean = np.mean(X, axis=0)
    X_centered = X - global_mean
    ss_total = np.sum(X_centered**2) + EPS

    ss_between = 0.0
    for c in classes:
        mask = y == c
        n_c = np.sum(mask)
        if n_c == 0:
            continue
        class_mean = np.mean(X[mask], axis=0)
        ss_between += n_c * np.sum((class_mean - global_mean) ** 2)

    return float(ss_between / ss_total)


def supervised_alignment(
    X: np.ndarray,
    y: np.ndarray,
    metric: Literal["cosine", "correlation"] = "correlation",
    seed: Optional[int] = None,
    max_samples: int = 300,
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
) -> Union[float, dict]:
    """
    Supervised RDM Alignment: correlation between model RDM and ideal label RDM.

    Measures how well the representation's distance structure aligns with
    task-defined similarity (same class = similar, different class = dissimilar).

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    y : np.ndarray
        Class labels of shape (n_samples,).
    metric : str
        Distance metric for model RDM.
    seed : int, optional
        Random seed for subsampling.
    max_samples : int
        Subsample to this many samples (RDM computation is O(n^2)).
    n_bootstrap_ci : int, optional
        If provided, compute bootstrap confidence interval by resampling
        the input data this many times.
    ci : float, default=0.95
        Confidence level for the interval.

    Returns
    -------
    float or dict
        If n_bootstrap_ci is None: Spearman correlation. Range: [-1, 1].
        If n_bootstrap_ci is set: dict with keys 'mean', 'ci_low', 'ci_high',
        'std', 'n_bootstraps', 'ci_level'.
    """
    validate_metric(metric, ("cosine", "correlation"))
    validate_positive_int(max_samples, "max_samples")
    validate_positive_int(n_bootstrap_ci, "n_bootstrap_ci", allow_none=True)
    validate_ci(ci)
    X = as_2d_array(X, "X")
    y = as_label_array(y, X.shape[0])

    if len(np.unique(y)) < 2:
        return np.nan
    validate_class_counts(y)

    if n_bootstrap_ci is not None:
        return bootstrap_ci(
            supervised_alignment,
            n_bootstrap_ci,
            ci,
            seed,
            X,
            y,
            skip_value_errors=True,
            metric=metric,
            seed=seed,
            max_samples=max_samples,
        )

    rng = np.random.default_rng(seed)

    if len(X) > max_samples:
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    # Center for correlation distance
    X = X - np.mean(X, axis=0)

    # Model RDM
    model_rdm = pdist(X, metric=metric)

    # Ideal RDM from labels (Hamming distance on labels)
    ideal_rdm = pdist(y.reshape(-1, 1), metric="hamming")

    rho, _ = spearmanr(model_rdm, ideal_rdm)
    return float(rho) if np.isfinite(rho) else np.nan


def class_separation_ratio(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 50,
    subsample_frac: float = 0.5,
    metric: Literal["cosine", "euclidean"] = "euclidean",
    seed: Optional[int] = None,
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
) -> Union[float, dict]:
    """
    Class Separation Ratio: ratio of between-class to within-class distances.

    Measures how well-separated classes are in the representation space.
    Uses bootstrap subsampling for computational efficiency and stability.
    Related to Fisher's discriminant ratio but operates in distance space.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    y : np.ndarray
        Class labels of shape (n_samples,).
    n_bootstrap : int
        Number of bootstrap iterations for stability.
    subsample_frac : float
        Fraction of samples to use per bootstrap (0.0-1.0).
    metric : str
        Distance metric: 'cosine' or 'euclidean'.
    seed : int, optional
        Random seed for reproducibility.
    n_bootstrap_ci : int, optional
        If provided, compute bootstrap confidence interval by resampling
        the input data this many times.
    ci : float, default=0.95
        Confidence level for the interval.

    Returns
    -------
    float or dict
        If n_bootstrap_ci is None: mean separation ratio. Range: [0, inf).
        If n_bootstrap_ci is set: dict with keys 'mean', 'ci_low', 'ci_high',
        'std', 'n_bootstraps', 'ci_level'.

    Examples
    --------
    >>> # Well-separated classes
    >>> X = np.vstack([np.random.randn(100, 10),
    ...                np.random.randn(100, 10) + 5])
    >>> y = np.array([0]*100 + [1]*100)
    >>> ratio = class_separation_ratio(X, y)
    >>> print(f"Separation: {ratio:.2f}")  # High value

    Notes
    -----
    Higher values indicate representations where same-class samples are closer
    together than different-class samples, suggesting good discriminability.
    """
    validate_positive_int(n_bootstrap, "n_bootstrap")
    validate_fraction(subsample_frac, "subsample_frac")
    validate_metric(metric, ("cosine", "euclidean"))
    validate_positive_int(n_bootstrap_ci, "n_bootstrap_ci", allow_none=True)
    validate_ci(ci)
    X = as_2d_array(X, "X")
    y = as_label_array(y, X.shape[0])

    if len(np.unique(y)) < 2:
        return np.nan
    validate_class_counts(y)

    if n_bootstrap_ci is not None:
        return bootstrap_ci(
            class_separation_ratio,
            n_bootstrap_ci,
            ci,
            seed,
            X,
            y,
            skip_value_errors=True,
            n_bootstrap=n_bootstrap,
            subsample_frac=subsample_frac,
            metric=metric,
            seed=seed,
        )

    rng = np.random.default_rng(seed)

    if metric == "cosine":
        # Normalize for cosine distance
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, EPS)

    ratios = []
    n_samples = int(len(X) * subsample_frac)

    for _ in range(n_bootstrap):
        # Subsample
        idx = rng.choice(len(X), n_samples, replace=False)
        X_sub, y_sub = X[idx], y[idx]

        # Skip if any class is missing
        if len(np.unique(y_sub)) < 2:
            continue

        # Compute pairwise distances
        dists = cdist(X_sub, X_sub, metric="euclidean" if metric == "euclidean" else "cosine")

        # Within-class distances (same label)
        within_dists = []
        for label in np.unique(y_sub):
            mask = y_sub == label
            if np.sum(mask) < 2:
                continue
            class_dists = dists[mask][:, mask]
            # Upper triangle only (avoid diagonal)
            within_dists.extend(class_dists[np.triu_indices_from(class_dists, k=1)])

        # Between-class distances (different labels)
        between_dists = []
        for i, label_i in enumerate(np.unique(y_sub)):
            for label_j in np.unique(y_sub):
                if label_i >= label_j:
                    continue
                mask_i = y_sub == label_i
                mask_j = y_sub == label_j
                between_dists.extend(dists[mask_i][:, mask_j].flatten())

        if len(within_dists) == 0 or len(between_dists) == 0:
            continue

        mean_between = np.mean(between_dists)
        mean_within = np.mean(within_dists)

        if mean_within > EPS:
            ratios.append(mean_between / mean_within)

    return float(np.mean(ratios)) if len(ratios) > 0 else np.nan


def lda_stability(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 50,
    subsample_frac: float = 0.5,
    seed: Optional[int] = None,
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
) -> Union[float, dict]:
    """
    LDA Subspace Stability: consistency of linear discriminant direction.

    Measures whether the optimal linear decision boundary is robust to sampling
    variation. Computes LDA on full dataset and bootstrapped subsamples, then
    measures alignment of discriminant vectors.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    y : np.ndarray
        Binary class labels of shape (n_samples,). Must have exactly 2 classes.
    n_bootstrap : int
        Number of bootstrap iterations.
    subsample_frac : float
        Fraction of samples to use per bootstrap (0.0-1.0).
    seed : int, optional
        Random seed for reproducibility.
    n_bootstrap_ci : int, optional
        If provided, compute bootstrap confidence interval by resampling
        the input data this many times.
    ci : float, default=0.95
        Confidence level for the interval.

    Returns
    -------
    float or dict
        If n_bootstrap_ci is None: mean absolute cosine similarity. Range: [0, 1].
        If n_bootstrap_ci is set: dict with keys 'mean', 'ci_low', 'ci_high',
        'std', 'n_bootstraps', 'ci_level'.

    Examples
    --------
    >>> # Create well-separated binary classification data
    >>> X = np.vstack([np.random.randn(100, 10),
    ...                np.random.randn(100, 10) + 3])
    >>> y = np.array([0]*100 + [1]*100)
    >>> stability = lda_stability(X, y)
    >>> print(f"LDA Stability: {stability:.3f}")  # Should be high

    Notes
    -----
    Low values suggest the discriminant subspace is unstable, potentially
    indicating overfitting to source domain structure. This metric is
    particularly useful for predicting transfer learning performance.

    Only works for binary classification. For multi-class, consider using
    class_separation_ratio instead.
    """
    validate_positive_int(n_bootstrap, "n_bootstrap")
    validate_fraction(subsample_frac, "subsample_frac")
    validate_positive_int(n_bootstrap_ci, "n_bootstrap_ci", allow_none=True)
    validate_ci(ci)
    X = as_2d_array(X, "X")
    y = as_label_array(y, X.shape[0])

    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"LDA stability requires exactly 2 classes, got {len(classes)}")
    validate_class_counts(y)

    if n_bootstrap_ci is not None:
        return bootstrap_ci(
            lda_stability,
            n_bootstrap_ci,
            ci,
            seed,
            X,
            y,
            skip_value_errors=True,
            n_bootstrap=n_bootstrap,
            subsample_frac=subsample_frac,
            seed=seed,
        )

    rng = np.random.default_rng(seed)

    # Compute full discriminant vector
    try:
        # Compute class means
        mean_0 = np.mean(X[y == classes[0]], axis=0)
        mean_1 = np.mean(X[y == classes[1]], axis=0)

        # Compute pooled within-class covariance
        X_0_centered = X[y == classes[0]] - mean_0
        X_1_centered = X[y == classes[1]] - mean_1
        S_w = (X_0_centered.T @ X_0_centered + X_1_centered.T @ X_1_centered) / len(X)

        # Add regularization for numerical stability
        S_w += np.eye(X.shape[1]) * 1e-6

        # Compute discriminant direction: S_w^{-1} (μ_1 - μ_0)
        mean_diff = mean_1 - mean_0
        w_full = np.linalg.solve(S_w, mean_diff)
        w_full = w_full / (np.linalg.norm(w_full) + EPS)
    except np.linalg.LinAlgError:
        return np.nan

    # Bootstrap
    similarities = []
    n_samples = int(len(X) * subsample_frac)

    for _ in range(n_bootstrap):
        # Subsample with stratification
        idx_0 = rng.choice(np.where(y == classes[0])[0], n_samples // 2, replace=True)
        idx_1 = rng.choice(np.where(y == classes[1])[0], n_samples // 2, replace=True)
        idx = np.concatenate([idx_0, idx_1])

        X_boot, y_boot = X[idx], y[idx]

        try:
            # Compute bootstrap discriminant
            mean_0_boot = np.mean(X_boot[y_boot == classes[0]], axis=0)
            mean_1_boot = np.mean(X_boot[y_boot == classes[1]], axis=0)

            X_0_boot_centered = X_boot[y_boot == classes[0]] - mean_0_boot
            X_1_boot_centered = X_boot[y_boot == classes[1]] - mean_1_boot
            S_w_boot = (
                X_0_boot_centered.T @ X_0_boot_centered + X_1_boot_centered.T @ X_1_boot_centered
            ) / len(X_boot)
            S_w_boot += np.eye(X_boot.shape[1]) * 1e-6

            mean_diff_boot = mean_1_boot - mean_0_boot
            w_boot = np.linalg.solve(S_w_boot, mean_diff_boot)
            w_boot = w_boot / (np.linalg.norm(w_boot) + EPS)

            # Absolute cosine similarity (sign ambiguity in discriminant)
            sim = np.abs(np.dot(w_full, w_boot))
            similarities.append(sim)
        except np.linalg.LinAlgError:
            continue

    return float(np.mean(similarities)) if len(similarities) > 0 else np.nan


# =============================================================================
# Drift Metrics
# =============================================================================


def rdm_similarity(
    X: np.ndarray,
    Y: np.ndarray,
    method: Literal["spearman", "pearson"] = "spearman",
    metric: Literal["cosine", "correlation", "euclidean"] = "cosine",
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
    seed: Optional[int] = None,
    nan_policy: Literal["replace", "raise", "omit", "propagate"] = "replace",
) -> Union[float, dict]:
    """
    Compute RDM similarity between two representations.

    Measures how similar the pairwise distance structures are between two
    representations. Useful for measuring representational drift, comparing
    models, or tracking changes during training.

    Parameters
    ----------
    X : np.ndarray
        First representation matrix of shape (n_samples, n_features_x).
    Y : np.ndarray
        Second representation matrix of shape (n_samples, n_features_y).
        Must have the same number of samples as X.
    method : str
        Correlation method: 'spearman' (rank-based, default) or 'pearson'.
    metric : str
        Distance metric for RDM computation: 'cosine', 'correlation', or 'euclidean'.
    n_bootstrap_ci : int, optional
        If provided, compute bootstrap confidence interval by resampling
        the input data this many times.
    ci : float, default=0.95
        Confidence level for the interval.
    seed : int, optional
        Random seed for bootstrap reproducibility.
    nan_policy : {'replace', 'raise', 'omit', 'propagate'}, default='replace'
        How to handle undefined distances. The default will change to
        ``raise`` in 0.3.0.

    Returns
    -------
    float or dict
        If n_bootstrap_ci is None: correlation between RDMs. Range: [-1, 1].
        If n_bootstrap_ci is set: dict with keys 'mean', 'ci_low', 'ci_high',
        'std', 'n_bootstraps', 'ci_level'.

    Examples
    --------
    >>> # Compare representations before and after training
    >>> X_before = model_before.encode(data)
    >>> X_after = model_after.encode(data)
    >>> similarity = rdm_similarity(X_before, X_after)
    >>> print(f"RDM similarity: {similarity:.3f}")

    >>> # Compare two different models
    >>> X_model1 = model1.encode(data)
    >>> X_model2 = model2.encode(data)
    >>> similarity = rdm_similarity(X_model1, X_model2, method='pearson')

    Notes
    -----
    - Spearman (default) is more robust to outliers and non-linear relationships
    - Pearson captures linear relationships in distance magnitudes
    - The representations can have different feature dimensions (only sample
      count must match)
    - Fewer than 3 samples is unestimable and returns NaN
    - This function shares its implementation with ``shesha.sim.rdm_similarity``
    """
    validate_positive_int(n_bootstrap_ci, "n_bootstrap_ci", allow_none=True)
    validate_ci(ci)
    X = as_2d_array(X, "X")
    Y = as_2d_array(Y, "Y")
    validate_paired_samples(X, Y)

    if n_bootstrap_ci is not None:
        return bootstrap_ci_two_sample(
            rdm_similarity,
            n_bootstrap_ci,
            ci,
            seed,
            X,
            Y,
            method=method,
            metric=metric,
            nan_policy=nan_policy,
        )
    return rdm_similarity_impl(X, Y, metric=metric, method=method, nan_policy=nan_policy)


def rdm_drift(
    X: np.ndarray,
    Y: np.ndarray,
    method: Literal["spearman", "pearson"] = "spearman",
    metric: Literal["cosine", "correlation", "euclidean"] = "cosine",
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
    seed: Optional[int] = None,
    nan_policy: Literal["replace", "raise", "omit", "propagate"] = "replace",
) -> Union[float, dict]:
    """
    Compute representational drift between two representations.

    Drift is defined as 1 - rdm_similarity, so higher values indicate
    more change in geometric structure. This is useful for tracking
    how much a representation has changed over time or due to some
    intervention (fine-tuning, perturbation, etc.).

    Parameters
    ----------
    X : np.ndarray
        First (baseline/before) representation of shape (n_samples, n_features_x).
    Y : np.ndarray
        Second (comparison/after) representation of shape (n_samples, n_features_y).
        Must have the same number of samples as X.
    method : str
        Correlation method: 'spearman' (rank-based, default) or 'pearson'.
    metric : str
        Distance metric for RDM computation.
    n_bootstrap_ci : int, optional
        If provided, compute bootstrap confidence interval by resampling
        the input data this many times.
    ci : float, default=0.95
        Confidence level for the interval.
    seed : int, optional
        Random seed for bootstrap reproducibility.
    nan_policy : {'replace', 'raise', 'omit', 'propagate'}, default='replace'
        How to handle undefined distances. The default will change to
        ``raise`` in 0.3.0.

    Returns
    -------
    float or dict
        If n_bootstrap_ci is None: drift score. Range: [0, 2].
        If n_bootstrap_ci is set: dict with keys 'mean', 'ci_low', 'ci_high',
        'std', 'n_bootstraps', 'ci_level'.

    Examples
    --------
    >>> # Track drift during training
    >>> X_epoch0 = model.encode(data)
    >>> for epoch in range(10):
    ...     train_one_epoch(model)
    ...     X_current = model.encode(data)
    ...     drift = rdm_drift(X_epoch0, X_current)
    ...     print(f"Epoch {epoch+1}: drift = {drift:.3f}")

    >>> # Measure drift due to noise perturbation
    >>> X_clean = model.encode(clean_data)
    >>> X_noisy = model.encode(noisy_data)
    >>> drift = rdm_drift(X_clean, X_noisy)
    >>> print(f"Noise-induced drift: {drift:.3f}")

    See Also
    --------
    rdm_similarity : The inverse metric (similarity instead of drift)
    """
    validate_positive_int(n_bootstrap_ci, "n_bootstrap_ci", allow_none=True)
    validate_ci(ci)
    X = as_2d_array(X, "X")
    Y = as_2d_array(Y, "Y")
    validate_paired_samples(X, Y)

    if n_bootstrap_ci is not None:
        return bootstrap_ci_two_sample(
            rdm_drift,
            n_bootstrap_ci,
            ci,
            seed,
            X,
            Y,
            method=method,
            metric=metric,
            nan_policy=nan_policy,
        )
    similarity = rdm_similarity(X, Y, method=method, metric=metric, nan_policy=nan_policy)

    if np.isnan(similarity):
        return np.nan

    return 1.0 - similarity


# =============================================================================
# Convenience function
# =============================================================================


def shesha(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    variant: Literal[
        "feature_split", "sample_split", "anchor", "variance", "supervised"
    ] = "feature_split",
    **kwargs,
) -> float:
    """
    Unified interface for computing Shesha stability metrics.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    y : np.ndarray, optional
        Class labels (required for supervised variants).
    variant : str
        Which Shesha variant to compute:

        - 'feature_split': Unsupervised, partitions features
        - 'sample_split': Unsupervised, bootstrap resampling (invalid estimand;
          emits FutureWarning; do not use for inference)
        - 'anchor': Unsupervised, anchor-based stability (invalid estimand;
          emits FutureWarning; do not use for inference)
        - 'variance': Supervised, variance ratio
        - 'supervised': Supervised, RDM alignment
    **kwargs
        Additional arguments passed to the specific variant function.

    Returns
    -------
    float
        Shesha stability score.

    Examples
    --------
    >>> # Unsupervised
    >>> stability = shesha(X, variant='feature_split', n_splits=30, seed=320)

    >>> # Supervised
    >>> alignment = shesha(X, y, variant='supervised')
    """
    if variant == "feature_split":
        return feature_split(X, **kwargs)
    elif variant == "sample_split":
        return sample_split(X, **kwargs)
    elif variant == "anchor":
        return anchor_stability(X, **kwargs)
    elif variant == "variance":
        if y is None:
            raise ValueError("Labels required for variance_ratio")
        return variance_ratio(X, y)
    elif variant == "supervised":
        if y is None:
            raise ValueError("Labels required for supervised_alignment")
        return supervised_alignment(X, y, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
