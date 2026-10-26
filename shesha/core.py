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

from ._rdm import (
    compute_rdm_impl,
    rdm_similarity_impl,
    resampled_pair_index,
    validate_rdm_metric,
)
from ._result import (
    StabilityResult,
    bootstrap_result,
    bootstrap_summary,
    result_metadata,
    validate_return_type,
)
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
    validate_positive_float,
    validate_positive_int,
)
from .replicate import paired_rdm_result

SAMPLE_SPLIT_WARNING = (
    "sample_split correlates unmatched RDM entries from independently drawn "
    "subsamples. Corresponding distances do not refer to the same observation "
    "pairs, so the estimand is not scientifically identifiable. Do not use these "
    "outputs for scientific inference. Use shesha.replicate_stability with matched "
    "replicates instead. sample_split is deprecated and will be removed in a "
    "future release."
)

ANCHOR_STABILITY_WARNING = (
    "anchor_stability correlates flattened distance profiles whose columns "
    "refer to different probe observations. Rank-normalizing does not restore "
    "correspondence, so the estimand is not scientifically identifiable. Do not "
    "use these outputs for scientific inference. Use "
    "shesha.anchor_profile_stability with matched replicates instead. "
    "anchor_stability is deprecated and will be removed in a future release."
)

NOT_CONVERGED_PREFIX = "feature_split did not reach"

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
    nan_policy: Literal["raise", "omit", "propagate", "replace"] = "raise",
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
    nan_policy : {'raise', 'omit', 'propagate', 'replace'}, default='raise'
        How to handle undefined distances (for example cosine distance of a
        zero vector). ``propagate`` returns the RDM with NaN entries,
        ``replace`` fills them with 1.0 and warns, and ``omit`` raises if
        omission would break condensed-RDM indexing.

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
    nan_policy: Literal["raise", "omit", "propagate", "replace"] = "raise",
    return_type: Literal["float", "result"] = "float",
    target_mc_se: Optional[float] = None,
    max_splits: Optional[int] = None,
) -> Union[float, dict, StabilityResult, List]:
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
        Number of random feature partitions to average over. With
        ``target_mc_se``, the minimum number of partitions.
    metric : str
        Distance metric for RDM computation.
    seed : int, optional
        Random seed for reproducibility.
    max_samples : int, optional
        Subsample to this many samples if exceeded (for efficiency).
    n_bootstrap_ci : int, optional
        If provided, compute bootstrap confidence interval by resampling
        the input data this many times (e.g. 1000 or 10000). Combining it
        with ``target_mc_se`` requires ``return_type='result'``.
    ci : float, default=0.95
        Confidence level for the interval (only used when n_bootstrap_ci is set).
    return_all_splits : bool, default=False
        If True, return a dict with the mean score and per-split correlation
        scores instead of only the mean score.
    nan_policy : {'raise', 'omit', 'propagate', 'replace'}, default='raise'
        How to handle undefined distances (for example a feature half that is
        all zeros for some sample under cosine distance). ``omit`` drops the
        affected pairs per split, ``propagate`` makes the whole result NaN, and
        ``replace`` fills them with 1.0 and warns.
    return_type : {'float', 'result'}, default='float'
        ``'result'`` returns a :class:`~shesha.StabilityResult` whose
        ``estimate`` is the full-data score, with the partition standard
        deviation, Monte Carlo standard error, per-split scores, and (when
        ``n_bootstrap_ci`` is set) a data interval. That interval resamples
        observations while holding the full-data feature partitions fixed,
        and drops pairs that refer to the same observation twice.
        ``return_all_splits`` is ignored for ``'result'``.
    target_mc_se : float, optional
        Convergence target. After ``n_splits`` partitions, keep drawing
        partitions until the Monte Carlo standard error of the mean
        (sample SD / sqrt(valid splits)) is at most ``target_mc_se`` or
        ``max_splits`` is reached. A ``UserWarning`` (float return) or a
        result warning and ``metadata['converged']`` report non-convergence.
    max_splits : int, optional
        Upper bound on partitions when ``target_mc_se`` is set. Defaults to
        ``max(n_splits, 500)``.

    Returns
    -------
    float or dict or StabilityResult or list
        If X is a single array and n_bootstrap_ci is None: mean Spearman
        correlation in [-1, 1].
        If X is a single array and return_all_splits is True: dict with keys
        'mean' and 'split_scores'.
        If X is a single array and n_bootstrap_ci is set: dict with keys
        'mean', 'ci_low', 'ci_high', 'std', 'n_bootstraps', 'ci_level'. Here
        'mean' is the mean of bootstrap replicates, not the full-data
        estimate; use ``return_type='result'`` for the full-data estimate.
        If return_type is 'result': a StabilityResult.
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

    >>> # Structured result: full-data estimate with separated uncertainty
    >>> result = feature_split(X, seed=320, n_bootstrap_ci=200, return_type="result")
    >>> result.estimate, result.monte_carlo_se, result.data_ci_low, result.data_ci_high

    >>> # Draw partitions until the Monte Carlo error is below 0.005
    >>> result = feature_split(X, seed=320, target_mc_se=0.005, return_type="result")
    >>> result.n_valid_splits, result.metadata["converged"]
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
        "return_type": return_type,
        "target_mc_se": target_mc_se,
        "max_splits": max_splits,
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
    return_type: str,
    target_mc_se: Optional[float],
    max_splits: Optional[int],
) -> Union[float, dict, StabilityResult]:
    """Validate and dispatch a single feature-split input."""
    validate_positive_int(n_splits, "n_splits")
    validate_rdm_metric(metric, ("cosine", "correlation"))
    validate_positive_int(max_samples, "max_samples", allow_none=True)
    validate_positive_int(n_bootstrap_ci, "n_bootstrap_ci", allow_none=True)
    validate_ci(ci)
    validate_nan_policy(nan_policy)
    validate_return_type(return_type)
    convergence = _validate_convergence(n_splits, target_mc_se, max_splits)

    X = as_2d_array(X, "X")

    if return_type == "result":
        return _feature_split_stability_result(
            X, n_splits, metric, seed, max_samples, n_bootstrap_ci, ci, nan_policy, convergence
        )
    if convergence is not None:
        if n_bootstrap_ci is not None:
            raise ValueError("target_mc_se with n_bootstrap_ci requires return_type='result'.")
        return _feature_split_converged_float(
            X, n_splits, metric, seed, max_samples, nan_policy, convergence, return_all_splits
        )

    if n_bootstrap_ci is not None and return_all_splits:
        raise ValueError("return_all_splits cannot be used with n_bootstrap_ci.")

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
    _, correlations, propagated = _run_feature_partitions(X, metric, nan_policy, rng, n_splits)
    return _feature_split_result([] if propagated else correlations, return_all_splits)


def _validate_convergence(
    n_splits: int, target_mc_se: Optional[float], max_splits: Optional[int]
) -> Optional[tuple]:
    """Return ``(target_mc_se, max_splits)`` or None when convergence is off."""
    if target_mc_se is None:
        if max_splits is not None:
            raise ValueError("max_splits requires target_mc_se")
        return None
    validate_positive_float(target_mc_se, "target_mc_se")
    validate_positive_int(max_splits, "max_splits", allow_none=True)
    if max_splits is None:
        max_splits = max(n_splits, 500)
    if max_splits < n_splits:
        raise ValueError(f"max_splits ({max_splits}) must be >= n_splits ({n_splits})")
    return float(target_mc_se), int(max_splits)


def _monte_carlo_se(scores: List[float]) -> float:
    if len(scores) < 2:
        return np.nan
    return float(np.std(scores, ddof=1) / np.sqrt(len(scores)))


def _run_feature_partitions(
    X: np.ndarray,
    metric: str,
    nan_policy: str,
    rng: np.random.Generator,
    n_splits: int,
    convergence: Optional[tuple] = None,
) -> tuple:
    """
    Draw and score random feature partitions.

    Returns ``(partitions, scores, propagated)``. Without ``convergence``
    exactly ``n_splits`` partitions are drawn; otherwise drawing continues
    past ``n_splits`` until the Monte Carlo SE reaches the target or the
    partition limit is hit.
    """
    target, limit = convergence if convergence is not None else (None, n_splits)
    partitions, scores = [], []
    while len(partitions) < limit:
        first, second = _draw_feature_partition(X.shape[1], rng)
        partitions.append((first, second))
        score, should_propagate = _score_rdm_pair(
            pdist(X[:, first], metric=metric), pdist(X[:, second], metric=metric), nan_policy
        )
        if should_propagate:
            return partitions, scores, True
        if score is not None:
            scores.append(score)
        if target is not None and len(partitions) >= n_splits:
            if _monte_carlo_se(scores) <= target:
                break
    return partitions, scores, False


def _feature_split_converged_float(
    X, n_splits, metric, seed, max_samples, nan_policy, convergence, return_all_splits
):
    """Float/dict return for convergence-based feature splits."""
    result = _feature_split_stability_result(
        X, n_splits, metric, seed, max_samples, None, 0.95, nan_policy, convergence
    )
    for note in result.warnings:
        if note.startswith(NOT_CONVERGED_PREFIX):
            warnings.warn(note, UserWarning, stacklevel=4)
    return _feature_split_result(list(result.component_scores), return_all_splits)


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


def _draw_feature_partition(n_features: int, rng: np.random.Generator) -> tuple:
    """Draw two disjoint, equal-size random feature halves."""
    permutation = rng.permutation(n_features)
    midpoint = n_features // 2
    return permutation[:midpoint], permutation[midpoint : 2 * midpoint]


def _score_rdm_pair(rdm1: np.ndarray, rdm2: np.ndarray, nan_policy: str) -> tuple:
    """Spearman correlation of two aligned RDMs, or ``(None, propagate)``."""
    rdms = apply_nan_policy(rdm1, rdm2, nan_policy=nan_policy)
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


def _score_feature_partitions(
    X: np.ndarray,
    partitions: list,
    metric: str,
    nan_policy: str,
    rows: Optional[np.ndarray] = None,
    pairs: Optional[tuple] = None,
) -> tuple:
    """
    Score fixed feature partitions on ``X`` or on a bootstrap resample of it.

    ``rows`` are resampled row indices; distances are computed once per unique
    row and pairs that refer to the same row twice are dropped.
    """
    pair_index = None
    if rows is not None:
        unique, inverse = np.unique(rows, return_inverse=True)
        X = X[unique]
        pair_index = resampled_pair_index(inverse, unique.size, pairs)
    scores = []
    for first, second in partitions:
        rdm1 = pdist(X[:, first], metric=metric)
        rdm2 = pdist(X[:, second], metric=metric)
        if pair_index is not None:
            rdm1, rdm2 = rdm1[pair_index], rdm2[pair_index]
        score, should_propagate = _score_rdm_pair(rdm1, rdm2, nan_policy)
        if should_propagate:
            return scores, True
        if score is not None:
            scores.append(score)
    return scores, False


def _feature_split_stability_result(
    X: np.ndarray,
    n_splits: int,
    metric: str,
    seed: Optional[int],
    max_samples: Optional[int],
    n_bootstrap_ci: Optional[int],
    ci: float,
    nan_policy: str,
    convergence: Optional[tuple] = None,
) -> StabilityResult:
    """Feature-split estimate with partition and dataset uncertainty separated."""
    metadata = result_metadata(
        "feature_split",
        metric=metric,
        nan_policy=nan_policy,
        seed=seed,
        n_splits=n_splits,
        max_samples=max_samples,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        bootstrap_unit="observations (feature partitions held fixed)",
    )
    if convergence is not None:
        metadata["target_mc_se"], metadata["max_splits"] = convergence
    if X.shape[0] < 4 or X.shape[1] < 4:
        return StabilityResult(
            estimate=np.nan,
            warnings=("feature_split requires at least 4 samples and 4 features.",),
            metadata=metadata,
        )

    rng = np.random.default_rng(seed)
    X = _prepare_feature_split_data(X, metric, max_samples, rng)
    metadata["n_samples_used"] = X.shape[0]
    partitions, scores, propagated = _run_feature_partitions(
        X, metric, nan_policy, rng, n_splits, convergence
    )
    metadata["n_partitions_drawn"] = len(partitions)
    if propagated:
        return StabilityResult(
            estimate=np.nan,
            warnings=("Undefined distances propagated to the result (nan_policy='propagate').",),
            metadata=metadata,
        )

    boot_scores = []
    if n_bootstrap_ci is not None:
        boot_scores = _bootstrap_feature_split(
            X, partitions, metric, nan_policy, n_bootstrap_ci, seed
        )
    ci_low, ci_high, ci_level, boot_notes = bootstrap_summary(boot_scores, n_bootstrap_ci, ci)
    partition_sd = float(np.std(scores, ddof=1)) if len(scores) >= 2 else np.nan
    return StabilityResult(
        estimate=float(np.mean(scores)) if scores else np.nan,
        data_ci_low=ci_low,
        data_ci_high=ci_high,
        ci_level=ci_level,
        partition_sd=partition_sd,
        monte_carlo_se=_monte_carlo_se(scores),
        n_valid_splits=len(scores),
        n_bootstraps=len(boot_scores),
        component_scores=tuple(float(score) for score in scores),
        warnings=_partition_notes(scores, len(partitions), convergence, metadata) + boot_notes,
        metadata=metadata,
    )


def _partition_notes(scores, n_drawn, convergence, metadata) -> tuple:
    """Warnings for skipped partitions and convergence; records ``converged``."""
    notes = ()
    if len(scores) < n_drawn:
        notes += (
            f"{n_drawn - len(scores)} of {n_drawn} feature partitions were "
            "unestimable and skipped.",
        )
    if convergence is not None:
        target, limit = convergence
        mc_se = _monte_carlo_se(scores)
        metadata["converged"] = bool(mc_se <= target)
        if not metadata["converged"]:
            notes += (
                f"{NOT_CONVERGED_PREFIX} target_mc_se={target:g} after {n_drawn} "
                f"partitions (max_splits={limit}); Monte Carlo SE is {mc_se:.4g}.",
            )
    return notes


def _bootstrap_feature_split(X, partitions, metric, nan_policy, n_bootstrap, seed):
    """Observation bootstrap conditional on the full-data feature partitions."""
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    n_samples = X.shape[0]
    pairs = np.triu_indices(n_samples, k=1)
    boot_scores = []
    for _ in range(n_bootstrap):
        rows = rng.choice(n_samples, n_samples, replace=True)
        scores, propagated = _score_feature_partitions(
            X, partitions, metric, nan_policy, rows=rows, pairs=pairs
        )
        if scores and not propagated:
            boot_scores.append(float(np.mean(scores)))
    return boot_scores


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
    nan_policy: Literal["raise", "omit", "propagate", "replace"] = "raise",
) -> Union[float, dict]:
    """
    Sample-Split Shesha (deprecated; invalid estimand).

    .. deprecated:: 0.3.0
       Use :func:`shesha.replicate_stability` with matched replicates.
       ``sample_split`` will be removed in a future release.

    .. warning::
       This function correlates unmatched RDM entries from independently drawn
       subsamples. Corresponding distances do **not** refer to the same
       observation pairs. Outputs must not be used for scientific inference.

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
    nan_policy : {'raise', 'omit', 'propagate', 'replace'}, default='raise'
        How to handle undefined distances. ``omit`` drops pairs undefined in
        either RDM, ``propagate`` returns NaN, and ``replace`` fills them with
        1.0 and warns.

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
    nan_policy: Literal["raise", "omit", "propagate", "replace"] = "raise",
) -> Union[float, dict]:
    """
    Anchor-based Shesha (deprecated; invalid estimand).

    .. deprecated:: 0.3.0
       Use :func:`shesha.anchor_profile_stability` with matched replicates.
       ``anchor_stability`` will be removed in a future release.

    .. warning::
       This function correlates flattened distance profiles whose columns
       refer to different probe observations. Rank-normalizing does not
       restore correspondence. Outputs must not be used for scientific
       inference.

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
    nan_policy : {'raise', 'omit', 'propagate', 'replace'}, default='raise'
        How to handle undefined distances. ``omit`` drops pairs undefined in
        either RDM, ``propagate`` returns NaN, and ``replace`` fills them with
        1.0 and warns.

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

    n_anchors, n_per_split = _legacy_anchor_sizes(n_samples, n_anchors, n_per_split)
    if n_samples < n_anchors + 2 * n_per_split:
        return np.nan

    anchor_idx = rng.choice(n_samples, n_anchors, replace=False)
    anchors = X[anchor_idx]
    remaining_idx = np.setdiff1d(np.arange(n_samples), anchor_idx)

    if len(remaining_idx) < 2 * n_per_split:
        return np.nan

    correlations = []
    for _ in range(n_splits):
        perm = rng.permutation(remaining_idx)
        D1 = cdist(anchors, X[perm[:n_per_split]], metric=metric)
        D2 = cdist(anchors, X[perm[n_per_split : 2 * n_per_split]], metric=metric)
        rho, should_propagate = _score_legacy_anchor_split(D1, D2, rank_normalize, nan_policy)
        if should_propagate:
            return np.nan
        if rho is not None:
            correlations.append(rho)

    return float(np.mean(correlations)) if correlations else np.nan


def _legacy_anchor_sizes(n_samples: int, n_anchors: int, n_per_split: int) -> tuple:
    """Shrink anchor and split sizes proportionally when samples are scarce."""
    min_required = n_anchors + 2 * n_per_split
    if n_samples < min_required:
        scale = n_samples / min_required * 0.9
        n_anchors = max(10, int(n_anchors * scale))
        n_per_split = max(20, int(n_per_split * scale))
    return n_anchors, n_per_split


def _score_legacy_anchor_split(
    D1: np.ndarray, D2: np.ndarray, rank_normalize: bool, nan_policy: str
) -> tuple:
    """Correlate two anchor-to-probe distance matrices, or ``(None, propagate)``."""
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
        return None, True
    if handled[0].shape != D1.ravel().shape:
        # Omission retains the within-anchor ranking performed above.
        rho, _ = spearmanr(handled[0], handled[1])
        return (rho, False) if np.isfinite(rho) else (None, False)
    D1 = handled[0].reshape(D1.shape)
    D2 = handled[1].reshape(D2.shape)

    if rank_normalize:
        D1 = np.apply_along_axis(rankdata, 1, D1)
        D2 = np.apply_along_axis(rankdata, 1, D2)

    rho, _ = spearmanr(D1.ravel(), D2.ravel())
    return (rho, False) if np.isfinite(rho) else (None, False)


# =============================================================================
# Supervised Variants
# =============================================================================


def variance_ratio(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
    seed: Optional[int] = None,
    return_type: Literal["float", "result"] = "float",
) -> Union[float, dict, StabilityResult]:
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
    return_type : {'float', 'result'}, default='float'
        ``'result'`` returns a :class:`~shesha.StabilityResult` with the
        full-data ``estimate`` and, when ``n_bootstrap_ci`` is set, its data
        interval.

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
    validate_return_type(return_type)
    if return_type == "result":
        return bootstrap_result(
            variance_ratio,
            "variance_ratio",
            (
                X,
                y,
            ),
            {"seed": seed},
            n_bootstrap_ci,
            ci,
            bootstrap_unit="observations (rows of X and y resampled together)",
            monte_carlo_se=0.0,
        )

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
    return_type: Literal["float", "result"] = "float",
) -> Union[float, dict, StabilityResult]:
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
    return_type : {'float', 'result'}, default='float'
        ``'result'`` returns a :class:`~shesha.StabilityResult` with the
        full-data ``estimate`` and, when ``n_bootstrap_ci`` is set, its data
        interval.

    Returns
    -------
    float or dict
        If n_bootstrap_ci is None: Spearman correlation. Range: [-1, 1].
        If n_bootstrap_ci is set: dict with keys 'mean', 'ci_low', 'ci_high',
        'std', 'n_bootstraps', 'ci_level'.
    """
    validate_return_type(return_type)
    if return_type == "result":
        return bootstrap_result(
            supervised_alignment,
            "supervised_alignment",
            (
                X,
                y,
            ),
            {"metric": metric, "seed": seed, "max_samples": max_samples},
            n_bootstrap_ci,
            ci,
            bootstrap_unit="observations (rows of X and y resampled together)",
            monte_carlo_se=0.0 if len(X) <= max_samples else np.nan,
        )

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
    return_type: Literal["float", "result"] = "float",
) -> Union[float, dict, StabilityResult]:
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
    return_type : {'float', 'result'}, default='float'
        ``'result'`` returns a :class:`~shesha.StabilityResult` with the
        full-data ``estimate`` and, when ``n_bootstrap_ci`` is set, its data
        interval.

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
    validate_return_type(return_type)
    if return_type == "result":
        return bootstrap_result(
            class_separation_ratio,
            "class_separation_ratio",
            (
                X,
                y,
            ),
            {
                "n_bootstrap": n_bootstrap,
                "subsample_frac": subsample_frac,
                "metric": metric,
                "seed": seed,
            },
            n_bootstrap_ci,
            ci,
            bootstrap_unit="observations (rows of X and y resampled together)",
            monte_carlo_se=np.nan,
        )

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
        idx = rng.choice(len(X), n_samples, replace=False)
        ratio = _class_separation_once(X[idx], y[idx], metric)
        if ratio is not None:
            ratios.append(ratio)

    return float(np.mean(ratios)) if len(ratios) > 0 else np.nan


def _class_separation_once(X_sub: np.ndarray, y_sub: np.ndarray, metric: str) -> Optional[float]:
    """Mean between-class over mean within-class distance for one subsample."""
    labels = np.unique(y_sub)
    if len(labels) < 2:
        return None

    dists = cdist(X_sub, X_sub, metric="euclidean" if metric == "euclidean" else "cosine")
    masks = [y_sub == label for label in labels]

    within_dists = []
    for mask in masks:
        if np.sum(mask) < 2:
            continue
        class_dists = dists[mask][:, mask]
        within_dists.extend(class_dists[np.triu_indices_from(class_dists, k=1)])

    between_dists = []
    for i, mask_i in enumerate(masks):
        for mask_j in masks[i + 1 :]:
            between_dists.extend(dists[mask_i][:, mask_j].flatten())

    if len(within_dists) == 0 or len(between_dists) == 0:
        return None
    mean_within = np.mean(within_dists)
    if mean_within <= EPS:
        return None
    return np.mean(between_dists) / mean_within


def lda_stability(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 50,
    subsample_frac: float = 0.5,
    seed: Optional[int] = None,
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
    return_type: Literal["float", "result"] = "float",
) -> Union[float, dict, StabilityResult]:
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
    return_type : {'float', 'result'}, default='float'
        ``'result'`` returns a :class:`~shesha.StabilityResult` with the
        full-data ``estimate`` and, when ``n_bootstrap_ci`` is set, its data
        interval.

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
    validate_return_type(return_type)
    if return_type == "result":
        return bootstrap_result(
            lda_stability,
            "lda_stability",
            (
                X,
                y,
            ),
            {"n_bootstrap": n_bootstrap, "subsample_frac": subsample_frac, "seed": seed},
            n_bootstrap_ci,
            ci,
            bootstrap_unit="observations (rows of X and y resampled together)",
            monte_carlo_se=np.nan,
        )

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
    nan_policy: Literal["raise", "omit", "propagate", "replace"] = "raise",
    return_type: Literal["float", "result"] = "float",
) -> Union[float, dict, StabilityResult]:
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
    nan_policy : {'raise', 'omit', 'propagate', 'replace'}, default='raise'
        How to handle undefined distances. ``omit`` drops pairs undefined in
        either RDM, ``propagate`` returns NaN, and ``replace`` fills them with
        1.0 and warns.
    return_type : {'float', 'result'}, default='float'
        ``'result'`` returns a :class:`~shesha.StabilityResult` with the
        full-data ``estimate`` and, when ``n_bootstrap_ci`` is set, a data
        interval from an observation bootstrap that drops pairs referring
        to the same observation twice.

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

    validate_return_type(return_type)
    if return_type == "result":
        estimate = rdm_similarity(X, Y, method=method, metric=metric, nan_policy=nan_policy)
        return paired_rdm_result(
            "rdm_similarity",
            estimate,
            X,
            Y,
            metric=metric,
            method=method,
            nan_policy=nan_policy,
            n_bootstrap_ci=n_bootstrap_ci,
            ci=ci,
            seed=seed,
            drift=False,
        )

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
    nan_policy: Literal["raise", "omit", "propagate", "replace"] = "raise",
    return_type: Literal["float", "result"] = "float",
) -> Union[float, dict, StabilityResult]:
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
    nan_policy : {'raise', 'omit', 'propagate', 'replace'}, default='raise'
        How to handle undefined distances. ``omit`` drops pairs undefined in
        either RDM, ``propagate`` returns NaN, and ``replace`` fills them with
        1.0 and warns.
    return_type : {'float', 'result'}, default='float'
        ``'result'`` returns a :class:`~shesha.StabilityResult` with the
        full-data ``estimate`` and, when ``n_bootstrap_ci`` is set, a data
        interval from an observation bootstrap that drops pairs referring
        to the same observation twice.

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

    validate_return_type(return_type)
    if return_type == "result":
        estimate = rdm_drift(X, Y, method=method, metric=metric, nan_policy=nan_policy)
        return paired_rdm_result(
            "rdm_drift",
            estimate,
            X,
            Y,
            metric=metric,
            method=method,
            nan_policy=nan_policy,
            n_bootstrap_ci=n_bootstrap_ci,
            ci=ci,
            seed=seed,
            drift=True,
        )

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
        - 'sample_split': Deprecated invalid estimand (emits FutureWarning; do
          not use for inference). Use :func:`shesha.replicate_stability`.
        - 'anchor': Deprecated invalid estimand (emits FutureWarning; do not
          use for inference). Use :func:`shesha.anchor_profile_stability`.
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
