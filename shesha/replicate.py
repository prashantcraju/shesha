"""
Matched-replicate stability estimators.

These estimators compare two representations of the *same* observations, for
example repeated measurements of the same stimuli or cell conditions, or two
fitted models evaluated on a shared test set. Every compared distance refers
to the same pair of observations in both inputs, so the estimand is
identifiable. They replace the unmatched ``sample_split`` and
``anchor_stability`` estimators.
"""

from typing import Optional, Sequence, Union

import numpy as np
from scipy.spatial.distance import cdist, pdist

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ._rdm import CORR_METHODS, RDM_METRICS, resampled_pair_index, rowwise_correlation
from ._result import (
    StabilityResult,
    bootstrap_summary,
    result_metadata,
    validate_return_type,
)
from ._validate import (
    align_paired_ids,
    apply_nan_policy,
    as_2d_array,
    resolve_positions,
    validate_ci,
    validate_metric,
    validate_nan_policy,
    validate_positive_int,
)

__all__ = ["replicate_stability", "anchor_profile_stability"]

PROPAGATED_WARNING = "Undefined distances propagated to the result (nan_policy='propagate')."


def _validate_options(metric, method, n_bootstrap_ci, ci, nan_policy, return_type) -> None:
    validate_metric(metric, RDM_METRICS)
    validate_metric(method, CORR_METHODS, name="method")
    validate_positive_int(n_bootstrap_ci, "n_bootstrap_ci", allow_none=True)
    validate_ci(ci)
    validate_nan_policy(nan_policy)
    validate_return_type(return_type)
    if n_bootstrap_ci is not None and return_type == "float":
        raise ValueError(
            "n_bootstrap_ci requires return_type='result', which returns the "
            "full-data estimate and its data interval as a StabilityResult."
        )


def _aligned_inputs(X_a, X_b, ids_a, ids_b):
    X_a = as_2d_array(X_a, "X_a")
    X_b = as_2d_array(X_b, "X_b")
    return align_paired_ids(X_a, X_b, ids_a, ids_b)


def _apply_pair_policy(D_a: np.ndarray, D_b: np.ndarray, nan_policy: str):
    """
    Apply ``nan_policy`` to aligned distance arrays of any shape.

    Returns ``(D_a, D_b, valid)`` where ``valid`` marks entries defined in both
    inputs, or ``None`` when the estimator must propagate NaN.
    """
    handled = apply_nan_policy(D_a.ravel(), D_b.ravel(), nan_policy=nan_policy)
    if handled is None:
        return None
    if nan_policy == "replace":
        D_a = handled[0].reshape(D_a.shape)
        D_b = handled[1].reshape(D_b.shape)
    return D_a, D_b, np.isfinite(D_a) & np.isfinite(D_b)


def _finish(estimate, return_type, boot_scores, n_bootstrap_ci, ci, metadata, notes, components=()):
    if return_type == "float":
        return estimate
    ci_low, ci_high, ci_level, boot_notes = bootstrap_summary(boot_scores, n_bootstrap_ci, ci)
    return StabilityResult(
        estimate=estimate,
        data_ci_low=ci_low,
        data_ci_high=ci_high,
        ci_level=ci_level,
        partition_sd=np.nan,
        monte_carlo_se=0.0,
        n_valid_splits=None,
        n_bootstraps=len(boot_scores),
        component_scores=tuple(float(score) for score in components),
        warnings=tuple(notes) + boot_notes,
        metadata=metadata,
    )


# =============================================================================
# Matched replicate RDM stability
# =============================================================================


def replicate_stability(
    X_a: np.ndarray,
    X_b: np.ndarray,
    *,
    ids_a: Optional[Sequence] = None,
    ids_b: Optional[Sequence] = None,
    metric: Literal["cosine", "correlation", "euclidean"] = "cosine",
    method: Literal["spearman", "pearson"] = "spearman",
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
    seed: Optional[int] = None,
    nan_policy: Literal["raise", "omit", "propagate", "replace"] = "raise",
    return_type: Literal["float", "result"] = "float",
) -> Union[float, StabilityResult]:
    """
    Matched-replicate RDM stability.

    Correlates the RDM of ``X_a`` with the RDM of ``X_b`` where row ``i`` of
    both inputs represents the same observation (stimulus, cell condition,
    sample) measured in two replicates. Corresponding RDM entries therefore
    describe the same observation pair.

    Parameters
    ----------
    X_a, X_b : np.ndarray
        Replicate representations of shape (n_samples, n_features_a) and
        (n_samples, n_features_b). Feature dimensions may differ.
    ids_a, ids_b : sequence, optional
        Unique observation identifiers for the rows of ``X_a`` and ``X_b``.
        When given, both must contain the same identifiers and ``X_b`` is
        reordered to match ``X_a``. When omitted, rows are assumed to already
        correspond; row order cannot be verified without identifiers.
    metric : {'cosine', 'correlation', 'euclidean'}, default='cosine'
        Distance metric for both RDMs.
    method : {'spearman', 'pearson'}, default='spearman'
        Correlation between the two RDMs.
    n_bootstrap_ci : int, optional
        Number of observation bootstrap resamples for a data interval.
        Requires ``return_type='result'``. Pairs that refer to the same
        observation twice are excluded from each resample.
    ci : float, default=0.95
        Confidence level for the data interval.
    seed : int, optional
        Seed for the bootstrap.
    nan_policy : {'raise', 'omit', 'propagate', 'replace'}, default='raise'
        Handling of undefined distances (for example cosine distance to a
        zero vector). ``omit`` drops pairs undefined in either replicate.
    return_type : {'float', 'result'}, default='float'
        ``'result'`` returns a :class:`~shesha.StabilityResult`.

    Returns
    -------
    float or StabilityResult
        RDM correlation in [-1, 1]. NaN when either RDM is constant.

    Raises
    ------
    ValueError
        On mismatched sample counts or identifiers, fewer than 3 matched
        observations, invalid options, or undefined distances under
        ``nan_policy='raise'``.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> latent = rng.standard_normal((200, 10))
    >>> X_a = latent + 0.3 * rng.standard_normal((200, 10))
    >>> X_b = latent + 0.3 * rng.standard_normal((200, 10))
    >>> score = replicate_stability(X_a, X_b)

    >>> # Align replicates recorded in different orders
    >>> order = rng.permutation(200)
    >>> score = replicate_stability(
    ...     X_a, X_b[order], ids_a=np.arange(200), ids_b=order
    ... )

    >>> result = replicate_stability(X_a, X_b, n_bootstrap_ci=500, seed=0,
    ...                              return_type="result")
    >>> result.estimate, result.data_ci_low, result.data_ci_high
    """
    _validate_options(metric, method, n_bootstrap_ci, ci, nan_policy, return_type)
    X_a, X_b, ids = _aligned_inputs(X_a, X_b, ids_a, ids_b)
    n_samples = X_a.shape[0]
    if n_samples < 3:
        raise ValueError("replicate_stability requires at least 3 matched observations")

    metadata = result_metadata(
        "replicate_stability",
        metric=metric,
        method=method,
        nan_policy=nan_policy,
        seed=seed,
        n_samples=n_samples,
        n_features_a=X_a.shape[1],
        n_features_b=X_b.shape[1],
        aligned_by_ids=ids is not None,
    )
    handled = _apply_pair_policy(pdist(X_a, metric=metric), pdist(X_b, metric=metric), nan_policy)
    if handled is None:
        return _finish(np.nan, return_type, [], n_bootstrap_ci, ci, metadata, (PROPAGATED_WARNING,))
    rdm_a, rdm_b, valid = handled
    metadata["n_pairs"] = int(valid.sum())
    estimate = float(rowwise_correlation(rdm_a[valid], rdm_b[valid], method)[0])

    boot_scores = []
    if n_bootstrap_ci is not None:
        boot_scores = _bootstrap_replicate(
            rdm_a, rdm_b, valid, n_samples, n_bootstrap_ci, method, seed
        )
    return _finish(estimate, return_type, boot_scores, n_bootstrap_ci, ci, metadata, ())


def paired_rdm_result(
    estimator: str,
    estimate: float,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    metric: str,
    method: str,
    nan_policy: str,
    n_bootstrap_ci: Optional[int],
    ci: float,
    seed: Optional[int],
    drift: bool = False,
) -> StabilityResult:
    """
    Structured result for RDM similarity (or drift) of two paired inputs.

    The data interval uses the same duplicate-aware observation bootstrap as
    :func:`replicate_stability`.
    """
    boot_scores = []
    if n_bootstrap_ci is not None and X.shape[0] >= 3:
        handled = _apply_pair_policy(pdist(X, metric=metric), pdist(Y, metric=metric), nan_policy)
        if handled is not None:
            boot_scores = _bootstrap_replicate(*handled, X.shape[0], n_bootstrap_ci, method, seed)
    if drift:
        boot_scores = [1.0 - score for score in boot_scores]
    metadata = result_metadata(
        estimator,
        metric=metric,
        method=method,
        nan_policy=nan_policy,
        seed=seed,
        n_samples=X.shape[0],
        n_features_a=X.shape[1],
        n_features_b=Y.shape[1],
        bootstrap_unit="observations (paired; duplicated-observation pairs dropped)",
    )
    return _finish(estimate, "result", boot_scores, n_bootstrap_ci, ci, metadata, ())


def _bootstrap_replicate(rdm_a, rdm_b, valid, n_samples, n_bootstrap, method, seed):
    rng = np.random.default_rng(seed)
    pairs = np.triu_indices(n_samples, k=1)
    scores = []
    for _ in range(n_bootstrap):
        rows = rng.choice(n_samples, n_samples, replace=True)
        index = resampled_pair_index(rows, n_samples, pairs)
        index = index[valid[index]]
        score = rowwise_correlation(rdm_a[index], rdm_b[index], method)[0]
        if np.isfinite(score):
            scores.append(float(score))
    return scores


# =============================================================================
# Matched anchor-profile stability
# =============================================================================


def anchor_profile_stability(
    X_a: np.ndarray,
    X_b: np.ndarray,
    anchor_ids: Sequence,
    probe_ids: Optional[Sequence] = None,
    *,
    ids_a: Optional[Sequence] = None,
    ids_b: Optional[Sequence] = None,
    metric: Literal["cosine", "correlation", "euclidean"] = "cosine",
    method: Literal["spearman", "pearson"] = "spearman",
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
    seed: Optional[int] = None,
    nan_policy: Literal["raise", "omit", "propagate", "replace"] = "raise",
    return_type: Literal["float", "result"] = "float",
) -> Union[float, StabilityResult]:
    """
    Matched anchor-profile stability.

    For each anchor, computes its distance profile to a shared set of probe
    observations in both replicates and correlates the two profiles across
    probes. The estimate is the mean per-anchor correlation. Anchors and
    probes refer to the same observations in ``X_a`` and ``X_b``, so every
    compared distance describes the same anchor-probe pair.

    Parameters
    ----------
    X_a, X_b : np.ndarray
        Replicate representations of shape (n_samples, n_features_a) and
        (n_samples, n_features_b).
    anchor_ids : sequence
        Anchor observations. Identifiers from ``ids_a`` when identifiers are
        given, otherwise integer row indices.
    probe_ids : sequence, optional
        Probe observations, specified like ``anchor_ids``. Defaults to every
        non-anchor observation. Must be disjoint from ``anchor_ids`` and
        contain at least 3 observations.
    ids_a, ids_b : sequence, optional
        Unique observation identifiers used to align ``X_b`` to ``X_a``. See
        :func:`replicate_stability`.
    metric : {'cosine', 'correlation', 'euclidean'}, default='cosine'
        Anchor-probe distance metric.
    method : {'spearman', 'pearson'}, default='spearman'
        Correlation between matched anchor profiles.
    n_bootstrap_ci : int, optional
        Number of probe bootstrap resamples for a data interval. Anchors are
        treated as fixed reference observations. Requires
        ``return_type='result'``.
    ci : float, default=0.95
        Confidence level for the data interval.
    seed : int, optional
        Seed for the bootstrap.
    nan_policy : {'raise', 'omit', 'propagate', 'replace'}, default='raise'
        Handling of undefined anchor-probe distances.
    return_type : {'float', 'result'}, default='float'
        ``'result'`` returns a :class:`~shesha.StabilityResult` whose
        ``component_scores`` are the per-anchor correlations.

    Returns
    -------
    float or StabilityResult
        Mean per-anchor profile correlation in [-1, 1]. Anchors with a
        constant profile in either replicate are excluded; NaN if none remain.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> latent = rng.standard_normal((300, 10))
    >>> X_a = latent + 0.3 * rng.standard_normal((300, 10))
    >>> X_b = latent + 0.3 * rng.standard_normal((300, 10))
    >>> score = anchor_profile_stability(X_a, X_b, anchor_ids=np.arange(20))
    """
    _validate_options(metric, method, n_bootstrap_ci, ci, nan_policy, return_type)
    X_a, X_b, ids = _aligned_inputs(X_a, X_b, ids_a, ids_b)
    anchors, probes = _anchor_probe_positions(anchor_ids, probe_ids, ids, X_a.shape[0])

    metadata = result_metadata(
        "anchor_profile_stability",
        metric=metric,
        method=method,
        nan_policy=nan_policy,
        seed=seed,
        n_samples=X_a.shape[0],
        n_features_a=X_a.shape[1],
        n_features_b=X_b.shape[1],
        n_anchors=int(anchors.size),
        n_probes=int(probes.size),
        aligned_by_ids=ids is not None,
        bootstrap_unit="probes",
    )
    handled = _apply_pair_policy(
        cdist(X_a[anchors], X_a[probes], metric=metric),
        cdist(X_b[anchors], X_b[probes], metric=metric),
        nan_policy,
    )
    if handled is None:
        return _finish(np.nan, return_type, [], n_bootstrap_ci, ci, metadata, (PROPAGATED_WARNING,))
    D_a, D_b, valid = handled
    anchor_scores = _anchor_scores(D_a, D_b, valid, method)
    defined = np.isfinite(anchor_scores)
    metadata["n_valid_anchors"] = int(defined.sum())
    estimate = float(np.mean(anchor_scores[defined])) if defined.any() else np.nan
    notes = ()
    if not defined.all():
        notes = (
            f"{int((~defined).sum())} of {anchors.size} anchors had an unestimable "
            "profile and were excluded.",
        )

    boot_scores = []
    if n_bootstrap_ci is not None:
        boot_scores = _bootstrap_anchor(D_a, D_b, valid, n_bootstrap_ci, method, seed)
    return _finish(
        estimate, return_type, boot_scores, n_bootstrap_ci, ci, metadata, notes, anchor_scores
    )


def _anchor_probe_positions(anchor_ids, probe_ids, ids, n_samples):
    anchors = resolve_positions(anchor_ids, ids, n_samples, "anchor_ids")
    if probe_ids is None:
        probes = np.setdiff1d(np.arange(n_samples), anchors)
    else:
        probes = resolve_positions(probe_ids, ids, n_samples, "probe_ids")
        if np.intersect1d(anchors, probes).size:
            raise ValueError("anchor_ids and probe_ids must be disjoint")
    if probes.size < 3:
        raise ValueError("anchor_profile_stability requires at least 3 probe observations")
    return anchors, probes


def _anchor_scores(D_a, D_b, valid, method):
    """Per-anchor profile correlations over jointly defined probes."""
    if valid.all():
        return rowwise_correlation(D_a, D_b, method)
    return np.array(
        [
            rowwise_correlation(D_a[row, valid[row]], D_b[row, valid[row]], method)[0]
            for row in range(D_a.shape[0])
        ]
    )


def _bootstrap_anchor(D_a, D_b, valid, n_bootstrap, method, seed):
    rng = np.random.default_rng(seed)
    n_probes = D_a.shape[1]
    scores = []
    for _ in range(n_bootstrap):
        columns = rng.choice(n_probes, n_probes, replace=True)
        per_anchor = _anchor_scores(D_a[:, columns], D_b[:, columns], valid[:, columns], method)
        per_anchor = per_anchor[np.isfinite(per_anchor)]
        if per_anchor.size:
            scores.append(float(np.mean(per_anchor)))
    return scores
