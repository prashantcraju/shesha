"""
Structured estimator results.

``StabilityResult`` separates the full-data point estimate from dataset
(bootstrap) uncertainty and from Monte Carlo variability across random
partitions, and records the configuration needed to reconstruct a result.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

RETURN_TYPES = ("float", "result")


@dataclass(frozen=True)
class StabilityResult:
    """
    Structured result returned when ``return_type="result"``.

    Attributes
    ----------
    estimate : float
        Point estimate computed on the full (possibly subsampled) input. This
        is never the mean of bootstrap replicates.
    data_ci_low, data_ci_high : float
        Percentile bootstrap interval over observations. ``NaN`` when no
        bootstrap was requested.
    ci_level : float or None
        Confidence level of the data interval.
    partition_sd : float
        Standard deviation of per-split scores (random feature partitions).
        ``NaN`` for estimators without random partitions.
    monte_carlo_se : float
        Monte Carlo standard error of ``estimate`` from the finite number of
        partitions (``partition_sd / sqrt(n_valid_splits)``). ``0.0`` for
        deterministic estimators; NaN when an estimator's internal random
        subsampling is not summarized.
    n_valid_splits : int or None
        Number of partitions that produced a finite score. ``None`` when the
        estimator has no partitions.
    n_bootstraps : int
        Number of bootstrap replicates that produced a finite score.
    component_scores : tuple of float
        Per-split scores (``feature_split``) or per-anchor scores
        (``anchor_profile_stability``).
    warnings : tuple of str
        Estimability warnings, for example skipped partitions or resamples.
    metadata : dict
        Estimator name, configuration, input shape, seed, and package version.
    """

    estimate: float
    data_ci_low: float = np.nan
    data_ci_high: float = np.nan
    ci_level: Optional[float] = None
    partition_sd: float = np.nan
    monte_carlo_se: float = np.nan
    n_valid_splits: Optional[int] = None
    n_bootstraps: int = 0
    component_scores: Tuple[float, ...] = ()
    warnings: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.estimate)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary (tuples become lists)."""
        data = asdict(self)
        data["component_scores"] = list(self.component_scores)
        data["warnings"] = list(self.warnings)
        return data


def validate_return_type(return_type: str) -> None:
    if return_type not in RETURN_TYPES:
        allowed = ", ".join(repr(item) for item in RETURN_TYPES)
        raise ValueError(f"Unknown return_type: {return_type!r}. Use one of: {allowed}")


def percentile_interval(scores, ci: float) -> Tuple[float, float]:
    """Percentile interval of finite bootstrap scores, or ``(nan, nan)``."""
    if len(scores) == 0:
        return np.nan, np.nan
    alpha = 1.0 - ci
    return (
        float(np.percentile(scores, 100 * alpha / 2)),
        float(np.percentile(scores, 100 * (1 - alpha / 2))),
    )


def bootstrap_summary(
    scores, n_requested: Optional[int], ci: float
) -> Tuple[float, float, Optional[float], Tuple[str, ...]]:
    """Return ``(ci_low, ci_high, ci_level, warnings)`` for bootstrap scores."""
    if not n_requested:
        return np.nan, np.nan, None, ()
    low, high = percentile_interval(scores, ci)
    skipped = n_requested - len(scores)
    notes = ()
    if skipped:
        notes = (f"{skipped} of {n_requested} bootstrap resamples were unestimable and skipped.",)
    return low, high, ci, notes


def bootstrap_result(
    func: Callable,
    estimator: str,
    args: tuple,
    kwargs: Dict[str, Any],
    n_bootstrap_ci: Optional[int],
    ci: float,
    *,
    bootstrap_unit: str,
    monte_carlo_se: float = 0.0,
) -> StabilityResult:
    """
    Structured result for a metric with a float return and a dict bootstrap.

    ``func(*args, **kwargs)`` gives the full-data estimate. When requested, the
    data interval comes from ``func``'s own bootstrap. ``monte_carlo_se`` is
    0.0 for deterministic metrics and NaN when internal random subsampling
    makes it unestimated.
    """
    estimate = float(func(*args, **kwargs))
    notes = ()
    ci_low = ci_high = np.nan
    n_bootstraps = 0
    if n_bootstrap_ci is not None:
        boot = func(*args, n_bootstrap_ci=n_bootstrap_ci, ci=ci, **kwargs)
        if not isinstance(boot, dict):
            # The metric returned NaN before bootstrapping (unestimable input).
            boot = {"ci_low": np.nan, "ci_high": np.nan, "n_bootstraps": 0}
        ci_low, ci_high, n_bootstraps = boot["ci_low"], boot["ci_high"], boot["n_bootstraps"]
        if n_bootstraps < n_bootstrap_ci:
            notes = (
                f"{n_bootstrap_ci - n_bootstraps} of {n_bootstrap_ci} bootstrap resamples "
                "were unestimable and skipped.",
            )
    metadata = result_metadata(
        estimator,
        input_shapes=[list(np.shape(arg)) for arg in args],
        bootstrap_unit=bootstrap_unit,
        **kwargs,
    )
    return StabilityResult(
        estimate=estimate,
        data_ci_low=ci_low,
        data_ci_high=ci_high,
        ci_level=ci if n_bootstrap_ci is not None else None,
        monte_carlo_se=monte_carlo_se,
        n_bootstraps=n_bootstraps,
        warnings=notes,
        metadata=metadata,
    )


def result_metadata(estimator: str, **config) -> Dict[str, Any]:
    from . import __version__

    metadata = {"estimator": estimator, "package_version": __version__}
    metadata.update(config)
    return metadata
