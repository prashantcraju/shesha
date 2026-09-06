"""
Shared input validation and NaN-policy helpers.

These helpers raise on invalid configuration. Valid-but-unestimable data
(for example too few samples) is left to the caller to return NaN.
"""

import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


NanPolicy = Literal["replace", "raise", "omit", "propagate"]

NAN_POLICIES = ("replace", "raise", "omit", "propagate")
NAN_REPLACE_FILL = 1.0
NAN_REPLACE_WARNING = (
    "Undefined distances were replaced with 1.0 (nan_policy='replace'). "
    "This silently treats undefined pairs as maximally dissimilar and can invent "
    "structure. The default will change to nan_policy='raise' in shesha-geometry "
    "0.3.0. Pass nan_policy='raise', 'omit', or 'propagate' to opt in now."
)


def as_2d_array(
    X,
    name: str = "X",
    *,
    require_finite: bool = True,
) -> np.ndarray:
    """Convert ``X`` to a finite 2-D float64 array."""
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-dimensional array, got shape {array.shape}")
    if require_finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def as_label_array(y, n_samples: int, name: str = "y") -> np.ndarray:
    """Convert labels to a 1-D array whose length matches ``n_samples``."""
    labels = np.asarray(y)
    if labels.ndim != 1:
        labels = np.ravel(labels)
    if labels.shape[0] != n_samples:
        raise ValueError(
            f"{name} length ({labels.shape[0]}) must match number of samples ({n_samples})"
        )
    return labels


def validate_paired_samples(
    X: np.ndarray, Y: np.ndarray, names: Tuple[str, str] = ("X", "Y")
) -> None:
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"{names[0]} and {names[1]} must have the same number of samples: "
            f"{names[0]} has {X.shape[0]}, {names[1]} has {Y.shape[0]}"
        )


def validate_metric(metric: str, allowed: Sequence[str], name: str = "metric") -> None:
    if metric not in allowed:
        allowed_str = ", ".join(repr(item) for item in allowed)
        raise ValueError(f"Unknown {name}: {metric!r}. Use one of: {allowed_str}")


def validate_positive_int(
    value: Optional[int],
    name: str,
    *,
    minimum: int = 1,
    allow_none: bool = False,
) -> None:
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{name} is required")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def validate_fraction(value: float, name: str, *, open_low: bool = True) -> None:
    if open_low:
        ok = 0.0 < float(value) <= 1.0
        interval = "(0, 1]"
    else:
        ok = 0.0 <= float(value) <= 1.0
        interval = "[0, 1]"
    if not ok:
        raise ValueError(f"{name} must be in {interval}, got {value!r}")


def validate_ci(ci: float) -> None:
    if not (0.0 < float(ci) < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")


def validate_nan_policy(nan_policy: str) -> None:
    if nan_policy not in NAN_POLICIES:
        allowed_str = ", ".join(repr(item) for item in NAN_POLICIES)
        raise ValueError(f"Unknown nan_policy: {nan_policy!r}. Use one of: {allowed_str}")


def apply_nan_policy(
    *arrays: np.ndarray,
    nan_policy: str = "replace",
    fill_value: float = NAN_REPLACE_FILL,
) -> Union[Tuple[np.ndarray, ...], None]:
    """
    Apply ``nan_policy`` to one or more aligned condensed distance arrays.

    Returns the processed arrays. For ``omit``, corresponding undefined
    entries are dropped from every array. Returns ``None`` only when
    ``propagate`` is requested and a caller-level NaN result is preferred
    because every entry is undefined.
    """
    validate_nan_policy(nan_policy)
    processed = [np.asarray(array, dtype=np.float64) for array in arrays]
    if not processed:
        return tuple()

    nan_mask = np.zeros(processed[0].shape, dtype=bool)
    for array in processed:
        if array.shape != processed[0].shape:
            raise ValueError("All arrays passed to apply_nan_policy must have the same shape")
        nan_mask |= ~np.isfinite(array)

    if not nan_mask.any():
        return tuple(processed)

    if nan_policy == "raise":
        n_undefined = int(np.sum(nan_mask))
        raise ValueError(
            f"Undefined distances encountered ({n_undefined} entries). "
            "This often occurs with zero or constant vectors under cosine or "
            "correlation distance. Pass nan_policy='omit', 'propagate', or "
            "'replace' to handle them."
        )

    if nan_policy == "replace":
        warnings.warn(NAN_REPLACE_WARNING, FutureWarning, stacklevel=3)
        filled = [
            np.nan_to_num(array, nan=fill_value, posinf=fill_value, neginf=fill_value)
            for array in processed
        ]
        return tuple(filled)

    if nan_policy == "propagate":
        if np.all(nan_mask):
            return None
        return tuple(processed)

    omitted = int(np.sum(nan_mask))
    warnings.warn(
        f"Omitted {omitted} undefined distance pair(s) (nan_policy='omit').",
        UserWarning,
        stacklevel=3,
    )
    keep = ~nan_mask
    return tuple(array[keep] for array in processed)
