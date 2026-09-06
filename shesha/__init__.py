"""
Shesha: Self-consistency Metrics for Representational Stability

A framework for measuring geometric stability via self-consistency of
Representational Dissimilarity Matrices (RDMs).

Basic usage:
    >>> import shesha
    >>> stability = shesha.feature_split(X, n_splits=30, seed=320)

    >>> # Or with labels
    >>> alignment = shesha.supervised_alignment(X, y)

    >>> # Unified interface
    >>> score = shesha.shesha(X, variant='feature_split')

    >>> # Measure drift between representations
    >>> similarity = shesha.rdm_similarity(X_before, X_after)
    >>> drift = shesha.rdm_drift(X_before, X_after)

    >>> # Biological perturbation analysis
    >>> from shesha.bio import perturbation_coherence
    >>> coherence = perturbation_coherence(X_control, X_perturbed)
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

from . import bio, sim
from .core import (
    anchor_stability,
    class_separation_ratio,
    # Utilities
    compute_rdm,
    # Unsupervised variants
    feature_split,
    lda_stability,
    rdm_drift,
    # Drift metrics
    rdm_similarity,
    sample_split,
    # Main function
    shesha,
    supervised_alignment,
    # Supervised variants
    variance_ratio,
)


def _package_version() -> str:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.is_file():
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("version") and "=" in stripped:
                return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    try:
        return _pkg_version("shesha-geometry")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _package_version()
__author__ = "Prashant Raju"

__all__ = [
    "shesha",
    "feature_split",
    "sample_split",
    "anchor_stability",
    "variance_ratio",
    "supervised_alignment",
    "class_separation_ratio",
    "lda_stability",
    "rdm_similarity",
    "rdm_drift",
    "compute_rdm",
    "bio",
    "sim",
]
