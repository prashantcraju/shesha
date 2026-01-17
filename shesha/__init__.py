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
    >>> from shesha.bio import perturbation_stability
    >>> stability = perturbation_stability(X_control, X_perturbed)
"""

from .core import (
    # Main function
    shesha,
    # Unsupervised variants
    feature_split,
    sample_split,
    anchor_stability,
    # Supervised variants
    variance_ratio,
    supervised_alignment,
    # Drift metrics
    rdm_similarity,
    rdm_drift,
    # Utilities
    compute_rdm,
)

from . import bio

__version__ = "0.1.2"
__author__ = "Prashant Raju"

__all__ = [
    "shesha",
    "feature_split",
    "sample_split",
    "anchor_stability",
    "variance_ratio",
    "supervised_alignment",
    "rdm_similarity",
    "rdm_drift",
    "compute_rdm",
    "bio",
]