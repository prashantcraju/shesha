"""
Shesha Bio: Stability metrics for biological perturbation experiments.

This module provides Shesha variants for single-cell and perturbation biology,
measuring the consistency of perturbation effects across individual cells.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ._utils import bootstrap_ci_bio

try:
    from anndata import AnnData
except ImportError:
    AnnData = None

__all__ = [
    "perturbation_stability", 
    "perturbation_stability_whitened",
    "perturbation_stability_knn",
    "perturbation_effect_size", 
    "compute_stability",
    "compute_stability_whitened",
    "compute_stability_knn",
    "compute_magnitude",
    "split_half_reproducibility",
    "magnitude_matched_comparison",
    "discordance",
]

EPS = 1e-12


def perturbation_stability(
    X_control: np.ndarray,
    X_perturbed: np.ndarray,
    method: Literal["standard", "whitened", "knn"] = "standard",
    metric: Literal["cosine", "euclidean"] = "cosine",
    k: int = 50,
    regularization: float = 1e-6,
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1000,
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
) -> Union[float, dict]:
    """
    Perturbation stability: consistency of perturbation effects across samples.
    
    Measures whether individual perturbed samples shift in a consistent direction
    relative to the control population. High values indicate that the perturbation
    has a coherent, reproducible effect; low values suggest heterogeneous or noisy
    responses.
    
    Parameters
    ----------
    X_control : np.ndarray
        Control population embeddings, shape (n_control, n_features).
    X_perturbed : np.ndarray
        Perturbed population embeddings, shape (n_perturbed, n_features).
    method : {'standard', 'whitened', 'knn'}, default='standard'
        Method for computing stability:
        - 'standard': Global control centroid (default)
        - 'whitened': Mahalanobis-scaled using control covariance
        - 'knn': Local k-NN matched control centroids
    metric : {'cosine', 'euclidean'}, default='cosine'
        How to measure directional consistency (used for 'standard' and 'knn' methods).
    k : int, default=50
        Number of nearest neighbors (only used when method='knn').
    regularization : float, default=1e-6
        Regularization for covariance (only used when method='whitened').
    seed : int, optional
        Random seed for subsampling reproducibility.
    max_samples : int, optional
        Subsample perturbed population if exceeded.
    n_bootstrap_ci : int, optional
        If provided, compute bootstrap confidence interval by resampling
        control and perturbed populations this many times.
    ci : float, default=0.95
        Confidence level for the interval.
    
    Returns
    -------
    float or dict
        If n_bootstrap_ci is None: stability score in [-1, 1].
        Higher = more consistent perturbation effect.
        If n_bootstrap_ci is set: dict with keys 'mean', 'ci_low', 'ci_high',
        'std', 'n_bootstraps', 'ci_level'.
    
    Examples
    --------
    >>> # Control and perturbed cell populations
    >>> X_ctrl = np.random.randn(500, 50)  # 500 control cells, 50 genes
    >>> shift = np.random.randn(50)  # consistent direction
    >>> X_pert = X_ctrl + shift + np.random.randn(500, 50) * 0.1
    >>> 
    >>> # Standard stability
    >>> stability = perturbation_stability(X_ctrl, X_pert, method='standard')
    >>> 
    >>> # With bootstrap CI
    >>> result = perturbation_stability(X_ctrl, X_pert, n_bootstrap_ci=1000)
    >>> print(f"{result['mean']:.3f} [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
    
    Notes
    -----
    Method selection:
    - 'standard': Best for homogeneous controls, computationally fastest
    - 'whitened': Better when features have different scales or are correlated
    - 'knn': Best for heterogeneous controls with multiple cell types/states
    
    The control reference is computed differently for each method:
    - Standard: Global centroid of all control cells
    - Whitened: Mahalanobis-scaled space accounting for control covariance
    - k-NN: Local centroid of k nearest control cells for each perturbed cell
    """
    X_control = np.asarray(X_control, dtype=np.float64)
    X_perturbed = np.asarray(X_perturbed, dtype=np.float64)

    if n_bootstrap_ci is not None:
        return bootstrap_ci_bio(
            perturbation_stability, n_bootstrap_ci, ci, seed,
            X_control, X_perturbed,
            method=method, metric=metric, k=k,
            regularization=regularization, max_samples=max_samples,
        )
    
    if X_control.shape[1] != X_perturbed.shape[1]:
        raise ValueError(
            f"Feature dimensions must match: control has {X_control.shape[1]}, "
            f"perturbed has {X_perturbed.shape[1]}"
        )
    
    if len(X_control) < 5:
        return np.nan
    if len(X_perturbed) < 5:
        return np.nan
    
    # Dispatch to appropriate method
    if method == "standard":
        return _perturbation_stability_standard(
            X_control, X_perturbed, metric, seed, max_samples
        )
    elif method == "whitened":
        return _perturbation_stability_whitened(
            X_control, X_perturbed, regularization, seed, max_samples
        )
    elif method == "knn":
        return _perturbation_stability_knn(
            X_control, X_perturbed, k, metric, seed, max_samples
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'standard', 'whitened', or 'knn'."
        )


def _perturbation_stability_standard(
    X_control: np.ndarray,
    X_perturbed: np.ndarray,
    metric: str = "cosine",
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1000,
) -> float:
    """Internal implementation of standard perturbation stability."""
    rng = np.random.default_rng(seed)
    
    # Subsample perturbed if needed
    if max_samples is not None and len(X_perturbed) > max_samples:
        idx = rng.choice(len(X_perturbed), max_samples, replace=False)
        X_perturbed = X_perturbed[idx]
    
    # Compute control centroid
    control_centroid = np.mean(X_control, axis=0)
    
    # Compute shift vectors for each perturbed sample
    shift_vectors = X_perturbed - control_centroid
    
    # Compute mean shift direction
    mean_shift = np.mean(shift_vectors, axis=0)
    mean_shift_norm = np.linalg.norm(mean_shift)
    
    if mean_shift_norm < EPS:
        # No net shift - perturbation has no coherent effect
        return 0.0
    
    if metric == "cosine":
        # Normalize mean shift
        mean_shift_unit = mean_shift / mean_shift_norm
        
        # Compute cosine similarity of each shift to mean direction
        shift_norms = np.linalg.norm(shift_vectors, axis=1, keepdims=True)
        shift_norms = np.maximum(shift_norms, EPS)
        shift_unit = shift_vectors / shift_norms
        
        # Cosine similarities
        cosines = shift_unit @ mean_shift_unit
        
        return float(np.mean(cosines))
    
    elif metric == "euclidean":
        # Euclidean-based consistency: how tight are shifts around mean?
        # Normalized by expected variance under random shifts
        deviations = shift_vectors - mean_shift
        deviation_var = np.mean(np.sum(deviations ** 2, axis=1))
        total_var = np.mean(np.sum(shift_vectors ** 2, axis=1))
        
        if total_var < EPS:
            return np.nan
        
        # 1 - (deviation / total) gives consistency score
        consistency = 1.0 - (deviation_var / total_var)
        return float(np.clip(consistency, -1, 1))
    
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'.")

def _get_array(adata: "AnnData", mask, layer: Optional[str]) -> np.ndarray:
    """Extract a dense numpy array from an AnnData slice."""
    X = adata[mask].layers[layer] if layer else adata[mask].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X


def _iter_perturbations(adata, perturbation_key, control_label, layer):
    """Yield (perturbation_name, X_ctrl, X_pert) for each non-control perturbation."""
    ctrl_mask = adata.obs[perturbation_key] == control_label
    X_ctrl = _get_array(adata, ctrl_mask, layer)
    for pert in adata.obs[perturbation_key].unique():
        if pert == control_label:
            continue
        pert_mask = adata.obs[perturbation_key] == pert
        yield pert, X_ctrl, _get_array(adata, pert_mask, layer)


def compute_stability(
    adata: "AnnData",
    perturbation_key: str,
    control_label: str = "control",
    layer: Optional[str] = None,
    method: Literal["standard", "whitened", "knn"] = "standard",
    **kwargs
) -> dict:
    """
    Scanpy-compatible wrapper for perturbation stability.
    
    Computes stability for all perturbations in an AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    perturbation_key : str
        Column in adata.obs containing perturbation labels (e.g. 'guide_id').
    control_label : str
        The label in perturbation_key representing control cells (e.g. 'NT').
    layer : str, optional
        Layer to use for computation. If None, uses .X.
    method : {'standard', 'whitened', 'knn'}, default='standard'
        Method for computing stability:
        - 'standard': Global control centroid
        - 'whitened': Mahalanobis-scaled using control covariance
        - 'knn': Local k-NN matched control centroids
    **kwargs
        Additional arguments passed to perturbation_stability()
        (e.g., k=50 for knn, regularization=1e-6 for whitened).
    
    Returns
    -------
    dict
        Dictionary mapping perturbation names to stability scores.
    
    Examples
    --------
    >>> import shesha.bio as bio
    >>> # Standard stability
    >>> stability = bio.compute_stability(adata, "perturbation")
    >>> # Whitened stability
    >>> stability_w = bio.compute_stability(adata, "perturbation", method="whitened")
    >>> # k-NN stability
    >>> stability_knn = bio.compute_stability(adata, "perturbation", method="knn", k=50)
    """
    if AnnData is None or not isinstance(adata, AnnData):
        raise ImportError("anndata is required for this function.")
    return {
        pert: perturbation_stability(X_ctrl, X_pert, method=method, **kwargs)
        for pert, X_ctrl, X_pert in _iter_perturbations(
            adata, perturbation_key, control_label, layer
        )
    }


def perturbation_effect_size(
    X_control: np.ndarray,
    X_perturbed: np.ndarray,
    metric: Literal["euclidean", "cohen"] = "euclidean",
    n_bootstrap_ci: Optional[int] = None,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Union[float, dict]:
    """
    Compute the magnitude of the perturbation effect.
    
    Parameters
    ----------
    X_control : np.ndarray
        Control population embeddings.
    X_perturbed : np.ndarray
        Perturbed population embeddings.
    metric : str, default="euclidean"
        - 'euclidean': Raw L2 distance between centroids (Magnitude). 
           Use this for geometric plots (Stability vs Magnitude).
        - 'cohen': Standardized effect size (Magnitude / Pooled SD).
           Use this for statistical power analysis.
    n_bootstrap_ci : int, optional
        If provided, compute bootstrap confidence interval by resampling
        control and perturbed populations this many times.
    ci : float, default=0.95
        Confidence level for the interval.
    seed : int, optional
        Random seed for bootstrap reproducibility.
    
    Returns
    -------
    float or dict
        If n_bootstrap_ci is None: the calculated magnitude/effect size.
        If n_bootstrap_ci is set: dict with keys 'mean', 'ci_low', 'ci_high',
        'std', 'n_bootstraps', 'ci_level'.
    """
    X_control = np.asarray(X_control, dtype=np.float64)
    X_perturbed = np.asarray(X_perturbed, dtype=np.float64)

    if n_bootstrap_ci is not None:
        return bootstrap_ci_bio(
            perturbation_effect_size, n_bootstrap_ci, ci, seed,
            X_control, X_perturbed,
            metric=metric,
        )
    
    control_centroid = np.mean(X_control, axis=0)
    perturbed_centroid = np.mean(X_perturbed, axis=0)
    
    # 1. Raw Magnitude (Euclidean Distance)
    shift_magnitude = np.linalg.norm(perturbed_centroid - control_centroid)
    
    if metric == "euclidean":
        return float(shift_magnitude)
        
    elif metric == "cohen":
        # 2. Standardized Effect Size (Cohen's d-like)
        # Pooled standard deviation (averaged across features)
        control_var = np.var(X_control, axis=0, ddof=1)
        perturbed_var = np.var(X_perturbed, axis=0, ddof=1)
        
        # Average variance across features to get a scalar scale
        pooled_var = np.mean((control_var + perturbed_var) / 2)
        pooled_std = np.sqrt(pooled_var) + EPS
        
        return float(shift_magnitude / pooled_std)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_magnitude(
    adata: "AnnData",
    perturbation_key: str,
    control_label: str = "control",
    metric: str = "euclidean",
    layer: Optional[str] = None,
) -> dict:
    """
    Scanpy-compatible wrapper for perturbation magnitude.
    """
    if AnnData is None or not isinstance(adata, AnnData):
        raise ImportError("anndata is required for this function.")
    return {
        pert: perturbation_effect_size(X_ctrl, X_pert, metric=metric)
        for pert, X_ctrl, X_pert in _iter_perturbations(
            adata, perturbation_key, control_label, layer
        )
    }


def compute_stability_whitened(
    adata: "AnnData",
    perturbation_key: str,
    control_label: str = "control",
    layer: Optional[str] = None,
    regularization: float = 1e-6,
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1000,
) -> dict:
    """
    Scanpy-compatible wrapper for whitened perturbation stability.
    
    Convenience wrapper for compute_stability(..., method='whitened').
    Consider using the unified interface instead.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell data.
    perturbation_key : str
        Column in adata.obs containing perturbation labels.
    control_label : str, default="control"
        Label identifying control/unperturbed cells.
    layer : str, optional
        Layer in adata.layers to use. If None, uses adata.X.
    regularization : float, default=1e-6
        Regularization added to covariance diagonal for numerical stability.
    seed : int, optional
        Random seed for subsampling reproducibility.
    max_samples : int, optional
        Subsample perturbed population if exceeded.
    
    Returns
    -------
    dict
        Dictionary mapping perturbation names to whitened stability scores.
    
    See Also
    --------
    compute_stability : Unified interface with method='whitened'
    """
    return compute_stability(
        adata,
        perturbation_key,
        control_label=control_label,
        layer=layer,
        method='whitened',
        regularization=regularization,
        seed=seed,
        max_samples=max_samples
    )


def compute_stability_knn(
    adata: "AnnData",
    perturbation_key: str,
    control_label: str = "control",
    layer: Optional[str] = None,
    k: int = 50,
    metric: Literal["cosine", "euclidean"] = "euclidean",
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1000,
) -> dict:
    """
    Scanpy-compatible wrapper for k-NN matched control stability.
    
    Convenience wrapper for compute_stability(..., method='knn').
    Consider using the unified interface instead.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell data.
    perturbation_key : str
        Column in adata.obs containing perturbation labels.
    control_label : str, default="control"
        Label identifying control/unperturbed cells.
    layer : str, optional
        Layer in adata.layers to use. If None, uses adata.X.
    k : int, default=50
        Number of nearest control neighbors to use for local centroid.
    metric : str, default="euclidean"
        Distance metric for k-NN matching: 'cosine' or 'euclidean'.
    seed : int, optional
        Random seed for subsampling reproducibility.
    max_samples : int, optional
        Subsample perturbed population if exceeded.
    
    Returns
    -------
    dict
        Dictionary mapping perturbation names to k-NN matched stability scores.
    
    See Also
    --------
    compute_stability : Unified interface with method='knn'
    """
    return compute_stability(
        adata,
        perturbation_key,
        control_label=control_label,
        layer=layer,
        method='knn',
        k=k,
        metric=metric,
        seed=seed,
        max_samples=max_samples
    )


def _perturbation_stability_whitened(
    X_control: np.ndarray,
    X_perturbed: np.ndarray,
    regularization: float = 1e-6,
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1000,
) -> float:
    """Internal implementation of whitened perturbation stability."""
    # Subsample if needed
    if max_samples and len(X_perturbed) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_perturbed), max_samples, replace=False)
        X_perturbed = X_perturbed[idx]
    
    # Compute control centroid
    control_centroid = np.mean(X_control, axis=0)
    
    # Compute control covariance
    control_cov = np.cov(X_control.T)
    control_cov_reg = control_cov + regularization * np.eye(control_cov.shape[0])
    
    try:
        # Compute whitening matrix via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(control_cov_reg)
        eigvals = np.maximum(eigvals, regularization)
        W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        # Apply whitening
        control_centroid_w = W @ control_centroid
        pert_matrix_w = (W @ X_perturbed.T).T
        
        # Compute shift vectors in whitened space
        shift_vectors = pert_matrix_w - control_centroid_w
    except np.linalg.LinAlgError:
        # Fall back to unwhitened if whitening fails
        shift_vectors = X_perturbed - control_centroid
    
    # Compute mean shift direction
    mean_shift = np.mean(shift_vectors, axis=0)
    mean_magnitude = np.linalg.norm(mean_shift)
    
    if mean_magnitude < EPS:
        return 0.0
    
    # Normalize shift vectors
    norms = np.linalg.norm(shift_vectors, axis=1)
    valid_idx = norms > EPS
    
    if np.sum(valid_idx) < 5:
        return 0.0
    
    # Compute stability as mean cosine similarity to mean direction
    unit_mean = mean_shift / mean_magnitude
    cosine_sims = np.dot(shift_vectors[valid_idx], unit_mean) / norms[valid_idx]
    stability = np.mean(cosine_sims)
    
    return float(stability)


def perturbation_stability_whitened(
    X_control: np.ndarray,
    X_perturbed: np.ndarray,
    regularization: float = 1e-6,
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1000,
) -> float:
    """
    Whitened (Mahalanobis) perturbation stability.
    
    Convenience wrapper for perturbation_stability(..., method='whitened').
    Consider using the unified interface instead.
    
    Parameters
    ----------
    X_control : np.ndarray
        Control population embeddings, shape (n_control, n_features).
    X_perturbed : np.ndarray
        Perturbed population embeddings, shape (n_perturbed, n_features).
    regularization : float
        Regularization added to covariance diagonal for numerical stability.
    seed : int, optional
        Random seed for subsampling reproducibility.
    max_samples : int, optional
        Subsample perturbed population if exceeded.
    
    Returns
    -------
    float
        Whitened stability score in [-1, 1].
    
    See Also
    --------
    perturbation_stability : Unified interface with method='whitened'
    """
    return perturbation_stability(
        X_control, X_perturbed, 
        method='whitened',
        regularization=regularization,
        seed=seed,
        max_samples=max_samples
    )


def _perturbation_stability_knn(
    X_control: np.ndarray,
    X_perturbed: np.ndarray,
    k: int = 50,
    metric: str = "euclidean",
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1000,
) -> float:
    """Internal implementation of k-NN perturbation stability."""
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        raise ImportError(
            "perturbation_stability with method='knn' requires scikit-learn. "
            "Install with: pip install scikit-learn"
        )
    
    if len(X_control) < k:
        k = len(X_control)
    
    # Subsample if needed
    if max_samples and len(X_perturbed) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_perturbed), max_samples, replace=False)
        X_perturbed = X_perturbed[idx]
    
    # Fit k-NN on control population
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X_control)
    
    # Find k nearest controls for each perturbed cell
    _, indices = nn.kneighbors(X_perturbed)
    
    # Compute shift vectors relative to local control centroids
    shift_vectors = []
    for i, idx in enumerate(indices):
        local_ctrl_centroid = np.mean(X_control[idx], axis=0)
        shift_vectors.append(X_perturbed[i] - local_ctrl_centroid)
    
    shift_vectors = np.array(shift_vectors)
    
    # Compute mean shift direction
    mean_shift = np.mean(shift_vectors, axis=0)
    mean_magnitude = np.linalg.norm(mean_shift)
    
    if mean_magnitude < EPS:
        return 0.0
    
    # Normalize shift vectors
    norms = np.linalg.norm(shift_vectors, axis=1)
    valid_idx = norms > EPS
    
    if np.sum(valid_idx) < 5:
        return 0.0
    
    # Compute stability as mean cosine similarity to mean direction
    unit_mean = mean_shift / mean_magnitude
    cosine_sims = np.dot(shift_vectors[valid_idx], unit_mean) / norms[valid_idx]
    stability = np.mean(cosine_sims)
    
    return float(stability)


def perturbation_stability_knn(
    X_control: np.ndarray,
    X_perturbed: np.ndarray,
    k: int = 50,
    metric: Literal["cosine", "euclidean"] = "euclidean",
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1000,
) -> float:
    """
    k-NN matched control perturbation stability.
    
    Convenience wrapper for perturbation_stability(..., method='knn').
    Consider using the unified interface instead.
    
    Parameters
    ----------
    X_control : np.ndarray
        Control population embeddings, shape (n_control, n_features).
    X_perturbed : np.ndarray
        Perturbed population embeddings, shape (n_perturbed, n_features).
    k : int
        Number of nearest control neighbors to use for local centroid.
    metric : str
        Distance metric for k-NN matching: 'cosine' or 'euclidean'.
    seed : int, optional
        Random seed for subsampling reproducibility.
    max_samples : int, optional
        Subsample perturbed population if exceeded.
    
    Returns
    -------
    float
        k-NN matched stability score in [-1, 1].
    
    See Also
    --------
    perturbation_stability : Unified interface with method='knn'
    """
    return perturbation_stability(
        X_control, X_perturbed,
        method='knn',
        k=k,
        metric=metric,
        seed=seed,
        max_samples=max_samples
    )


# =============================================================================
# Split-Half Reproducibility
# =============================================================================

def _split_half_cosine(
    X_pert: np.ndarray,
    ctrl_centroid: np.ndarray,
    n_splits: int = 50,
    seed: int = 0,
    min_cells: int = 30,
) -> float:
    """
    Split perturbation cells 50/50 repeatedly and return mean cosine similarity
    between the two half-shift vectors (relative to the control centroid).

    Parameters
    ----------
    X_pert : np.ndarray
        Perturbed cell embeddings, shape (n_cells, n_features).
    ctrl_centroid : np.ndarray
        Control centroid, shape (n_features,).
    n_splits : int
        Number of random splits.
    seed : int
        Random seed for reproducibility.
    min_cells : int
        Minimum cells required (need >= min_cells/2 per half).

    Returns
    -------
    float
        Mean cosine similarity across splits, or NaN if too few cells.
    """
    n_cells = X_pert.shape[0]
    if n_cells < min_cells:
        return np.nan

    rng = np.random.default_rng(seed=seed)
    cosines = np.empty(n_splits)

    for i in range(n_splits):
        perm = rng.permutation(n_cells)
        half = n_cells // 2
        idx_a, idx_b = perm[:half], perm[half:2 * half]

        shift_a = (X_pert[idx_a] - ctrl_centroid).mean(axis=0)
        shift_b = (X_pert[idx_b] - ctrl_centroid).mean(axis=0)

        norm_a = np.linalg.norm(shift_a)
        norm_b = np.linalg.norm(shift_b)

        if norm_a < EPS or norm_b < EPS:
            cosines[i] = 0.0
        else:
            cosines[i] = np.dot(shift_a, shift_b) / (norm_a * norm_b)

    return float(np.mean(cosines))


def split_half_reproducibility(
    adata: "AnnData",
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    n_splits: int = 50,
    random_state: int = 320,
    min_cells: int = 30,
    layer: Optional[str] = None,
) -> "pd.DataFrame":
    """
    Split-half reproducibility for each perturbation in an AnnData object.

    For each perturbation with enough cells, randomly splits cells 50/50,
    computes independent shift vectors relative to the control centroid,
    and measures cosine similarity between the halves. This is a direct
    measure of effect-direction reproducibility: perturbations whose
    individual cells shift coherently will have high split-half cosine.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cells x features). Operates on adata.X or
        the specified layer.
    perturbation_key : str, default="perturbation"
        Column in adata.obs containing perturbation labels.
    control_label : str, default="control"
        Label identifying control/unperturbed cells.
    n_splits : int, default=50
        Number of random 50/50 splits per perturbation.
    random_state : int, default=320
        Base random seed. Each perturbation gets a unique derived seed.
    min_cells : int, default=30
        Minimum cells required for a perturbation to be included.
    layer : str, optional
        Layer in adata.layers to use. If None, uses adata.X.

    Returns
    -------
    pd.DataFrame
        Columns: perturbation, split_half_cosine, n_cells.
        Indexed by perturbation name.

    Examples
    --------
    >>> from shesha.bio import split_half_reproducibility
    >>> repro = split_half_reproducibility(
    ...     adata,
    ...     perturbation_key="perturbation",
    ...     control_label="control",
    ...     n_splits=50,
    ...     random_state=320,
    ... )
    """
    if AnnData is None or not isinstance(adata, AnnData):
        raise ImportError("anndata is required for this function.")

    labels = adata.obs[perturbation_key].astype(str).values
    ctrl_mask = labels == control_label
    X_ctrl = _get_array(adata, ctrl_mask, layer)
    ctrl_centroid = X_ctrl.mean(axis=0)

    perturbations = [p for p in np.unique(labels) if p != control_label]

    rows = []
    for pert in perturbations:
        pert_mask = labels == pert
        n_cells = int(pert_mask.sum())
        if n_cells < min_cells:
            continue

        X_pert = _get_array(adata, pert_mask, layer)
        pert_seed = random_state + hash(pert) % 100_000

        cosine = _split_half_cosine(
            X_pert, ctrl_centroid,
            n_splits=n_splits,
            seed=pert_seed,
            min_cells=min_cells,
        )
        rows.append({
            "perturbation": pert,
            "split_half_cosine": cosine,
            "n_cells": n_cells,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.set_index("perturbation")
    return df


def magnitude_matched_comparison(
    repro_df: "pd.DataFrame",
    stability_col: str = "Sp",
    repro_col: str = "split_half_cosine",
    magnitude_col: str = "Mp",
    n_bins: int = 4,
) -> "pd.DataFrame":
    """
    Magnitude-matched comparison of high-stability vs low-stability groups.

    Bins perturbations by magnitude, then within each bin splits at the
    stability median to compare reproducibility between the high- and
    low-stability halves. This controls for the confound that larger-effect
    perturbations may appear more reproducible simply due to higher SNR.

    Parameters
    ----------
    repro_df : pd.DataFrame
        DataFrame containing at least the columns specified by stability_col,
        repro_col, and magnitude_col.
    stability_col : str, default="Sp"
        Column with stability scores.
    repro_col : str, default="split_half_cosine"
        Column with reproducibility scores (e.g. split-half cosine).
    magnitude_col : str, default="Mp"
        Column with magnitude/effect-size scores for binning.
    n_bins : int, default=4
        Number of magnitude bins (quartiles by default).

    Returns
    -------
    pd.DataFrame
        One row per magnitude bin with columns:
        mag_bin, n, mag_min, mag_max, high_stability_mean, low_stability_mean,
        difference, within_bin_rho, within_bin_pvalue.

    Examples
    --------
    >>> from shesha.bio import magnitude_matched_comparison
    >>> bins = magnitude_matched_comparison(
    ...     repro_df,
    ...     stability_col="Sp",
    ...     repro_col="split_half_cosine",
    ...     magnitude_col="Mp",
    ...     n_bins=4,
    ... )
    """
    from scipy.stats import spearmanr

    df = repro_df.dropna(subset=[stability_col, repro_col, magnitude_col]).copy()

    if len(df) < n_bins * 4:
        raise ValueError(
            f"Too few perturbations ({len(df)}) for {n_bins} bins. "
            f"Need at least {n_bins * 4}."
        )

    bin_labels = [f"Q{i+1}" for i in range(n_bins)]
    df["_mag_bin"] = pd.qcut(
        df[magnitude_col], q=n_bins, labels=bin_labels, duplicates="drop"
    )

    results = []
    for q in bin_labels:
        subset = df[df["_mag_bin"] == q]
        if len(subset) < 6:
            continue

        sp_median = subset[stability_col].median()
        high = subset[subset[stability_col] >= sp_median]
        low = subset[subset[stability_col] < sp_median]

        mean_high = high[repro_col].mean()
        mean_low = low[repro_col].mean()

        rho, pval = spearmanr(subset[stability_col], subset[repro_col])

        results.append({
            "mag_bin": q,
            "n": len(subset),
            "mag_min": float(subset[magnitude_col].min()),
            "mag_max": float(subset[magnitude_col].max()),
            "high_stability_mean": float(mean_high),
            "low_stability_mean": float(mean_low),
            "difference": float(mean_high - mean_low),
            "within_bin_rho": float(rho),
            "within_bin_pvalue": float(pval),
        })

    return pd.DataFrame(results)


# =============================================================================
# Discordance
# =============================================================================

def discordance(
    df: "pd.DataFrame",
    stability_col: str = "Sp",
    magnitude_col: str = "Mp",
    method: Literal["linear", "rank", "loess"] = "linear",
    loess_frac: float = 0.3,
) -> "pd.Series":
    """
    Compute discordance scores: how much a perturbation deviates from the
    expected stability-magnitude relationship.

    High discordance (positive values) identifies perturbations that are
    less stable than expected given their effect size — candidates for
    pleiotropic or heterogeneous effects.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the columns specified by stability_col
        and magnitude_col.
    stability_col : str, default="Sp"
        Column with stability scores.
    magnitude_col : str, default="Mp"
        Column with magnitude/effect-size scores.
    method : {'linear', 'rank', 'loess'}, default='linear'
        How to model the expected stability-magnitude relationship:
        - 'linear': OLS residual, sign-flipped, z-scored.
        - 'rank': rank(Mp) - rank(Sp), z-scored.
        - 'loess': LOESS residual (local regression), sign-flipped, z-scored.
          Captures nonlinear magnitude-stability trends.
    loess_frac : float, default=0.3
        Fraction of data used for each local regression window (only used
        when method='loess'). Smaller values follow the data more closely;
        larger values produce smoother fits.

    Returns
    -------
    pd.Series
        Z-scored discordance scores indexed like the input DataFrame.
        Positive = less stable than expected (discordant).
        Negative = more stable than expected (concordant).

    Examples
    --------
    >>> from shesha.bio import discordance
    >>> df["disc_linear"] = discordance(df, stability_col="Sp", magnitude_col="Mp")
    >>> df["disc_loess"] = discordance(df, method="loess", loess_frac=0.3)
    >>> # Top discordant perturbations
    >>> df.nlargest(10, "disc_loess")
    """
    from scipy.stats import rankdata

    sub = df[[stability_col, magnitude_col]].dropna()
    mag = sub[magnitude_col].values.astype(np.float64)
    stab = sub[stability_col].values.astype(np.float64)

    if len(sub) < 10:
        raise ValueError(
            f"Too few valid observations ({len(sub)}). Need at least 10."
        )

    if method == "linear":
        X = np.column_stack([np.ones_like(mag), mag])
        beta = np.linalg.lstsq(X, stab, rcond=None)[0]
        fitted = X @ beta
        resid = stab - fitted
        d = -resid

    elif method == "rank":
        rank_m = rankdata(mag)
        rank_s = rankdata(stab)
        d = rank_m - rank_s

    elif method == "loess":
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
        except ImportError:
            raise ImportError(
                "method='loess' requires statsmodels. "
                "Install with: pip install statsmodels"
            )
        fitted = lowess(stab, mag, frac=loess_frac, return_sorted=False)
        resid = stab - fitted
        d = -resid

    else:
        raise ValueError(
            f"Unknown method: {method!r}. Use 'linear', 'rank', or 'loess'."
        )

    std = d.std()
    if std < EPS:
        z = np.zeros_like(d)
    else:
        z = (d - d.mean()) / std

    return pd.Series(z, index=sub.index, name="discordance")