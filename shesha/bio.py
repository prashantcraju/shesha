"""
Shesha Bio: Stability metrics for biological perturbation experiments.

This module provides Shesha variants for single-cell and perturbation biology,
measuring the consistency of perturbation effects across individual cells.
"""

import numpy as np
from typing import Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

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
    "compute_magnitude"
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
) -> float:
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
    
    Returns
    -------
    float
        Stability score in [-1, 1]. Higher = more consistent perturbation effect.
        Values near 1 indicate all samples shift in the same direction;
        values near 0 indicate random/inconsistent shifts.
    
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
    >>> # Whitened (accounts for feature correlations)
    >>> stability_w = perturbation_stability(X_ctrl, X_pert, method='whitened')
    >>> 
    >>> # k-NN (for heterogeneous control populations)
    >>> stability_knn = perturbation_stability(X_ctrl, X_pert, method='knn', k=50)
    
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
    
    # Get control data
    ctrl_mask = adata.obs[perturbation_key] == control_label
    if layer:
        X_ctrl = adata[ctrl_mask].layers[layer]
    else:
        X_ctrl = adata[ctrl_mask].X
        
    # Handle sparse matrices
    if hasattr(X_ctrl, "toarray"):
        X_ctrl = X_ctrl.toarray()
        
    results = {}
    perturbations = adata.obs[perturbation_key].unique()
    
    for pert in perturbations:
        if pert == control_label:
            continue
            
        pert_mask = adata.obs[perturbation_key] == pert
        if layer:
            X_pert = adata[pert_mask].layers[layer]
        else:
            X_pert = adata[pert_mask].X
            
        if hasattr(X_pert, "toarray"):
            X_pert = X_pert.toarray()
            
        score = perturbation_stability(X_ctrl, X_pert, method=method, **kwargs)
        results[pert] = score
        
    return results


def perturbation_effect_size(
    X_control: np.ndarray,
    X_perturbed: np.ndarray,
    metric: Literal["euclidean", "cohen"] = "euclidean"
) -> float:
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
    
    Returns
    -------
    float
        The calculated magnitude/effect size.
    """
    X_control = np.asarray(X_control, dtype=np.float64)
    X_perturbed = np.asarray(X_perturbed, dtype=np.float64)
    
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
    
    # Get control data
    ctrl_mask = adata.obs[perturbation_key] == control_label
    if layer:
        X_ctrl = adata[ctrl_mask].layers[layer]
    else:
        X_ctrl = adata[ctrl_mask].X
        
    if hasattr(X_ctrl, "toarray"):
        X_ctrl = X_ctrl.toarray()
        
    results = {}
    perturbations = adata.obs[perturbation_key].unique()
    
    for pert in perturbations:
        if pert == control_label:
            continue
            
        pert_mask = adata.obs[perturbation_key] == pert
        if layer:
            X_pert = adata[pert_mask].layers[layer]
        else:
            X_pert = adata[pert_mask].X
            
        if hasattr(X_pert, "toarray"):
            X_pert = X_pert.toarray()
            
        score = perturbation_effect_size(X_ctrl, X_pert, metric=metric)
        results[pert] = score
        
    return results


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