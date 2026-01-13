"""
Shesha Bio: Stability metrics for biological perturbation experiments.

This module provides Shesha variants for single-cell and perturbation biology,
measuring the consistency of perturbation effects across individual cells.
"""

import numpy as np
from typing import Optional, Literal

__all__ = [
    "perturbation_stability", 
    "perturbation_effect_size", 
    "compute_stability", 
    "compute_magnitude"
]

EPS = 1e-12


def perturbation_stability(
    X_control: np.ndarray,
    X_perturbed: np.ndarray,
    metric: Literal["cosine", "euclidean"] = "cosine",
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1000,
) -> float:
    """
    Perturbation stability: consistency of perturbation effects across samples.
    
    Measures whether individual perturbed samples shift in a consistent direction
    relative to the control population. High values indicate that the perturbation
    has a coherent, reproducible effect; low values suggest heterogeneous or noisy
    responses.
    
    The metric computes the mean cosine similarity between each perturbed sample's
    shift vector (relative to the control centroid) and the mean shift direction.
    
    Parameters
    ----------
    X_control : np.ndarray
        Control population embeddings, shape (n_control, n_features).
    X_perturbed : np.ndarray
        Perturbed population embeddings, shape (n_perturbed, n_features).
    metric : str
        How to measure directional consistency:
        - 'cosine': Cosine similarity of shift vectors to mean direction (default)
        - 'euclidean': Normalized euclidean consistency
    seed : int, optional
        Random seed for subsampling reproducibility.
    max_samples : int, optional
        Subsample perturbed population if exceeded.
    
    Returns
    -------
    float
        Stability score in [-1, 1] for cosine metric. Higher = more consistent
        perturbation effect. Values near 1 indicate all samples shift in the
        same direction; values near 0 indicate random/inconsistent shifts.
    
    Examples
    --------
    >>> # Control and perturbed cell populations
    >>> X_ctrl = np.random.randn(500, 50)  # 500 control cells, 50 genes
    >>> 
    >>> # Coherent perturbation: all cells shift similarly
    >>> shift = np.random.randn(50)  # consistent direction
    >>> X_pert_coherent = X_ctrl + shift + np.random.randn(500, 50) * 0.1
    >>> stability = perturbation_stability(X_ctrl, X_pert_coherent)
    >>> print(f"Coherent perturbation: {stability:.3f}")  # High value
    >>> 
    >>> # Incoherent perturbation: cells shift randomly
    >>> X_pert_random = X_ctrl + np.random.randn(500, 50)
    >>> stability = perturbation_stability(X_ctrl, X_pert_random)
    >>> print(f"Random perturbation: {stability:.3f}")  # Low value
    
    Notes
    -----
    This metric is designed for single-cell perturbation experiments (e.g., 
    Perturb-seq, CRISPR screens) where you want to assess whether a genetic
    perturbation produces a consistent phenotypic shift across cells.
    
    The control centroid is used as the reference point. Each perturbed cell's
    shift vector is computed as (x_perturbed - centroid_control), and these
    are compared to the mean shift direction.
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

try:
    from anndata import AnnData
except ImportError:
    AnnData = None

def compute_stability(
    adata: "AnnData",
    perturbation_key: str,
    control_label: str = "control",
    layer: Optional[str] = None,
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
    
    Returns
    -------
    dict
        Dictionary mapping perturbation names to stability scores.
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
            
        score = perturbation_stability(X_ctrl, X_pert, **kwargs)
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