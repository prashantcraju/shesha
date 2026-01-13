"""
Shesha: Self-consistency Metrics for Representational Stability

Core implementations of Shesha variants for measuring geometric stability
of high-dimensional representations.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, cdist
from typing import Optional, Literal, Union

__all__ = [
    # Unsupervised variants
    "feature_split",
    "sample_split", 
    "anchor_stability",
    # Supervised variants
    "variance_ratio",
    "supervised_alignment",
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
    
    Returns
    -------
    np.ndarray
        Condensed distance vector (upper triangle of RDM).
    """
    X = np.asarray(X, dtype=np.float64)
    
    if normalize and metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, EPS)
    
    return pdist(X, metric=metric)


# =============================================================================
# Unsupervised Variants
# =============================================================================

def feature_split(
    X: np.ndarray,
    n_splits: int = 30,
    metric: Literal["cosine", "correlation"] = "cosine",
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1600,
) -> float:
    """
    Feature-Split Shesha: measures internal geometric consistency.
    
    Partitions feature dimensions into random disjoint halves, computes RDMs
    on each half, and measures their rank correlation. High values indicate
    that geometric structure is distributed across features (redundant encoding).
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    n_splits : int
        Number of random feature partitions to average over.
    metric : str
        Distance metric for RDM computation.
    seed : int, optional
        Random seed for reproducibility.
    max_samples : int, optional
        Subsample to this many samples if exceeded (for efficiency).
    
    Returns
    -------
    float
        Mean Spearman correlation between split-half RDMs. Range: [-1, 1].
    
    Examples
    --------
    >>> X = np.random.randn(500, 768)  # 500 samples, 768-dim embeddings
    >>> stability = feature_split(X, n_splits=30, seed=320)
    >>> print(f"Feature-split stability: {stability:.3f}")
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    
    if n_features < 4:
        return np.nan
    if n_samples < 4:
        return np.nan
    
    rng = np.random.default_rng(seed)
    
    # Subsample if needed
    if max_samples is not None and n_samples > max_samples:
        idx = rng.choice(n_samples, max_samples, replace=False)
        X = X[idx]
        n_samples = max_samples
    
    # L2 normalize for cosine metric
    if metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, EPS)
    
    correlations = []
    
    for i in range(n_splits):
        # Random partition of features
        perm = rng.permutation(n_features)
        mid = n_features // 2
        feat1, feat2 = perm[:mid], perm[mid:2*mid]
        
        X1, X2 = X[:, feat1], X[:, feat2]
        
        # Compute RDMs
        rdm1 = pdist(X1, metric=metric)
        rdm2 = pdist(X2, metric=metric)
        
        # Handle NaN distances (can occur with zero vectors)
        rdm1 = np.nan_to_num(rdm1, nan=1.0)
        rdm2 = np.nan_to_num(rdm2, nan=1.0)
        
        # Check for constant RDMs
        if np.std(rdm1) < EPS or np.std(rdm2) < EPS:
            continue
        
        rho, _ = spearmanr(rdm1, rdm2)
        if np.isfinite(rho):
            correlations.append(rho)
    
    return float(np.mean(correlations)) if correlations else np.nan


def sample_split(
    X: np.ndarray,
    n_splits: int = 30,
    subsample_fraction: float = 0.4,
    metric: Literal["cosine", "correlation"] = "cosine",
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1500,
) -> float:
    """
    Sample-Split Shesha (Bootstrap RDM): measures robustness to input variation.
    
    Creates random subsamples of data points, computes RDMs on each, and 
    measures their correlation. Assesses whether distance structure generalizes
    across different subsets of the data.
    
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
    
    Returns
    -------
    float
        Mean Spearman correlation between bootstrap RDMs. Range: [-1, 1].
    
    Examples
    --------
    >>> X = np.random.randn(1000, 384)
    >>> stability = sample_split(X, n_splits=50, seed=320)
    """
    X = np.asarray(X, dtype=np.float64)
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
        
        if np.std(rdm1) < EPS or np.std(rdm2) < EPS:
            continue
        
        rho, _ = spearmanr(rdm1, rdm2)
        if np.isfinite(rho):
            correlations.append(rho)
    
    return float(np.mean(correlations)) if correlations else np.nan


def anchor_stability(
    X: np.ndarray,
    n_splits: int = 30,
    n_anchors: int = 100,
    n_per_split: int = 200,
    metric: Literal["cosine", "euclidean"] = "cosine",
    rank_normalize: bool = True,
    seed: Optional[int] = None,
    max_samples: Optional[int] = 1500,
) -> float:
    """
    Anchor-based Shesha: measures stability of distance profiles from fixed anchors.
    
    Selects fixed anchor points, then measures consistency of distance profiles
    from anchors to random data splits. More robust to sampling variation than
    pure bootstrap approaches.
    
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
    
    Returns
    -------
    float
        Mean correlation of anchor distance profiles across splits.
    """
    X = np.asarray(X, dtype=np.float64)
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
        split2_idx = perm[n_per_split:2*n_per_split]
        
        # Distance matrices: anchors x split_samples
        D1 = cdist(anchors, X[split1_idx], metric=metric)
        D2 = cdist(anchors, X[split2_idx], metric=metric)
        
        if rank_normalize:
            # Rank within each anchor's distances
            from scipy.stats import rankdata
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
) -> float:
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
    
    Returns
    -------
    float
        Between-class variance / total variance. Range: [0, 1].
    
    Examples
    --------
    >>> X = np.random.randn(500, 768)
    >>> y = np.random.randint(0, 10, 500)
    >>> vr = variance_ratio(X, y)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    
    classes = np.unique(y)
    if len(classes) < 2:
        return np.nan
    
    global_mean = np.mean(X, axis=0)
    X_centered = X - global_mean
    ss_total = np.sum(X_centered ** 2) + EPS
    
    ss_between = 0.0
    for c in classes:
        mask = (y == c)
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
) -> float:
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
    
    Returns
    -------
    float
        Spearman correlation between model and ideal RDMs. Range: [-1, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    
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


# =============================================================================
# Drift Metrics
# =============================================================================

def rdm_similarity(
    X: np.ndarray,
    Y: np.ndarray,
    method: Literal["spearman", "pearson"] = "spearman",
    metric: Literal["cosine", "correlation", "euclidean"] = "cosine",
) -> float:
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
    
    Returns
    -------
    float
        Correlation between RDMs. Range: [-1, 1].
        Higher values indicate more similar geometric structure.
    
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
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Sample counts must match: X has {X.shape[0]}, Y has {Y.shape[0]}")
    
    if X.shape[0] < 3:
        return np.nan
    
    # Compute RDMs
    rdm_x = pdist(X, metric=metric)
    rdm_y = pdist(Y, metric=metric)
    
    # Handle NaN values
    rdm_x = np.nan_to_num(rdm_x, nan=1.0)
    rdm_y = np.nan_to_num(rdm_y, nan=1.0)
    
    # Check for constant RDMs
    if np.std(rdm_x) < EPS or np.std(rdm_y) < EPS:
        return 0.0
    
    # Compute correlation
    if method == "spearman":
        rho = spearmanr(rdm_x, rdm_y).correlation
    elif method == "pearson":
        rho, _ = pearsonr(rdm_x, rdm_y)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'pearson'")
    
    return float(rho) if np.isfinite(rho) else 0.0


def rdm_drift(
    X: np.ndarray,
    Y: np.ndarray,
    method: Literal["spearman", "pearson"] = "spearman",
    metric: Literal["cosine", "correlation", "euclidean"] = "cosine",
) -> float:
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
    
    Returns
    -------
    float
        Drift score: 1 - RDM_correlation. Range: [0, 2].
        - 0: Identical geometric structure
        - 1: Uncorrelated (random relationship)
        - 2: Perfectly anti-correlated (inverted structure)
    
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
    similarity = rdm_similarity(X, Y, method=method, metric=metric)
    
    if np.isnan(similarity):
        return np.nan
    
    return 1.0 - similarity


# =============================================================================
# Convenience function
# =============================================================================

def shesha(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    variant: Literal["feature_split", "sample_split", "anchor", "variance", "supervised"] = "feature_split",
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
        - 'sample_split': Unsupervised, bootstrap resampling
        - 'anchor': Unsupervised, anchor-based stability
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