"""
Shesha Similarity: Representational Similarity Metrics

This module provides metrics for measuring similarity between representations,
complementing the stability metrics in shesha.core. While stability measures
intrinsic geometric robustness, similarity measures extrinsic alignment.

Key distinction from the paper:
- Similarity is an *extrinsic* property (how one representation aligns with another)
- Stability is an *intrinsic* property (how robust a representation's geometry is)
- These are empirically uncorrelated (ρ ≈ 0.01)
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr
from scipy.linalg import orthogonal_procrustes
from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


__all__ = [
    "cka",
    "cka_linear",
    "cka_debiased",
    "procrustes_similarity",
    "rdm_similarity",
]

EPS = 1e-12


# =============================================================================
# CKA (Centered Kernel Alignment)
# =============================================================================

def cka_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment (CKA) - Standard version.
    
    Measures similarity between two representations using linear kernels.
    This is the standard (non-debiased) version which is simpler and more
    numerically stable, recommended for most use cases.
    
    Parameters
    ----------
    X : np.ndarray
        First representation matrix of shape (n_samples, n_features_x).
    Y : np.ndarray
        Second representation matrix of shape (n_samples, n_features_y).
        Must have same number of samples as X.
    
    Returns
    -------
    float
        CKA similarity score in [0, 1]. Higher values indicate more similar
        representational structure. 1.0 means identical structure (up to
        linear transformation).
    
    Examples
    --------
    >>> import numpy as np
    >>> from shesha.similarity import cka_linear
    >>> 
    >>> # Two representations of the same data
    >>> X = np.random.randn(100, 50)
    >>> Y = np.random.randn(100, 30)
    >>> 
    >>> similarity = cka_linear(X, Y)
    >>> print(f"CKA: {similarity:.3f}")
    >>> 
    >>> # Self-similarity should be 1.0
    >>> self_sim = cka_linear(X, X)
    >>> print(f"Self-similarity: {self_sim:.3f}")  # Should be ~1.0
    
    Notes
    -----
    CKA is invariant to:
    - Orthogonal transformations
    - Isotropic scaling
    
    CKA measures the similarity of Gram matrices (X @ X.T and Y @ Y.T),
    which capture the pairwise similarities between samples in each
    representation space.
    
    References
    ----------
    Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).
    Similarity of neural network representations revisited.
    ICML 2019.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have same number of samples: "
            f"X has {X.shape[0]}, Y has {Y.shape[0]}"
        )
    
    # Center the data (subtract column means)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    
    # Compute HSIC using Frobenius norm of cross-Gram matrix
    # HSIC(X, Y) = ||X^T Y||_F^2
    num = np.linalg.norm(X.T @ Y, 'fro') ** 2
    
    # Normalize by self-similarities
    # CKA = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    den = np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')
    
    return float(num / (den + EPS))


def cka_debiased(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Debiased Centered Kernel Alignment (CKA).
    
    Unbiased estimator of CKA that corrects for finite sample effects.
    More accurate for small sample sizes but computationally more expensive.
    
    Parameters
    ----------
    X : np.ndarray
        First representation matrix of shape (n_samples, n_features_x).
    Y : np.ndarray
        Second representation matrix of shape (n_samples, n_features_y).
        Must have same number of samples as X.
    
    Returns
    -------
    float
        Debiased CKA similarity score in [0, 1]. Higher values indicate more
        similar representational structure.
    
    Examples
    --------
    >>> import numpy as np
    >>> from shesha.similarity import cka_debiased
    >>> 
    >>> # For small sample sizes, debiased version is more accurate
    >>> X = np.random.randn(50, 20)
    >>> Y = np.random.randn(50, 15)
    >>> 
    >>> # Compare standard vs debiased
    >>> from shesha.similarity import cka_linear
    >>> std_cka = cka_linear(X, Y)
    >>> debiased_cka = cka_debiased(X, Y)
    >>> 
    >>> print(f"Standard: {std_cka:.3f}")
    >>> print(f"Debiased: {debiased_cka:.3f}")
    
    Notes
    -----
    For n < 4, falls back to standard CKA as debiasing is not well-defined.
    
    The debiased estimator uses the unbiased HSIC estimator from Kornblith
    et al. (2019), which removes diagonal terms and applies correction factors.
    
    Recommended when:
    - Sample size is small (n < 100)
    - Exact statistical properties are important
    - Computing statistical significance
    
    References
    ----------
    Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).
    Similarity of neural network representations revisited.
    ICML 2019.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have same number of samples: "
            f"X has {X.shape[0]}, Y has {Y.shape[0]}"
        )
    
    # Center the data
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    
    n = X.shape[0]
    
    # For very small samples, fall back to standard CKA
    if n < 4:
        num = np.linalg.norm(X.T @ Y, 'fro') ** 2
        den = np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')
        return float(num / (den + EPS))
    
    # Helper function to center Gram matrix
    def center_gram_matrix(G):
        """Center a Gram matrix: H @ G @ H where H is centering matrix."""
        row_means = G.mean(axis=1, keepdims=True)
        col_means = G.mean(axis=0, keepdims=True)
        grand_mean = G.mean()
        return G - row_means - col_means + grand_mean
    
    # Compute and center Gram matrices
    K = center_gram_matrix(X @ X.T)
    L = center_gram_matrix(Y @ Y.T)
    
    # Zero out diagonals for debiasing terms
    K_no_diag = K.copy()
    L_no_diag = L.copy()
    np.fill_diagonal(K_no_diag, 0)
    np.fill_diagonal(L_no_diag, 0)
    
    # Debiased HSIC estimator (Kornblith et al., 2019)
    # Removes bias from diagonal terms
    hsic = (
        np.sum(K * L) 
        + (np.sum(K_no_diag) * np.sum(L_no_diag)) / ((n - 1) * (n - 2))
        - 2 * np.sum(np.sum(K_no_diag, axis=1) * np.sum(L_no_diag, axis=1)) / (n - 2)
    ) / (n * (n - 3))
    
    # Self-HSIC for normalization (also debiased)
    hsic_xx = (
        np.sum(K * K) 
        + np.sum(K_no_diag)**2 / ((n - 1) * (n - 2))
        - 2 * np.sum(np.sum(K_no_diag, axis=1)**2) / (n - 2)
    ) / (n * (n - 3))
    
    hsic_yy = (
        np.sum(L * L) 
        + np.sum(L_no_diag)**2 / ((n - 1) * (n - 2))
        - 2 * np.sum(np.sum(L_no_diag, axis=1)**2) / (n - 2)
    ) / (n * (n - 3))
    
    # Avoid division by zero or negative values (can happen due to numerical issues)
    if hsic_xx <= 0 or hsic_yy <= 0:
        return 0.0
    
    return float(hsic / np.sqrt(hsic_xx * hsic_yy))


def cka(
    X: np.ndarray, 
    Y: np.ndarray,
    debiased: bool = False
) -> float:
    """
    Centered Kernel Alignment (CKA) - Unified interface.
    
    Convenience function that selects between standard and debiased CKA.
    
    Parameters
    ----------
    X : np.ndarray
        First representation matrix of shape (n_samples, n_features_x).
    Y : np.ndarray
        Second representation matrix of shape (n_samples, n_features_y).
    debiased : bool, default=False
        If True, use debiased estimator. Recommended for small sample sizes.
    
    Returns
    -------
    float
        CKA similarity score in [0, 1].
    
    Examples
    --------
    >>> from shesha.similarity import cka
    >>> 
    >>> X = np.random.randn(100, 50)
    >>> Y = np.random.randn(100, 30)
    >>> 
    >>> # Standard CKA (default, faster)
    >>> sim = cka(X, Y)
    >>> 
    >>> # Debiased CKA (more accurate for small n)
    >>> sim_debiased = cka(X, Y, debiased=True)
    
    See Also
    --------
    cka_linear : Standard CKA implementation
    cka_debiased : Debiased CKA implementation
    """
    if debiased:
        return cka_debiased(X, Y)
    else:
        return cka_linear(X, Y)


# =============================================================================
# Procrustes Similarity
# =============================================================================

def procrustes_similarity(
    X: np.ndarray,
    Y: np.ndarray,
    center: bool = True,
    scale: bool = True,
) -> float:
    """
    Procrustes similarity between two representations.
    
    Finds the optimal orthogonal transformation that aligns Y to X and
    returns the similarity (1 - disparity). Unlike CKA, Procrustes attempts
    to directly align the representations in their original spaces.
    
    Parameters
    ----------
    X : np.ndarray
        First representation matrix of shape (n_samples, n_features).
    Y : np.ndarray
        Second representation matrix of shape (n_samples, n_features).
        Must have same shape as X.
    center : bool, default=True
        If True, center both matrices before alignment.
    scale : bool, default=True
        If True, scale to unit Frobenius norm before alignment.
    
    Returns
    -------
    float
        Procrustes similarity in [0, 1]. Higher values indicate better alignment.
        1.0 means perfect alignment (identical up to rotation/reflection).
    
    Examples
    --------
    >>> import numpy as np
    >>> from shesha.similarity import procrustes_similarity
    >>> 
    >>> # Two representations that differ by a rotation
    >>> X = np.random.randn(100, 50)
    >>> Q = np.linalg.qr(np.random.randn(50, 50))[0]  # Random rotation
    >>> Y = X @ Q
    >>> 
    >>> similarity = procrustes_similarity(X, Y)
    >>> print(f"Procrustes: {similarity:.3f}")  # Should be ~1.0
    
    Notes
    -----
    Procrustes is more sensitive to outliers and noise than CKA, which can
    be both an advantage (detects small changes) and disadvantage (more false
    alarms). The paper shows CKA is often preferred for representation analysis.
    
    If dimensions don't match, returns NaN. Unlike CKA, Procrustes requires
    representations to live in the same dimensional space.
    
    References
    ----------
    Schönemann, P. H. (1966). A generalized solution of the orthogonal
    Procrustes problem. Psychometrika, 31(1), 1-10.
    """
    try:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        
        if X.shape != Y.shape:
            raise ValueError(
                f"X and Y must have same shape for Procrustes: "
                f"X is {X.shape}, Y is {Y.shape}"
            )
        
        # Check for NaN/Inf values
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            return np.nan
        if np.any(np.isinf(X)) or np.any(np.isinf(Y)):
            return np.nan
        
        # Check for degenerate cases (all-zero or constant columns)
        X_std = X.std(axis=0)
        Y_std = Y.std(axis=0)
        if np.any(X_std < 1e-12) or np.any(Y_std < 1e-12):
            # Add small noise to break degeneracy
            rng = np.random.default_rng(320)
            X = X + rng.normal(0, 1e-8, X.shape)
            Y = Y + rng.normal(0, 1e-8, Y.shape)
        
        if center:
            # Center the data
            X_mean = X.mean(axis=0)
            Y_mean = Y.mean(axis=0)
            X_centered = X - X_mean
            Y_centered = Y - Y_mean
        else:
            X_centered = X.copy()
            Y_centered = Y.copy()
        
        # Check if matrices are degenerate after centering
        X_norm = np.linalg.norm(X_centered, 'fro')
        Y_norm = np.linalg.norm(Y_centered, 'fro')
        
        if X_norm < 1e-12 or Y_norm < 1e-12:
            return np.nan
        
        if scale:
            # Scale to unit Frobenius norm
            X_scaled = X_centered / X_norm
            Y_scaled = Y_centered / Y_norm
        else:
            X_scaled = X_centered
            Y_scaled = Y_centered
        
        # Find optimal orthogonal transformation
        R, scale_factor = orthogonal_procrustes(X_scaled, Y_scaled)
        
        # Apply transformation
        Y_aligned = Y_scaled @ R
        
        # Compute disparity (mean squared error)
        disparity = np.mean((X_scaled - Y_aligned) ** 2)
        
        # Convert disparity to similarity
        # Disparity ranges from 0 (perfect) to ~2 (opposite)
        # Map to similarity in [0, 1]
        similarity = 1.0 - min(disparity / 2.0, 1.0)
        
        if not np.isfinite(similarity):
            return np.nan
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


# =============================================================================
# RDM-based Similarity
# =============================================================================

def rdm_similarity(
    X: np.ndarray,
    Y: np.ndarray,
    metric: Literal["cosine", "correlation", "euclidean"] = "cosine",
    method: Literal["spearman", "pearson"] = "spearman",
) -> float:
    """
    RDM-based similarity using correlation of pairwise distances.
    
    Computes Representational Dissimilarity Matrices (RDMs) for X and Y,
    then measures their correlation. This is the same approach used in
    shesha.rdm_similarity but available here for comparison with CKA.
    
    Parameters
    ----------
    X : np.ndarray
        First representation matrix of shape (n_samples, n_features_x).
    Y : np.ndarray
        Second representation matrix of shape (n_samples, n_features_y).
        Must have same number of samples as X.
    metric : str, default="cosine"
        Distance metric for RDM: 'cosine', 'correlation', or 'euclidean'.
    method : str, default="spearman"
        Correlation method: 'spearman' (rank-based) or 'pearson' (linear).
    
    Returns
    -------
    float
        RDM similarity in [-1, 1]. Higher values indicate more similar
        pairwise distance structure. Spearman is more robust to outliers.
    
    Examples
    --------
    >>> import numpy as np
    >>> from shesha.similarity import rdm_similarity
    >>> 
    >>> X = np.random.randn(100, 50)
    >>> Y = np.random.randn(100, 30)
    >>> 
    >>> # Spearman correlation (robust, rank-based)
    >>> sim_spearman = rdm_similarity(X, Y, method='spearman')
    >>> 
    >>> # Pearson correlation (linear)
    >>> sim_pearson = rdm_similarity(X, Y, method='pearson')
    
    Notes
    -----
    RDM similarity is similar to RSA (Representational Similarity Analysis).
    Spearman correlation is preferred as it's robust to monotonic transformations
    of distances and less sensitive to outliers.
    
    Unlike CKA, RDM similarity operates on pairwise distances rather than
    Gram matrices, making it more interpretable but potentially less sensitive.
    
    See Also
    --------
    shesha.rdm_similarity : Identical implementation in core module
    cka : Alternative similarity metric using kernel alignment
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have same number of samples: "
            f"X has {X.shape[0]}, Y has {Y.shape[0]}"
        )
    
    # Compute RDMs (condensed form - upper triangle only)
    rdm_x = pdist(X, metric=metric)
    rdm_y = pdist(Y, metric=metric)
    
    # Compute correlation
    if method == "spearman":
        corr = spearmanr(rdm_x, rdm_y).correlation
    elif method == "pearson":
        corr, _ = pearsonr(rdm_x, rdm_y)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'pearson'")
    
    return float(corr) if np.isfinite(corr) else 0.0
