# Changelog

All notable changes to the `shesha` package will be documented in this file.

## [0.1.4] - 2026-02-09

### Added

#### New Module: `shesha.sim`

A dedicated module for representational **similarity** metrics, complementing the **stability** metrics in `shesha.core`. This separation emphasizes the key distinction from the paper: similarity is *extrinsic* (how representations align), while stability is *intrinsic* (how robust a representation's geometry is).

**Similarity Metrics:**

1. **`cka_linear(X, Y)`** - Standard Centered Kernel Alignment
   - Linear kernel-based similarity metric
   - Invariant to orthogonal transformations and isotropic scaling
   - Fast and numerically stable
   - Range: [0, 1], with 1.0 meaning identical structure

2. **`cka_debiased(X, Y)`** - Debiased Centered Kernel Alignment
   - Unbiased estimator correcting for finite sample effects
   - More accurate for small sample sizes
   - Uses debiased HSIC estimator from Kornblith et al. (2019)
   - Recommended for n < 100

3. **`cka(X, Y, debiased=False)`** - Unified CKA interface
   - Convenience function selecting between standard and debiased CKA
   - Single entry point for CKA computation

4. **`procrustes_similarity(X, Y, center=True, scale=True)`**
   - Orthogonal Procrustes similarity
   - Finds optimal rotation/reflection alignment
   - More sensitive to outliers than CKA
   - Returns 1 - disparity as similarity score

5. **`rdm_similarity(X, Y, metric='cosine', method='spearman')`**
   - RDM-based similarity using correlation of pairwise distances
   - Same approach as RSA (Representational Similarity Analysis)
   - Supports cosine, correlation, and euclidean distance metrics
   - Supports Spearman (rank-based) and Pearson (linear) correlation

#### Core Module (`shesha.core`)

**New Supervised Stability Metrics:**

6. **`class_separation_ratio(X, y, n_bootstrap=50, subsample_frac=0.5, metric='euclidean', seed=None)`**
   - Measures ratio of between-class to within-class distances
   - Uses bootstrap subsampling for computational efficiency
   - Related to Fisher's discriminant ratio but operates in distance space
   - Higher values indicate better class separation
   - Useful for transfer learning and model selection

7. **`lda_stability(X, y, n_bootstrap=50, subsample_frac=0.5, seed=None)`**
   - Measures consistency of linear discriminant direction under resampling
   - Binary classification only (2 classes required)
   - Returns absolute cosine similarity between full and bootstrap discriminant vectors
   - Values near 1 indicate stable discriminant subspace
   - Predicts transfer learning performance (ρ = 0.89-0.96 in paper)
   - Low values suggest overfitting to source domain

#### Bio Module (`shesha.bio`)

**Enhanced Perturbation Stability Methods:**

8. **`perturbation_stability_whitened(X_control, X_perturbed, regularization=1e-6, seed=None, max_samples=1000)`**
   - Mahalanobis-scaled (whitened) perturbation stability
   - Accounts for feature correlations and covariance structure
   - More robust when features have different scales or are highly correlated
   - Applies whitening transformation: W = V @ diag(1/sqrt(λ)) @ V.T
   - Useful for reducing batch effect sensitivity

9. **`perturbation_stability_knn(X_control, X_perturbed, k=50, metric='euclidean', seed=None, max_samples=1000)`**
   - k-NN matched control perturbation stability
   - Matches each perturbed cell to k nearest control cells
   - Computes shift relative to local control centroid
   - Reduces sensitivity to population heterogeneity
   - Particularly useful when control population has multiple cell types/states

**AnnData Integration (Scanpy-compatible):**

10. **`compute_stability_whitened(adata, perturbation_key, control_label='control', layer=None, ...)`**
    - AnnData wrapper for whitened perturbation stability
    - Compatible with scanpy workflows
    - Supports sparse matrices and custom layers
    - Returns dictionary of stability scores for all perturbations

11. **`compute_stability_knn(adata, perturbation_key, control_label='control', layer=None, k=50, ...)`**
    - AnnData wrapper for k-NN matched stability
    - Compatible with scanpy workflows
    - Supports sparse matrices and custom layers
    - Returns dictionary of stability scores for all perturbations

### Motivation

From the paper: **Stability and similarity are empirically uncorrelated** (ρ ≈ 0.01) and measure fundamentally different properties:
- **Similarity (extrinsic)**: How one representation aligns with another
- **Stability (intrinsic)**: How robust a representation's internal geometry is

Having both in a unified package enables researchers to:
- Reproduce the ρ ≈ 0.01 finding
- Compare stability vs. similarity for model selection
- Identify models with high similarity but low stability (the "geometric tax")
- Perform complete supervised stability analysis
- Conduct robust biological perturbation analysis with better handling of heterogeneous populations and batch effects
- Improve transfer learning prediction capabilities

### Technical Details

- **Backward compatibility**: Added `typing_extensions` support for Python < 3.8 (Literal import)
- **Performance**: All new functions use efficient bootstrap subsampling
- **Testing**: Comprehensive test suite with 15+ tests in `tests/test_similarity.py`, `tests/test_new_features.py`, and `tests/test_anndata_integration.py`
- **Dependencies**: NumPy + SciPy only (no additional dependencies for core functionality)
- All implementations ported from the paper's experiment code (drift/, distinction/, transfer_learning/, crispr/)

### References

- Raju, P. C. (2026). "Geometric Stability: The Missing Axis of Representations." arXiv:2601.09173
- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. ICML 2019.
- Schönemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem. Psychometrika

---

## [0.1.32] - 2026-01-12

### Initial Release

- Core unsupervised metrics: `feature_split`, `sample_split`, `anchor_stability`
- Supervised metrics: `variance_ratio`, `supervised_alignment`
- Drift metrics: `rdm_similarity`, `rdm_drift`
- Bio module: `perturbation_stability`, `perturbation_effect_size`
- AnnData integration via `compute_stability`, `compute_magnitude`
