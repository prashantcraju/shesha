# Changelog

All notable changes to the `shesha` package will be documented in this file.

## [0.2.21] - 2026-06-14

### Added

- **`split_half_reproducibility`** (`shesha.bio`): AnnData-compatible function that measures
  per-perturbation effect-direction reproducibility via repeated 50/50 random cell splits.
  For each perturbation, cells are split into two independent halves, a shift vector relative
  to the control centroid is computed for each half, and the cosine similarity between the two
  halves is averaged over `n_splits` (default 50). Returns a `pd.DataFrame` indexed by
  perturbation with columns `split_half_cosine` and `n_cells`.
- **`magnitude_matched_comparison`** (`shesha.bio`): Confound-control utility that bins
  perturbations by effect magnitude, then within each bin compares mean split-half cosine
  between the high-stability and low-stability halves. Returns a `pd.DataFrame` with per-bin
  statistics including `difference`, `within_bin_rho`, and `within_bin_pvalue`. Controls for
  the SNR confound where larger-effect perturbations may appear more reproducible.
- **`_split_half_cosine`** (`shesha.bio`): Low-level numpy implementation of the split-half
  cosine kernel, reusable independently of AnnData.
- `tests/test_split_half.py`: 20 pytest tests covering `_split_half_cosine`,
  `split_half_reproducibility`, and `magnitude_matched_comparison` (signal recovery,
  determinism, min-cells filtering, column validation, NaN handling, custom column names).

### Changed

- `pyproject.toml`: added `pandas>=1.3` as an explicit dependency (previously a transitive
  dependency via `anndata`; now required directly by `shesha.bio`).

---

## [0.2.20] - 2026-05-28

### Added

- **Bootstrap confidence intervals** (outer bootstrap on input data): optional `n_bootstrap_ci` and `ci` parameters on all public metrics in `shesha.core`, `shesha.bio`, and `shesha.sim`. When `n_bootstrap_ci` is set, functions return a dict with `mean`, `ci_low`, `ci_high`, `std`, `n_bootstraps`, and `ci_level`; default behaviour (no `n_bootstrap_ci`) still returns a `float`.
- `shesha/_utils.py`: shared helpers `bootstrap_ci`, `bootstrap_ci_two_sample`, and `bootstrap_ci_bio` for single-matrix, paired two-sample, and independent two-population resampling.
- `tests/test_bootstrap_ci.py`: tests for CI dict structure, backward compatibility, determinism, and coverage across core, bio, and sim.
- Read the Docs user guide `docs/guide/bootstrap_ci.rst`; CI examples added to existing guides and quickstart.

### Changed

- Core metrics with CI support: `feature_split`, `sample_split`, `anchor_stability`, `variance_ratio`, `supervised_alignment`, `class_separation_ratio`, `lda_stability`, `rdm_similarity`, `rdm_drift`.
- Bio metrics with CI support: `perturbation_stability`, `perturbation_effect_size`.
- Sim metrics with CI support: `cka`, `cka_linear`, `cka_debiased`, `procrustes_similarity`, `rdm_similarity`.

---

## [0.2.18] - 2026-05-26

### Changed

- `tutorials/crispr_tutorial.ipynb`: restored `pt.dt.norman_2019()` loading and standard optional install cell. Added notes that the Norman 2019 dataset (~111k cells) is too large for the standard free Google Colab runtime; recommend Colab Pro, a local machine, or more RAM.

---

## [0.2.17] - 2026-05-25

### Changed

- Simplified `README.md` and `README_PYPI.md`: removed inline API reference, variants, examples, and testing sections in favour of a link to the ReadTheDocs documentation site. Tutorials and citation blocks retained.
- Updated logo reference in `README.md` to use the local `assets/shesha-logo.jpg` path; `README_PYPI.md` retains the absolute URL for PyPI rendering.

### Fixed

- `paper/paper.bib`: corrected `vandenBosch2025` entry — changed `publisher = {openRxiv}` to `journal = {bioRxiv}`, updated URL scheme to `https://doi.org/`, and lowercased the `month` macro.

---


## [0.2.15] - 2026-04-15

### Added

- Read the Docs documentation site with full Sphinx setup (`docs/`)
- API reference pages for `shesha.core`, `shesha.bio`, and `shesha.sim`
- User guides for unsupervised, supervised, drift, bio, and similarity metrics
- `.readthedocs.yaml` configuration for automated doc builds on RTD

---

## [0.2.14] - 2026-03-31

### Fixed

- **`shesha/bio.py`**: Moved `anndata` import to the top of the module alongside other imports. Previously the `try/except ImportError` block was placed mid-file between function definitions, causing static analysis tools to misparse the following function's docstring as a floating duplicate string literal.
- **`shesha/bio.py`**: Eliminated duplicate AnnData extraction logic in `compute_stability` and `compute_magnitude` by extracting two private helpers:
  - `_get_array(adata, mask, layer)` — extracts a dense numpy array from an AnnData slice, handling sparse matrices in one place
  - `_iter_perturbations(adata, perturbation_key, control_label, layer)` — generator yielding `(pert_name, X_ctrl, X_pert)` for each non-control perturbation

---

## [0.2.0] - 2026-03-20
Added the `shesha.sim` similarity module (CKA, debiased CKA, Procrustes, RDM) and two new supervised metrics in `shesha.core` (`class_separation_ratio`, `lda_stability`). Extended `shesha.bio` perturbation stability with whitened and k-NN matched methods.

### Testing
Expanded the test suite with four new test files covering all v0.2.0 features: `test_new_features.py` (supervised metrics and bio methods), `test_similarity.py` (full `shesha.sim` coverage), `test_anndata_integration.py` (AnnData workflows for all three stability methods), and `test_v020_features.py` (edge cases, error handling, and dispatch consistency). CI now runs across Python 3.8–3.12 on Ubuntu, macOS, and Windows via GitHub Actions, with a separate coverage-reporting job.

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
