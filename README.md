[![PyPI version](https://img.shields.io/pypi/v/shesha-geometry.svg?cache=bust)](https://pypi.org/project/shesha-geometry/)
[![Tests](https://github.com/prashantcraju/shesha/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/prashantcraju/shesha/actions/workflows/test.yml)
[![Socket Badge](https://badge.socket.dev/pypi/package/shesha-geometry/0.2.12?artifact_id=tar-gz)](https://badge.socket.dev/pypi/package/shesha-geometry/0.2.12?artifact_id=tar-gz)
[![CodeFactor](https://www.codefactor.io/repository/github/prashantcraju/shesha/badge)](https://www.codefactor.io/repository/github/prashantcraju/shesha)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18227453.svg)](https://doi.org/10.5281/zenodo.18227453)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/prashantcraju/shesha/blob/main/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/shesha-geometry?period=total&units=NONE&left_color=GRAY&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/shesha-geometry)
<p align="center">
    <img src="https://i.imgur.com/oJ5YhBo.jpg" alt="Shesha Logo" width="300">
</p>

# Shesha

Self-consistency metrics for representational stability analysis.

Shesha measures the geometric stability of high-dimensional representations by quantifying the self-consistency of their pairwise distance structure (RDMs) under controlled internal perturbations.

## Installation

```bash
pip install shesha-geometry
```

## Quick Start

```python
import numpy as np
import shesha

# Your embeddings: (n_samples, n_features)
X = np.random.randn(500, 768)

# Feature-split stability (unsupervised)
stability = shesha.feature_split(X, n_splits=30, seed=320)
print(f"Feature-split stability: {stability:.3f}")
```

With labels:

```python
y = np.random.randint(0, 10, 500)
alignment = shesha.supervised_alignment(X, y)
print(f"Supervised alignment: {alignment:.3f}")
```

Measuring drift between representations:

```python
X_before = np.random.randn(100, 256)
X_after = X_before + np.random.randn(100, 256) * 0.3  # Add noise

# Compare before/after fine-tuning
similarity = shesha.rdm_similarity(X_before, X_after)
drift = shesha.rdm_drift(X_before, X_after)
print(f"RDM similarity: {similarity:.3f}, drift: {drift:.3f}")
```


## Variants

### Unsupervised (no labels required)

**`feature_split(X, n_splits=30, metric='cosine', seed=None)`**

Correlates RDMs from random feature partitions. Use for internal consistency and drift detection.

**`sample_split(X, n_splits=30, subsample_fraction=0.4, seed=None)`**

Correlates RDMs from bootstrap samples. Use for robustness to sampling.

**`anchor_stability(X, n_splits=30, n_anchors=100, seed=None)`**

Distance profile consistency from fixed anchors. Use for large-scale stability.

### Supervised (labels required)

**`variance_ratio(X, y)`**

Between-class / total variance. Use for quick separability check.

**`supervised_alignment(X, y, metric='correlation', seed=None)`**

Correlation with ideal label RDM. Use for task alignment.

**`class_separation_ratio(X, y, n_bootstrap=50, subsample_frac=0.5)`**

Between-class to within-class distance ratio.

**`lda_stability(X, y, n_bootstrap=50, subsample_frac=0.5)`**

Consistency of discriminant direction under resampling.

### Drift Metrics (comparing two representations)

**`rdm_similarity(X, Y, method='spearman', metric='cosine')`**

RDM correlation between two representations. Use for comparing models or tracking changes.

**`rdm_drift(X, Y, method='spearman', metric='cosine')`**

Representational drift (1 - similarity). Use for quantifying how much geometry has changed.

### Similarity Metrics (`shesha.sim`)

**`cka(X, Y, debiased=False)`**

Centered Kernel Alignment - the most popular similarity metric for neural representations. Invariant to orthogonal transformations.

```python
import shesha.sim as sim

# Compare two model layers
similarity = sim.cka(model_layer_12, model_layer_18)
```

**`procrustes_similarity(X, Y, center=True, scale=True)`**

Orthogonal Procrustes alignment. More sensitive to outliers than CKA (6x more false alarms in stable regimes).

**`rdm_similarity(X, Y, metric='cosine', method='spearman')`**

RSA-style RDM correlation. Rank-based and robust to transformations.

## Examples

### Comparing model stability

```python
import numpy as np
import shesha

# Example embeddings from two different models
embeddings_a = np.random.randn(500, 768)  # Model A embeddings
embeddings_b = np.random.randn(500, 768)  # Model B embeddings

models = {'model_a': embeddings_a, 'model_b': embeddings_b}

for name, X in models.items():
    fs = shesha.feature_split(X, seed=320)
    print(f"{name}: {fs:.3f}")
```

### Monitoring fine-tuning drift

```python
import shesha

X_initial = model.encode(data)

for epoch in range(10):
    train_one_epoch(model)
    X_current = model.encode(data)
    
    # Internal stability
    stability = shesha.feature_split(X_current, seed=320)
    
    # Drift from initial
    drift = shesha.rdm_drift(X_initial, X_current)
    
    print(f"Epoch {epoch}: stability={stability:.3f}, drift={drift:.3f}")
```

### Comparing two models

```python
import shesha

X_model1 = model1.encode(data)
X_model2 = model2.encode(data)

# How similar are their geometric structures?
similarity = shesha.rdm_similarity(X_model1, X_model2)
print(f"Model similarity: {similarity:.3f}")
```

### Analyzing single-cell perturbations

Measure the geometric consistency of CRISPR/drug screens directly from `AnnData` objects:

```python
import numpy as np
from shesha.bio import compute_stability, compute_magnitude
from anndata import AnnData


# 1. Setup mock single-cell data (1000 cells, 50 PCA features)
n_cells = 1000
n_genes = 2000  # Original feature space (genes)
n_pcs = 50

# Create a mock AnnData object
# Note: Shesha works best on PCA coordinates (latent space), not raw counts
adata = AnnData(X=np.random.randn(n_cells, n_genes))  # Raw counts (unused)
adata.obsm['X_pca'] = np.random.randn(n_cells, n_pcs)  # PCA embeddings
adata.obs['guide_id'] = ['NT'] * 800 + ['KLF1'] * 200  # Metadata


# Create a proxy for PCA coordinates (Recommended for robust geometry)
adata_pca = AnnData(X=adata.obsm['X_pca'], obs=adata.obs)


# Compute Stability (Consistency of the phenotype)
stability = compute_stability(
    adata_pca, 
    perturbation_key='guide_id', 
    control_label='NT',
    metric='cosine'
)

# Compute Magnitude (Strength of the phenotype)
magnitude = compute_magnitude(
    adata_pca, 
    perturbation_key='guide_id', 
    control_label='NT', 
    metric='euclidean'
)

print(f"KLF1 Stability: {stability['KLF1']:.3f}")  # e.g., 0.85 (High = Consistent)
print(f"KLF1 Magnitude: {magnitude['KLF1']:.3f}")  # e.g., 2.40 (High = Strong)
```


## Tutorials

Explore `shesha` with these interactive notebooks (each takes < 5 minutes to run):


*  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/llm_embeddings_tutorial.ipynb) **LLM Embeddings** - **Geometric Stability:** Analyze embedding stability across layers and models using `feature_split`. 
*  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/steering_vectors_tutorial.ipynb) **Steering Vectors** -  **Consistency Analysis:** Compute steering vectors from contrastive pairs and measure their effectiveness and consistency. 
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/vision_models_tutorial.ipynb) **Vision Models** -  **Architecture Comparison:** Compare geometric stability and class separability across ResNets, ViTs, and other vision architectures. 
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/drift_tutorial.ipynb) **Representational Drift** - **Perturbation Analysis:** Measure drift caused by Gaussian noise injection and LoRA fine-tuning using `rdm_drift`. 
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/training_drift_tutorial.ipynb) **Training Dynamics** - **Live Monitoring:** Track geometric stability during model training to detect representation collapse or divergence. 
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/crispr_tutorial.ipynb) **CRISPR (Bio)** - **Single-Cell Analysis:** Use `shesha.bio` to analyze stability and effect sizes in single-cell CRISPR perturbation experiments. 


## API Reference

### `shesha.feature_split(X, n_splits=30, metric='cosine', seed=None, max_samples=1600)`

Measures internal geometric consistency by correlating RDMs computed from random, disjoint subsets of feature dimensions.

**Parameters:**
- `X` - array of shape (n_samples, n_features)
- `n_splits` - number of random partitions to average
- `metric` - 'cosine' or 'correlation'
- `seed` - random seed for reproducibility
- `max_samples` - subsample if exceeded

**Returns:** float in [-1, 1], higher = more stable

### `shesha.sample_split(X, n_splits=30, subsample_fraction=0.4, metric='cosine', seed=None, max_samples=1500)`

Measures robustness to input variation via bootstrap resampling.

**Parameters:**
- `X` - array of shape (n_samples, n_features)
- `n_splits` - number of bootstrap iterations
- `subsample_fraction` - fraction of samples per bootstrap
- `metric` - 'cosine' or 'correlation'
- `seed` - random seed for reproducibility
- `max_samples` - subsample if exceeded

**Returns:** float in [-1, 1], higher = more stable

### `shesha.anchor_stability(X, n_splits=30, n_anchors=100, n_per_split=200, metric='cosine', rank_normalize=True, seed=None, max_samples=1500)`

Measures stability of distance profiles from fixed anchor points.

**Parameters:**
- `X` - array of shape (n_samples, n_features)
- `n_splits` - number of random splits
- `n_anchors` - number of fixed anchor points
- `n_per_split` - samples per split
- `metric` - 'cosine' or 'euclidean'
- `rank_normalize` - rank-normalize distances within each anchor
- `seed` - random seed for reproducibility
- `max_samples` - subsample if exceeded

**Returns:** float in [-1, 1], higher = more stable

### `shesha.variance_ratio(X, y)`

Ratio of between-class to total variance.

**Parameters:**
- `X` - array of shape (n_samples, n_features)
- `y` - array of shape (n_samples,) with class labels

**Returns:** float in [0, 1], higher = better class separation

### `shesha.supervised_alignment(X, y, metric='correlation', seed=None, max_samples=300)`

Spearman correlation between model RDM and ideal label-based RDM.

**Parameters:**
- `X` - array of shape (n_samples, n_features)
- `y` - array of shape (n_samples,) with class labels
- `metric` - 'cosine' or 'correlation'
- `seed` - random seed for reproducibility
- `max_samples` - subsample if exceeded (RDM is O(n^2))

**Returns:** float in [-1, 1], higher = better task alignment

### `shesha.class_separation_ratio(X, y, n_bootstrap=50, subsample_frac=0.5, metric='euclidean', seed=None)`

Ratio of between-class to within-class distances with bootstrap subsampling.

**Parameters:**
- `X` - array of shape (n_samples, n_features)
- `y` - array of shape (n_samples,) with class labels
- `n_bootstrap` - number of bootstrap iterations
- `subsample_frac` - fraction of samples per bootstrap
- `metric` - 'cosine' or 'euclidean'

**Returns:** float > 0, higher = better class separation. Typically in [0.5, 5.0].

### `shesha.lda_stability(X, y, n_bootstrap=50, subsample_frac=0.5, seed=None)`

Consistency of LDA discriminant direction under resampling. Binary classification only.

**Parameters:**
- `X` - array of shape (n_samples, n_features)
- `y` - array of shape (n_samples,) with **binary** class labels (exactly 2 classes)
- `n_bootstrap` - number of bootstrap iterations
- `subsample_frac` - fraction of samples per bootstrap

**Returns:** float in [0, 1], higher = more stable discriminant. Predicts steerability (ρ=0.89-0.96).

### `shesha.rdm_similarity(X, Y, method='spearman', metric='cosine')`

Computes RDM correlation between two representations. Useful for comparing models, tracking drift during training, or measuring the effect of interventions.

**Parameters:**
- `X` - array of shape (n_samples, n_features_x), first representation
- `Y` - array of shape (n_samples, n_features_y), second representation (same n_samples)
- `method` - 'spearman' (rank-based, default) or 'pearson' (linear)
- `metric` - 'cosine', 'correlation', or 'euclidean'

**Returns:** float in [-1, 1], higher = more similar geometric structure

### `shesha.rdm_drift(X, Y, method='spearman', metric='cosine')`

Computes representational drift as 1 - rdm_similarity. Useful for quantifying how much a representation has changed.

**Parameters:**
- `X` - array of shape (n_samples, n_features_x), baseline representation
- `Y` - array of shape (n_samples, n_features_y), comparison representation
- `method` - 'spearman' (rank-based, default) or 'pearson' (linear)
- `metric` - 'cosine', 'correlation', or 'euclidean'

**Returns:** float in [0, 2], where 0 = identical, 1 = uncorrelated, 2 = inverted

## Biological Perturbation Analysis

The `shesha.bio` module provides metrics for single-cell perturbation experiments (e.g., Perturb-seq, CRISPR screens).

### `shesha.bio.perturbation_stability(X_control, X_perturbed, method='standard', metric='cosine', k=50, regularization=1e-6, seed=None, max_samples=1000)`

Unified interface for measuring consistency of perturbation effects across samples.

**Parameters:**
- `X_control` - array of shape (n_control, n_features), control population
- `X_perturbed` - array of shape (n_perturbed, n_features), perturbed population
- `method` - 'standard' (default), 'whitened', or 'knn'
  - `'standard'`: Global control centroid (fastest)
  - `'whitened'`: Mahalanobis-scaled for batch effects
  - `'knn'`: Local k-NN matched controls for heterogeneity
- `metric` - 'cosine' (default) or 'euclidean' (for standard/knn methods)
- `k` - number of neighbors (only for method='knn')
- `regularization` - covariance regularization (only for method='whitened')
- `seed` - random seed for reproducibility
- `max_samples` - subsample perturbed population if exceeded

**Returns:** float in [-1, 1], higher = more consistent perturbation

### `shesha.bio.perturbation_effect_size(X_control, X_perturbed)`

Cohen's d-like effect size measuring magnitude of perturbation shift.

**Parameters:**
- `X_control` - array of shape (n_control, n_features)
- `X_perturbed` - array of shape (n_perturbed, n_features)

**Returns:** float >= 0, higher = larger perturbation effect

### Scanpy / AnnData Integration

For single-cell analysis, Shesha provides high-level wrappers that work directly with `AnnData` objects.

### `shesha.bio.compute_stability(adata, perturbation_key, control_label, layer=None, method='standard', **kwargs)`

Computes the geometric stability for every perturbation in the dataset. Unified interface supporting multiple methods.

**Parameters:**
- `adata` - AnnData object.
- `perturbation_key` - Column in `adata.obs` identifying the perturbation (e.g., `'guide_id'`).
- `control_label` - The label in that column representing control cells (e.g., `'NT'`).
- `layer` - (Optional) Layer to use (e.g., `'pca'`). If None, uses `.X`.
- `method` - `'standard'` (default), `'whitened'`, or `'knn'`.
- `**kwargs` - Additional arguments: `k=50` for knn, `regularization=1e-6` for whitened, `metric='cosine'` for standard/knn.

**Returns:** Dictionary `{perturbation_name: stability_score}`.

### `shesha.bio.compute_magnitude(adata, perturbation_key, control_label, layer=None, metric='euclidean')`

Computes the magnitude (effect size) for every perturbation.

**Parameters:**
- `adata` - AnnData object.
- `metric` - `'euclidean'` (default, raw distance) or `'cohen'` (standardized effect size).

**Returns:** Dictionary `{perturbation_name: magnitude_score}`.

### Enhanced Perturbation Methods

**`shesha.bio.perturbation_stability(X_control, X_perturbed, method='standard', ...)`**

Unified interface with three methods for computing perturbation stability:
- `method='standard'` - Global control centroid (default, fastest)
- `method='whitened'` - Mahalanobis-scaled for batch effects
- `method='knn'` - k-NN matched for heterogeneous controls

**`shesha.bio.compute_stability(adata, perturbation_key, method='standard', ...)`**

AnnData wrapper supporting all three methods.

```python
import shesha.bio as bio

# Standard stability
stability = bio.compute_stability(adata, "perturbation", method="standard")

# Whitened (for batch effects)
stability_white = bio.compute_stability(adata, "perturbation", method="whitened")

# k-NN (for heterogeneous controls)
stability_knn = bio.compute_stability(adata, "perturbation", method="knn", k=50)

# Low-level API also supports method parameter
from shesha.bio import perturbation_stability
score = perturbation_stability(X_ctrl, X_pert, method="whitened")
```

**Backward compatibility:** Old function names (`perturbation_stability_whitened`, `perturbation_stability_knn`, `compute_stability_whitened`, `compute_stability_knn`) still work as convenience wrappers.


## Testing

Shesha has a comprehensive test suite to ensure scientific reliability and correctness.

### Running Tests Locally

Install development dependencies:

```bash
pip install -e .[dev]
```

Run all tests:

```bash
pytest tests/
```

Run with coverage report:

```bash
pytest tests/ --cov=shesha --cov-report=term
```

### Continuous Integration

All tests are automatically run on:
- **Multiple Python versions:** 3.8, 3.9, 3.10, 3.11, 3.12
- **Multiple operating systems:** Ubuntu, macOS, Windows

See the [Tests workflow](https://github.com/prashantcraju/shesha/actions) for current status.


## Citation

If you use `shesha-geometry`, please cite:
```bibtex
@software{shesha2026,
  title = {Shesha: Self-Consistency Metrics for Representational Stability},
  author = {Raju, Prashant C.},
  year = {2026},
  howpublished = {Zenodo},
  doi = {10.5281/zenodo.18227453},
  url = {https://doi.org/10.5281/zenodo.18227453},
  copyright = {MIT License}
}

@article{raju2026geometric,
  title = {Geometric Stability: The Missing Axis of Representations},
  author = {Raju, Prashant C.},
  journal = {arXiv preprint arXiv:2601.09173},
  year = {2026}
}
```

## License

MIT

---

<sub>Logo generated by [Nano Banana Pro](https://nanobananapro.com)</sub>
