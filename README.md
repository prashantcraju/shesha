[![PyPI version](https://img.shields.io/pypi/v/shesha-geometry.svg)](https://pypi.org/project/shesha-geometry/)
[![DOI](https://zenodo.org/badge/1133185691.svg)](https://doi.org/10.5281/zenodo.18227453)
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

### Drift Metrics (comparing two representations)

**`rdm_similarity(X, Y, method='spearman', metric='cosine')`**

RDM correlation between two representations. Use for comparing models or tracking changes.

**`rdm_drift(X, Y, method='spearman', metric='cosine')`**

Representational drift (1 - similarity). Use for quantifying how much geometry has changed.

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

Explore `shesha` with these interactive notebooks:

*  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/llm_embeddings_tutorial.ipynb) **LLM Embeddings**: **Geometric Stability:** Analyze embedding stability across layers and models using `feature_split`. 
*  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/steering_vectors_tutorial.ipynb) **Steering Vectors**:  **Consistency Analysis:** Compute steering vectors from contrastive pairs and measure their effectiveness and consistency. 
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/vision_models_tutorial.ipynb) **Vision Models**:  **Architecture Comparison:** Compare geometric stability and class separability across ResNets, ViTs, and other vision architectures. 
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/drift_tutorial.ipynb) **Representational Drift**: **Perturbation Analysis:** Measure drift caused by Gaussian noise injection and LoRA fine-tuning using `rdm_drift`. 
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/training_drift_tutorial.ipynb) **Training Dynamics**: **Live Monitoring:** Track geometric stability during model training to detect representation collapse or divergence. 
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/crispr_tutorial.ipynb) **CRISPR (Bio)**: **Single-Cell Analysis:** Use `shesha.bio` to analyze stability and effect sizes in single-cell CRISPR perturbation experiments. 


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

### `shesha.bio.perturbation_stability(X_control, X_perturbed, metric='cosine', seed=None, max_samples=1000)`

Measures consistency of perturbation effects across samples. High values indicate coherent, reproducible perturbation effects.

**Parameters:**
- `X_control` - array of shape (n_control, n_features), control population
- `X_perturbed` - array of shape (n_perturbed, n_features), perturbed population
- `metric` - 'cosine' (default) or 'euclidean'
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

### `shesha.bio.compute_stability(adata, perturbation_key, control_label, layer=None, metric='cosine')`

Computes the geometric stability for every perturbation in the dataset.

**Parameters:**
- `adata` - AnnData object.
- `perturbation_key` - Column in `adata.obs` identifying the perturbation (e.g., `'guide_id'`).
- `control_label` - The label in that column representing control cells (e.g., `'NT'`).
- `layer` - (Optional) Layer to use (e.g., `'pca'`). If None, uses `.X`.
- `metric` - `'cosine'` (default) or `'euclidean'`.

**Returns:** Dictionary `{perturbation_name: stability_score}`.

### `shesha.bio.compute_magnitude(adata, perturbation_key, control_label, layer=None, metric='euclidean')`

Computes the magnitude (effect size) for every perturbation.

**Parameters:**
- `adata` - AnnData object.
- `metric` - `'euclidean'` (default, raw distance) or `'cohen'` (standardized effect size).

**Returns:** Dictionary `{perturbation_name: magnitude_score}`.


## Citation

If you use `shesha-geometry`, please cite:
```bibtex
@software{shesha2026,
    title = {Shesha: Self-consistency Metrics for Representational Stability},
    author = {Prashant C. Raju},
    year = {2026},
    publisher = {Zenodo},
    doi = {10.5281/zenodo.18227453},
    url = {https://doi.org/10.5281/zenodo.18227453},
    copyright = {MIT License}
}

@article{raju2026geometric,
  title={Geometric Stability: The Missing Axis of Representations},
  author={Raju, Prashant C.},
  journal={arXiv preprint arXiv:2601.09173},
  year={2026}
}
```

## License

MIT

---

<sub>Logo generated by [Nano Banana Pro](https://nanobananapro.com)</sub>
