[![PyPI version](https://img.shields.io/pypi/v/shesha-geometry.svg?cache=bust)](https://pypi.org/project/shesha-geometry/)
[![Tests](https://github.com/prashantcraju/shesha/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/prashantcraju/shesha/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/shesha-geometry/badge/?version=latest)](https://shesha-geometry.readthedocs.io/en/latest/?badge=latest)
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

**[Full documentation at shesha-geometry.readthedocs.io](https://shesha-geometry.readthedocs.io/en/latest/)**

## Installation

```bash
pip install shesha-geometry
```

For single-cell biology workflows (`shesha.bio`), install with the `bio` extra, which adds `anndata` and `scikit-learn`:

```bash
pip install shesha-geometry[bio]
```

## Quick Start

```python
import numpy as np
import shesha

X = np.random.randn(500, 768)  # (n_samples, n_features)

stability = shesha.feature_split(X, n_splits=30, seed=320)
print(f"Feature-split stability: {stability:.3f}")
```

For the full API reference, installation guide, and usage examples, see the [documentation](https://shesha-geometry.readthedocs.io/en/latest/).

## Tutorials

Explore `shesha` with these interactive notebooks (each takes < 5 minutes to run):

* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/llm_embeddings_tutorial.ipynb) **LLM Embeddings** - Analyze embedding stability across layers and models using `feature_split`.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/steering_vectors_tutorial.ipynb) **Steering Vectors** - Compute steering vectors from contrastive pairs and measure their consistency.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/vision_models_tutorial.ipynb) **Vision Models** - Compare geometric stability across ResNets, ViTs, and other architectures.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/drift_tutorial.ipynb) **Representational Drift** - Measure drift from Gaussian noise injection and LoRA fine-tuning.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/training_drift_tutorial.ipynb) **Training Dynamics** - Track geometric stability during training to detect representation collapse.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/crispr_tutorial.ipynb) **CRISPR (Bio)** - Use `shesha.bio` to analyze stability in single-cell CRISPR perturbation experiments.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashantcraju/shesha/blob/main/tutorials/crispr_split_half_tutorial.ipynb) **CRISPR Split-Half Reproducibility (Bio)** - Measure effect-direction reproducibility with `split_half_reproducibility` and control for magnitude confounds with `magnitude_matched_comparison`.

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

If you use the supervised variants (`supervised_alignment`, `lda_stability`, `variance_ratio`, `class_separation_ratio`), please also cite:
```bibtex
@inproceedings{raju2026canary,
  title = {The Geometric Canary: Predicting Steerability and Detecting Drift via Representational Stability},
  author = {Raju, Prashant C.},
  booktitle= {Mechanistic Interpretability Workshop at ICML 2026},
  year= {2026}
}
```

If you use the `shesha.bio` module, please also cite:
```bibtex
@article{raju2026crispr,
  title = {Geometric Coherence of Single-Cell CRISPR Perturbations Reveals Regulatory Architecture and Predicts Cellular Stress},
  author = {Raju, Prashant C.},
  journal = {arXiv preprint arXiv:2604.16642},
  year = {2026}
}
```

## License

MIT

---

<sub>Logo generated by [Nano Banana Pro](https://nanobananapro.com)</sub>
