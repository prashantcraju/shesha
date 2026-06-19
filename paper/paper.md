---
title: 'Shesha: Self-Consistency Metrics for Representational Stability'
tags:
  - Python
  - machine learning
  - representation learning
  - single-cell biology
  - perturbation analysis
  - manifold analysis
  - interpretability
authors:
  - name: Prashant C. Raju
    orcid: 0000-0003-3778-4788
    affiliation: "1"
affiliations:
  - name: Independent Researcher
    index: 1
date: 31 December 2026
bibliography: paper.bib
---

# Summary

Analysis of learned representations has historically focused on **similarity**, measuring how closely internal embeddings align with external references. However, similarity only reveals *what* is represented, not whether that structure is robust to internal or external perturbations. `Shesha` is an open-source Python framework designed to quantify **geometric stability**, a distinct dimension of representational quality that measures how reliably a model's internal geometry holds under perturbation.

The framework unifies three lines of research into a single package: (1) foundational unsupervised stability metrics for general representation analysis and LLM drift detection [@raju2026geometric]; (2) supervised alignment metrics and LLM steerability prediction [@raju2026canary]; and (3) geometric coherence analysis for single-cell perturbation biology experiments [@raju2026crispr]. Implementations are available in `shesha.core` (ML) and `shesha.bio` (single-cell biology with native `AnnData` support).

# Mathematical Framework

`Shesha` measures geometric stability by quantifying the self-consistency of pairwise distance structure under controlled internal perturbations. Given a representation matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ with $n$ samples and $d$ features, a Representational Dissimilarity Matrix (RDM) [@Kriegeskorte2008] $\mathbf{D} \in \mathbb{R}^{n \times n}$ captures pairwise dissimilarities:

$$D_{ij} = 1 - \frac{\mathbf{x}_i^\top \mathbf{x}_j}{\|\mathbf{x}_i\| \|\mathbf{x}_j\|}$$

The core principle is to construct two RDMs from complementary "views" of $\mathbf{X}$ and measure their agreement via Spearman rank correlation:

$$\text{Shesha}(\mathbf{X}) = \rho_s\bigl(\text{vec}(\mathbf{D}^{(1)}), \text{vec}(\mathbf{D}^{(2)})\bigr)$$

where $\text{vec}(\cdot)$ extracts the upper triangular elements. The choice of how views are constructed defines three families of variants, each grounded in a distinct paper.

## Unsupervised Variants (Foundations Paper)

$\text{Shesha}_{\text{FS}}$ (`feature_split`) assesses internal geometric consistency by partitioning feature dimensions into random disjoint halves:

$$\text{Shesha}_{\text{FS}}(\mathbf{X}) = \frac{1}{K} \sum_{k=1}^{K} \rho_s\bigl(\text{vec}(\mathbf{D}_{\mathcal{F}_k^{(1)}}), \text{vec}(\mathbf{D}_{\mathcal{F}_k^{(2)}})\bigr)$$

where $\mathcal{F}_k^{(1)}, \mathcal{F}_k^{(2)} \subset \{1, \ldots, d\}$ are disjoint feature partitions at split $k$, averaged over $K=30$ random splits. High values indicate that geometric structure is evenly distributed across dimensions rather than concentrated in a few dominant features.

$\text{Shesha}_{\text{SS}}$ (`sample_split`) measures robustness to data variation by partitioning samples into disjoint subsets and comparing RDMs computed on each. This variant assesses whether the distance structure generalizes across different subsets of the data.

$\text{Shesha}_{\text{anc}}$ (`anchor_stability`) measures consistency of distance profiles from fixed anchor points, providing an efficient approximation for large-scale datasets where full $O(n^2)$ RDM computation is intractable.

## Supervised Variants (LLM Paper)

$\text{Shesha}_{\text{sup}}$ (`supervised_alignment`) quantifies task-aligned stability when class labels $\mathbf{y}$ are available:

$$\text{Shesha}_{\text{sup}}(\mathbf{X}, \mathbf{y}) = \rho_s\bigl(\text{vec}(\mathbf{D}_\mathbf{X}), \text{vec}(\mathbf{D}_\mathbf{y})\bigr)$$

where $\mathbf{D}_\mathbf{y}$ encodes label dissimilarity. This variant was introduced to predict steerability and detect functional drift in large language models [@Touvron2023LLaMAOA], where ground-truth labels from steering tasks serve as the reference RDM.

Additional supervised metrics introduced in this line of work include the variance ratio (between-class to total variance), class separation ratio (between-class to within-class distances), and LDA subspace stability (consistency of discriminant direction under resampling, was previously shown to predict steerability with $\rho = 0.89$–$0.96$ [@raju2026canary]).

## Biological Perturbation Variants (CRISPR Paper)

$\text{Shesha}_{\text{pert}}$ (`perturbation_stability` in `shesha.bio`) measures directional consistency of biological perturbation effects:

$$\text{Shesha}_{\text{pert}} = \frac{1}{|\mathcal{P}|} \sum_{i \in \mathcal{P}} \frac{(\mathbf{x}_i - \boldsymbol{\mu}_{\text{ctrl}})^\top \bar{\mathbf{d}}}{\|\mathbf{x}_i - \boldsymbol{\mu}_{\text{ctrl}}\| \|\bar{\mathbf{d}}\|}$$

where $\bar{\mathbf{d}}$ is the mean shift direction from control to perturbed population. High values indicate coherent, reproducible perturbation effects rather than stochastic noise [@raju2026crispr]. Three implementations handle different experimental regimes: `standard` (global control centroid, fastest), `whitened` (Mahalanobis-scaled for batch effects), and `knn` (local k-NN matched controls for heterogeneous populations).

All variants use Spearman correlation by default (robust to monotonic transformations and outliers) and subsample to $n_{\max}=1600$ when computational constraints require.

![**A**: Core geometric stability computation for $\text{Shesha}_{\text{FS}}$: the representation matrix $\mathbf{X}$ is split into complementary feature subsets, RDMs are computed on each, and their Spearman rank correlation yields the stability score. **B**: Modular software architecture of `shesha-geometry`, showing the dependency structure across `shesha.core`, `shesha.bio`, and the scverse ecosystem.](overview.png){ width=100% }

# Statement of Need

Geometric stability is an **intrinsic property** of a representation's manifold that measures how consistently it preserves internal geometric structure across varied feature subsets or under perturbation. Unlike similarity metrics such as CKA [@kornblith2019similarity] and RSA [@Kriegeskorte2008], which are **extrinsic properties** measuring how one representation aligns with another—stability is reference-independent. It asks whether the internal "geometry" of the representational space is robust or brittle. Shesha provides a unified API for geometric stability analysis, with applications spanning representation learning [@raju2026geometric], model steerability and drift detection [@raju2026canary], and single-cell perturbation biology [@raju2026crispr].

## Stability vs. Similarity: A Distinction

Consider two libraries holding identical collections. A content audit confirms they are equivalent—the same books are in both. Now suppose one library strictly maintains its ordering, but in the other, readers returns books to the wrong shelves. In the first, the indexing system is robust: nearby books remain topically related and retrieval degrades only slightly. In the second, even these minor displacements cascade: the shelving logic is brittle and locating a book becomes unreliable. Though the inventories remain identical, the libraries are no longer functionally equivalent. For learned representations, geometric stability measures exactly this resilience. Standard similarity metrics confirm that two representations contain the same features, but fail to detect when the fine-grained relationships between them fracture under minor perturbation.


## Critical Gaps Addressed by Shesha

`Shesha` addresses three key tooling challenges, each motivated by a distinct research application:

1. **Foundational Stability Analysis**: unsupervised, reference-free metrics (`feature_split`, `sample_split`, `anchor_stability`, `rdm_drift`) for manifold consistency across language, vision, and general-purpose embeddings, overcoming rigid distance metrics like Procrustes [@Schnemann1966].

2. **LLM Safety and Steerability**: the supervised variant and LDA stability metric act as a highly sensitive "geometric canary" that predicts whether an LLM's representations are steerable and detects early functional degradation during fine-tuning or alignment interventions [@Touvron2023LLaMAOA]. Evaluating intrinsic geometric robustness of vision models also falls within this domain.

3. **Biological Coherence**: the `shesha.bio` module enables native processing of single-cell perturbation data, quantifying the structural coherence of transcriptomic shifts to isolate reproducible state-space trajectories from stochastic noise.

`Shesha` provides a unified API across all three domains.

# State of the Field

Existing tools for representational analysis, such as `rsatoolbox` [@vandenBosch2025] for RSA, `scipy` [@2020SciPy-NMeth] for Procrustes, and various CKA implementations, focus primarily on measuring alignment between two different representations. While these are vital for model comparison, they lack native support for internal self-consistency testing: the ability to measure whether a model's own geometry remains stable under controlled perturbations.

`Shesha` fills this niche by providing specialized methods like `feature_split` (which partitions feature dimensions to test geometric coherence) and `anchor_stability` (which measures consistency across data subsamples). Unlike general-purpose manifold learning tools (e.g., `UMAP` [@McInnes2018], `scikit-learn` [@scikit-learn]), `Shesha` is specifically optimized for high-dimensional stability testing at scale. It also includes a dedicated `shesha.bio` module with high-level wrappers for `AnnData` objects [@Virshup2023; @Virshup2024], making it directly compatible with the single-cell biology ecosystem used by tools like `scanpy` [@Wolf2018] and `pertpy` [@Heumos2025].

# Software Design

`Shesha` is built around a unified API with several deliberate design trade-offs:

- **Modular Architecture**: The three-module structure (`shesha.core`, `shesha.bio`, `shesha.sim`) reflects a dependency isolation strategy: `shesha.bio` requires `anndata` and encodes perturbation-biology semantics (control/treated populations, AnnData accessors), while `shesha.core` carries only NumPy/SciPy dependencies. This allows ML researchers to use stability metrics without the bioinformatics stack, and isolates domain-specific logic from general metrics.

- **Computational Design**: Spearman rank correlation is used in preference to Pearson because RDM entries span heterogeneous scales across metrics and datasets; rank transformation provides invariance to monotonic rescaling and robustness to the outlier distances common in high-dimensional spaces. The $O(n^2)$ pairwise distance bottleneck is addressed via `max_samples` (default 1600): random subsampling preserves RDM rank structure at a fraction of memory cost, with the default chosen empirically to balance statistical fidelity and computational cost for typical single-cell dataset sizes.

- **Self-Consistency Estimator**: Split-half self-consistency was chosen over test-retest reliability because it requires only a single pass over a pre-computed embedding matrix—no repeated measurements or access to training infrastructure—making stability assessment applicable to any archived representation.

- **Ecosystem Integration**: The software relies on the standard scientific Python stack (`NumPy` [@harris2020array], `SciPy` [@2020SciPy-NMeth], `scikit-learn` [@scikit-learn]). The `shesha.bio` module works natively with `AnnData` [@Virshup2023; @Virshup2024] objects and is compatible with `scanpy` [@Wolf2018] and `pertpy` [@Heumos2025] workflows, enabling stability analysis in existing single-cell pipelines without additional preprocessing.

# Research Impact Statement

`Shesha` has been used as the primary analysis framework in three lines of published research, each corresponding to a distinct module. All analyses in [@raju2026geometric], [@raju2026canary], and [@raju2026crispr] were conducted using `Shesha`, with reproducible artifacts available as six Colab tutorials (each executable in under 5 minutes).

**General Representation Analysis**: `Shesha`'s unsupervised metrics were used to evaluate manifold consistency across varied model architectures, demonstrating that geometric stability predicts generalization independently of benchmark performance.

**LLM Alignment and Steerability**: The LDA stability score was previously shown [@raju2026canary] to predict steerability of fine-tuned and RLHF-trained LLMs [@Touvron2023LLaMAOA] with $\rho = 0.89$–$0.96$, providing a fast geometry-based diagnostic that complements behavioral evaluations.

**Single-Cell Perturbation Analysis**: `shesha.bio` was used to distinguish reproducible genetic perturbation effects from stochastic transcriptomic noise in large-scale CRISPR screens, demonstrating applicability beyond ML to experimental biology.

The package is available on PyPI (`shesha-geometry`) and archived on Zenodo [@raju2026shesha] for long-term reproducibility.


# AI Usage Disclosure

In accordance with JOSS AI policy, the author discloses the use of generative AI during the development of this software and paper:

- **Software Creation & Refactoring**: Claude (Sonnet and Opus) and Gemini were used to refactor experimental research scripts into a production-grade Python package, including implementation of design patterns, code optimization for readability, and generation of `pytest` scaffolding.
- **Debugging & Documentation**: AI assistants supported identification of edge cases in high-dimensional distance calculations and assisted in drafting docstrings and API reference material.
- **Formatting & Manuscript Editing**: AI tools helped with formatting and copy-editing of the paper text.
- **Visual Assets**: The software logo was generated using Nano Banana Pro.

**Confirmation of Review**: The author has reviewed, edited, and validated all AI-assisted outputs to ensure technical accuracy and adherence to the core geometric principles of the Shesha framework. All core design decisions, architectural choices, and research insights are attributable to the human author, who maintains full responsibility for the code's integrity and the manuscript's content.

# References


