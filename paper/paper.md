---
title: 'Shesha: Self-Consistency Metrics for Representational Stability Analysis'
tags:
  - Python
  - machine learning
  - representation learning
  - single-cell biology
  - interpretability
authors:
  - name: Prashant C. Raju
    orcid: 0000-0003-3778-4788
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 15 March 2026
bibliography: paper.bib
---

# Summary

Analysis of learned representations has historically focused on **similarity**, measuring how closely internal embeddings align with external references. However, similarity only reveals *what* is represented, not whether that structure is robust to internal or external perturbations. `Shesha` is an open-source Python framework designed to quantify **geometric stability**, a distinct dimension of representational quality that measures how reliably a model's internal geometry holds under perturbation [@raju2026geometric].

# Mathematical Framework

`Shesha` measures geometric stability by quantifying the self-consistency of pairwise distance structure under controlled internal perturbations. Given a representation matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ with $n$ samples and $d$ features, a Representational Dissimilarity Matrix (RDM) [@Kriegeskorte2008] $\mathbf{D} \in \mathbb{R}^{n \times n}$ captures pairwise dissimilarities:

$$D_{ij} = 1 - \frac{\mathbf{x}_i^\top \mathbf{x}_j}{\|\mathbf{x}_i\| \|\mathbf{x}_j\|}$$

The core principle is to construct two RDMs from complementary "views" of $\mathbf{X}$ and measure their agreement via Spearman rank correlation:

$$\text{Shesha}(\mathbf{X}) = \rho_s\bigl(\text{vec}(\mathbf{D}^{(1)}), \text{vec}(\mathbf{D}^{(2)})\bigr)$$

where $\text{vec}(\cdot)$ extracts the upper triangular elements. The choice of how views are constructed defines distinct variants.

## Core Variants

**Feature-Split Shesha** (`feature_split`) assesses internal geometric consistency by partitioning feature dimensions into random disjoint halves:

$$\text{Shesha}_{\text{FS}}(\mathbf{X}) = \frac{1}{K} \sum_{k=1}^{K} \rho_s\bigl(\text{vec}(\mathbf{D}_{\mathcal{F}_k^{(1)}}), \text{vec}(\mathbf{D}_{\mathcal{F}_k^{(2)}})\bigr)$$

where $\mathcal{F}_k^{(1)}, \mathcal{F}_k^{(2)} \subset \{1, \ldots, d\}$ are disjoint feature partitions at split $k$, averaged over $K=30$ random splits. High values indicate that geometric structure is evenly distributed across dimensions rather than concentrated in a few dominant features.

**Sample-Split Shesha** (`sample_split`) measures robustness to data variation by partitioning samples into disjoint subsets and comparing RDMs computed on each. This variant assesses whether the distance structure generalizes across different subsets of the data.

**Supervised Shesha** (`supervised_alignment`) quantifies task-aligned stability when class labels $\mathbf{y}$ are available:

$$\text{Shesha}_{\text{sup}}(\mathbf{X}, \mathbf{y}) = \rho_s\bigl(\text{vec}(\mathbf{D}_\mathbf{X}), \text{vec}(\mathbf{D}_\mathbf{y})\bigr)$$

where $\mathbf{D}_\mathbf{y}$ encodes label dissimilarity. Additional supervised metrics include the variance ratio (between-class to total variance), class separation ratio (between-class to within-class distances), and LDA subspace stability (consistency of discriminant direction under resampling).

**Perturbation Stability** (`perturbation_stability` in `shesha.bio`) measures directional consistency of biological perturbation effects:

$$\text{Shesha}_{\text{pert}} = \frac{1}{|\mathcal{P}|} \sum_{i \in \mathcal{P}} \frac{(\mathbf{x}_i - \boldsymbol{\mu}_{\text{ctrl}})^\top \bar{\mathbf{d}}}{\|\mathbf{x}_i - \boldsymbol{\mu}_{\text{ctrl}}\| \|\bar{\mathbf{d}}\|}$$

where $\bar{\mathbf{d}}$ is the mean shift direction from control to perturbed population. High values indicate coherent, reproducible perturbation effects rather than stochastic noise.

All variants use Spearman correlation by default (robust to monotonic transformations and outliers) and subsample to $n_{\max}=1600$ when computational constraints require. Implementations are available in `shesha.core` (general ML) and `shesha.bio` (single-cell biology with native `AnnData` support).

# Statement of Need

Geometric stability is an **intrinsic property** of a representation's manifold that measures how consistently it preserves internal geometric structure across varied feature subsets or under perturbation. Unlike similarity metrics such as Centered Kernel Alignment (CKA) [@kornblith2019similarity] and Representational Similarity Analysis (RSA) [@Kriegeskorte2008]—which are **extrinsic properties** measuring how one representation aligns with another—stability is reference-independent. It asks whether the internal "geometry" of the representational space is robust or brittle.

## Stability vs. Similarity: A Mechanistic Distinction

A library analogy illustrates this distinction: suppose every book in a library is randomly moved to different shelves while the overall catalog remains unchanged. An inventory check would verify that everything is still present and the collection is unchanged. However, the organizational structure that allows readers to locate specific books has collapsed. The books that were once grouped carefully are now scattered unpredictably. Similarity metrics function like the inventory check by verifying that representational content **exists**. Geometric stability, on the other hand, plays the role of the index by checking whether the functional relationships that make features discoverable and interpretable remains **coherent**.

This distinction is mechanistically grounded: similarity metrics often collapse when the top principal components (the dominant variance) are removed, whereas geometric stability retains sensitivity to fine-grained manifold structure even after filtering out the dominant axes. Empirically, stability and similarity are uncorrelated ($\rho \approx 0.01$) across diverse domains [@raju2026geometric], meaning a model can appear highly similar to a reference while having a completely degraded internal geometry.

## Critical Gaps Addressed by Shesha

`Shesha` addresses three key tooling challenges in current research workflows:

1. **Safety and Drift Monitoring**: provides a robust, noise-filtered metric for monitoring structural drift in large language models (LLMs) [@Touvron2023LLaMAOA]. By filtering out non-functional variance, the software acts as a highly sensitive "geometric canary" that overcomes the limitations of rigid distance metrics like Procrustes [@Schnemann1966], allowing researchers to programmatically detect early functional degradation.

2. **Large-Scale Vision Benchmarks**: Evaluating the intrinsic geometric robustness of state-of-the-art vision models (e.g., DINOv2 [@oquab2024dinov]) requires processing massive, high-dimensional datasets. `Shesha` is architected to scale efficiently, enabling researchers to systematically quantify the relationship between task transferability and internal manifold consistency across extensive model suites.

3. **Biological Coherence**: Beyond machine learning, the `shesha.bio` module enables the native processing of single-cell perturbation data (e.g., CRISPR screens). The software quantifies the structural coherence of transcriptomic shifts, equipping biologists with an accessible tool to isolate reproducible state-space trajectories from stochastic noise.

`Shesha` provides researchers with a unified API to compute these metrics across diverse domains, including language, vision, and biology. The package includes six interactive tutorials (each < 5 minutes to run) covering LLM embeddings, steering vectors, vision models, representational drift, training dynamics, and single-cell CRISPR analysis.

# State of the Field

Existing tools for representational analysis, such as `rsatoolbox` [@vandenBosch2025] for RSA, `scipy` [@2020SciPy-NMeth] for Procrustes, and various CKA implementations, focus primarily on measuring alignment between two different representations. While these are vital for model comparison, they lack native support for internal self-consistency testing—the ability to measure whether a model's own geometry remains stable under controlled perturbations.

`Shesha` fills this niche by providing specialized methods like `feature_split` (which partitions feature dimensions to test geometric coherence) and `anchor_stability` (which measures consistency across data subsamples). Unlike general-purpose manifold learning tools (e.g., `UMAP` [@McInnes2018], `scikit-learn` [@scikit-learn]), `Shesha` is specifically optimized for high-dimensional stability testing at scale. It also includes a dedicated `shesha.bio` module with high-level wrappers for `AnnData` objects [@Virshup2023] [@Virshup2024], making it directly compatible with the single-cell biology ecosystem used by tools like `scanpy` [@Wolf2018] and `pertpy` [@Heumos2025].

# Software Design

`Shesha` is built around a unified API that balances computational performance with research flexibility:

- **Modular Architecture**: The package is organized into three main components: core unsupervised metrics (e.g., `feature_split`, `rdm_drift`), supervised alignment methods (e.g., `anchored_stability`), and a specialized biological perturbation module (`shesha.bio`) for single-cell analysis.

- **Computational Efficiency**: Core representational dissimilarity matrix (RDM) operations are optimized for high-dimensional arrays using vectorized NumPy and SciPy routines. To ensure $O(n^2)$ pairwise distance calculations remain tractable for large datasets, the package implements intelligent subsampling strategies (via the `max_samples` parameter) that preserve statistical properties while reducing computational burden.

- **Ecosystem Integration**: The software relies on the standard scientific Python stack (`NumPy` [@harris2020array], `SciPy` [@2020SciPy-NMeth], `scikit-learn` [@scikit-learn]) and provides seamless integration with the bioinformatics ecosystem. The `shesha.bio` module works natively with `AnnData` [@Virshup2023] [@Virshup2024] objects and is compatible with `scanpy` [@Wolf2018] and `pertpy` [@Heumos2025] workflows, enabling researchers to incorporate stability analysis into existing single-cell pipelines without additional data preprocessing.

# Applications

`Shesha` is designed to be highly adaptable across diverse research domains. In computer vision and natural language processing, researchers utilize `Shesha`'s core unsupervised metrics to evaluate manifold consistency across varied model architectures. The framework enables the systematic comparison of how different training paradigms impact the internal geometric robustness of a model, independently of its performance on external similarity benchmarks.

In computational biology, the `shesha.bio` module natively interfaces with AnnData objects to streamline the analysis of large-scale single-cell datasets, such as the Norman et al. (2019) CRISPRa dataset [@norman2019exploring]. The framework provides specialized functions to evaluate the stability of genetic regulators, allowing researchers to programmatically distinguish reproducible biological perturbations from stochastic transcriptomic noise without requiring complex, out-of-ecosystem preprocessing pipelines.

The package is publicly available on PyPI as `shesha-geometry`, has been archived on Zenodo for long-term preservation, and features comprehensive documentation. Its focus on accessibility and usability is evidenced by six interactive Colab tutorials that allow researchers to explore LLM embeddings, steering vectors, vision model architectures, representational drift, training dynamics, and single-cell CRISPR analysis. All tutorials are designed to be executable in under 5 minutes each.


# AI Usage Disclosure

In accordance with JOSS AI policy, the author discloses the use of generative AI during the development of this software and paper:

- **Software Creation & Refactoring**: Claude 4.5 (Sonnet/Opus) and Gemini 3 Pro were used to refactor experimental research scripts into a production-grade Python package, including implementation of design patterns, code optimization for readability, and generation of `pytest` scaffolding.
- **Debugging & Documentation**: AI assistants supported identification of edge cases in high-dimensional distance calculations and assisted in drafting docstrings and API reference material.
- **Formatting & Manuscript Editing**: AI tools helped with LaTeX/Markdown formatting and copy-editing of the paper text.
- **Visual Assets**: The software logo was generated using Nano Banana Pro.

**Confirmation of Review**: The author has reviewed, edited, and validated all AI-assisted outputs to ensure technical accuracy and adherence to the core geometric principles of the Shesha framework. All core design decisions, architectural choices, and research insights are attributable to the human author, who maintains full responsibility for the code's integrity and the manuscript's content.

# References
