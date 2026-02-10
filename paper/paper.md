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
date: 9 February 2026
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

`Shesha` addresses three key challenges in current research workflows:

1. **Safety and Drift Monitoring**: Stability serves as a "geometric canary." In instruction-tuned models like Llama [@Touvron2023LLaMAOA], `Shesha` detects structural drift nearly 2x more sensitively than CKA (up to 5.23x) [@raju2026geometric], providing an earlier warning of functional degradation before task failure occurs. Unlike rigid distance metrics such as Procrustes [@Schnemann1966], `Shesha` filters out non-functional noise, resulting in 6x fewer false alarms in stable regimes [@raju2026geometric].

2. **The Geometric Tax in Vision**: In large-scale vision benchmarks, `Shesha` revealed a fundamental trade-off between task transferability and internal geometric robustness. State-of-the-art models like DINOv2 [@oquab2024dinov] often rank first in extrinsic similarity and downstream performance but last in intrinsic geometric stability, a phenomenon termed the "geometric tax" [@raju2026geometric]. This suggests that optimizing solely for rich, adaptable features can inadvertently collapse the fine-grained manifold structure required for predictable, consistent behavior.

3. **Biological Coherence**: Beyond machine learning, `Shesha` enables the analysis of single-cell perturbation data (e.g., CRISPR screens). It quantifies the structural coherence of transcriptomic shifts, where high stability scores identify perturbations that induce reproducible state-space trajectories rather than stochastic noise.

`Shesha` provides researchers with a unified API to compute these metrics across diverse domains, including language, vision, and biology. The package includes six interactive tutorials (each < 5 minutes to run) covering LLM embeddings, steering vectors, vision models, representational drift, training dynamics, and single-cell CRISPR analysis.

# State of the Field

Existing tools for representational analysis, such as `rsatoolbox` [@vandenBosch2025] for RSA, `scipy` [@2020SciPy-NMeth] for Procrustes, and various CKA implementations, focus primarily on measuring alignment between two different representations. While these are vital for model comparison, they lack native support for internal self-consistency testing—the ability to measure whether a model's own geometry remains stable under controlled perturbations.

`Shesha` fills this niche by providing specialized methods like `feature_split` (which partitions feature dimensions to test geometric coherence) and `anchor_stability` (which measures consistency across data subsamples). Unlike general-purpose manifold learning tools (e.g., `UMAP` [@McInnes2018], `scikit-learn` [@scikit-learn]), `Shesha` is specifically optimized for high-dimensional stability testing at scale. It also includes a dedicated `shesha.bio` module with high-level wrappers for `AnnData` objects [@Virshup2023] [@Virshup2024], making it directly compatible with the single-cell biology ecosystem used by tools like `scanpy` [@Wolf2018] and `pertpy` [@Heumos2025].

# Software Design

`Shesha` is built around a unified API that balances computational performance with research flexibility:

- **Modular Architecture**: The package is organized into three main components: core unsupervised metrics (e.g., `feature_split`, `rdm_drift`), supervised alignment methods (e.g., `anchored_stability`), and a specialized biological perturbation module (`shesha.bio`) for single-cell analysis.

- **Computational Efficiency**: Core representational dissimilarity matrix (RDM) operations are optimized for high-dimensional arrays using vectorized NumPy and SciPy routines. To ensure $O(n^2)$ pairwise distance calculations remain tractable for large datasets, the package implements intelligent subsampling strategies (via the `max_samples` parameter) that preserve statistical properties while reducing computational burden.

- **Ecosystem Integration**: The software relies on the standard scientific Python stack (`NumPy` [@harris2020array], `SciPy` [@2020SciPy-NMeth], `scikit-learn` [@scikit-learn]) and provides seamless integration with the bioinformatics ecosystem. The `shesha.bio` module works natively with `AnnData` [@Virshup2023] [@Virshup2024] objects and is compatible with `scanpy` [@Wolf2018] and `pertpy` [@Heumos2025] workflows, enabling researchers to incorporate stability analysis into existing single-cell pipelines without additional data preprocessing.

# Research Impact

`Shesha` has demonstrated impact across multiple research domains. In computer vision, the framework identified a fundamental "geometric tax" [@raju2026geometric]: a trade-off where state-of-the-art models like DINOv2 achieve high task transferability but exhibit lower geometric stability. This finding suggests that optimizing solely for rich feature representations may come at the cost of manifold consistency required for reliable, predictable behavior.

In computational biology, `Shesha` has been applied to the analysis of the Norman et al. (2019) CRISPRa datase [@norman2019exploring], where it successfully identified stable genetic regulators such as *KLF1* based on their manifold consistency scores. These findings demonstrate the framework's utility for distinguishing reproducible biological perturbations from stochastic transcriptomic noise.

The package is publicly available on PyPI as `shesha-geometry`, has been archived on Zenodo for long-term preservation, and features comprehensive documentation. Its focus on accessibility and usability is evidenced by six interactive Colab tutorials that allow researchers to explore LLM embeddings, steering vectors, vision model architectures, representational drift, training dynamics, and single-cell CRISPR analysis. All executable in under 5 minutes each.

# AI Usage Disclosure

In accordance with JOSS AI policy, the author discloses the use of generative AI during the development of this software and paper:

- **Software Creation & Refactoring**: Claude 4.5 (Sonnet/Opus) and Gemini 3 Pro were used to refactor experimental research scripts into a production-grade Python package, including implementation of design patterns, code optimization for readability, and generation of `pytest` scaffolding.
- **Debugging & Documentation**: AI assistants supported identification of edge cases in high-dimensional distance calculations and assisted in drafting docstrings and API reference material.
- **Formatting & Manuscript Editing**: AI tools helped with LaTeX/Markdown formatting and copy-editing of the paper text.
- **Visual Assets**: The software logo was generated using Nano Banana Pro.

**Confirmation of Review**: The author has reviewed, edited, and validated all AI-assisted outputs to ensure technical accuracy and adherence to the core geometric principles of the Shesha framework. All core design decisions, architectural choices, and research insights are attributable to the human author, who maintains full responsibility for the code's integrity and the manuscript's content.

# References
