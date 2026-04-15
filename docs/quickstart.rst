Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install shesha-geometry

Feature-split stability (unsupervised)
---------------------------------------

.. code-block:: python

   import numpy as np
   import shesha

   X = np.random.randn(500, 768)
   stability = shesha.feature_split(X, n_splits=30, seed=320)
   print(f"Feature-split stability: {stability:.3f}")

With labels
-----------

.. code-block:: python

   y = np.random.randint(0, 10, 500)
   alignment = shesha.supervised_alignment(X, y)
   print(f"Supervised alignment: {alignment:.3f}")

Measuring drift
---------------

.. code-block:: python

   X_before = np.random.randn(100, 256)
   X_after = X_before + np.random.randn(100, 256) * 0.3

   similarity = shesha.rdm_similarity(X_before, X_after)
   drift = shesha.rdm_drift(X_before, X_after)
   print(f"RDM similarity: {similarity:.3f}, drift: {drift:.3f}")

Single-cell perturbation analysis (Bio)
----------------------------------------

For CRISPR or drug screens stored as `AnnData <https://anndata.readthedocs.io>`_ objects:

.. code-block:: python

   import numpy as np
   from anndata import AnnData
   from shesha.bio import compute_stability, compute_magnitude

   # Mock single-cell data: 1000 cells, 50 PCA features
   n_cells, n_pcs = 1000, 50
   adata = AnnData(X=np.random.randn(n_cells, n_pcs))
   adata.obs['guide_id'] = ['NT'] * 800 + ['KLF1'] * 200

   # Compute geometric consistency of each perturbation
   stability = compute_stability(
       adata,
       perturbation_key='guide_id',
       control_label='NT',
       metric='cosine',
   )

   # Compute effect size relative to control
   magnitude = compute_magnitude(
       adata,
       perturbation_key='guide_id',
       control_label='NT',
       metric='euclidean',
   )

   print(f"KLF1 stability: {stability['KLF1']:.3f}")  # e.g. 0.85 — consistent phenotype
   print(f"KLF1 magnitude: {magnitude['KLF1']:.3f}")  # e.g. 2.40 — strong effect
