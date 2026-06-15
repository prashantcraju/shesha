Biological Perturbation Analysis
=================================

``shesha.bio`` provides metrics for single-cell and CRISPR perturbation experiments.
It works natively with `AnnData <https://anndata.readthedocs.io>`_ objects.

Compute stability
-----------------

:func:`shesha.bio.compute_stability` measures per-perturbation geometric consistency
relative to a control population.

.. code-block:: python

   from shesha.bio import compute_stability

   stability = compute_stability(
       adata_pca,
       perturbation_key='guide_id',
       control_label='NT',
       metric='cosine',
   )
   print(stability['KLF1'])   # e.g. 0.85

Compute magnitude
-----------------

:func:`shesha.bio.compute_magnitude` measures the average distance of perturbed cells
from the centroid of the control population.

.. code-block:: python

   from shesha.bio import compute_magnitude

   magnitude = compute_magnitude(
       adata_pca,
       perturbation_key='guide_id',
       control_label='NT',
       metric='euclidean',
   )
   print(magnitude['KLF1'])   # e.g. 2.40

Bootstrap confidence intervals
------------------------------

The low-level functions :func:`~shesha.bio.perturbation_stability` and
:func:`~shesha.bio.perturbation_effect_size` support bootstrap CIs via
``n_bootstrap_ci``. Control and perturbed populations are resampled independently.
See :doc:`bootstrap_ci` for full details.

.. code-block:: python

   from shesha.bio import perturbation_stability

   result = perturbation_stability(X_ctrl, X_pert, n_bootstrap_ci=1000, seed=320)
   print(f"{result['mean']:.3f} [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")

Split-half reproducibility
--------------------------

:func:`shesha.bio.split_half_reproducibility` measures effect-direction reproducibility
for each perturbation by repeated 50/50 random cell splits. High values indicate that the
perturbation's directional shift is consistent across independent subsets of cells — a
direct assay of biological reproducibility that is distinct from effect magnitude.

.. code-block:: python

   from shesha.bio import split_half_reproducibility

   repro = split_half_reproducibility(
       adata,
       perturbation_key="perturbation",
       control_label="control",
       n_splits=50,
       random_state=320,
   )
   # Returns DataFrame: index=perturbation, columns=[split_half_cosine, n_cells]
   print(repro.sort_values("split_half_cosine", ascending=False).head())

Magnitude-matched comparison
-----------------------------

:func:`shesha.bio.magnitude_matched_comparison` tests whether stability predicts
reproducibility *within* magnitude bins, controlling for the SNR confound. Perturbations
are binned by effect size and, within each bin, the mean split-half cosine is compared
between the high-stability and low-stability halves.

.. code-block:: python

   from shesha.bio import compute_stability, compute_magnitude, magnitude_matched_comparison
   import pandas as pd

   sp = compute_stability(adata, perturbation_key="perturbation", control_label="control")
   mp = compute_magnitude(adata, perturbation_key="perturbation", control_label="control")

   df = repro.copy()
   df["Sp"] = pd.Series(sp)
   df["Mp"] = pd.Series(mp)

   bins = magnitude_matched_comparison(
       df,
       stability_col="Sp",
       repro_col="split_half_cosine",
       magnitude_col="Mp",
       n_bins=4,
   )
   print(bins[["mag_bin", "n", "high_stability_mean", "low_stability_mean", "difference"]])
