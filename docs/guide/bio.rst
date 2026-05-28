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
