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
