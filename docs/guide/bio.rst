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

Discordance
-----------

:func:`shesha.bio.discordance` identifies perturbations that deviate from the expected
stability-magnitude relationship. High discordance scores flag perturbations that are
less stable than expected given their effect size — candidates for pleiotropic or
heterogeneous effects.

Three methods are available:

- **linear** (default): OLS residual, sign-flipped and z-scored. Fast and interpretable.
- **rank**: rank(Mp) - rank(Sp), z-scored. Non-parametric; robust to outliers.
- **loess**: Local regression (LOWESS) residual, sign-flipped and z-scored.
  Captures nonlinear magnitude-stability trends where the relationship curves at low
  magnitudes. Requires ``statsmodels``.

.. code-block:: python

   from shesha.bio import discordance

   # Linear (default)
   df["disc_linear"] = discordance(df, stability_col="Sp", magnitude_col="Mp")

   # LOESS — captures nonlinear curvature
   df["disc_loess"] = discordance(
       df,
       stability_col="Sp",
       magnitude_col="Mp",
       method="loess",
       loess_frac=0.3,
   )

   # Top 10 most discordant perturbations
   print(df.nlargest(10, "disc_loess")[["Sp", "Mp", "disc_loess"]])

The ``loess_frac`` parameter controls smoothness: smaller values (e.g. 0.2) follow the
data more closely, while larger values (e.g. 0.5) produce smoother expected curves.
The default of 0.3 balances sensitivity and stability for typical CRISPR screen sizes.

.. note::

   ``method='loess'`` requires `statsmodels <https://www.statsmodels.org>`_.
   Install with ``pip install statsmodels``.
