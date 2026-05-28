Bootstrap Confidence Intervals
==============================

Every public metric in Shesha supports an optional **outer bootstrap** for
computing confidence intervals on the point estimate. Instead of returning a
single float, the function returns a dictionary with the mean, lower/upper CI
bounds, standard deviation, and metadata.

How it works
------------

1. The input data is resampled **with replacement** (rows/samples).
2. The metric is recomputed on each resampled dataset.
3. Percentile-based confidence intervals are computed from the distribution of
   bootstrap estimates.

This "outer bootstrap" is independent of any internal iterations the metric
already performs (e.g. ``n_splits`` in ``feature_split``). It quantifies
uncertainty due to the *finite sample* of observations.

Usage
-----

Pass ``n_bootstrap_ci`` (number of resamples) and optionally ``ci`` (confidence
level, default 0.95) to any metric function:

.. code-block:: python

   import shesha

   X = np.random.randn(500, 768)

   # Point estimate (default behaviour, returns float)
   stability = shesha.feature_split(X, n_splits=30, seed=320)

   # With 95% bootstrap CI (returns dict)
   result = shesha.feature_split(X, n_splits=30, seed=320, n_bootstrap_ci=1000)
   print(result)
   # {'mean': 0.42, 'ci_low': 0.38, 'ci_high': 0.46,
   #  'std': 0.021, 'n_bootstraps': 1000, 'ci_level': 0.95}

   # 99% CI
   result_99 = shesha.feature_split(X, n_splits=30, seed=320,
                                     n_bootstrap_ci=1000, ci=0.99)

Return format
-------------

When ``n_bootstrap_ci`` is set, the function returns a dictionary:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Key
     - Description
   * - ``mean``
     - Mean of the bootstrap distribution
   * - ``ci_low``
     - Lower bound of the confidence interval
   * - ``ci_high``
     - Upper bound of the confidence interval
   * - ``std``
     - Standard deviation of bootstrap estimates
   * - ``n_bootstraps``
     - Number of successful resamples (may be < ``n_bootstrap_ci`` if some
       resamples yield NaN)
   * - ``ci_level``
     - The confidence level used (e.g. 0.95)

Examples across modules
-----------------------

**Core (unsupervised)**

.. code-block:: python

   result = shesha.feature_split(X, n_splits=30, n_bootstrap_ci=1000, seed=320)
   result = shesha.sample_split(X, n_splits=30, n_bootstrap_ci=1000, seed=320)
   result = shesha.anchor_stability(X, n_bootstrap_ci=1000, seed=320)

**Core (supervised)**

.. code-block:: python

   result = shesha.variance_ratio(X, y, n_bootstrap_ci=1000, seed=320)
   result = shesha.supervised_alignment(X, y, n_bootstrap_ci=1000, seed=320)
   result = shesha.class_separation_ratio(X, y, n_bootstrap_ci=1000, seed=320)
   result = shesha.lda_stability(X, y, n_bootstrap_ci=1000, seed=320)

**Core (drift)**

.. code-block:: python

   result = shesha.rdm_similarity(X, Y, n_bootstrap_ci=1000, seed=320)
   result = shesha.rdm_drift(X, Y, n_bootstrap_ci=1000, seed=320)

**Bio (perturbation analysis)**

.. code-block:: python

   from shesha.bio import perturbation_stability, perturbation_effect_size

   result = perturbation_stability(X_ctrl, X_pert, n_bootstrap_ci=1000, seed=320)
   result = perturbation_effect_size(X_ctrl, X_pert, n_bootstrap_ci=1000, seed=320)

**Sim (similarity metrics)**

.. code-block:: python

   from shesha.sim import cka, cka_linear, cka_debiased
   from shesha.sim import procrustes_similarity, rdm_similarity

   result = cka(X, Y, n_bootstrap_ci=1000, seed=320)
   result = cka_linear(X, Y, n_bootstrap_ci=1000, seed=320)
   result = cka_debiased(X, Y, n_bootstrap_ci=1000, seed=320)
   result = procrustes_similarity(X, Y, n_bootstrap_ci=1000, seed=320)
   result = rdm_similarity(X, Y, n_bootstrap_ci=1000, seed=320)

Choosing ``n_bootstrap_ci``
---------------------------

- **Quick exploration**: 200–500 resamples
- **Publication-quality**: 1000–10000 resamples
- **Computational cost**: scales linearly with ``n_bootstrap_ci``. Each resample
  runs the full metric computation, so expensive metrics (e.g.
  ``anchor_stability`` on large data) will take proportionally longer.

Resampling strategy
-------------------

- **Single-matrix metrics** (``feature_split``, ``sample_split``, etc.): rows of
  ``X`` (and ``y`` if supervised) are resampled together with the same indices.
- **Two-matrix metrics** (``rdm_similarity``, ``cka``, etc.): both ``X`` and
  ``Y`` are resampled with the **same** indices (paired bootstrap).
- **Bio metrics** (``perturbation_stability``, ``perturbation_effect_size``):
  control and perturbed populations are resampled **independently**.

Reproducibility
---------------

Pass ``seed`` for deterministic results:

.. code-block:: python

   result1 = shesha.feature_split(X, n_bootstrap_ci=1000, seed=320)
   result2 = shesha.feature_split(X, n_bootstrap_ci=1000, seed=320)
   assert result1 == result2  # identical
