Unsupervised Metrics
====================

These metrics require no labels and measure the internal geometric consistency
of a representation.

Feature split
-------------

:func:`shesha.feature_split` randomly partitions features into two halves, computes
an RDM for each half, and returns their Spearman correlation. High values indicate
that relational structure is distributed across the observed coordinate axes.

This is **not** a basis-invariant robustness score. Orthogonal rotations can
change ``feature_split`` while leaving pairwise Euclidean geometry unchanged.
See :doc:`caveats`.

.. code-block:: python

   import shesha
   stability = shesha.feature_split(X, n_splits=30, metric='cosine', seed=320)

Undefined distances raise by default (``nan_policy="raise"``). Sparse or
axis-aligned data can produce feature halves that are all zeros for some
samples; pass ``nan_policy="omit"`` to drop those pairs per split.

Uncertainty
-----------

``return_type="result"`` returns a :class:`shesha.StabilityResult` that
separates the sources of variation:

* ``estimate``: the full-data score (identical to the float return value).
* ``partition_sd`` and ``monte_carlo_se``: variability across random feature
  partitions, and the resulting Monte Carlo error of ``estimate``. Increase
  ``n_splits`` to reduce ``monte_carlo_se``.
* ``data_ci_low`` / ``data_ci_high``: an observation bootstrap
  (``n_bootstrap_ci``) that holds the full-data feature partitions fixed and
  drops pairs that refer to the same observation twice.

.. code-block:: python

   result = shesha.feature_split(
       X, n_splits=30, seed=320, n_bootstrap_ci=1000, return_type="result"
   )
   print(f"{result.estimate:.3f} ± {result.monte_carlo_se:.3f} (MC), "
         f"data CI [{result.data_ci_low:.3f}, {result.data_ci_high:.3f}]")

Rather than guessing whether ``n_splits=30`` is enough, set an accuracy
target. ``n_splits`` becomes the minimum, and partitions are drawn until the
Monte Carlo standard error is at most ``target_mc_se`` or ``max_splits``
(default ``max(n_splits, 500)``) is reached:

.. code-block:: python

   result = shesha.feature_split(
       X, n_splits=30, seed=320, target_mc_se=0.005, return_type="result"
   )
   print(result.n_valid_splits, result.metadata["converged"])

The older dictionary return (``n_bootstrap_ci`` without ``return_type``)
is unchanged; its ``mean`` key is the mean of bootstrap replicates, not the
full-data estimate. See :doc:`bootstrap_ci`.

Deprecated: sample split and anchor stability
---------------------------------------------

.. warning::

   :func:`shesha.sample_split` and :func:`shesha.anchor_stability` are
   **deprecated and not scientifically valid**. They correlate distances
   between different observations. Both emit a ``FutureWarning`` and will be
   removed in a future release. Use :func:`shesha.replicate_stability` and
   :func:`shesha.anchor_profile_stability` with matched replicates instead;
   see :doc:`replicate`.
