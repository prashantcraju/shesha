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

Sample split
------------

.. warning::

   :func:`shesha.sample_split` is **not scientifically valid** in its current
   form. It correlates unmatched RDM entries from independently drawn
   subsamples. Do not use the outputs for scientific inference. The function
   emits a ``FutureWarning`` and will be replaced in 0.3.0 by a
   matched-replicate API.

.. code-block:: python

   # Compatibility only — do not use for inference
   stability = shesha.sample_split(X, n_splits=30, subsample_fraction=0.4)

Anchor stability
----------------

.. warning::

   :func:`shesha.anchor_stability` is **not scientifically valid** in its
   current form. Corresponding columns in the two anchor-to-probe distance
   matrices refer to different observations. Do not use the outputs for
   scientific inference. The function emits a ``FutureWarning`` and will be
   replaced in 0.3.0 by a matched-observation API.

.. code-block:: python

   # Compatibility only — do not use for inference
   stability = shesha.anchor_stability(X, n_splits=30, n_anchors=100)

Bootstrap confidence intervals
------------------------------

``feature_split`` supports optional bootstrap CIs via ``n_bootstrap_ci``.
See :doc:`bootstrap_ci` for full details. CIs on ``sample_split`` and
``anchor_stability`` are available for compatibility but inherit the same
invalid estimand.

.. code-block:: python

   result = shesha.feature_split(X, n_splits=30, seed=320, n_bootstrap_ci=1000)
   print(f"{result['mean']:.3f} [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
