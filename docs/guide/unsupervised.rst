Unsupervised Metrics
====================

These metrics require no labels and measure the internal geometric consistency
of a representation.

Feature split
-------------

:func:`shesha.feature_split` randomly partitions features into two halves, computes
an RDM for each half, and returns their Spearman correlation. High values indicate
that the geometry is robust to which features you use — a sign of stable structure.

.. code-block:: python

   import shesha
   stability = shesha.feature_split(X, n_splits=30, metric='cosine', seed=320)

Sample split
------------

:func:`shesha.sample_split` bootstraps the sample dimension instead. It measures
robustness to which data points are observed.

.. code-block:: python

   stability = shesha.sample_split(X, n_splits=30, subsample_fraction=0.4)

Anchor stability
----------------

:func:`shesha.anchor_stability` computes distance profiles from a fixed set of anchor
points and correlates them across two bootstrap draws. Suitable for very large datasets.

.. code-block:: python

   stability = shesha.anchor_stability(X, n_splits=30, n_anchors=100)
