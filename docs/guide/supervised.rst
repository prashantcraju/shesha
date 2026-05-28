Supervised Metrics
==================

These metrics require class labels ``y`` alongside the embedding matrix ``X``.

Variance ratio
--------------

:func:`shesha.variance_ratio` computes the between-class to total variance ratio
(analogous to the F-statistic). Quick proxy for linear separability.

.. code-block:: python

   score = shesha.variance_ratio(X, y)

Supervised alignment
--------------------

:func:`shesha.supervised_alignment` correlates the empirical RDM with the ideal label
RDM (same-class = 0, different-class = 1). High values mean geometry respects labels.

.. code-block:: python

   alignment = shesha.supervised_alignment(X, y)

Class separation ratio
----------------------

:func:`shesha.class_separation_ratio` estimates the ratio of between-class to
within-class pairwise distances via bootstrap resampling.

.. code-block:: python

   ratio = shesha.class_separation_ratio(X, y, n_bootstrap=50)

LDA stability
-------------

:func:`shesha.lda_stability` measures how consistently LDA finds the same discriminant
direction across bootstrap resamplings of the data.

.. code-block:: python

   stability = shesha.lda_stability(X, y, n_bootstrap=50)

Bootstrap confidence intervals
------------------------------

All supervised metrics support optional bootstrap CIs via ``n_bootstrap_ci``.
See :doc:`bootstrap_ci` for full details.

.. code-block:: python

   result = shesha.variance_ratio(X, y, n_bootstrap_ci=1000, seed=320)
   print(f"{result['mean']:.3f} [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
