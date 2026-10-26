Matched-Replicate Stability
===========================

A single data matrix cannot show whether geometry generalizes to new
measurements through an RDM correlation. That needs **matched observations**:
the same stimuli, cell conditions, or test inputs measured twice, for
example in two recording sessions, two biological replicates, or two
fitted models evaluated on a shared test set.

These estimators replace :func:`shesha.sample_split` and
:func:`shesha.anchor_stability`. Those functions correlated distances
between *different* observations (see :doc:`caveats`).

Replicate stability
-------------------

:func:`shesha.replicate_stability` correlates the RDM of replicate A with
the RDM of replicate B. Row ``i`` of both inputs must represent the same
observation, so every compared distance refers to the same pair.

.. code-block:: python

   import shesha

   score = shesha.replicate_stability(X_rep1, X_rep2, metric="cosine")

If the replicates are stored in different orders, pass identifiers. Both
identifier sets must be unique and identical; ``X_b`` is reordered to match
``X_a``. Without identifiers, rows are assumed to correspond, and a
misaligned order cannot be detected.

.. code-block:: python

   score = shesha.replicate_stability(
       X_rep1, X_rep2, ids_a=stimulus_ids_rep1, ids_b=stimulus_ids_rep2
   )

Anchor-profile stability
------------------------

:func:`shesha.anchor_profile_stability` fixes a set of anchor observations
and a disjoint set of probe observations. For each anchor it correlates the
anchor-to-probe distance profile in replicate A with the same profile in
replicate B, then averages over anchors. Anchors and probes are selected by
identifier (or by row index when no identifiers are given).

.. code-block:: python

   score = shesha.anchor_profile_stability(
       X_rep1,
       X_rep2,
       anchor_ids=reference_ids,
       probe_ids=probe_ids,  # defaults to every non-anchor observation
       ids_a=ids_rep1,
       ids_b=ids_rep2,
   )

Structured results and uncertainty
----------------------------------

Pass ``return_type="result"`` to get a :class:`shesha.StabilityResult`.
Its ``estimate`` is always the full-data value, never the mean of bootstrap
replicates. A data interval requires ``return_type="result"``:

.. code-block:: python

   result = shesha.replicate_stability(
       X_rep1, X_rep2, n_bootstrap_ci=1000, seed=320, return_type="result"
   )
   print(result.estimate, result.data_ci_low, result.data_ci_high)
   result.to_dict()  # JSON-serializable, including configuration metadata

* ``replicate_stability`` resamples observations. Pairs that refer to the
  same observation twice are dropped from each resample, because their zero
  self-distance would appear in both RDMs and inflate agreement.
* ``anchor_profile_stability`` resamples probes and treats anchors as fixed
  reference observations. ``component_scores`` holds the per-anchor
  correlations.

Undefined distances
-------------------

Both estimators default to ``nan_policy="raise"``. ``omit`` drops entries
undefined in either replicate (anchors whose whole profile is undefined are
excluded and reported in ``warnings``), ``propagate`` returns NaN, and
``replace`` fills them with ``1.0`` and warns.
