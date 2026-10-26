What Shesha does not establish
==============================

Shesha scores are self-consistency statistics. They are useful, but they
are easy to over-interpret. This page states the limits that matter for
scientific use.

What ``feature_split`` measures
-------------------------------

:func:`shesha.feature_split` measures how consistently pairwise relations
are distributed across the **observed coordinate axes**. High values mean
the same relational structure is recoverable from random feature halves.

That is a narrower claim than "the representation is geometrically robust":

* It is **coordinate-basis dependent**. An orthogonal rotation can change
  the score even when pairwise Euclidean geometry is unchanged.
* It does **not** measure basis-invariant geometric robustness.
* It does **not** measure information retention, latent-feature
  recoverability, or functional preservation.

What high or low scores do not mean
-----------------------------------

* High Shesha does **not** establish that the representation contains
  correct information.
* Low Shesha does **not** establish information loss.
* Shesha does **not** establish causal use of a representation.
* Orthogonal transformations preserve an RDM but can change feature-split
  Shesha.
* Stability, similarity, decodability, and causal efficacy must be
  reported separately.

Invalid unmatched-sample estimators
-----------------------------------

:func:`shesha.sample_split` and :func:`shesha.anchor_stability` are
deprecated. Their estimands are **not scientifically identifiable**: they
correlate distance entries that do not refer to the same observation pairs.

Both functions emit a :class:`FutureWarning` on every call and will be
removed in a future release. Do not use their outputs, or confidence
intervals on them, for scientific inference. Use the matched-replicate
estimators in :doc:`replicate` instead.

Undefined distances
-------------------

Cosine and correlation distances are undefined for zero or constant
vectors. Since 0.3.0 every RDM path defaults to ``nan_policy='raise'``.
``replace`` fills undefined entries with ``1.0`` and emits a
:class:`UserWarning`; it treats those pairs as maximally dissimilar and can
invent structure that was not present in the data. Prefer ``omit`` or
``propagate``.

``propagate`` makes an estimator undefined if any required distance is
undefined. ``compute_rdm`` cannot omit individual distances while retaining
valid condensed-RDM indexing, so it raises if ``omit`` would remove entries.
