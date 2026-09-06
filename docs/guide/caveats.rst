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
retained for compatibility, but their current estimands are **not
scientifically identifiable**. They correlate distance entries that do
not refer to the same observation pairs.

Both functions emit a :class:`FutureWarning` on every call. Do not use
their outputs for scientific inference. shesha-geometry 0.3.0 will
replace them with matched-replicate APIs.

Bootstrap confidence intervals on these two functions inherit the same
invalid estimand.

Undefined distances
-------------------

Cosine and correlation distances are undefined for zero or constant
vectors. In 0.2.29 the default ``nan_policy='replace'`` still fills those
entries with ``1.0`` so existing pipelines keep their numbers. That
replacement can invent structure that was not present in the data.

When a replacement actually occurs, Shesha emits a :class:`FutureWarning`.
The default will change to ``nan_policy='raise'`` in 0.3.0. Prefer
``raise``, ``omit``, or ``propagate`` now if you want the safer behavior.
