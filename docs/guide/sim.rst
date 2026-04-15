Similarity Metrics
==================

``shesha.sim`` provides extrinsic similarity metrics — measuring how well two
representations align with each other. These are distinct from (and empirically
uncorrelated with) the intrinsic stability metrics in the core module.

CKA
---

:func:`shesha.sim.cka` computes Centered Kernel Alignment. Invariant to orthogonal
transformations and isotropic scaling; widely used for comparing neural network layers.

.. code-block:: python

   import shesha.sim as sim

   similarity = sim.cka(layer_a, layer_b)
   similarity_debiased = sim.cka(layer_a, layer_b, debiased=True)

Procrustes similarity
---------------------

:func:`shesha.sim.procrustes_similarity` finds the best orthogonal alignment between
two representations and returns the residual similarity. More sensitive to outliers
than CKA (~6× more false alarms in stable regimes).

.. code-block:: python

   similarity = sim.procrustes_similarity(X, Y)

RDM similarity (sim module)
----------------------------

:func:`shesha.sim.rdm_similarity` is an RSA-style RDM correlation. Rank-based and
robust to monotone transformations.

.. code-block:: python

   similarity = sim.rdm_similarity(X, Y, metric='cosine', method='spearman')
