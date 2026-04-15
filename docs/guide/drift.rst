Drift Metrics
=============

Drift metrics compare two representations ``X`` and ``Y`` (e.g. before/after fine-tuning).

RDM similarity
--------------

:func:`shesha.rdm_similarity` computes the Spearman correlation between the RDMs of
``X`` and ``Y``. Returns a value in [-1, 1]; higher means more similar geometry.

.. code-block:: python

   similarity = shesha.rdm_similarity(X_before, X_after)

RDM drift
---------

:func:`shesha.rdm_drift` is simply ``1 - rdm_similarity``. Returns a value in [0, 2];
higher means more representational change.

.. code-block:: python

   drift = shesha.rdm_drift(X_before, X_after)

Monitoring fine-tuning
----------------------

.. code-block:: python

   import shesha

   X_initial = model.encode(data)
   for epoch in range(10):
       train_one_epoch(model)
       X_current = model.encode(data)
       stability = shesha.feature_split(X_current, seed=320)
       drift = shesha.rdm_drift(X_initial, X_current)
       print(f"Epoch {epoch}: stability={stability:.3f}, drift={drift:.3f}")
