Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install shesha-geometry

Feature-split stability (unsupervised)
---------------------------------------

.. code-block:: python

   import numpy as np
   import shesha

   X = np.random.randn(500, 768)
   stability = shesha.feature_split(X, n_splits=30, seed=320)
   print(f"Feature-split stability: {stability:.3f}")

With labels
-----------

.. code-block:: python

   y = np.random.randint(0, 10, 500)
   alignment = shesha.supervised_alignment(X, y)
   print(f"Supervised alignment: {alignment:.3f}")

Measuring drift
---------------

.. code-block:: python

   X_before = np.random.randn(100, 256)
   X_after = X_before + np.random.randn(100, 256) * 0.3

   similarity = shesha.rdm_similarity(X_before, X_after)
   drift = shesha.rdm_drift(X_before, X_after)
   print(f"RDM similarity: {similarity:.3f}, drift: {drift:.3f}")
