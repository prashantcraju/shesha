"""
Example: Comparing Stability vs. Similarity

This synthetic example illustrates the key distinction from the paper:
feature-split stability and cross-representation similarity measure different
properties. It does not estimate their population-level correlation.

A model can have:
- High similarity (aligns well with reference)
- Low stability (brittle internal geometry)

The examples use controlled constructions whose expected properties can be
checked directly.
"""

import numpy as np

import shesha
import shesha.sim as sim


def create_representations():
    """Create controlled representations with known metric relationships."""
    rng = np.random.default_rng(320)
    n_samples, n_features, n_latent = 400, 80, 5

    # Distributed redundant encoding: random feature halves recover the same
    # latent relational structure, so feature_split should be high.
    latent = rng.standard_normal((n_samples, n_latent))
    projection = rng.standard_normal((n_latent, n_features))
    X_reference = latent @ projection
    X_reference += 0.01 * rng.standard_normal(X_reference.shape)

    # Scenario 1: an arbitrary orthogonal rotation preserves CKA, Euclidean
    # RDMs, and Procrustes alignment.
    Q = np.linalg.qr(rng.standard_normal((n_features, n_features)))[0]
    X_rotated = X_reference @ Q

    # Scenario 2: rotate into the principal-axis basis. The representation has
    # exactly the same geometry, but information is concentrated in a handful
    # of coordinates, so feature_split becomes low.
    _, _, Vt = np.linalg.svd(X_reference, full_matrices=False)
    X_concentrated = X_reference @ Vt.T

    # Scenario 3: an independent latent geometry with the same distributed,
    # redundant encoding pattern. It is internally stable but dissimilar to
    # the reference.
    independent_latent = rng.standard_normal((n_samples, n_latent))
    independent_projection = rng.standard_normal((n_latent, n_features))
    X_independent = independent_latent @ independent_projection
    X_independent += 0.01 * rng.standard_normal(X_independent.shape)

    return X_reference, X_rotated, X_concentrated, X_independent


def main():
    print("=" * 70)
    print("Stability vs. Similarity: Demonstrating the Distinction")
    print("=" * 70)
    print("\nFrom the paper:")
    print("  Stability = INTRINSIC property (internal geometric robustness)")
    print("  Similarity = EXTRINSIC property (alignment with reference)")
    print("  They must be evaluated separately.\n")

    # Create representations
    X_ref, X_rot, X_concentrated, X_indep = create_representations()

    print("\nScenario 1: Rotated Representation (Ideal Case)")
    print("-" * 70)

    # Measure stability (intrinsic)
    stab_ref = shesha.feature_split(X_ref, n_splits=30, seed=320)
    stab_rot = shesha.feature_split(X_rot, n_splits=30, seed=320)

    # Measure similarity (extrinsic)
    cka_rot = sim.cka(X_ref, X_rot)
    rdm_rot = sim.rdm_similarity(X_ref, X_rot)
    proc_rot = sim.procrustes_similarity(X_ref, X_rot)

    print(f"  Reference Stability:  {stab_ref:.3f}")
    print(f"  Rotated Stability:    {stab_rot:.3f}  (feature_split can change under rotation)")
    print(f"  CKA Similarity:       {cka_rot:.3f}  (should be high ~1.0)")
    print(f"  RDM Similarity:       {rdm_rot:.3f}  (should be high)")
    print(f"  Procrustes Similarity:{proc_rot:.3f}  (should be high ~1.0)")
    print("\n  [+] Rotation preserves RDM similarity; feature_split is basis-dependent")

    print("\n\nScenario 2: Same Geometry, Concentrated Coordinates")
    print("-" * 70)

    # Measure stability
    stab_concentrated = shesha.feature_split(X_concentrated, n_splits=30, seed=320)

    # Measure similarity
    cka_concentrated = sim.cka(X_ref, X_concentrated)
    rdm_concentrated = sim.rdm_similarity(X_ref, X_concentrated)
    proc_concentrated = sim.procrustes_similarity(X_ref, X_concentrated)

    print(f"  Reference Stability:  {stab_ref:.3f}")
    print(f"  Concentrated Stability:{stab_concentrated:.3f}  (should be much lower)")
    print(f"  CKA Similarity:       {cka_concentrated:.3f}  (should be ~1.0)")
    print(f"  RDM Similarity:       {rdm_concentrated:.3f}  (should be ~1.0)")
    print(f"  Procrustes Similarity:{proc_concentrated:.3f}  (should be ~1.0)")
    print("\n  [!] Same geometry, different feature redundancy")

    print("\n\nScenario 3: Independent Representation")
    print("-" * 70)

    # Measure stability
    stab_indep = shesha.feature_split(X_indep, n_splits=30, seed=320)

    # Measure similarity
    cka_indep = sim.cka(X_ref, X_indep)
    rdm_indep = sim.rdm_similarity(X_ref, X_indep)
    proc_indep = sim.procrustes_similarity(X_ref, X_indep)

    # Keep the narrative executable: fail if a future metric change makes the
    # controlled scenarios stop exhibiting their stated properties.
    assert stab_ref > 0.8 and stab_indep > 0.8
    assert stab_concentrated < 0.2
    assert min(cka_rot, rdm_rot, proc_rot) > 0.99
    assert min(cka_concentrated, rdm_concentrated, proc_concentrated) > 0.99
    assert max(cka_indep, abs(rdm_indep), proc_indep) < 0.2

    print(f"  Reference Stability:   {stab_ref:.3f}")
    print(f"  Independent Stability: {stab_indep:.3f}  (should be high)")
    print(f"  CKA Similarity:        {cka_indep:.3f}  (should be LOW)")
    print(f"  RDM Similarity:        {rdm_indep:.3f}  (should be LOW)")
    print(f"  Procrustes Similarity: {proc_indep:.3f}  (should be LOW)")
    print("\n  [o] Different relational geometry, still internally stable")

    print("\n\n" + "=" * 70)
    print("Summary: Comparison Table")
    print("=" * 70)
    print(f"{'Scenario':<25s} | {'Stability':>10s} | {'CKA':>10s} | {'RDM':>10s}")
    print("-" * 70)
    print(f"{'Reference':<25s} | {stab_ref:>10.3f} | {'    -':>10s} | {'    -':>10s}")
    print(f"{'Rotated (Ideal)':<25s} | {stab_rot:>10.3f} | {cka_rot:>10.3f} | {rdm_rot:>10.3f}")
    print(
        f"{'Concentrated Coordinates':<25s} | {stab_concentrated:>10.3f} | "
        f"{cka_concentrated:>10.3f} | {rdm_concentrated:>10.3f}"
    )
    print(f"{'Independent':<25s} | {stab_indep:>10.3f} | {cka_indep:>10.3f} | {rdm_indep:>10.3f}")

    print("\n\nKey Insight from Paper:")
    print("-" * 70)
    print("""
Models like DINOv2 show the "geometric tax":
  - High CKA similarity (dominant structure intact)
  - High transfer performance
  - BUT low geometric stability (fine-grained geometry brittle)

This demonstrates that stability and similarity are DISTINCT properties.
Measuring both is essential for complete model evaluation.

Usage in Practice:
  - Use CKA/RDM for: Model comparison, architecture search
  - Use Stability for: Safety monitoring, drift detection, steering prediction
  - Use BOTH for: Complete understanding of representational quality
""")

    print("=" * 70)


if __name__ == "__main__":
    main()
