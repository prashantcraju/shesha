"""
Example: Comparing Stability vs. Similarity

This example demonstrates the key distinction from the paper:
Stability and similarity are UNCORRELATED (ρ ≈ 0.01).

A model can have:
- High similarity (aligns well with reference)
- Low stability (brittle internal geometry)

This is the "geometric tax" identified in the paper.
"""

import numpy as np
import shesha
import shesha.sim as sim


def create_representations():
    """
    Create example representations demonstrating the stability-similarity distinction.
    """
    np.random.seed(320)
    
    # Original representation
    X_original = np.random.randn(200, 100)
    
    # Scenario 1: High similarity, high stability (ideal case)
    # Just a rotation - preserves both similarity and stability
    Q = np.linalg.qr(np.random.randn(100, 100))[0]
    X_rotated = X_original @ Q
    
    # Scenario 2: High similarity, low stability (geometric tax)
    # Dominant structure preserved but fine-grained geometry corrupted
    U, S, Vt = np.linalg.svd(X_original, full_matrices=False)
    # Keep top PCs for similarity, but shuffle within each PC contribution
    S_corrupted = S.copy()
    for i in range(len(S)):
        # Add noise that preserves magnitude but corrupts fine structure
        noise = np.random.randn(*U[:, i].shape) * 0.5
        U[:, i] = U[:, i] + noise
    X_corrupted = (U * S_corrupted) @ Vt
    
    # Scenario 3: Low similarity, high stability
    # Independent but internally consistent representation
    X_independent = np.random.randn(200, 100)
    X_independent = X_independent / np.linalg.norm(X_independent, axis=1, keepdims=True)
    
    return X_original, X_rotated, X_corrupted, X_independent


def main():
    print("=" * 70)
    print("Stability vs. Similarity: Demonstrating the Distinction")
    print("=" * 70)
    print("\nFrom the paper:")
    print("  Stability = INTRINSIC property (internal geometric robustness)")
    print("  Similarity = EXTRINSIC property (alignment with reference)")
    print("  These are UNCORRELATED (rho ~= 0.01)\n")
    
    # Create representations
    X_ref, X_rot, X_corrupt, X_indep = create_representations()
    
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
    print(f"  Rotated Stability:    {stab_rot:.3f}  (should be similar)")
    print(f"  CKA Similarity:       {cka_rot:.3f}  (should be high ~1.0)")
    print(f"  RDM Similarity:       {rdm_rot:.3f}  (should be high)")
    print(f"  Procrustes Similarity:{proc_rot:.3f}  (should be high ~1.0)")
    print("\n  [+] High Stability + High Similarity = Ideal")
    
    print("\n\nScenario 2: Corrupted Representation (Geometric Tax)")
    print("-" * 70)
    
    # Measure stability
    stab_corrupt = shesha.feature_split(X_corrupt, n_splits=30, seed=320)
    
    # Measure similarity
    cka_corrupt = sim.cka(X_ref, X_corrupt)
    rdm_corrupt = sim.rdm_similarity(X_ref, X_corrupt)
    proc_corrupt = sim.procrustes_similarity(X_ref, X_corrupt)
    
    print(f"  Reference Stability:  {stab_ref:.3f}")
    print(f"  Corrupted Stability:  {stab_corrupt:.3f}  (should be LOWER)")
    print(f"  CKA Similarity:       {cka_corrupt:.3f}  (should be moderate-high)")
    print(f"  RDM Similarity:       {rdm_corrupt:.3f}")
    print(f"  Procrustes Similarity:{proc_corrupt:.3f}")
    print("\n  [!] Lower Stability + High Similarity = Geometric Tax")
    print("  (Dominant structure preserved, but fine-grained geometry brittle)")
    
    print("\n\nScenario 3: Independent Representation")
    print("-" * 70)
    
    # Measure stability
    stab_indep = shesha.feature_split(X_indep, n_splits=30, seed=320)
    
    # Measure similarity  
    cka_indep = sim.cka(X_ref, X_indep)
    rdm_indep = sim.rdm_similarity(X_ref, X_indep)
    proc_indep = sim.procrustes_similarity(X_ref, X_indep)
    
    print(f"  Reference Stability:   {stab_ref:.3f}")
    print(f"  Independent Stability: {stab_indep:.3f}  (internally consistent)")
    print(f"  CKA Similarity:        {cka_indep:.3f}  (should be LOW)")
    print(f"  RDM Similarity:        {rdm_indep:.3f}  (should be LOW)")
    print(f"  Procrustes Similarity: {proc_indep:.3f}  (should be LOW)")
    print("\n  [o] Different representation, still internally stable")
    
    print("\n\n" + "=" * 70)
    print("Summary: Comparison Table")
    print("=" * 70)
    print(f"{'Scenario':<25s} | {'Stability':>10s} | {'CKA':>10s} | {'RDM':>10s}")
    print("-" * 70)
    print(f"{'Reference':<25s} | {stab_ref:>10.3f} | {'    -':>10s} | {'    -':>10s}")
    print(f"{'Rotated (Ideal)':<25s} | {stab_rot:>10.3f} | {cka_rot:>10.3f} | {rdm_rot:>10.3f}")
    print(f"{'Corrupted (Geom. Tax)':<25s} | {stab_corrupt:>10.3f} | {cka_corrupt:>10.3f} | {rdm_corrupt:>10.3f}")
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
