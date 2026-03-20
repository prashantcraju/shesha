"""
Tests for similarity metrics module.
"""

import numpy as np
import pytest
import shesha.sim as sim


def test_cka_linear_basic():
    """Test basic CKA linear computation."""
    np.random.seed(320)
    
    # Two random representations
    X = np.random.randn(100, 50)
    Y = np.random.randn(100, 30)
    
    cka_val = sim.cka_linear(X, Y)
    
    assert isinstance(cka_val, float)
    assert 0 <= cka_val <= 1
    print(f"[PASS] CKA linear basic: {cka_val:.3f}")


def test_cka_self_similarity():
    """Test that CKA(X, X) = 1.0."""
    np.random.seed(320)
    
    X = np.random.randn(100, 50)
    
    cka_val = sim.cka_linear(X, X)
    
    assert abs(cka_val - 1.0) < 1e-6
    print(f"[PASS] CKA self-similarity: {cka_val:.6f}")


def test_cka_orthogonal_invariance():
    """Test that CKA is invariant to orthogonal transformations."""
    np.random.seed(320)
    
    X = np.random.randn(100, 50)
    
    # Apply random orthogonal transformation
    Q = np.linalg.qr(np.random.randn(50, 50))[0]
    X_rotated = X @ Q
    
    cka_val = sim.cka_linear(X, X_rotated)
    
    # Should be very close to 1.0
    assert abs(cka_val - 1.0) < 0.01
    print(f"[PASS] CKA orthogonal invariance: {cka_val:.6f}")


def test_cka_debiased():
    """Test debiased CKA."""
    np.random.seed(320)
    
    X = np.random.randn(50, 20)
    Y = np.random.randn(50, 15)
    
    cka_std = sim.cka_linear(X, Y)
    cka_deb = sim.cka_debiased(X, Y)
    
    assert isinstance(cka_deb, float)
    assert 0 <= cka_deb <= 1
    
    # Should be reasonably similar
    assert abs(cka_std - cka_deb) < 0.3
    
    print(f"[PASS] CKA standard: {cka_std:.3f}, debiased: {cka_deb:.3f}")


def test_cka_unified_interface():
    """Test unified cka() interface."""
    np.random.seed(320)
    
    X = np.random.randn(100, 50)
    Y = np.random.randn(100, 30)
    
    # Standard
    cka_std = sim.cka(X, Y, debiased=False)
    cka_lin = sim.cka_linear(X, Y)
    assert abs(cka_std - cka_lin) < 1e-10
    
    # Debiased
    cka_deb1 = sim.cka(X, Y, debiased=True)
    cka_deb2 = sim.cka_debiased(X, Y)
    assert abs(cka_deb1 - cka_deb2) < 1e-10
    
    print("[PASS] CKA unified interface")


def test_cka_dimension_mismatch_error():
    """Test that CKA raises error for mismatched sample sizes."""
    X = np.random.randn(100, 50)
    Y = np.random.randn(90, 30)  # Different number of samples
    
    with pytest.raises(ValueError, match="same number of samples"):
        sim.cka_linear(X, Y)
    
    print("[PASS] CKA dimension mismatch error")


def test_procrustes_basic():
    """Test basic Procrustes similarity."""
    np.random.seed(320)
    
    X = np.random.randn(100, 50)
    Y = np.random.randn(100, 50)
    
    proc_sim = sim.procrustes_similarity(X, Y)
    
    assert isinstance(proc_sim, float)
    assert 0 <= proc_sim <= 1 or np.isnan(proc_sim)
    
    print(f"[PASS] Procrustes basic: {proc_sim:.3f}")


def test_procrustes_rotation():
    """Test that Procrustes detects perfect alignment under rotation."""
    np.random.seed(320)
    
    X = np.random.randn(100, 50)
    
    # Apply random rotation
    Q = np.linalg.qr(np.random.randn(50, 50))[0]
    Y = X @ Q
    
    proc_sim = sim.procrustes_similarity(X, Y)
    
    # Should be very close to 1.0
    assert proc_sim > 0.99
    print(f"[PASS] Procrustes rotation: {proc_sim:.6f}")


def test_procrustes_self_similarity():
    """Test that Procrustes(X, X) = 1.0."""
    np.random.seed(320)
    
    X = np.random.randn(100, 50)
    
    proc_sim = sim.procrustes_similarity(X, X)
    
    assert abs(proc_sim - 1.0) < 1e-6
    print(f"[PASS] Procrustes self-similarity: {proc_sim:.6f}")


def test_procrustes_dimension_mismatch():
    """Test that Procrustes returns NaN for dimension mismatch."""
    X = np.random.randn(100, 50)
    Y = np.random.randn(100, 30)  # Different feature dimensions
    
    proc_sim = sim.procrustes_similarity(X, Y)
    
    assert np.isnan(proc_sim)
    print("[PASS] Procrustes dimension mismatch returns NaN")


def test_rdm_similarity_basic():
    """Test basic RDM similarity."""
    np.random.seed(320)
    
    X = np.random.randn(100, 50)
    Y = np.random.randn(100, 30)
    
    rdm_sim = sim.rdm_similarity(X, Y)
    
    assert isinstance(rdm_sim, float)
    assert -1 <= rdm_sim <= 1
    
    print(f"[PASS] RDM similarity basic: {rdm_sim:.3f}")


def test_rdm_similarity_methods():
    """Test different RDM similarity methods."""
    np.random.seed(320)
    
    X = np.random.randn(100, 50)
    Y = np.random.randn(100, 30)
    
    # Spearman (default)
    rdm_spearman = sim.rdm_similarity(X, Y, method='spearman')
    
    # Pearson
    rdm_pearson = sim.rdm_similarity(X, Y, method='pearson')
    
    assert isinstance(rdm_spearman, float)
    assert isinstance(rdm_pearson, float)
    
    # Should be reasonably similar but not identical
    assert abs(rdm_spearman - rdm_pearson) < 0.5
    
    print(f"[PASS] RDM Spearman: {rdm_spearman:.3f}, Pearson: {rdm_pearson:.3f}")


def test_rdm_similarity_metrics():
    """Test different distance metrics for RDM."""
    np.random.seed(320)
    
    X = np.random.randn(100, 50)
    Y = np.random.randn(100, 30)
    
    # Different distance metrics
    rdm_cosine = sim.rdm_similarity(X, Y, metric='cosine')
    rdm_euclidean = sim.rdm_similarity(X, Y, metric='euclidean')
    rdm_correlation = sim.rdm_similarity(X, Y, metric='correlation')
    
    assert isinstance(rdm_cosine, float)
    assert isinstance(rdm_euclidean, float)
    assert isinstance(rdm_correlation, float)
    
    print(f"[PASS] RDM metrics - cosine: {rdm_cosine:.3f}, "
          f"euclidean: {rdm_euclidean:.3f}, correlation: {rdm_correlation:.3f}")


def test_comparison_cka_vs_rdm():
    """Compare CKA and RDM similarity on same data."""
    np.random.seed(320)
    
    X = np.random.randn(100, 50)
    Y = np.random.randn(100, 30)
    
    cka_val = sim.cka_linear(X, Y)
    rdm_val = sim.rdm_similarity(X, Y)
    
    # Both should be positive for random data, but no strict relationship
    assert cka_val > 0
    assert rdm_val > -0.5  # Can be negative but usually positive
    
    print(f"[PASS] CKA: {cka_val:.3f}, RDM: {rdm_val:.3f}")


def test_imports():
    """Test that all functions are properly exported."""
    import shesha
    
    # Should be accessible via shesha.sim
    assert hasattr(shesha.sim, 'cka')
    assert hasattr(shesha.sim, 'cka_linear')
    assert hasattr(shesha.sim, 'cka_debiased')
    assert hasattr(shesha.sim, 'procrustes_similarity')
    assert hasattr(shesha.sim, 'rdm_similarity')
    
    print("[PASS] All imports accessible")


if __name__ == "__main__":
    print("\nTesting similarity metrics module...\n")
    test_cka_linear_basic()
    test_cka_self_similarity()
    test_cka_orthogonal_invariance()
    test_cka_debiased()
    test_cka_unified_interface()
    test_cka_dimension_mismatch_error()
    test_procrustes_basic()
    test_procrustes_rotation()
    test_procrustes_self_similarity()
    test_procrustes_dimension_mismatch()
    test_rdm_similarity_basic()
    test_rdm_similarity_methods()
    test_rdm_similarity_metrics()
    test_comparison_cka_vs_rdm()
    test_imports()
    print("\n[SUCCESS] All similarity tests passed!")
