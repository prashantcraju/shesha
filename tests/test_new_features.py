"""
Tests for newly added features from paper code.
"""

import numpy as np
import pytest
import shesha


def test_class_separation_ratio():
    """Test class separation ratio metric."""
    # Well-separated classes
    np.random.seed(320)
    X = np.vstack([
        np.random.randn(100, 10),
        np.random.randn(100, 10) + 5
    ])
    y = np.array([0]*100 + [1]*100)
    
    ratio = shesha.class_separation_ratio(X, y, n_bootstrap=10, seed=320)
    
    assert isinstance(ratio, float)
    assert not np.isnan(ratio)
    assert ratio > 1.0  # Between-class should be larger than within-class
    print(f"[PASS] Class separation ratio: {ratio:.3f}")


def test_lda_stability():
    """Test LDA subspace stability metric."""
    # Well-separated binary classification
    np.random.seed(320)
    X = np.vstack([
        np.random.randn(100, 10),
        np.random.randn(100, 10) + 3
    ])
    y = np.array([0]*100 + [1]*100)
    
    stability = shesha.lda_stability(X, y, n_bootstrap=10, seed=320)
    
    assert isinstance(stability, float)
    assert not np.isnan(stability)
    assert 0 <= stability <= 1  # Should be between 0 and 1
    assert stability > 0.7  # Should be high for well-separated data
    print(f"[PASS] LDA stability: {stability:.3f}")


def test_lda_stability_multiclass_error():
    """Test that LDA stability raises error for multiclass."""
    X = np.random.randn(100, 10)
    y = np.array([0]*33 + [1]*33 + [2]*34)
    
    with pytest.raises(ValueError, match="exactly 2 classes"):
        shesha.lda_stability(X, y)
    print("[PASS] LDA stability correctly rejects multiclass")


def test_perturbation_stability_whitened():
    """Test whitened perturbation stability."""
    np.random.seed(320)
    
    # Control population
    X_ctrl = np.random.randn(500, 50)
    
    # Coherent perturbation
    shift = np.random.randn(50)
    X_pert = X_ctrl[:250] + shift + np.random.randn(250, 50) * 0.1
    
    stability = shesha.bio.perturbation_stability_whitened(
        X_ctrl, X_pert, max_samples=200, seed=320
    )
    
    assert isinstance(stability, float)
    assert not np.isnan(stability)
    assert -1 <= stability <= 1
    assert stability > 0.5  # Should be high for coherent perturbation
    print(f"[PASS] Whitened perturbation stability: {stability:.3f}")


def test_perturbation_stability_knn():
    """Test k-NN matched control perturbation stability."""
    np.random.seed(320)
    
    # Heterogeneous control population (two subpopulations)
    X_ctrl = np.vstack([
        np.random.randn(250, 50),
        np.random.randn(250, 50) + 2
    ])
    
    # Coherent perturbation
    shift = np.random.randn(50)
    X_pert = X_ctrl[:200] + shift + np.random.randn(200, 50) * 0.1
    
    stability = shesha.bio.perturbation_stability_knn(
        X_ctrl, X_pert, k=50, max_samples=150, seed=320
    )
    
    assert isinstance(stability, float)
    assert not np.isnan(stability)
    assert -1 <= stability <= 1
    assert stability > 0.5  # Should be high for coherent perturbation
    print(f"[PASS] k-NN perturbation stability: {stability:.3f}")


def test_comparison_standard_vs_whitened():
    """Compare standard and whitened perturbation stability."""
    np.random.seed(320)
    
    X_ctrl = np.random.randn(300, 30)
    shift = np.random.randn(30)
    X_pert = X_ctrl[:150] + shift + np.random.randn(150, 30) * 0.2
    
    # Standard
    std_stab = shesha.bio.perturbation_stability(X_ctrl, X_pert, max_samples=100, seed=320)
    
    # Whitened
    white_stab = shesha.bio.perturbation_stability_whitened(X_ctrl, X_pert, max_samples=100, seed=320)
    
    # Both should be positive and reasonably similar
    assert std_stab > 0
    assert white_stab > 0
    assert abs(std_stab - white_stab) < 0.5  # Should be in same ballpark
    
    print(f"[PASS] Standard stability: {std_stab:.3f}")
    print(f"[PASS] Whitened stability: {white_stab:.3f}")


if __name__ == "__main__":
    print("\nTesting newly added features...\n")
    test_class_separation_ratio()
    test_lda_stability()
    test_lda_stability_multiclass_error()
    test_perturbation_stability_whitened()
    test_perturbation_stability_knn()
    test_comparison_standard_vs_whitened()
    print("\n[SUCCESS] All tests passed!")
