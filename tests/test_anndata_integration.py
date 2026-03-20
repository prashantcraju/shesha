"""
Tests for AnnData integration of new bio features.
"""

import numpy as np
import pytest

# Try to import AnnData
try:
    from anndata import AnnData
    import pandas as pd
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors  # noqa: F401
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import shesha


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")
def test_compute_stability_whitened_anndata():
    """Test whitened stability computation with AnnData."""
    np.random.seed(320)
    
    # Create synthetic single-cell data
    n_ctrl = 300
    n_pert1 = 150
    n_pert2 = 150
    n_features = 50
    
    # Control cells
    X_ctrl = np.random.randn(n_ctrl, n_features)
    
    # Perturbation 1: coherent shift
    shift1 = np.random.randn(n_features)
    X_pert1 = X_ctrl[:n_pert1] + shift1 + np.random.randn(n_pert1, n_features) * 0.1
    
    # Perturbation 2: incoherent/noisy
    X_pert2 = X_ctrl[:n_pert2] + np.random.randn(n_pert2, n_features) * 2
    
    # Combine into single matrix
    X = np.vstack([X_ctrl, X_pert1, X_pert2])
    
    # Create perturbation labels
    pert_labels = (
        ["control"] * n_ctrl + 
        ["perturbation_1"] * n_pert1 + 
        ["perturbation_2"] * n_pert2
    )
    
    # Create AnnData object
    adata = AnnData(X=X)
    adata.obs["perturbation"] = pert_labels
    
    # Compute whitened stability
    results = shesha.bio.compute_stability_whitened(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        regularization=1e-6,
        seed=320,
        max_samples=100
    )
    
    # Check results
    assert isinstance(results, dict)
    assert "perturbation_1" in results
    assert "perturbation_2" in results
    assert "control" not in results  # Control should not be in results
    
    # Coherent perturbation should have higher stability
    assert results["perturbation_1"] > results["perturbation_2"]
    assert results["perturbation_1"] > 0.5  # Should be reasonably high
    
    print(f"[PASS] Whitened stability (AnnData):")
    print(f"  Perturbation 1 (coherent): {results['perturbation_1']:.3f}")
    print(f"  Perturbation 2 (incoherent): {results['perturbation_2']:.3f}")


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")
@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_compute_stability_knn_anndata():
    """Test k-NN matched stability computation with AnnData."""
    np.random.seed(320)
    
    # Create synthetic data with heterogeneous control population
    n_ctrl_pop1 = 150
    n_ctrl_pop2 = 150
    n_pert = 100
    n_features = 40
    
    # Heterogeneous control: two subpopulations
    X_ctrl_pop1 = np.random.randn(n_ctrl_pop1, n_features)
    X_ctrl_pop2 = np.random.randn(n_ctrl_pop2, n_features) + 2
    X_ctrl = np.vstack([X_ctrl_pop1, X_ctrl_pop2])
    
    # Coherent perturbation from first population
    shift = np.random.randn(n_features)
    X_pert = X_ctrl[:n_pert] + shift + np.random.randn(n_pert, n_features) * 0.1
    
    # Combine
    X = np.vstack([X_ctrl, X_pert])
    
    # Create labels
    pert_labels = ["control"] * (n_ctrl_pop1 + n_ctrl_pop2) + ["gene_knockout"] * n_pert
    
    # Create AnnData
    adata = AnnData(X=X)
    adata.obs["perturbation"] = pert_labels
    
    # Compute k-NN matched stability
    results = shesha.bio.compute_stability_knn(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        k=30,
        metric="euclidean",
        seed=320,
        max_samples=80
    )
    
    # Check results
    assert isinstance(results, dict)
    assert "gene_knockout" in results
    assert "control" not in results
    
    # Should detect coherent perturbation despite heterogeneous control
    assert results["gene_knockout"] > 0.4
    
    print(f"[PASS] k-NN stability (AnnData): {results['gene_knockout']:.3f}")


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")
@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_anndata_with_sparse_matrix():
    """Test that sparse matrices are handled correctly."""
    try:
        from scipy.sparse import csr_matrix
    except ImportError:
        pytest.skip("scipy not available")
    
    np.random.seed(320)
    
    # Create sparse data
    X_dense = np.random.randn(200, 50)
    X_dense[X_dense < 0.5] = 0  # Make sparse
    X_sparse = csr_matrix(X_dense)
    
    # Create labels
    pert_labels = ["control"] * 100 + ["treatment"] * 100
    
    # Create AnnData with sparse matrix
    adata = AnnData(X=X_sparse)
    adata.obs["perturbation"] = pert_labels
    
    # Should handle sparse matrix automatically
    results_whitened = shesha.bio.compute_stability_whitened(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        max_samples=50,
        seed=320
    )
    
    results_knn = shesha.bio.compute_stability_knn(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        k=20,
        max_samples=50,
        seed=320
    )
    
    assert isinstance(results_whitened, dict)
    assert isinstance(results_knn, dict)
    assert "treatment" in results_whitened
    assert "treatment" in results_knn
    
    print("[PASS] Sparse matrix handling works")


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")
def test_anndata_with_layer():
    """Test using a specific layer instead of X."""
    np.random.seed(320)
    
    n_cells = 200
    n_genes = 100
    n_pcs = 30
    
    # Raw counts
    X_raw = np.random.poisson(10, size=(n_cells, n_genes)).astype(float)
    
    # PCA representation
    X_pca = np.random.randn(n_cells, n_pcs)
    
    # Create labels
    pert_labels = ["control"] * 100 + ["crispr_a"] * 50 + ["crispr_b"] * 50
    
    # Create AnnData with layer
    adata = AnnData(X=X_raw)
    adata.layers["X_pca"] = X_pca
    adata.obs["perturbation"] = pert_labels
    
    # Compute on PCA layer
    results = shesha.bio.compute_stability_whitened(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        layer="X_pca",
        max_samples=40,
        seed=320
    )
    
    assert isinstance(results, dict)
    assert "crispr_a" in results
    assert "crispr_b" in results
    
    print("[PASS] Layer specification works")
    print(f"  CRISPR A: {results['crispr_a']:.3f}")
    print(f"  CRISPR B: {results['crispr_b']:.3f}")


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not installed")
@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_comparison_all_methods():
    """Compare all three stability computation methods on same data."""
    np.random.seed(320)
    
    # Create data
    n_ctrl = 300
    n_pert = 150
    n_features = 50
    
    X_ctrl = np.random.randn(n_ctrl, n_features)
    shift = np.random.randn(n_features) * 2
    X_pert = X_ctrl[:n_pert] + shift + np.random.randn(n_pert, n_features) * 0.3
    
    X = np.vstack([X_ctrl, X_pert])
    pert_labels = ["control"] * n_ctrl + ["treatment"] * n_pert
    
    adata = AnnData(X=X)
    adata.obs["perturbation"] = pert_labels
    
    # Compute with all three methods
    std_results = shesha.bio.compute_stability(
        adata, "perturbation", max_samples=100, seed=320
    )
    
    white_results = shesha.bio.compute_stability_whitened(
        adata, "perturbation", max_samples=100, seed=320
    )
    
    knn_results = shesha.bio.compute_stability_knn(
        adata, "perturbation", k=50, max_samples=100, seed=320
    )
    
    # All should return similar results for this simple case
    std_val = std_results["treatment"]
    white_val = white_results["treatment"]
    knn_val = knn_results["treatment"]
    
    assert std_val > 0
    assert white_val > 0
    assert knn_val > 0
    
    print("[PASS] All three methods comparison:")
    print(f"  Standard:  {std_val:.3f}")
    print(f"  Whitened:  {white_val:.3f}")
    print(f"  k-NN:      {knn_val:.3f}")


if __name__ == "__main__":
    if not ANNDATA_AVAILABLE:
        print("AnnData not available, skipping tests")
    else:
        print("\nTesting AnnData integration for new bio features...\n")
        test_compute_stability_whitened_anndata()
        test_compute_stability_knn_anndata()
        test_anndata_with_sparse_matrix()
        test_anndata_with_layer()
        test_comparison_all_methods()
        print("\n[SUCCESS] All AnnData integration tests passed!")
