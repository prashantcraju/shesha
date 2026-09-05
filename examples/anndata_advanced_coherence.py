"""
Example: Advanced coherence analysis with AnnData integration

This example demonstrates how to use the enhanced perturbation coherence
methods (whitened and k-NN) with AnnData objects in a scanpy-compatible workflow.
"""

import numpy as np

# Try to import required packages
try:
    from anndata import AnnData
    import pandas as pd
    ANNDATA_AVAILABLE = True
except ImportError:
    print("Error: This example requires anndata and pandas")
    print("Install with: pip install anndata pandas")
    exit(1)

import shesha


def create_synthetic_perturbation_data():
    """
    Create synthetic single-cell perturbation data for demonstration.
    
    Simulates a Perturb-seq or CRISPR screen experiment with:
    - Control (non-targeting) cells
    - Multiple perturbations with varying coherence
    """
    np.random.seed(320)
    
    n_features = 50  # Could be PCs or selected genes
    
    # Control population (heterogeneous: two cell states)
    n_ctrl_state1 = 200
    n_ctrl_state2 = 200
    X_ctrl_state1 = np.random.randn(n_ctrl_state1, n_features)
    X_ctrl_state2 = np.random.randn(n_ctrl_state2, n_features) + 1.5
    X_ctrl = np.vstack([X_ctrl_state1, X_ctrl_state2])
    
    # Perturbation 1: Strong, coherent effect (e.g., essential gene knockout)
    n_pert1 = 150
    shift1 = np.random.randn(n_features) * 3
    X_pert1 = X_ctrl[:n_pert1] + shift1 + np.random.randn(n_pert1, n_features) * 0.2
    
    # Perturbation 2: Moderate, coherent effect
    n_pert2 = 150
    shift2 = np.random.randn(n_features) * 1.5
    X_pert2 = X_ctrl[:n_pert2] + shift2 + np.random.randn(n_pert2, n_features) * 0.3
    
    # Perturbation 3: Weak/incoherent effect (e.g., off-target or no effect)
    n_pert3 = 150
    X_pert3 = X_ctrl[:n_pert3] + np.random.randn(n_pert3, n_features) * 0.8
    
    # Combine all data
    X = np.vstack([X_ctrl, X_pert1, X_pert2, X_pert3])
    
    # Create perturbation labels
    perturbations = (
        ["non-targeting"] * (n_ctrl_state1 + n_ctrl_state2) +
        ["KLF1_knockout"] * n_pert1 +
        ["GATA1_knockout"] * n_pert2 +
        ["weak_perturbation"] * n_pert3
    )
    
    # Create AnnData object
    adata = AnnData(X=X)
    adata.obs["perturbation"] = perturbations
    adata.obs["perturbation"] = adata.obs["perturbation"].astype("category")
    
    # Add some metadata
    adata.obs["n_counts"] = np.random.poisson(5000, len(perturbations))
    adata.var_names = [f"PC_{i+1}" for i in range(n_features)]
    
    return adata


def main():
    print("=" * 70)
    print("Advanced Perturbation Coherence Analysis with AnnData")
    print("=" * 70)
    
    # Create synthetic data
    print("\n1. Creating synthetic Perturb-seq data...")
    adata = create_synthetic_perturbation_data()
    print(f"   Created AnnData: {adata.shape[0]} cells x {adata.shape[1]} features")
    print(f"   Perturbations: {adata.obs['perturbation'].value_counts().to_dict()}")
    
    # Method 1: Standard coherence (global control centroid)
    print("\n2. Computing standard perturbation coherence...")
    std_coherence = shesha.bio.compute_coherence(
        adata,
        perturbation_key="perturbation",
        control_label="non-targeting",
        max_samples=100,
        seed=320
    )
    
    print("   Standard Coherence Scores:")
    for pert, score in sorted(std_coherence.items(), key=lambda x: -x[1]):
        print(f"     {pert:25s}: {score:.3f}")
    
    # Method 2: Whitened coherence (accounts for feature correlations)
    print("\n3. Computing whitened (Mahalanobis) coherence...")
    white_coherence = shesha.bio.compute_coherence_whitened(
        adata,
        perturbation_key="perturbation",
        control_label="non-targeting",
        regularization=1e-6,
        max_samples=100,
        seed=320
    )
    
    print("   Whitened Coherence Scores:")
    for pert, score in sorted(white_coherence.items(), key=lambda x: -x[1]):
        print(f"     {pert:25s}: {score:.3f}")
    
    # Method 3: k-NN matched coherence (local control baseline)
    print("\n4. Computing k-NN matched coherence...")
    knn_coherence = shesha.bio.compute_coherence_knn(
        adata,
        perturbation_key="perturbation",
        control_label="non-targeting",
        k=50,
        metric="euclidean",
        max_samples=100,
        seed=320
    )
    
    print("   k-NN Matched Coherence Scores:")
    for pert, score in sorted(knn_coherence.items(), key=lambda x: -x[1]):
        print(f"     {pert:25s}: {score:.3f}")
    
    # Comparison
    print("\n5. Method Comparison:")
    print("   " + "-" * 66)
    print(f"   {'Perturbation':<25s} | {'Standard':>10s} | {'Whitened':>10s} | {'k-NN':>10s}")
    print("   " + "-" * 66)
    
    for pert in sorted(std_coherence.keys()):
        print(f"   {pert:<25s} | {std_coherence[pert]:>10.3f} | "
              f"{white_coherence[pert]:>10.3f} | {knn_coherence[pert]:>10.3f}")
    
    # Add results to AnnData
    print("\n6. Storing results in AnnData.uns...")
    adata.uns["coherence_standard"] = std_coherence
    adata.uns["coherence_whitened"] = white_coherence
    adata.uns["coherence_knn"] = knn_coherence
    
    print("   Results stored in:")
    print("     - adata.uns['coherence_standard']")
    print("     - adata.uns['coherence_whitened']")
    print("     - adata.uns['coherence_knn']")
    
    # Interpretation
    print("\n7. Interpretation:")
    print("   " + "=" * 66)
    print("""
   High coherence scores (> 0.7): Coherent, reproducible perturbation effect
   Medium coherence (0.4-0.7):    Moderate effect with some heterogeneity
   Low coherence (< 0.4):         Weak/incoherent effect or high noise
   
   When to use each method:
   
   Standard:  Default choice for most analyses
   Whitened:  When features have different scales or correlations
              (e.g., raw gene expression, batch effects)
   k-NN:      When control population is heterogeneous
              (e.g., multiple cell types, developmental stages)
    """)
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
