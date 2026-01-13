"""
Test shesha.bio on CRISPR perturbation data.

Uses pertpy's Norman et al 2019 dataset (CRISPRa screen).
Install dependencies: pip install pertpy scanpy

Expected behavior:
- Strong perturbations should have higher stability (cells respond consistently)
- Weak/noisy perturbations should have lower stability
"""

import numpy as np
import pandas as pd

# Check for required dependencies
try:
    import pertpy as pt
    import scanpy as sc
except ImportError:
    print("This test requires pertpy and scanpy.")
    print("Install with: pip install pertpy scanpy")
    exit(1)

from shesha.bio import perturbation_stability, perturbation_effect_size

# Configuration
SEED = 320
np.random.seed(SEED)
N_PCA_DIMS = 50  # Use PCA-reduced space like in the paper


def load_norman_2019():
    """Load Norman 2019 CRISPRa dataset."""
    print("Loading Norman 2019 dataset...")
    adata = pt.dt.norman_2019()
    print(f"  Shape: {adata.shape}")
    
    # Basic preprocessing if not already done
    if 'X_pca' not in adata.obsm:
        print("  Running PCA...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        sc.pp.pca(adata, n_comps=N_PCA_DIMS)
    
    return adata


def get_perturbation_groups(adata):
    """Extract control and perturbation groups."""
    # Norman 2019 uses 'guide_ids' or 'perturbation' column
    if 'perturbation' in adata.obs.columns:
        pert_col = 'perturbation'
    elif 'guide_ids' in adata.obs.columns:
        pert_col = 'guide_ids'
    else:
        # Try to find a suitable column
        candidates = [c for c in adata.obs.columns if 'pert' in c.lower() or 'guide' in c.lower()]
        if candidates:
            pert_col = candidates[0]
        else:
            raise ValueError(f"Could not find perturbation column. Available: {list(adata.obs.columns)}")
    
    print(f"  Using perturbation column: {pert_col}")
    
    # Identify control cells
    all_perts = adata.obs[pert_col].unique()
    control_keywords = ['control', 'ctrl', 'neg', 'nt', 'non-targeting', 'unperturbed', 'nan']
    
    control_perts = []
    for p in all_perts:
        p_lower = str(p).lower()
        if any(kw in p_lower for kw in control_keywords) or p_lower == 'nan' or pd.isna(p):
            control_perts.append(p)
    
    if not control_perts:
        # Fallback: look for most common perturbation (often control)
        counts = adata.obs[pert_col].value_counts()
        control_perts = [counts.index[0]]
        print(f"  Warning: No obvious control found, using most common: {control_perts}")
    
    print(f"  Control perturbations: {control_perts}")
    
    # Get control cells
    control_mask = adata.obs[pert_col].isin(control_perts)
    
    # Get non-control perturbations
    other_perts = [p for p in all_perts if p not in control_perts]
    
    return pert_col, control_mask, other_perts


def test_on_real_data():
    """Main test function."""
    
    # Load data
    adata = load_norman_2019()
    
    # Get perturbation groups
    pert_col, control_mask, perturbations = get_perturbation_groups(adata)
    
    # Get embeddings
    X_pca = adata.obsm['X_pca'][:, :N_PCA_DIMS]
    X_control = X_pca[control_mask]
    
    print(f"\nControl cells: {X_control.shape[0]}")
    print(f"Perturbations to test: {len(perturbations)}")
    
    # Test on a subset of perturbations
    results = []
    n_test = min(20, len(perturbations))
    
    print(f"\nTesting {n_test} perturbations...")
    print("-" * 60)
    
    for pert in perturbations[:n_test]:
        pert_mask = adata.obs[pert_col] == pert
        n_cells = pert_mask.sum()
        
        if n_cells < 10:
            continue
        
        X_pert = X_pca[pert_mask]
        
        # Compute metrics using shesha.bio
        stability = perturbation_stability(X_control, X_pert, seed=SEED)
        effect = perturbation_effect_size(X_control, X_pert)
        
        results.append({
            'perturbation': str(pert)[:30],  # Truncate long names
            'n_cells': n_cells,
            'stability': stability,
            'effect_size': effect
        })
        
        print(f"{str(pert)[:30]:30s}  n={n_cells:4d}  stability={stability:.3f}  effect={effect:.2f}")
    
    # Summary statistics
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Perturbations tested: {len(df)}")
    print(f"Stability - mean: {df['stability'].mean():.3f}, std: {df['stability'].std():.3f}")
    print(f"Stability - min: {df['stability'].min():.3f}, max: {df['stability'].max():.3f}")
    print(f"Effect size - mean: {df['effect_size'].mean():.2f}, std: {df['effect_size'].std():.2f}")
    
    # Check correlation between stability and effect size
    from scipy.stats import spearmanr
    rho, p = spearmanr(df['stability'], df['effect_size'])
    print(f"\nCorrelation (stability vs effect): rho={rho:.3f}, p={p:.4f}")
    
    # Sanity checks
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    
    # Check that stability values are in expected range
    assert df['stability'].min() >= -1, "Stability below -1"
    assert df['stability'].max() <= 1, "Stability above 1"
    print("✓ Stability values in [-1, 1]")
    
    # Check that effect sizes are non-negative
    assert df['effect_size'].min() >= 0, "Negative effect size"
    print("✓ Effect sizes non-negative")
    
    # Check that we get reasonable variation (not all same value)
    assert df['stability'].std() > 0.01, "No variation in stability"
    print("✓ Reasonable variation in stability")
    
    # Most perturbations should have positive stability (coherent effect)
    frac_positive = (df['stability'] > 0).mean()
    print(f"✓ {frac_positive*100:.0f}% of perturbations have positive stability")
    
    print("\n✓ All sanity checks passed!")
    
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("SHESHA.BIO TEST ON REAL CRISPR DATA")
    print("=" * 60)
    print()
    
    df = test_on_real_data()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)