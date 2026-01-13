# %% [markdown]
# # Shesha Tutorial
# 
# This tutorial demonstrates how to use SHESHA (Self-consistency metrics for representational stability analysis) to measure geometric stability of high-dimensional representations.
# 
# **What you'll learn:**
# 1. Basic usage of unsupervised variants (feature_split, sample_split, anchor_stability)
# 2. Supervised variants with labels (variance_ratio, supervised_alignment)
# 3. Practical applications: comparing models, detecting drift, analyzing embeddings

# %% [markdown]
# ## Installation
# 
# ```bash
# pip install shesha-geometry
# ```

# %%
import numpy as np
import matplotlib.pyplot as plt
import shesha

print(f"shesha version: {shesha.__version__}")

# Set random seed for reproducibility
np.random.seed(320)

# %% [markdown]
# ## 1. Understanding Shesha: The Core Idea
# 
# Shesha measures **geometric stability** - whether a representation's distance structure 
# is internally consistent. Unlike similarity metrics (CKA, Procrustes) that compare 
# *between* representations, Shesha measures consistency *within* a single representation.
# 
# **Key insight:** A stable representation should produce similar pairwise distance patterns 
# (RDMs) when computed from different "views" of the same data.

# %% [markdown]
# ## 2. Feature-Split Shesha (Unsupervised)
# 
# The most common variant. Splits feature dimensions into random halves and checks if 
# both halves encode consistent distance relationships.
# 
# **High stability** = geometric structure is distributed across features (redundant encoding)
# **Low stability** = structure concentrated in few features or noisy

# %%
# Example 1: Structured data (high stability expected)
# Create data with low-rank structure - geometry should be consistent across feature subsets
n_samples, latent_dim, feature_dim = 500, 20, 768

latent = np.random.randn(n_samples, latent_dim)
projection = np.random.randn(latent_dim, feature_dim)
X_structured = latent @ projection

stability_structured = shesha.feature_split(X_structured, n_splits=30, seed=320)
print(f"Structured data stability: {stability_structured:.3f}")

# %%
# Example 2: Random noise (lower stability expected)
# Each feature is independent - no consistent structure across subsets
X_random = np.random.randn(n_samples, feature_dim)

stability_random = shesha.feature_split(X_random, n_splits=30, seed=320)
print(f"Random noise stability: {stability_random:.3f}")

# %%
# Visualize the difference
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Run multiple seeds to show distribution
stabilities_structured = [shesha.feature_split(X_structured, n_splits=30, seed=s) for s in range(10)]
stabilities_random = [shesha.feature_split(X_random, n_splits=30, seed=s) for s in range(10)]

axes[0].bar(['Structured', 'Random'], 
            [np.mean(stabilities_structured), np.mean(stabilities_random)],
            yerr=[np.std(stabilities_structured), np.std(stabilities_random)],
            capsize=5, color=['steelblue', 'coral'])
axes[0].set_ylabel('Feature-Split Stability')
axes[0].set_title('Stability Comparison')
axes[0].set_ylim(0, 1)

# Show how stability varies with latent dimension
latent_dims = [5, 10, 20, 50, 100, 200]
stabilities_by_dim = []

for ld in latent_dims:
    latent = np.random.randn(n_samples, ld)
    proj = np.random.randn(ld, feature_dim)
    X = latent @ proj
    stabilities_by_dim.append(shesha.feature_split(X, n_splits=30, seed=320))

axes[1].plot(latent_dims, stabilities_by_dim, 'o-', color='steelblue', linewidth=2)
axes[1].set_xlabel('Latent Dimensionality')
axes[1].set_ylabel('Feature-Split Stability')
axes[1].set_title('Stability vs. Latent Rank')
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('tutorial_feature_split.png', dpi=150)
plt.show()

# %% [markdown]
# ## 3. Sample-Split Shesha (Bootstrap)
# 
# Measures robustness to sampling variation by computing RDMs on different 
# random subsets of data points.

# %%
# Compare sample-split stability
sample_stability_structured = shesha.sample_split(X_structured, n_splits=30, seed=320)
sample_stability_random = shesha.sample_split(X_random, n_splits=30, seed=320)

print(f"Sample-split stability (structured): {sample_stability_structured:.3f}")
print(f"Sample-split stability (random): {sample_stability_random:.3f}")

# %% [markdown]
# ## 4. Anchor Stability
# 
# Uses fixed anchor points to measure distance profile consistency. 
# More robust for large datasets.

# %%
anchor_stability_structured = shesha.anchor_stability(X_structured, n_splits=30, seed=320)
anchor_stability_random = shesha.anchor_stability(X_random, n_splits=30, seed=320)

print(f"Anchor stability (structured): {anchor_stability_structured:.3f}")
print(f"Anchor stability (random): {anchor_stability_random:.3f}")

# %% [markdown]
# ## 5. Supervised Variants
# 
# When you have class labels, supervised variants measure task-relevant stability.

# %%
# Create labeled data with clear class structure
n_per_class = 100
n_classes = 5

# Well-separated clusters
X_separated = np.vstack([
    np.random.randn(n_per_class, 768) + np.random.randn(768) * 3
    for _ in range(n_classes)
])
y_separated = np.repeat(np.arange(n_classes), n_per_class)

# Overlapping clusters (harder to separate)
X_overlapping = np.vstack([
    np.random.randn(n_per_class, 768) + np.random.randn(768) * 0.5
    for _ in range(n_classes)
])
y_overlapping = np.repeat(np.arange(n_classes), n_per_class)

# %%
# Variance ratio: between-class / total variance
vr_separated = shesha.variance_ratio(X_separated, y_separated)
vr_overlapping = shesha.variance_ratio(X_overlapping, y_overlapping)

print(f"Variance ratio (well-separated): {vr_separated:.3f}")
print(f"Variance ratio (overlapping): {vr_overlapping:.3f}")

# %%
# Supervised alignment: correlation with ideal label-based RDM
align_separated = shesha.supervised_alignment(X_separated, y_separated, seed=320)
align_overlapping = shesha.supervised_alignment(X_overlapping, y_overlapping, seed=320)

print(f"Supervised alignment (well-separated): {align_separated:.3f}")
print(f"Supervised alignment (overlapping): {align_overlapping:.3f}")

# %% [markdown]
# ## 6. Practical Application: Comparing Embedding Models
# 
# A common use case: which embedding model has more stable representations?

# %%
# Simulate embeddings from different "models" with varying structure
def simulate_embeddings(n_samples, n_features, latent_dim, noise_level):
    """Simulate embeddings with controllable structure and noise."""
    latent = np.random.randn(n_samples, latent_dim)
    projection = np.random.randn(latent_dim, n_features)
    signal = latent @ projection
    signal = signal / np.std(signal)
    noise = np.random.randn(n_samples, n_features) * noise_level
    return signal + noise

# Simulate 4 "models" with different properties
models = {
    'Model A (high rank, low noise)': simulate_embeddings(500, 768, 100, 0.1),
    'Model B (high rank, high noise)': simulate_embeddings(500, 768, 100, 1.0),
    'Model C (low rank, low noise)': simulate_embeddings(500, 768, 20, 0.1),
    'Model D (low rank, high noise)': simulate_embeddings(500, 768, 20, 1.0),
}

# Compare stability across models
print("Model Comparison (Feature-Split Stability):")
print("-" * 50)

results = {}
for name, X in models.items():
    stability = shesha.feature_split(X, n_splits=30, seed=320)
    results[name] = stability
    print(f"{name}: {stability:.3f}")

# %%
# Visualize model comparison
fig, ax = plt.subplots(figsize=(10, 5))

names = list(results.keys())
values = list(results.values())
colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

bars = ax.barh(names, values, color=colors)
ax.set_xlabel('Feature-Split Stability')
ax.set_title('Embedding Model Stability Comparison')
ax.set_xlim(0, 1)

for bar, val in zip(bars, values):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
            f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('tutorial_model_comparison.png', dpi=150)
plt.show()

# %% [markdown]
# ## 7. Practical Application: Monitoring Training Drift
# 
# Track how representation stability changes during fine-tuning.

# %%
# Simulate embeddings at different "epochs" of training
# Early: random initialization, Late: structured representations

def simulate_training_trajectory(n_epochs=10):
    """Simulate how embeddings evolve during training."""
    n_samples, n_features = 500, 768
    
    # Start with noise, gradually add structure
    embeddings = []
    for epoch in range(n_epochs):
        structure_weight = epoch / (n_epochs - 1)  # 0 to 1
        
        # Structured component
        latent = np.random.randn(n_samples, 50)
        projection = np.random.randn(50, n_features)
        structured = latent @ projection
        structured = structured / np.std(structured)
        
        # Random component
        random = np.random.randn(n_samples, n_features)
        
        # Mix based on epoch
        X = structure_weight * structured + (1 - structure_weight) * random
        embeddings.append(X)
    
    return embeddings

# Generate trajectory
embeddings_over_time = simulate_training_trajectory(n_epochs=10)

# Track stability
stabilities = []
for epoch, X in enumerate(embeddings_over_time):
    stability = shesha.feature_split(X, n_splits=30, seed=320)
    stabilities.append(stability)
    print(f"Epoch {epoch}: stability = {stability:.3f}")

# %%
# Plot training trajectory
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(range(len(stabilities)), stabilities, 'o-', linewidth=2, markersize=8, color='steelblue')
ax.fill_between(range(len(stabilities)), stabilities, alpha=0.3, color='steelblue')

ax.set_xlabel('Epoch')
ax.set_ylabel('Feature-Split Stability')
ax.set_title('Representation Stability During Training')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tutorial_training_drift.png', dpi=150)
plt.show()

# %% [markdown]
# ## 8. Unified Interface
# 
# Use `shesha.shesha()` for a single entry point to all variants.

# %%
X = np.random.randn(500, 768)
y = np.random.randint(0, 5, 500)

# All variants through unified interface
print("Unified interface examples:")
print(f"  feature_split: {shesha.shesha(X, variant='feature_split', seed=320):.3f}")
print(f"  sample_split:  {shesha.shesha(X, variant='sample_split', seed=320):.3f}")
print(f"  anchor:        {shesha.shesha(X, variant='anchor', seed=320):.3f}")
print(f"  variance:      {shesha.shesha(X, y, variant='variance'):.3f}")
print(f"  supervised:    {shesha.shesha(X, y, variant='supervised', seed=320):.3f}")

# %% [markdown]
# ## 9. Tips and Best Practices
# 
# ### Choosing a variant:
# - **feature_split**: Default choice for unsupervised analysis. Good for drift detection, intrinsic quality.
# - **sample_split**: When you care about robustness to sampling. Good for small datasets.
# - **anchor_stability**: For very large datasets where feature_split is slow.
# - **variance_ratio**: Quick supervised check. Computationally cheap.
# - **supervised_alignment**: When you want RDM-based task alignment (more nuanced than variance_ratio).
# 
# ### Parameter recommendations:
# - `n_splits=30` is usually sufficient; increase to 50+ for publication-quality results
# - `seed` should always be set for reproducibility
# - `max_samples` prevents memory issues with large datasets
# 
# ### Interpretation:
# - **feature_split > 0.7**: Strong internal consistency, distributed structure
# - **feature_split 0.3-0.7**: Moderate consistency
# - **feature_split < 0.3**: Weak consistency, possibly noisy or sparse structure
# - **variance_ratio**: Directly interpretable as "fraction of variance explained by classes"

# %% [markdown]
# ## 10. Summary
# 
# | Variant | Supervised | Best For |
# |---------|------------|----------|
# | `feature_split` | No | General stability, drift detection |
# | `sample_split` | No | Sampling robustness |
# | `anchor_stability` | No | Large-scale analysis |
# | `variance_ratio` | Yes | Quick separability check |
# | `supervised_alignment` | Yes | Task alignment |

# %%
print("Tutorial complete!")
print("\nFor more information, see: https://github.com/prashantcraju/shesha")