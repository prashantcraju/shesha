import numpy as np
from shesha.bio import perturbation_stability, perturbation_effect_size

# Test 1: Basic functionality
print("=== Test 1: Basic functionality ===")
X_ctrl = np.random.randn(200, 50)
X_pert = np.random.randn(200, 50)
stability = perturbation_stability(X_ctrl, X_pert, seed=320)
effect = perturbation_effect_size(X_ctrl, X_pert)
print(f"Random data - Stability: {stability:.3f}, Effect size: {effect:.3f}")

# Test 2: Coherent perturbation (should have HIGH stability)
print("\n=== Test 2: Coherent perturbation ===")
X_ctrl = np.random.randn(200, 50)
shift = np.random.randn(50) * 3  # Same direction for all cells
X_pert = X_ctrl + shift + np.random.randn(200, 50) * 0.1  # Small noise
stability = perturbation_stability(X_ctrl, X_pert, seed=320)
effect = perturbation_effect_size(X_ctrl, X_pert)
print(f"Coherent shift - Stability: {stability:.3f} (should be >0.8), Effect: {effect:.3f}")

# Test 3: Incoherent perturbation (should have LOW stability)
print("\n=== Test 3: Incoherent perturbation ===")
X_ctrl = np.random.randn(200, 50)
X_pert = X_ctrl + np.random.randn(200, 50)  # Each cell shifts randomly
stability = perturbation_stability(X_ctrl, X_pert, seed=320)
effect = perturbation_effect_size(X_ctrl, X_pert)
print(f"Random shifts - Stability: {stability:.3f} (should be <0.5), Effect: {effect:.3f}")

# Test 4: Large vs small effect
print("\n=== Test 4: Effect size comparison ===")
X_ctrl = np.random.randn(200, 50)
X_pert_small = X_ctrl + 0.1  # Small shift
X_pert_large = X_ctrl + 5.0  # Large shift
effect_small = perturbation_effect_size(X_ctrl, X_pert_small)
effect_large = perturbation_effect_size(X_ctrl, X_pert_large)
print(f"Small shift effect: {effect_small:.3f}")
print(f"Large shift effect: {effect_large:.3f} (should be >> small)")

# Test 5: Determinism
print("\n=== Test 5: Determinism ===")
X_ctrl = np.random.randn(200, 50)
X_pert = np.random.randn(200, 50)
r1 = perturbation_stability(X_ctrl, X_pert, seed=320)
r2 = perturbation_stability(X_ctrl, X_pert, seed=320)
print(f"Run 1: {r1:.6f}")
print(f"Run 2: {r2:.6f}")
print(f"Deterministic: {r1 == r2}")

print("\n=== All tests complete ===")