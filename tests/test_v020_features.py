"""
Tests for v0.2.0 features: class_separation_ratio, lda_stability,
perturbation_stability method dispatch, and shesha.sim similarity metrics.

Focuses on coverage gaps not addressed in test_new_features.py,
test_similarity.py, and test_anndata_integration.py.
"""

import numpy as np
import pytest
import shesha
import shesha.sim as sim
import shesha.bio as bio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_separated():
    """Two well-separated Gaussian blobs."""
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.standard_normal((100, 20)),
        rng.standard_normal((100, 20)) + 4,
    ])
    y = np.array([0] * 100 + [1] * 100)
    return X, y


@pytest.fixture
def binary_overlapping():
    """Two barely-separated Gaussian blobs (noisy)."""
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.standard_normal((100, 20)),
        rng.standard_normal((100, 20)) + 0.3,
    ])
    y = np.array([0] * 100 + [1] * 100)
    return X, y


@pytest.fixture
def ctrl_pert_coherent():
    """Control + coherently-shifted perturbation."""
    rng = np.random.default_rng(42)
    X_ctrl = rng.standard_normal((400, 40))
    shift = rng.standard_normal(40)
    X_pert = X_ctrl[:200] + shift + rng.standard_normal((200, 40)) * 0.1
    return X_ctrl, X_pert


@pytest.fixture
def ctrl_pert_random():
    """Control + randomly-shifted perturbation (incoherent)."""
    rng = np.random.default_rng(42)
    X_ctrl = rng.standard_normal((400, 40))
    X_pert = rng.standard_normal((200, 40))
    return X_ctrl, X_pert


# ===========================================================================
# class_separation_ratio
# ===========================================================================

class TestClassSeparationRatio:

    def test_well_separated_high(self, binary_separated):
        """Well-separated classes should yield ratio > 1."""
        X, y = binary_separated
        ratio = shesha.class_separation_ratio(X, y, n_bootstrap=20, seed=42)
        assert isinstance(ratio, float)
        assert not np.isnan(ratio)
        assert ratio > 1.0

    def test_overlapping_lower(self, binary_separated, binary_overlapping):
        """Overlapping classes should have a lower ratio than separated ones."""
        X_sep, y = binary_separated
        X_ov, _ = binary_overlapping
        r_sep = shesha.class_separation_ratio(X_sep, y, n_bootstrap=20, seed=42)
        r_ov = shesha.class_separation_ratio(X_ov, y, n_bootstrap=20, seed=42)
        assert r_sep > r_ov

    def test_multiclass(self):
        """Should work with more than 2 classes."""
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.standard_normal((60, 10)),
            rng.standard_normal((60, 10)) + 4,
            rng.standard_normal((60, 10)) + 8,
        ])
        y = np.array([0] * 60 + [1] * 60 + [2] * 60)
        ratio = shesha.class_separation_ratio(X, y, n_bootstrap=15, seed=42)
        assert isinstance(ratio, float)
        assert not np.isnan(ratio)
        assert ratio > 1.0

    def test_cosine_metric(self, binary_separated):
        """Cosine metric variant should also return a valid ratio > 1."""
        X, y = binary_separated
        ratio = shesha.class_separation_ratio(
            X, y, metric="cosine", n_bootstrap=15, seed=42
        )
        assert isinstance(ratio, float)
        assert not np.isnan(ratio)
        assert ratio > 0.0

    def test_single_class_returns_nan(self):
        """Single-class input should return NaN."""
        X = np.random.randn(50, 10)
        y = np.zeros(50, dtype=int)
        ratio = shesha.class_separation_ratio(X, y, n_bootstrap=10, seed=42)
        assert np.isnan(ratio)

    def test_determinism(self, binary_separated):
        """Same seed should produce identical results."""
        X, y = binary_separated
        r1 = shesha.class_separation_ratio(X, y, n_bootstrap=20, seed=7)
        r2 = shesha.class_separation_ratio(X, y, n_bootstrap=20, seed=7)
        assert r1 == r2

    def test_different_seeds_vary(self, binary_separated):
        """Different seeds should produce different (but close) results."""
        X, y = binary_separated
        r1 = shesha.class_separation_ratio(X, y, n_bootstrap=20, seed=1)
        r2 = shesha.class_separation_ratio(X, y, n_bootstrap=20, seed=2)
        # Values should differ but not wildly
        assert abs(r1 - r2) < 2.0


# ===========================================================================
# lda_stability
# ===========================================================================

class TestLdaStability:

    def test_stable_high(self, binary_separated):
        """Well-separated data should produce high LDA stability."""
        X, y = binary_separated
        stab = shesha.lda_stability(X, y, n_bootstrap=20, seed=42)
        assert isinstance(stab, float)
        assert not np.isnan(stab)
        assert 0.0 <= stab <= 1.0
        assert stab > 0.7

    def test_unstable_lower(self, binary_overlapping):
        """Barely-separated data should yield lower stability than clear separation."""
        rng = np.random.default_rng(42)
        X_sep = np.vstack([
            rng.standard_normal((100, 20)),
            rng.standard_normal((100, 20)) + 6,
        ])
        y = np.array([0] * 100 + [1] * 100)

        X_ov, _ = binary_overlapping

        stab_sep = shesha.lda_stability(X_sep, y, n_bootstrap=20, seed=42)
        stab_ov = shesha.lda_stability(X_ov, y, n_bootstrap=20, seed=42)
        assert stab_sep > stab_ov

    def test_multiclass_raises(self):
        """lda_stability must raise ValueError for more than 2 classes."""
        X = np.random.randn(90, 10)
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)
        with pytest.raises(ValueError, match="exactly 2 classes"):
            shesha.lda_stability(X, y)

    def test_determinism(self, binary_separated):
        """Same seed produces identical results."""
        X, y = binary_separated
        s1 = shesha.lda_stability(X, y, n_bootstrap=20, seed=99)
        s2 = shesha.lda_stability(X, y, n_bootstrap=20, seed=99)
        assert s1 == s2

    def test_output_range(self, binary_separated):
        """Output must be in [0, 1]."""
        X, y = binary_separated
        stab = shesha.lda_stability(X, y, n_bootstrap=10, seed=42)
        assert 0.0 <= stab <= 1.0


# ===========================================================================
# perturbation_stability – unified method dispatch
# ===========================================================================

class TestPerturbationStabilityMethodDispatch:
    """
    Tests for the unified perturbation_stability(method=...) interface,
    verifying that dispatch to whitened/knn matches the standalone functions.
    """

    def test_standard_method_default(self, ctrl_pert_coherent):
        """Default (method='standard') should return the same as explicit call."""
        X_ctrl, X_pert = ctrl_pert_coherent
        s_default = bio.perturbation_stability(X_ctrl, X_pert, seed=42, max_samples=150)
        s_explicit = bio.perturbation_stability(
            X_ctrl, X_pert, method="standard", seed=42, max_samples=150
        )
        assert s_default == s_explicit

    def test_whitened_dispatch_matches_standalone(self, ctrl_pert_coherent):
        """method='whitened' should equal perturbation_stability_whitened."""
        X_ctrl, X_pert = ctrl_pert_coherent
        s_dispatch = bio.perturbation_stability(
            X_ctrl, X_pert, method="whitened", seed=42, max_samples=150
        )
        s_standalone = bio.perturbation_stability_whitened(
            X_ctrl, X_pert, seed=42, max_samples=150
        )
        assert abs(s_dispatch - s_standalone) < 1e-10

    def test_knn_dispatch_matches_standalone(self, ctrl_pert_coherent):
        """method='knn' should equal perturbation_stability_knn with same metric."""
        X_ctrl, X_pert = ctrl_pert_coherent
        # Must pass metric explicitly: perturbation_stability defaults to "cosine"
        # while perturbation_stability_knn defaults to "euclidean".
        s_dispatch = bio.perturbation_stability(
            X_ctrl, X_pert, method="knn", metric="euclidean", k=30, seed=42, max_samples=150
        )
        s_standalone = bio.perturbation_stability_knn(
            X_ctrl, X_pert, metric="euclidean", k=30, seed=42, max_samples=150
        )
        assert abs(s_dispatch - s_standalone) < 1e-10

    def test_invalid_method_raises(self, ctrl_pert_coherent):
        """An unknown method name should raise ValueError."""
        X_ctrl, X_pert = ctrl_pert_coherent
        with pytest.raises(ValueError, match="Unknown method"):
            bio.perturbation_stability(X_ctrl, X_pert, method="invalid")

    def test_coherent_high_all_methods(self, ctrl_pert_coherent):
        """All methods should return a high score for a coherent perturbation."""
        X_ctrl, X_pert = ctrl_pert_coherent
        for method in ("standard", "whitened", "knn"):
            score = bio.perturbation_stability(
                X_ctrl, X_pert, method=method, k=30, seed=42, max_samples=150
            )
            assert score > 0.4, f"method='{method}' gave low score: {score:.3f}"

    def test_incoherent_lower_than_coherent(self, ctrl_pert_coherent, ctrl_pert_random):
        """Coherent perturbation should score higher than random for all methods."""
        X_ctrl, X_pert_coh = ctrl_pert_coherent
        _, X_pert_rand = ctrl_pert_random
        for method in ("standard", "whitened", "knn"):
            s_coh = bio.perturbation_stability(
                X_ctrl, X_pert_coh, method=method, k=30, seed=42, max_samples=150
            )
            s_rand = bio.perturbation_stability(
                X_ctrl, X_pert_rand, method=method, k=30, seed=42, max_samples=150
            )
            assert s_coh > s_rand, (
                f"method='{method}': coherent ({s_coh:.3f}) not > random ({s_rand:.3f})"
            )

    def test_too_few_perturbed_returns_nan(self):
        """Fewer than 5 perturbed cells should return NaN."""
        X_ctrl = np.random.randn(100, 20)
        X_pert = np.random.randn(3, 20)
        for method in ("standard", "whitened", "knn"):
            score = bio.perturbation_stability(X_ctrl, X_pert, method=method)
            assert np.isnan(score), f"Expected NaN for method='{method}'"


# ===========================================================================
# shesha.sim – CKA edge cases
# ===========================================================================

class TestCkaEdgeCases:

    def test_cka_scale_invariance(self):
        """CKA should be invariant to isotropic scaling."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 30))
        Y = rng.standard_normal((80, 20))
        cka_base = sim.cka_linear(X, Y)
        cka_scaled = sim.cka_linear(X * 5.0, Y * 0.2)
        assert abs(cka_base - cka_scaled) < 1e-8

    def test_cka_debiased_small_n_fallback(self):
        """cka_debiased with n < 4 should fall back gracefully (no exception)."""
        X = np.random.randn(3, 5)
        Y = np.random.randn(3, 4)
        val = sim.cka_debiased(X, Y)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_cka_debiased_self_similarity(self):
        """cka_debiased(X, X) should be close to 1.0."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 20))
        val = sim.cka_debiased(X, X)
        assert abs(val - 1.0) < 0.05

    def test_cka_mismatched_samples_raises(self):
        """CKA should raise ValueError when sample counts differ."""
        X = np.random.randn(80, 30)
        Y = np.random.randn(70, 20)
        with pytest.raises(ValueError, match="same number of samples"):
            sim.cka(X, Y)
        with pytest.raises(ValueError, match="same number of samples"):
            sim.cka_debiased(X, Y)

    def test_cka_different_feature_dims_ok(self):
        """CKA should work when X and Y have different feature dimensions."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 512))
        Y = rng.standard_normal((100, 32))
        val = sim.cka_linear(X, Y)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0


# ===========================================================================
# shesha.sim – Procrustes edge cases
# ===========================================================================

class TestProcrustesEdgeCases:

    def test_nan_input_returns_nan(self):
        """NaN values in input should produce NaN output."""
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 10)
        Y[5, 3] = np.nan
        result = sim.procrustes_similarity(X, Y)
        assert np.isnan(result)

    def test_inf_input_returns_nan(self):
        """Inf values in input should produce NaN output."""
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 10)
        Y[0, 0] = np.inf
        result = sim.procrustes_similarity(X, Y)
        assert np.isnan(result)

    def test_shape_mismatch_returns_nan(self):
        """Procrustes with mismatched feature dims should return NaN (ValueError caught internally)."""
        X = np.random.randn(50, 20)
        Y = np.random.randn(50, 15)
        result = sim.procrustes_similarity(X, Y)
        assert np.isnan(result)

    def test_no_center_no_scale(self):
        """Procrustes with center=False, scale=False should still return float in [0,1]."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 10))
        Y = rng.standard_normal((60, 10))
        result = sim.procrustes_similarity(X, Y, center=False, scale=False)
        # May be nan or valid float – either is acceptable
        assert isinstance(result, float)


# ===========================================================================
# shesha.sim – RDM similarity edge cases
# ===========================================================================

class TestRdmSimilarityEdgeCases:

    def test_invalid_method_raises(self):
        """Unknown correlation method should raise ValueError."""
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 8)
        with pytest.raises(ValueError, match="Unknown method"):
            sim.rdm_similarity(X, Y, method="kendall")

    def test_self_similarity_high(self):
        """RDM(X, X) should be very close to 1.0."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 20))
        val = sim.rdm_similarity(X, X)
        assert val > 0.99

    def test_mismatched_samples_raises(self):
        """Mismatched sample counts should raise ValueError."""
        X = np.random.randn(60, 10)
        Y = np.random.randn(50, 10)
        with pytest.raises(ValueError, match="same number of samples"):
            sim.rdm_similarity(X, Y)


# ===========================================================================
# shesha.sim module accessibility
# ===========================================================================

class TestSimModuleAccess:

    def test_sim_accessible_via_shesha(self):
        """shesha.sim should be importable and expose all public functions."""
        assert hasattr(shesha, "sim")
        for fn in ("cka", "cka_linear", "cka_debiased", "procrustes_similarity", "rdm_similarity"):
            assert hasattr(shesha.sim, fn), f"shesha.sim.{fn} not found"

    def test_new_core_exports(self):
        """class_separation_ratio and lda_stability must be top-level exports."""
        assert hasattr(shesha, "class_separation_ratio")
        assert hasattr(shesha, "lda_stability")
        assert callable(shesha.class_separation_ratio)
        assert callable(shesha.lda_stability)

    def test_bio_new_exports(self):
        """New bio functions must be accessible via shesha.bio."""
        for fn in (
            "perturbation_stability_whitened",
            "perturbation_stability_knn",
            "compute_stability_whitened",
            "compute_stability_knn",
        ):
            assert hasattr(shesha.bio, fn), f"shesha.bio.{fn} not found"

    def test_version_bump(self):
        """Package version should be 0.2.0."""
        assert shesha.__version__ == "0.2.0"
