"""Tests for bootstrap confidence interval functionality."""

import numpy as np
import pytest
import shesha
from shesha.bio import perturbation_stability, perturbation_effect_size
from shesha.sim import cka, cka_linear, cka_debiased, procrustes_similarity
from shesha.sim import rdm_similarity as sim_rdm_similarity


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def X():
    rng = np.random.default_rng(320)
    return rng.standard_normal((100, 50))


@pytest.fixture
def X_structured():
    """Data with strong internal structure (high stability)."""
    rng = np.random.default_rng(320)
    latent = rng.standard_normal((100, 5))
    projection = rng.standard_normal((5, 50))
    return latent @ projection


@pytest.fixture
def y():
    rng = np.random.default_rng(320)
    return rng.integers(0, 3, size=100)


@pytest.fixture
def y_binary():
    return np.array([0] * 50 + [1] * 50)


@pytest.fixture
def X2():
    rng = np.random.default_rng(99)
    return rng.standard_normal((100, 50))


@pytest.fixture
def bio_data():
    rng = np.random.default_rng(320)
    X_ctrl = rng.standard_normal((200, 30))
    shift = rng.standard_normal(30) * 2
    X_pert = X_ctrl[:100] + shift + rng.standard_normal((100, 30)) * 0.3
    return X_ctrl, X_pert


# ============================================================================
# Helper to validate CI dict structure
# ============================================================================

def assert_valid_ci_dict(result, ci_level=0.95):
    """Check that the result is a well-formed CI dictionary."""
    assert isinstance(result, dict)
    required_keys = {"mean", "ci_low", "ci_high", "std", "n_bootstraps", "ci_level"}
    assert set(result.keys()) == required_keys

    assert isinstance(result["mean"], float)
    assert isinstance(result["ci_low"], float)
    assert isinstance(result["ci_high"], float)
    assert isinstance(result["std"], float)
    assert isinstance(result["n_bootstraps"], int)
    assert result["ci_level"] == ci_level

    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
    assert result["std"] >= 0
    assert result["n_bootstraps"] > 0


# ============================================================================
# Core module tests
# ============================================================================

class TestFeatureSplitCI:
    def test_returns_dict(self, X):
        result = shesha.feature_split(X, n_splits=5, seed=320, n_bootstrap_ci=20)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, X):
        result = shesha.feature_split(X, n_splits=5, seed=320)
        assert isinstance(result, float)

    def test_deterministic(self, X):
        r1 = shesha.feature_split(X, n_splits=5, seed=320, n_bootstrap_ci=20)
        r2 = shesha.feature_split(X, n_splits=5, seed=320, n_bootstrap_ci=20)
        assert r1 == r2

    def test_custom_ci_level(self, X):
        result = shesha.feature_split(X, n_splits=5, seed=320, n_bootstrap_ci=50, ci=0.99)
        assert_valid_ci_dict(result, ci_level=0.99)

    def test_wider_ci_at_higher_level(self, X):
        r95 = shesha.feature_split(X, n_splits=5, seed=320, n_bootstrap_ci=200)
        r99 = shesha.feature_split(X, n_splits=5, seed=320, n_bootstrap_ci=200, ci=0.99)
        width_95 = r95["ci_high"] - r95["ci_low"]
        width_99 = r99["ci_high"] - r99["ci_low"]
        assert width_99 >= width_95

    def test_structured_data_tight_ci(self, X_structured):
        result = shesha.feature_split(X_structured, n_splits=10, seed=320, n_bootstrap_ci=100)
        assert_valid_ci_dict(result)
        assert result["std"] < 0.1


class TestSampleSplitCI:
    def test_returns_dict(self, X):
        result = shesha.sample_split(X, n_splits=5, seed=320, n_bootstrap_ci=20)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, X):
        result = shesha.sample_split(X, n_splits=5, seed=320)
        assert isinstance(result, float)


class TestAnchorStabilityCI:
    def test_returns_dict(self, X):
        result = shesha.anchor_stability(X, n_splits=5, seed=320, n_bootstrap_ci=20)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, X):
        result = shesha.anchor_stability(X, n_splits=5, seed=320)
        assert isinstance(result, float)


class TestVarianceRatioCI:
    def test_returns_dict(self, X, y):
        result = shesha.variance_ratio(X, y, n_bootstrap_ci=20, seed=320)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, X, y):
        result = shesha.variance_ratio(X, y)
        assert isinstance(result, float)

    def test_deterministic(self, X, y):
        r1 = shesha.variance_ratio(X, y, n_bootstrap_ci=20, seed=320)
        r2 = shesha.variance_ratio(X, y, n_bootstrap_ci=20, seed=320)
        assert r1 == r2


class TestSupervisedAlignmentCI:
    def test_returns_dict(self, X, y):
        result = shesha.supervised_alignment(X, y, seed=320, n_bootstrap_ci=20)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, X, y):
        result = shesha.supervised_alignment(X, y, seed=320)
        assert isinstance(result, float)


class TestClassSeparationRatioCI:
    def test_returns_dict(self, X, y):
        result = shesha.class_separation_ratio(X, y, n_bootstrap=5, seed=320, n_bootstrap_ci=20)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, X, y):
        result = shesha.class_separation_ratio(X, y, n_bootstrap=5, seed=320)
        assert isinstance(result, float)


class TestLdaStabilityCI:
    def test_returns_dict(self, X, y_binary):
        result = shesha.lda_stability(X, y_binary, n_bootstrap=5, seed=320, n_bootstrap_ci=20)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, X, y_binary):
        result = shesha.lda_stability(X, y_binary, n_bootstrap=5, seed=320)
        assert isinstance(result, float)


class TestRdmSimilarityCI:
    def test_returns_dict(self, X, X2):
        result = shesha.rdm_similarity(X, X2, n_bootstrap_ci=20, seed=320)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, X, X2):
        result = shesha.rdm_similarity(X, X2)
        assert isinstance(result, float)

    def test_self_similarity_high(self, X):
        result = shesha.rdm_similarity(X, X, n_bootstrap_ci=50, seed=320)
        assert result["mean"] > 0.95


class TestRdmDriftCI:
    def test_returns_dict(self, X, X2):
        result = shesha.rdm_drift(X, X2, n_bootstrap_ci=20, seed=320)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, X, X2):
        result = shesha.rdm_drift(X, X2)
        assert isinstance(result, float)


# ============================================================================
# Bio module tests
# ============================================================================

class TestPerturbationStabilityCI:
    def test_returns_dict(self, bio_data):
        X_ctrl, X_pert = bio_data
        result = perturbation_stability(X_ctrl, X_pert, n_bootstrap_ci=20, seed=320)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, bio_data):
        X_ctrl, X_pert = bio_data
        result = perturbation_stability(X_ctrl, X_pert)
        assert isinstance(result, float)

    def test_strong_perturbation_high_stability(self, bio_data):
        X_ctrl, X_pert = bio_data
        result = perturbation_stability(X_ctrl, X_pert, n_bootstrap_ci=50, seed=320)
        assert result["mean"] > 0.7
        assert result["ci_low"] > 0.5

    def test_methods(self, bio_data):
        X_ctrl, X_pert = bio_data
        for method in ["standard", "whitened"]:
            result = perturbation_stability(
                X_ctrl, X_pert, method=method, n_bootstrap_ci=20, seed=320
            )
            assert_valid_ci_dict(result)


class TestPerturbationEffectSizeCI:
    def test_returns_dict(self, bio_data):
        X_ctrl, X_pert = bio_data
        result = perturbation_effect_size(X_ctrl, X_pert, n_bootstrap_ci=20, seed=320)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, bio_data):
        X_ctrl, X_pert = bio_data
        result = perturbation_effect_size(X_ctrl, X_pert)
        assert isinstance(result, float)

    def test_positive_effect_size(self, bio_data):
        X_ctrl, X_pert = bio_data
        result = perturbation_effect_size(X_ctrl, X_pert, n_bootstrap_ci=50, seed=320)
        assert result["mean"] > 0
        assert result["ci_low"] > 0


# ============================================================================
# Sim module tests
# ============================================================================

class TestCkaLinearCI:
    def test_returns_dict(self, X, X2):
        result = cka_linear(X, X2, n_bootstrap_ci=20, seed=320)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, X, X2):
        result = cka_linear(X, X2)
        assert isinstance(result, float)

    def test_self_similarity(self, X):
        result = cka_linear(X, X, n_bootstrap_ci=50, seed=320)
        assert result["mean"] > 0.95


class TestCkaDebiasedCI:
    def test_returns_dict(self, X, X2):
        result = cka_debiased(X, X2, n_bootstrap_ci=20, seed=320)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, X, X2):
        result = cka_debiased(X, X2)
        assert isinstance(result, float)


class TestCkaUnifiedCI:
    def test_standard(self, X, X2):
        result = cka(X, X2, n_bootstrap_ci=20, seed=320)
        assert_valid_ci_dict(result)

    def test_debiased(self, X, X2):
        result = cka(X, X2, debiased=True, n_bootstrap_ci=20, seed=320)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, X, X2):
        result = cka(X, X2)
        assert isinstance(result, float)


class TestProcrustesCI:
    def test_returns_dict(self, X, X2):
        result = procrustes_similarity(X, X2, n_bootstrap_ci=20, seed=320)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, X, X2):
        result = procrustes_similarity(X, X2)
        assert isinstance(result, float)

    def test_rotated_copy_high_similarity(self, X):
        rng = np.random.default_rng(7)
        Q = np.linalg.qr(rng.standard_normal((50, 50)))[0]
        Y = X @ Q
        result = procrustes_similarity(X, Y, n_bootstrap_ci=50, seed=320)
        assert result["mean"] > 0.95


class TestSimRdmSimilarityCI:
    def test_returns_dict(self, X, X2):
        result = sim_rdm_similarity(X, X2, n_bootstrap_ci=20, seed=320)
        assert_valid_ci_dict(result)

    def test_backward_compat(self, X, X2):
        result = sim_rdm_similarity(X, X2)
        assert isinstance(result, float)


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_n_bootstrap_ci_zero_returns_float(self, X):
        """n_bootstrap_ci=None should return float (not triggered)."""
        result = shesha.feature_split(X, n_splits=5, seed=320, n_bootstrap_ci=None)
        assert isinstance(result, float)

    def test_small_n_bootstrap(self, X):
        """Even very small n_bootstrap_ci should work."""
        result = shesha.feature_split(X, n_splits=5, seed=320, n_bootstrap_ci=3)
        assert_valid_ci_dict(result)

    def test_ci_bounds(self, X):
        """CI level 0.5 should give narrower interval than 0.99."""
        r50 = shesha.feature_split(X, n_splits=5, seed=320, n_bootstrap_ci=100, ci=0.50)
        r99 = shesha.feature_split(X, n_splits=5, seed=320, n_bootstrap_ci=100, ci=0.99)
        assert (r99["ci_high"] - r99["ci_low"]) >= (r50["ci_high"] - r50["ci_low"])
