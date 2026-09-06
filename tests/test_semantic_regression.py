"""
Semantic regression tests: estimators must measure their claimed properties.

These tests check scientific behavior, not just that a function returns a
bounded float.
"""

import inspect

import numpy as np
import pytest

import shesha
import shesha.sim as sim
from shesha._validate import NAN_REPLACE_WARNING


def _redundant_latent(n_samples=100, n_latent=5, n_features=100, seed=320):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_samples, n_latent))
    projection = rng.standard_normal((n_latent, n_features))
    return latent @ projection


def _concentrated_latent(n_samples=100, n_latent=5, n_features=100, n_used=8, seed=320):
    """Same latents, but copied into only ``n_used`` axes and noise elsewhere."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_samples, n_latent))
    X = rng.standard_normal((n_samples, n_features)) * 0.01
    block = latent @ rng.standard_normal((n_latent, n_used))
    X[:, :n_used] = block
    return X


def _random_orthogonal(n, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q


class TestFeatureSplitProperties:
    def test_redundant_structure_above_independent_noise(self):
        structured = _redundant_latent()
        noise = np.random.default_rng(7).standard_normal(structured.shape)
        s_struct = shesha.feature_split(structured, n_splits=40, seed=320)
        s_noise = shesha.feature_split(noise, n_splits=40, seed=320)
        assert s_struct > 0.5
        assert s_struct > s_noise + 0.2

    def test_concentrating_latents_lowers_score(self):
        distributed = _redundant_latent(n_features=80, seed=320)
        concentrated = _concentrated_latent(n_features=80, n_used=6, seed=320)
        s_dist = shesha.feature_split(distributed, n_splits=40, seed=320)
        s_conc = shesha.feature_split(concentrated, n_splits=40, seed=320)
        assert s_dist > s_conc

    @pytest.mark.filterwarnings("ignore:Undefined distances were replaced:FutureWarning")
    def test_orthogonal_rotation_can_change_score(self):
        rng = np.random.default_rng(320)
        # Axis-aligned: each independent latent occupies a single coordinate.
        # A random feature split then compares unmatched latents, so the score
        # is low. Rotation mixes every latent into every axis.
        n_samples, n_latent, n_features = 120, 5, 80
        X = np.zeros((n_samples, n_features))
        X[:, :n_latent] = rng.standard_normal((n_samples, n_latent))
        Q = _random_orthogonal(n_features, seed=99)
        X_rot = X @ Q
        s_axis = shesha.feature_split(X, n_splits=50, seed=320)
        s_rot = shesha.feature_split(X_rot, n_splits=50, seed=320)
        assert s_rot - s_axis > 0.4
        rdm = shesha.rdm_similarity(X, X_rot, metric="euclidean")
        assert rdm > 0.99

    def test_identical_feature_permutation_leaves_score_unchanged(self):
        X = _redundant_latent(n_features=64, seed=11)
        perm = np.random.default_rng(4).permutation(X.shape[1])
        s1 = shesha.feature_split(X, n_splits=80, seed=320)
        s2 = shesha.feature_split(X[:, perm], n_splits=80, seed=321)
        assert s1 == pytest.approx(s2, abs=0.08)

    def test_per_sample_scaling_does_not_affect_cosine(self):
        X = _redundant_latent(n_features=64, seed=21)
        scales = np.random.default_rng(3).uniform(0.5, 3.0, size=(X.shape[0], 1))
        s1 = shesha.feature_split(X, n_splits=30, seed=320, metric="cosine")
        s2 = shesha.feature_split(X * scales, n_splits=30, seed=320, metric="cosine")
        assert s1 == pytest.approx(s2, abs=1e-10)

    def test_more_splits_reduces_monte_carlo_error(self):
        X = _redundant_latent(n_features=64, seed=5)
        few = shesha.feature_split(X, n_splits=8, seed=320, return_all_splits=True)
        many = shesha.feature_split(X, n_splits=80, seed=320, return_all_splits=True)
        se_few = np.std(few["split_scores"]) / np.sqrt(len(few["split_scores"]))
        se_many = np.std(many["split_scores"]) / np.sqrt(len(many["split_scores"]))
        assert se_many < se_few


class TestInvalidEstimators:
    def test_sample_split_warns(self):
        X = np.random.default_rng(0).standard_normal((80, 32))
        with pytest.warns(FutureWarning, match="unmatched RDM entries"):
            shesha.sample_split(X, n_splits=5, seed=320)

    def test_anchor_stability_warns(self):
        X = np.random.default_rng(0).standard_normal((400, 32))
        with pytest.warns(FutureWarning, match="different probe observations"):
            shesha.anchor_stability(X, n_splits=5, seed=320)

    def test_estimand_warning_cannot_be_bypassed(self):
        assert "_skip_estimand_warning" not in inspect.signature(shesha.sample_split).parameters
        assert "_skip_estimand_warning" not in inspect.signature(shesha.anchor_stability).parameters

    def test_unified_wrapper_warns_for_sample_split(self):
        X = np.random.default_rng(0).standard_normal((80, 32))
        with pytest.warns(FutureWarning, match="unmatched RDM entries"):
            shesha.shesha(X, variant="sample_split", n_splits=5, seed=320)

    def test_redundant_geometry_sample_methods_near_zero(self):
        X = _redundant_latent(n_samples=120, n_latent=5, n_features=100, seed=320)
        feat = shesha.feature_split(X, n_splits=30, seed=320)
        with pytest.warns(FutureWarning):
            samp = shesha.sample_split(X, n_splits=20, seed=320)
        with pytest.warns(FutureWarning):
            anchor = shesha.anchor_stability(X, n_splits=15, n_anchors=20, n_per_split=30, seed=320)
        assert feat > 0.7
        assert abs(samp) < 0.15
        assert abs(anchor) < 0.15


class TestRdmSemantics:
    def test_identical_representations_near_one(self):
        X = _redundant_latent(n_features=40, seed=8)
        assert shesha.rdm_similarity(X, X) == pytest.approx(1.0, abs=1e-12)
        assert sim.rdm_similarity(X, X) == pytest.approx(1.0, abs=1e-12)

    def test_core_and_sim_agree_on_clean_inputs(self):
        rng = np.random.default_rng(320)
        X = rng.standard_normal((60, 24))
        Y = rng.standard_normal((60, 18))
        core = shesha.rdm_similarity(X, Y, method="spearman", metric="cosine")
        sim_val = sim.rdm_similarity(X, Y, metric="cosine", method="spearman")
        assert core == pytest.approx(sim_val, abs=1e-12)

    def test_positional_signatures_preserved(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 12))
        Y = rng.standard_normal((40, 9))
        # core: method before metric
        core_pos = shesha.rdm_similarity(X, Y, "pearson", "euclidean")
        core_kw = shesha.rdm_similarity(X, Y, method="pearson", metric="euclidean")
        assert core_pos == pytest.approx(core_kw, abs=1e-12)
        # sim: metric before method
        sim_pos = sim.rdm_similarity(X, Y, "euclidean", "pearson")
        sim_kw = sim.rdm_similarity(X, Y, metric="euclidean", method="pearson")
        assert sim_pos == pytest.approx(sim_kw, abs=1e-12)
        assert core_kw == pytest.approx(sim_kw, abs=1e-12)

    def test_legacy_scipy_metric_warns_but_still_runs(self):
        rng = np.random.default_rng(31)
        X = rng.standard_normal((30, 8))
        with pytest.warns(FutureWarning, match="0.2.x compatibility"):
            rdm = shesha.compute_rdm(X, metric="cityblock")
        assert rdm.shape == (435,)
        with pytest.warns(FutureWarning, match="0.2.x compatibility"):
            score = shesha.feature_split(X, n_splits=3, metric="cityblock", seed=1)
        assert np.isfinite(score)

    def test_fewer_than_three_samples_returns_nan(self):
        X = np.random.randn(2, 8)
        Y = np.random.randn(2, 8)
        assert np.isnan(shesha.rdm_similarity(X, Y))
        assert np.isnan(sim.rdm_similarity(X, Y))

    def test_nan_policy_raise(self):
        X = np.random.default_rng(2).standard_normal((30, 10))
        Y = X.copy()
        X[0] = 0.0
        with pytest.raises(ValueError, match="Undefined distances"):
            shesha.rdm_similarity(X, Y, metric="cosine", nan_policy="raise")

    def test_nan_policy_replace_warns(self):
        X = np.random.default_rng(2).standard_normal((30, 10))
        Y = X.copy()
        X[0] = 0.0
        with pytest.warns(FutureWarning, match="nan_policy='replace'"):
            val = shesha.rdm_similarity(X, Y, metric="cosine", nan_policy="replace")
        assert isinstance(val, float)

    def test_nan_policy_propagate(self):
        X = np.random.default_rng(2).standard_normal((30, 10))
        Y = X.copy()
        X[0] = 0.0
        val = shesha.rdm_similarity(X, Y, metric="cosine", nan_policy="propagate")
        assert np.isnan(val)

    def test_feature_split_partial_nan_propagates_entire_result(self):
        rng = np.random.default_rng(18)
        X = rng.standard_normal((20, 8))
        X[0] = [0, 0, 0, 0, 1, 2, 3, 4]
        result = shesha.feature_split(
            X,
            n_splits=8,
            seed=0,
            metric="correlation",
            nan_policy="propagate",
            return_all_splits=True,
        )
        assert np.isnan(result["mean"])
        assert result["split_scores"] == []

    def test_compute_rdm_omit_rejects_broken_condensed_shape(self):
        X = np.random.default_rng(2).standard_normal((20, 10))
        X[0] = 0.0
        with pytest.raises(ValueError, match="condensed-RDM shape"):
            shesha.compute_rdm(X, nan_policy="omit")

    @pytest.mark.parametrize(
        "func",
        [shesha.rdm_similarity, shesha.rdm_drift, sim.rdm_similarity],
    )
    def test_bootstrap_validates_paired_sample_counts(self, func):
        X = np.ones((10, 3))
        Y = np.ones((5, 3))
        with pytest.raises(ValueError, match="same number of samples"):
            func(X, Y, n_bootstrap_ci=3, seed=1)

    def test_feature_split_nan_policy_raise(self):
        X = np.random.default_rng(3).standard_normal((40, 20))
        X[0] = 0.0
        with pytest.raises(ValueError, match="Undefined distances"):
            shesha.feature_split(X, n_splits=5, seed=320, nan_policy="raise")


class TestInputValidation:
    def test_1d_input_raises(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            shesha.feature_split(np.random.randn(20), n_splits=5)

    def test_nonfinite_input_raises(self):
        X = np.random.randn(20, 10)
        X[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            shesha.feature_split(X, n_splits=5)

    def test_label_length_mismatch_raises(self):
        X = np.random.randn(20, 8)
        y = np.arange(19)
        with pytest.raises(ValueError, match="must match number of samples"):
            shesha.variance_ratio(X, y)

    def test_labels_must_be_one_dimensional(self):
        X = np.random.randn(20, 8)
        y = np.arange(20).reshape(2, 10)
        with pytest.raises(ValueError, match="1-dimensional"):
            shesha.variance_ratio(X, y)

    @pytest.mark.parametrize(
        "func",
        [shesha.variance_ratio, shesha.supervised_alignment, shesha.class_separation_ratio],
    )
    def test_singleton_class_raises(self, func):
        X = np.random.default_rng(0).standard_normal((5, 3))
        y = np.array([0, 1, 1, 1, 1])
        with pytest.raises(ValueError, match="at least 2 observations"):
            func(X, y)

    def test_lda_singleton_class_raises(self):
        X = np.random.default_rng(0).standard_normal((5, 3))
        y = np.array([0, 1, 1, 1, 1])
        with pytest.raises(ValueError, match="at least 2 observations"):
            shesha.lda_stability(X, y, n_bootstrap=3)

    def test_invalid_fraction_raises(self):
        X = np.random.randn(40, 12)
        with pytest.warns(FutureWarning):
            with pytest.raises(ValueError, match="subsample_fraction"):
                shesha.sample_split(X, subsample_fraction=0.0)

    def test_invalid_ci_type_raises_value_error(self):
        with pytest.raises(ValueError, match="ci must be"):
            shesha.variance_ratio(np.ones((4, 2)), [0, 0, 1, 1], ci="bad")

    def test_invalid_nan_policy_raises(self):
        X = np.random.randn(20, 10)
        with pytest.raises(ValueError, match="nan_policy"):
            shesha.feature_split(X, n_splits=5, nan_policy="drop")

    def test_unknown_metric_raises(self):
        X = np.random.randn(20, 10)
        with pytest.raises(ValueError, match="Unknown metric"):
            shesha.feature_split(X, n_splits=5, metric="manhattan")


def test_replace_warning_message_mentions_03():
    assert "0.3.0" in NAN_REPLACE_WARNING
