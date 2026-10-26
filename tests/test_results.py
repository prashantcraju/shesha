"""
StabilityResult support across metrics and convergence-based feature splits.
"""

import json

import numpy as np
import pytest

import shesha
from shesha import bio, sim


def _paired(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((80, 12))
    Y = X @ rng.standard_normal((12, 12)) + 0.5 * rng.standard_normal((80, 12))
    return X, Y


def _labeled(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((80, 12))
    y = np.repeat([0, 1], 40)
    X[y == 1] += 0.8
    return X, y


def _populations(seed=0):
    rng = np.random.default_rng(seed)
    X_ctrl = rng.standard_normal((60, 10))
    return X_ctrl, X_ctrl + 1.0 + 0.3 * rng.standard_normal((60, 10))


CASES = [
    ("variance_ratio", shesha.variance_ratio, _labeled, {}),
    ("supervised_alignment", shesha.supervised_alignment, _labeled, {"seed": 1}),
    (
        "class_separation_ratio",
        shesha.class_separation_ratio,
        _labeled,
        {"n_bootstrap": 5, "seed": 1},
    ),
    ("lda_stability", shesha.lda_stability, _labeled, {"n_bootstrap": 5, "seed": 1}),
    ("rdm_similarity", shesha.rdm_similarity, _paired, {}),
    ("rdm_drift", shesha.rdm_drift, _paired, {}),
    ("cka_linear", sim.cka_linear, _paired, {}),
    ("cka_debiased", sim.cka_debiased, _paired, {}),
    ("cka", sim.cka, _paired, {"debiased": True}),
    ("procrustes_similarity", sim.procrustes_similarity, _paired, {}),
    ("rdm_similarity", sim.rdm_similarity, _paired, {}),
    ("perturbation_coherence", bio.perturbation_coherence, _populations, {"seed": 1}),
    ("perturbation_effect_size", bio.perturbation_effect_size, _populations, {}),
]


@pytest.mark.parametrize("name, func, data, kwargs", CASES, ids=[c[0] for c in CASES])
class TestResultAcrossMetrics:
    def test_estimate_matches_float(self, name, func, data, kwargs):
        args = data()
        result = func(*args, return_type="result", **kwargs)
        assert isinstance(result, shesha.StabilityResult)
        assert result.estimate == pytest.approx(func(*args, **kwargs), abs=1e-12)
        assert result.metadata["estimator"] == name
        assert result.n_bootstraps == 0 and result.ci_level is None

    def test_data_interval(self, name, func, data, kwargs):
        args = data()
        kwargs = {key: value for key, value in kwargs.items() if key != "seed"}
        result = func(*args, n_bootstrap_ci=25, seed=3, ci=0.9, return_type="result", **kwargs)
        again = func(*args, n_bootstrap_ci=25, seed=3, ci=0.9, return_type="result", **kwargs)
        assert result == again
        assert result.ci_level == 0.9
        assert result.n_bootstraps == 25
        assert result.data_ci_low <= result.data_ci_high
        json.dumps(result.to_dict())

    def test_invalid_return_type(self, name, func, data, kwargs):
        with pytest.raises(ValueError, match="return_type"):
            func(*data(), return_type="dict", **kwargs)


class TestResultDetails:
    def test_rdm_interval_drops_duplicate_pairs(self):
        X, Y = _paired()
        result = shesha.rdm_similarity(X, Y, n_bootstrap_ci=40, seed=1, return_type="result")
        matched = shesha.replicate_stability(X, Y, n_bootstrap_ci=40, seed=1, return_type="result")
        assert (result.data_ci_low, result.data_ci_high) == (
            matched.data_ci_low,
            matched.data_ci_high,
        )

    def test_rdm_drift_interval_mirrors_similarity(self):
        X, Y = _paired()
        similarity = shesha.rdm_similarity(X, Y, n_bootstrap_ci=30, seed=2, return_type="result")
        drift = shesha.rdm_drift(X, Y, n_bootstrap_ci=30, seed=2, return_type="result")
        assert drift.estimate == pytest.approx(1 - similarity.estimate)
        assert drift.data_ci_low == pytest.approx(1 - similarity.data_ci_high)
        assert drift.data_ci_high == pytest.approx(1 - similarity.data_ci_low)

    def test_stochastic_metrics_report_unestimated_mc_se(self):
        X, y = _labeled()
        assert np.isnan(
            shesha.lda_stability(X, y, n_bootstrap=3, return_type="result").monte_carlo_se
        )
        assert shesha.variance_ratio(X, y, return_type="result").monte_carlo_se == 0.0

    def test_unestimable_input_reports_skipped_bootstrap(self):
        result = sim.procrustes_similarity(
            np.ones((10, 3)), np.ones((10, 3)), n_bootstrap_ci=5, return_type="result"
        )
        assert np.isnan(result.estimate)
        assert result.n_bootstraps == 0
        assert "5 of 5 bootstrap resamples" in result.warnings[0]


def _structured(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((150, 8)) @ rng.standard_normal((8, 64)) + rng.standard_normal(
        (150, 64)
    )


class TestFeatureSplitConvergence:
    def test_converges_to_target(self):
        result = shesha.feature_split(
            _structured(), n_splits=20, seed=1, target_mc_se=0.004, return_type="result"
        )
        assert result.metadata["converged"] is True
        assert result.monte_carlo_se <= 0.004
        assert result.n_valid_splits >= 20

    def test_loose_target_matches_fixed_splits(self):
        X = _structured()
        result = shesha.feature_split(
            X, n_splits=30, seed=1, target_mc_se=10.0, return_type="result"
        )
        assert result.n_valid_splits == 30
        assert result.estimate == shesha.feature_split(X, n_splits=30, seed=1)

    def test_prefix_of_partitions_is_shared(self):
        X = _structured()
        fixed = shesha.feature_split(X, n_splits=25, seed=4, return_type="result")
        converged = shesha.feature_split(
            X, n_splits=25, seed=4, target_mc_se=0.002, return_type="result"
        )
        assert converged.component_scores[:25] == fixed.component_scores

    def test_unconverged_warns_and_reports(self):
        X = _structured()
        with pytest.warns(UserWarning, match="did not reach target_mc_se"):
            value = shesha.feature_split(X, n_splits=5, seed=1, target_mc_se=1e-9, max_splits=12)
        assert np.isfinite(value)
        result = shesha.feature_split(
            X, n_splits=5, seed=1, target_mc_se=1e-9, max_splits=12, return_type="result"
        )
        assert result.metadata["converged"] is False
        assert result.metadata["n_partitions_drawn"] == 12
        assert any("did not reach" in note for note in result.warnings)

    def test_return_all_splits_with_convergence(self):
        out = shesha.feature_split(
            _structured(), n_splits=10, seed=1, target_mc_se=10.0, return_all_splits=True
        )
        assert len(out["split_scores"]) == 10

    def test_convergence_with_data_interval(self):
        result = shesha.feature_split(
            _structured(),
            n_splits=10,
            seed=1,
            target_mc_se=0.01,
            n_bootstrap_ci=10,
            return_type="result",
        )
        assert result.n_bootstraps == 10

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"max_splits": 50}, "max_splits requires target_mc_se"),
            ({"target_mc_se": 0.0}, "target_mc_se"),
            ({"target_mc_se": 0.01, "max_splits": 5}, "must be >= n_splits"),
            ({"target_mc_se": 0.01, "n_bootstrap_ci": 5}, "requires return_type='result'"),
        ],
    )
    def test_invalid_convergence_options(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            shesha.feature_split(_structured(), n_splits=10, **kwargs)
