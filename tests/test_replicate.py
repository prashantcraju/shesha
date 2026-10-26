"""
Semantic tests for matched-replicate estimators and StabilityResult.
"""

import json

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

import shesha
from shesha._rdm import resampled_pair_index
from shesha.replicate import _bootstrap_replicate


def _replicates(n_samples=150, n_latent=8, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_samples, n_latent))
    X_a = latent @ rng.standard_normal((n_latent, 40)) + noise * rng.standard_normal(
        (n_samples, 40)
    )
    X_b = latent @ rng.standard_normal((n_latent, 30)) + noise * rng.standard_normal(
        (n_samples, 30)
    )
    return X_a, X_b


class TestReplicateStability:
    def test_identical_replicates_score_one(self):
        X, _ = _replicates()
        assert shesha.replicate_stability(X, X) == pytest.approx(1.0, abs=1e-12)

    def test_independent_replicates_near_zero(self):
        rng = np.random.default_rng(1)
        score = shesha.replicate_stability(
            rng.standard_normal((150, 20)), rng.standard_normal((150, 20))
        )
        assert abs(score) < 0.05

    def test_noise_degrades_monotonically(self):
        scores = [
            shesha.replicate_stability(*_replicates(noise=noise, seed=3))
            for noise in (0.1, 1.0, 3.0, 10.0)
        ]
        assert all(a > b for a, b in zip(scores, scores[1:]))

    def test_matches_rdm_similarity_on_aligned_rows(self):
        X_a, X_b = _replicates()
        for metric in ("cosine", "correlation", "euclidean"):
            for method in ("spearman", "pearson"):
                assert shesha.replicate_stability(
                    X_a, X_b, metric=metric, method=method
                ) == pytest.approx(
                    shesha.rdm_similarity(X_a, X_b, method=method, metric=metric), abs=1e-10
                )

    def test_unaligned_rows_destroy_correspondence(self):
        X_a, X_b = _replicates()
        order = np.random.default_rng(4).permutation(X_a.shape[0])
        aligned = shesha.replicate_stability(X_a, X_b)
        shuffled = shesha.replicate_stability(X_a, X_b[order])
        assert aligned > 0.5
        assert abs(shuffled) < 0.1

    def test_ids_restore_alignment(self):
        X_a, X_b = _replicates()
        n = X_a.shape[0]
        order = np.random.default_rng(4).permutation(n)
        ids = np.array([f"stim_{i}" for i in range(n)])
        restored = shesha.replicate_stability(X_a, X_b[order], ids_a=ids, ids_b=ids[order])
        assert restored == pytest.approx(shesha.replicate_stability(X_a, X_b), abs=1e-12)

    def test_mismatched_id_sets_rejected(self):
        X_a, X_b = _replicates(n_samples=20)
        ids_b = np.arange(20)
        ids_b[0] = 99
        with pytest.raises(ValueError, match="same identifiers: 1 only in ids_a, 1 only in ids_b"):
            shesha.replicate_stability(X_a, X_b, ids_a=np.arange(20), ids_b=ids_b)

    def test_duplicate_ids_rejected(self):
        X_a, X_b = _replicates(n_samples=20)
        ids = np.arange(20)
        ids[1] = 0
        with pytest.raises(ValueError, match="duplicate"):
            shesha.replicate_stability(X_a, X_b, ids_a=ids, ids_b=np.arange(20))

    def test_one_sided_ids_rejected(self):
        X_a, X_b = _replicates(n_samples=20)
        with pytest.raises(ValueError, match="provided together"):
            shesha.replicate_stability(X_a, X_b, ids_a=np.arange(20))

    def test_sample_count_mismatch_rejected(self):
        X_a, X_b = _replicates(n_samples=20)
        with pytest.raises(ValueError, match="same number of samples"):
            shesha.replicate_stability(X_a, X_b[:-1])

    def test_too_few_observations_rejected(self):
        X_a, X_b = _replicates(n_samples=20)
        with pytest.raises(ValueError, match="at least 3"):
            shesha.replicate_stability(X_a[:2], X_b[:2])

    def test_bootstrap_requires_result_type(self):
        X_a, X_b = _replicates(n_samples=20)
        with pytest.raises(ValueError, match="return_type='result'"):
            shesha.replicate_stability(X_a, X_b, n_bootstrap_ci=10)

    def test_nan_policies(self):
        X_a, X_b = _replicates(n_samples=30)
        X_a[0] = 0.0
        with pytest.raises(ValueError, match="Undefined distances"):
            shesha.replicate_stability(X_a, X_b)
        assert np.isnan(shesha.replicate_stability(X_a, X_b, nan_policy="propagate"))
        with pytest.warns(UserWarning, match="Omitted 29"):
            omitted = shesha.replicate_stability(X_a, X_b, nan_policy="omit")
        expected = shesha.replicate_stability(X_a[1:], X_b[1:])
        assert omitted == pytest.approx(expected, abs=1e-12)

    def test_result_interval_and_determinism(self):
        X_a, X_b = _replicates()
        kwargs = {"n_bootstrap_ci": 100, "seed": 7, "return_type": "result"}
        r1 = shesha.replicate_stability(X_a, X_b, **kwargs)
        r2 = shesha.replicate_stability(X_a, X_b, **kwargs)
        assert r1 == r2
        assert r1.estimate == pytest.approx(shesha.replicate_stability(X_a, X_b), abs=1e-12)
        assert r1.data_ci_low < r1.estimate < r1.data_ci_high
        assert r1.n_bootstraps == 100
        assert r1.ci_level == 0.95
        assert r1.monte_carlo_se == 0.0
        assert r1.n_valid_splits is None
        assert r1.metadata["estimator"] == "replicate_stability"
        assert r1.metadata["n_pairs"] == 150 * 149 // 2

    def test_bootstrap_not_inflated_by_duplicate_observations(self):
        # Independent replicates: a bootstrap that keeps duplicated-observation
        # pairs shares zero self-distances across both RDMs and is biased upward.
        rng = np.random.default_rng(11)
        X_a = rng.standard_normal((60, 5))
        X_b = rng.standard_normal((60, 5))
        rdm_a, rdm_b = pdist(X_a), pdist(X_b)
        estimate = shesha.replicate_stability(X_a, X_b, metric="euclidean")
        matched = _bootstrap_replicate(
            rdm_a, rdm_b, np.ones(rdm_a.shape, bool), 60, 300, "spearman", 1
        )
        legacy = shesha.rdm_similarity(X_a, X_b, metric="euclidean", n_bootstrap_ci=300, seed=1)
        assert abs(np.mean(matched) - estimate) < 0.02
        assert legacy["mean"] - estimate > 0.03


class TestResampledPairIndex:
    def test_maps_to_condensed_entries_and_drops_self_pairs(self):
        X = np.random.default_rng(0).standard_normal((7, 3))
        condensed = pdist(X)
        square = squareform(condensed)
        rows = np.array([0, 3, 3, 6, 1, 0])
        pairs = np.triu_indices(rows.size, k=1)
        index = resampled_pair_index(rows, 7, pairs)
        expected = [square[rows[i], rows[j]] for i, j in zip(*pairs) if rows[i] != rows[j]]
        np.testing.assert_allclose(condensed[index], expected)
        assert index.size == pairs[0].size - 2


class TestAnchorProfileStability:
    def test_identical_replicates_score_one(self):
        X, _ = _replicates()
        score = shesha.anchor_profile_stability(X, X, anchor_ids=np.arange(10))
        assert score == pytest.approx(1.0, abs=1e-12)

    def test_independent_replicates_near_zero(self):
        rng = np.random.default_rng(2)
        score = shesha.anchor_profile_stability(
            rng.standard_normal((200, 10)),
            rng.standard_normal((200, 10)),
            anchor_ids=np.arange(20),
        )
        assert abs(score) < 0.1

    def test_noise_degrades_monotonically(self):
        scores = [
            shesha.anchor_profile_stability(
                *_replicates(noise=noise, seed=5), anchor_ids=np.arange(15)
            )
            for noise in (0.1, 1.0, 3.0, 10.0)
        ]
        assert all(a > b for a, b in zip(scores, scores[1:]))

    def test_ids_select_and_align(self):
        X_a, X_b = _replicates(n_samples=60)
        order = np.random.default_rng(6).permutation(60)
        ids = np.array([f"cell_{i}" for i in range(60)])
        anchors, probes = ids[:5], ids[5:40]
        expected = shesha.anchor_profile_stability(
            X_a, X_b, anchor_ids=np.arange(5), probe_ids=np.arange(5, 40)
        )
        restored = shesha.anchor_profile_stability(
            X_a, X_b[order], anchor_ids=anchors, probe_ids=probes, ids_a=ids, ids_b=ids[order]
        )
        assert restored == pytest.approx(expected, abs=1e-12)

    def test_result_reports_per_anchor_scores(self):
        X_a, X_b = _replicates()
        result = shesha.anchor_profile_stability(
            X_a, X_b, anchor_ids=np.arange(12), n_bootstrap_ci=50, seed=3, return_type="result"
        )
        assert len(result.component_scores) == 12
        assert result.estimate == pytest.approx(np.mean(result.component_scores), abs=1e-12)
        assert result.data_ci_low <= result.estimate <= result.data_ci_high
        assert result.metadata["n_probes"] == 138
        assert result.metadata["bootstrap_unit"] == "probes"

    def test_undefined_anchor_excluded(self):
        X_a, X_b = _replicates(n_samples=40)
        # A constant anchor has undefined correlation distance to every probe.
        X_a[2] = 1.0
        with pytest.warns(UserWarning, match="Omitted"):
            result = shesha.anchor_profile_stability(
                X_a,
                X_b,
                anchor_ids=[2, 3, 4],
                metric="correlation",
                nan_policy="omit",
                return_type="result",
            )
        assert result.metadata["n_valid_anchors"] == 2
        assert any("unestimable profile" in note for note in result.warnings)

    def test_overlapping_anchors_and_probes_rejected(self):
        X_a, X_b = _replicates(n_samples=30)
        with pytest.raises(ValueError, match="disjoint"):
            shesha.anchor_profile_stability(X_a, X_b, anchor_ids=[0, 1], probe_ids=[1, 2, 3])

    def test_too_few_probes_rejected(self):
        X_a, X_b = _replicates(n_samples=30)
        with pytest.raises(ValueError, match="at least 3 probe"):
            shesha.anchor_profile_stability(X_a, X_b, anchor_ids=[0], probe_ids=[1, 2])

    def test_unknown_ids_rejected(self):
        X_a, X_b = _replicates(n_samples=10)
        ids = np.arange(10)
        with pytest.raises(ValueError, match="unknown identifier"):
            shesha.anchor_profile_stability(X_a, X_b, anchor_ids=[42], ids_a=ids, ids_b=ids)

    def test_non_integer_positions_without_ids_rejected(self):
        X_a, X_b = _replicates(n_samples=10)
        with pytest.raises(ValueError, match="integer row indices"):
            shesha.anchor_profile_stability(X_a, X_b, anchor_ids=["a"])

    def test_out_of_range_positions_rejected(self):
        X_a, X_b = _replicates(n_samples=10)
        with pytest.raises(ValueError, match=r"\[0, 10\)"):
            shesha.anchor_profile_stability(X_a, X_b, anchor_ids=[10])


class TestMatchedVsLegacy:
    def test_redundant_geometry_recovered_by_matched_estimators(self):
        """The 0.2.x counterexample: legacy estimators score ~0, matched ones do not."""
        rng = np.random.default_rng(320)
        latent = rng.standard_normal((120, 5))
        projection = rng.standard_normal((5, 100))
        X_a = latent @ projection + 0.05 * rng.standard_normal((120, 100))
        X_b = latent @ projection + 0.05 * rng.standard_normal((120, 100))
        with pytest.warns(FutureWarning, match="replicate_stability"):
            legacy = shesha.sample_split(X_a, n_splits=20, seed=320)
        with pytest.warns(FutureWarning, match="anchor_profile_stability"):
            legacy_anchor = shesha.anchor_stability(
                X_a, n_splits=10, n_anchors=20, n_per_split=30, seed=320
            )
        assert abs(legacy) < 0.15
        assert abs(legacy_anchor) < 0.15
        assert shesha.replicate_stability(X_a, X_b) > 0.95
        assert shesha.anchor_profile_stability(X_a, X_b, anchor_ids=np.arange(20)) > 0.95


class TestFeatureSplitResult:
    def test_estimate_matches_float_path(self):
        X = _replicates()[0]
        result = shesha.feature_split(X, n_splits=20, seed=320, return_type="result")
        assert result.estimate == shesha.feature_split(X, n_splits=20, seed=320)
        legacy = shesha.feature_split(X, n_splits=20, seed=320, return_all_splits=True)
        assert list(result.component_scores) == legacy["split_scores"]

    def test_partition_uncertainty_fields(self):
        X = _replicates()[0]
        result = shesha.feature_split(X, n_splits=25, seed=320, return_type="result")
        scores = np.asarray(result.component_scores)
        assert result.n_valid_splits == 25
        assert result.partition_sd == pytest.approx(np.std(scores, ddof=1))
        assert result.monte_carlo_se == pytest.approx(np.std(scores, ddof=1) / 5.0)
        assert np.isnan(result.data_ci_low) and result.n_bootstraps == 0
        assert result.ci_level is None

    def test_data_interval_is_separate_and_deterministic(self):
        X = _replicates(n_samples=80)[0]
        kwargs = {"n_splits": 10, "seed": 3, "n_bootstrap_ci": 40, "return_type": "result"}
        r1 = shesha.feature_split(X, **kwargs)
        r2 = shesha.feature_split(X, **kwargs)
        assert r1 == r2
        assert r1.estimate == shesha.feature_split(X, n_splits=10, seed=3)
        assert r1.data_ci_low <= r1.estimate <= r1.data_ci_high
        assert r1.n_bootstraps == 40

    def test_skipped_partitions_reported(self):
        rng = np.random.default_rng(320)
        X = np.zeros((40, 40))
        X[:, :3] = rng.standard_normal((40, 3))
        with pytest.warns(UserWarning, match="Omitted"):
            result = shesha.feature_split(
                X, n_splits=10, seed=1, nan_policy="omit", return_type="result"
            )
        assert result.n_valid_splits < 10
        assert any("feature partitions were unestimable" in note for note in result.warnings)

    def test_list_input_returns_results(self):
        X = _replicates()[0]
        results = shesha.feature_split([X, X[:, :20]], n_splits=5, seed=1, return_type="result")
        assert all(isinstance(r, shesha.StabilityResult) for r in results)

    def test_propagate_returns_nan_result(self):
        X = np.random.default_rng(3).standard_normal((40, 20))
        X[0] = 0.0
        result = shesha.feature_split(
            X, n_splits=5, seed=1, nan_policy="propagate", return_type="result"
        )
        assert np.isnan(result.estimate)
        assert result.warnings

    def test_invalid_return_type_rejected(self):
        with pytest.raises(ValueError, match="return_type"):
            shesha.feature_split(np.random.randn(20, 10), return_type="dict")


class TestStabilityResult:
    def test_float_and_json_round_trip(self):
        X_a, X_b = _replicates(n_samples=30)
        result = shesha.replicate_stability(
            X_a, X_b, n_bootstrap_ci=10, seed=0, return_type="result"
        )
        assert float(result) == result.estimate
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["estimate"] == pytest.approx(result.estimate)
        assert payload["metadata"]["package_version"] == shesha.__version__
        assert isinstance(payload["warnings"], list)

    def test_immutable(self):
        result = shesha.StabilityResult(estimate=0.5)
        with pytest.raises(Exception):
            result.estimate = 0.1
