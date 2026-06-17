"""Tests for split-half reproducibility, magnitude-matched comparison, and discordance."""

import numpy as np
import pandas as pd
import pytest

from shesha.bio import (
    _split_half_cosine,
    split_half_reproducibility,
    magnitude_matched_comparison,
    discordance,
)


def _has_statsmodels():
    try:
        import statsmodels  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockAnnData:
    """Minimal AnnData stand-in for testing without anndata installed."""

    def __init__(self, X, obs):
        self.X = np.asarray(X)
        self.obs = obs
        self.layers = {}

    def __getitem__(self, mask):
        return MockAnnData(self.X[mask], self.obs[mask])


@pytest.fixture
def adata_simple():
    """AnnData-like object with one strong and one weak perturbation."""
    rng = np.random.default_rng(42)
    d = 50
    n_ctrl, n_strong, n_weak = 500, 80, 60

    X_ctrl = rng.standard_normal((n_ctrl, d))
    X_strong = rng.standard_normal((n_strong, d)) + np.ones(d) * 3
    X_weak = rng.standard_normal((n_weak, d)) * 2

    X = np.vstack([X_ctrl, X_strong, X_weak])
    labels = (
        ["control"] * n_ctrl
        + ["gene_strong"] * n_strong
        + ["gene_weak"] * n_weak
    )
    obs = pd.DataFrame({"perturbation": labels})
    return MockAnnData(X=X, obs=obs)


@pytest.fixture(autouse=True)
def patch_anndata(monkeypatch):
    """Ensure bio.AnnData resolves to our mock class."""
    import shesha.bio as bio
    monkeypatch.setattr(bio, "AnnData", MockAnnData)


# ---------------------------------------------------------------------------
# _split_half_cosine
# ---------------------------------------------------------------------------

class TestSplitHalfCosine:
    def test_strong_signal_high_cosine(self):
        rng = np.random.default_rng(0)
        ctrl_centroid = np.zeros(50)
        X_pert = rng.standard_normal((100, 50)) + np.ones(50) * 5
        cos = _split_half_cosine(X_pert, ctrl_centroid, n_splits=50, seed=320)
        assert cos > 0.95

    def test_noise_low_cosine(self):
        rng = np.random.default_rng(1)
        ctrl_centroid = np.zeros(50)
        X_pert = rng.standard_normal((100, 50))
        cos = _split_half_cosine(X_pert, ctrl_centroid, n_splits=50, seed=320)
        assert abs(cos) < 0.5

    def test_too_few_cells_returns_nan(self):
        ctrl_centroid = np.zeros(50)
        X_pert = np.random.randn(10, 50)
        cos = _split_half_cosine(X_pert, ctrl_centroid, n_splits=50, seed=0, min_cells=30)
        assert np.isnan(cos)

    def test_deterministic(self):
        rng = np.random.default_rng(7)
        ctrl_centroid = np.zeros(50)
        X_pert = rng.standard_normal((80, 50)) + np.ones(50)
        c1 = _split_half_cosine(X_pert, ctrl_centroid, n_splits=50, seed=123)
        c2 = _split_half_cosine(X_pert, ctrl_centroid, n_splits=50, seed=123)
        assert c1 == c2

    def test_different_seeds_differ(self):
        rng = np.random.default_rng(7)
        ctrl_centroid = np.zeros(50)
        X_pert = rng.standard_normal((80, 50)) + np.ones(50)
        c1 = _split_half_cosine(X_pert, ctrl_centroid, n_splits=50, seed=1)
        c2 = _split_half_cosine(X_pert, ctrl_centroid, n_splits=50, seed=2)
        assert c1 != c2

    def test_output_range(self):
        rng = np.random.default_rng(99)
        ctrl_centroid = rng.standard_normal(50)
        X_pert = rng.standard_normal((200, 50)) + ctrl_centroid + rng.standard_normal(50)
        cos = _split_half_cosine(X_pert, ctrl_centroid, n_splits=100, seed=0)
        assert -1.0 <= cos <= 1.0


# ---------------------------------------------------------------------------
# split_half_reproducibility (AnnData-level)
# ---------------------------------------------------------------------------

class TestSplitHalfReproducibility:
    def test_returns_dataframe(self, adata_simple):
        result = split_half_reproducibility(
            adata_simple, perturbation_key="perturbation", control_label="control"
        )
        assert isinstance(result, pd.DataFrame)
        assert "split_half_cosine" in result.columns
        assert "n_cells" in result.columns

    def test_index_is_perturbation(self, adata_simple):
        result = split_half_reproducibility(
            adata_simple, perturbation_key="perturbation", control_label="control"
        )
        assert result.index.name == "perturbation"
        assert "gene_strong" in result.index
        assert "gene_weak" in result.index

    def test_strong_beats_weak(self, adata_simple):
        result = split_half_reproducibility(
            adata_simple, perturbation_key="perturbation", control_label="control"
        )
        assert result.loc["gene_strong", "split_half_cosine"] > result.loc["gene_weak", "split_half_cosine"]

    def test_n_cells_correct(self, adata_simple):
        result = split_half_reproducibility(
            adata_simple, perturbation_key="perturbation", control_label="control"
        )
        assert result.loc["gene_strong", "n_cells"] == 80
        assert result.loc["gene_weak", "n_cells"] == 60

    def test_min_cells_filters(self, adata_simple):
        result = split_half_reproducibility(
            adata_simple,
            perturbation_key="perturbation",
            control_label="control",
            min_cells=100,
        )
        assert len(result) == 0

    def test_control_not_in_output(self, adata_simple):
        result = split_half_reproducibility(
            adata_simple, perturbation_key="perturbation", control_label="control"
        )
        assert "control" not in result.index

    def test_deterministic(self, adata_simple):
        r1 = split_half_reproducibility(
            adata_simple, perturbation_key="perturbation",
            control_label="control", random_state=42,
        )
        r2 = split_half_reproducibility(
            adata_simple, perturbation_key="perturbation",
            control_label="control", random_state=42,
        )
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# magnitude_matched_comparison
# ---------------------------------------------------------------------------

class TestMagnitudeMatchedComparison:
    @pytest.fixture
    def repro_df(self):
        rng = np.random.default_rng(10)
        n = 200
        sp = rng.standard_normal(n)
        mp = np.abs(rng.standard_normal(n)) + 0.1
        cos = 0.3 * sp + 0.2 * mp + rng.standard_normal(n) * 0.3
        return pd.DataFrame({"Sp": sp, "Mp": mp, "split_half_cosine": cos})

    def test_returns_dataframe(self, repro_df):
        result = magnitude_matched_comparison(repro_df)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, repro_df):
        result = magnitude_matched_comparison(repro_df)
        expected_cols = {
            "mag_bin", "n", "mag_min", "mag_max",
            "high_stability_mean", "low_stability_mean",
            "difference", "within_bin_rho", "within_bin_pvalue",
        }
        assert set(result.columns) == expected_cols

    def test_n_bins_matches(self, repro_df):
        result = magnitude_matched_comparison(repro_df, n_bins=4)
        assert len(result) == 4

        result3 = magnitude_matched_comparison(repro_df, n_bins=3)
        assert len(result3) == 3

    def test_difference_is_high_minus_low(self, repro_df):
        result = magnitude_matched_comparison(repro_df)
        for _, row in result.iterrows():
            expected = row["high_stability_mean"] - row["low_stability_mean"]
            assert abs(row["difference"] - expected) < 1e-10

    def test_too_few_raises(self):
        df = pd.DataFrame({"Sp": [1, 2], "Mp": [1, 2], "split_half_cosine": [0.5, 0.6]})
        with pytest.raises(ValueError, match="Too few"):
            magnitude_matched_comparison(df, n_bins=4)

    def test_custom_column_names(self):
        rng = np.random.default_rng(55)
        n = 100
        df = pd.DataFrame({
            "stability": rng.standard_normal(n),
            "magnitude": np.abs(rng.standard_normal(n)) + 0.1,
            "repro": rng.random(n),
        })
        result = magnitude_matched_comparison(
            df, stability_col="stability", repro_col="repro",
            magnitude_col="magnitude", n_bins=4,
        )
        assert len(result) == 4

    def test_nan_handling(self):
        rng = np.random.default_rng(77)
        n = 100
        df = pd.DataFrame({
            "Sp": rng.standard_normal(n),
            "Mp": np.abs(rng.standard_normal(n)) + 0.1,
            "split_half_cosine": rng.random(n),
        })
        df.loc[0:5, "Sp"] = np.nan
        result = magnitude_matched_comparison(df)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# discordance
# ---------------------------------------------------------------------------

class TestDiscordance:
    @pytest.fixture
    def base_df(self):
        rng = np.random.default_rng(42)
        n = 200
        mp = np.abs(rng.standard_normal(n)) + 0.5
        sp = 0.7 * mp + rng.standard_normal(n) * 0.3
        return pd.DataFrame({"Sp": sp, "Mp": mp})

    def test_returns_series(self, base_df):
        result = discordance(base_df)
        assert isinstance(result, pd.Series)
        assert result.name == "discordance"

    def test_output_length_matches_valid(self, base_df):
        result = discordance(base_df)
        assert len(result) == len(base_df)

    def test_z_scored(self, base_df):
        result = discordance(base_df)
        assert abs(result.mean()) < 1e-8
        assert abs(result.std(ddof=0) - 1.0) < 1e-8

    def test_linear_method(self, base_df):
        result = discordance(base_df, method="linear")
        assert len(result) == len(base_df)
        assert abs(result.mean()) < 1e-10

    def test_rank_method(self, base_df):
        result = discordance(base_df, method="rank")
        assert len(result) == len(base_df)
        assert abs(result.mean()) < 1e-10

    @pytest.mark.skipif(
        not _has_statsmodels(), reason="statsmodels not installed"
    )
    def test_loess_method(self, base_df):
        result = discordance(base_df, method="loess", loess_frac=0.3)
        assert len(result) == len(base_df)
        assert abs(result.mean()) < 1e-8

    @pytest.mark.skipif(
        not _has_statsmodels(), reason="statsmodels not installed"
    )
    def test_loess_frac_parameter(self, base_df):
        r1 = discordance(base_df, method="loess", loess_frac=0.2)
        r2 = discordance(base_df, method="loess", loess_frac=0.6)
        assert not r1.equals(r2)

    def test_methods_correlated_linear_rank(self, base_df):
        from scipy.stats import spearmanr
        d_lin = discordance(base_df, method="linear")
        d_rank = discordance(base_df, method="rank")
        rho_lr, _ = spearmanr(d_lin, d_rank)
        assert rho_lr > 0.7

    @pytest.mark.skipif(
        not _has_statsmodels(), reason="statsmodels not installed"
    )
    def test_methods_correlated_linear_loess(self, base_df):
        from scipy.stats import spearmanr
        d_lin = discordance(base_df, method="linear")
        d_loess = discordance(base_df, method="loess")
        rho_ll, _ = spearmanr(d_lin, d_loess)
        assert rho_ll > 0.7

    def test_invalid_method_raises(self, base_df):
        with pytest.raises(ValueError, match="Unknown method"):
            discordance(base_df, method="quadratic")

    def test_too_few_rows_raises(self):
        df = pd.DataFrame({"Sp": [1, 2, 3], "Mp": [1, 2, 3]})
        with pytest.raises(ValueError, match="Too few"):
            discordance(df)

    def test_custom_columns(self):
        rng = np.random.default_rng(99)
        n = 50
        df = pd.DataFrame({
            "stability": rng.standard_normal(n),
            "magnitude": np.abs(rng.standard_normal(n)) + 0.1,
        })
        result = discordance(df, stability_col="stability", magnitude_col="magnitude")
        assert len(result) == n

    def test_nan_handling(self):
        rng = np.random.default_rng(55)
        n = 100
        df = pd.DataFrame({
            "Sp": rng.standard_normal(n),
            "Mp": np.abs(rng.standard_normal(n)) + 0.1,
        })
        df.loc[0:5, "Sp"] = np.nan
        result = discordance(df)
        assert len(result) == n - 6  # NaN rows excluded

    def test_index_preserved(self, base_df):
        base_df.index = [f"gene_{i}" for i in range(len(base_df))]
        result = discordance(base_df)
        assert list(result.index) == list(base_df.index)

    def test_high_discordance_for_outlier(self):
        rng = np.random.default_rng(10)
        n = 100
        mp = np.abs(rng.standard_normal(n)) + 1.0
        sp = 0.8 * mp + rng.standard_normal(n) * 0.1
        sp[0] = 0.0  # high magnitude but very low stability -> discordant
        mp[0] = 5.0
        df = pd.DataFrame({"Sp": sp, "Mp": mp})
        result = discordance(df, method="linear")
        assert result.iloc[0] > 2.0  # should be a strong outlier
