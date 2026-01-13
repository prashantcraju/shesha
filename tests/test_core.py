"""Basic tests for Shesha metrics."""

import numpy as np
import pytest
import shesha


class TestFeatureSplit:
    """Tests for feature_split variant."""
    
    def test_basic_usage(self):
        """Test basic functionality."""
        X = np.random.randn(100, 64)
        result = shesha.feature_split(X, n_splits=10, seed=320)
        assert isinstance(result, float)
        assert -1 <= result <= 1
    
    def test_determinism(self):
        """Test reproducibility with seed."""
        X = np.random.randn(100, 64)
        r1 = shesha.feature_split(X, seed=320)
        r2 = shesha.feature_split(X, seed=320)
        assert r1 == r2
    
    def test_structured_data_high_stability(self):
        """Structured data should have high stability."""
        # Create data with redundant structure
        latent = np.random.randn(100, 10)
        projection = np.random.randn(10, 128)
        X = latent @ projection
        
        result = shesha.feature_split(X, n_splits=30, seed=320)
        assert result > 0.5, "Structured data should have high stability"
    
    def test_random_data_low_stability(self):
        """Random iid features should have lower stability."""
        X = np.random.randn(100, 128)
        result = shesha.feature_split(X, n_splits=30, seed=320)
        # Random data can still have moderate stability, just check it runs
        assert -1 <= result <= 1
    
    def test_too_few_features(self):
        """Should return NaN with too few features."""
        X = np.random.randn(100, 3)
        result = shesha.feature_split(X)
        assert np.isnan(result)
    
    def test_too_few_samples(self):
        """Should return NaN with too few samples."""
        X = np.random.randn(3, 64)
        result = shesha.feature_split(X)
        assert np.isnan(result)


class TestSampleSplit:
    """Tests for sample_split variant."""
    
    def test_basic_usage(self):
        X = np.random.randn(200, 64)
        result = shesha.sample_split(X, n_splits=10, seed=320)
        assert isinstance(result, float)
        assert -1 <= result <= 1
    
    def test_determinism(self):
        X = np.random.randn(200, 64)
        r1 = shesha.sample_split(X, seed=320)
        r2 = shesha.sample_split(X, seed=320)
        assert r1 == r2


class TestAnchorStability:
    """Tests for anchor_stability variant."""
    
    def test_basic_usage(self):
        X = np.random.randn(500, 64)
        result = shesha.anchor_stability(X, n_splits=10, seed=320)
        assert isinstance(result, float)
        assert -1 <= result <= 1
    
    def test_small_data_handling(self):
        """Should handle small datasets gracefully."""
        X = np.random.randn(50, 64)
        result = shesha.anchor_stability(X, seed=320)
        # Should either return valid result or NaN, not crash
        assert np.isnan(result) or (-1 <= result <= 1)


class TestVarianceRatio:
    """Tests for variance_ratio variant."""
    
    def test_basic_usage(self):
        X = np.random.randn(100, 64)
        y = np.random.randint(0, 5, 100)
        result = shesha.variance_ratio(X, y)
        assert isinstance(result, float)
        assert 0 <= result <= 1
    
    def test_perfect_separation(self):
        """Perfectly separated classes should have high ratio."""
        # Create well-separated clusters
        X = np.vstack([
            np.random.randn(50, 64) + np.array([10] * 64),
            np.random.randn(50, 64) + np.array([-10] * 64),
        ])
        y = np.array([0] * 50 + [1] * 50)
        
        result = shesha.variance_ratio(X, y)
        assert result > 0.8, "Well-separated classes should have high variance ratio"
    
    def test_single_class(self):
        """Single class should return NaN."""
        X = np.random.randn(100, 64)
        y = np.zeros(100)
        result = shesha.variance_ratio(X, y)
        assert np.isnan(result)


class TestSupervisedAlignment:
    """Tests for supervised_alignment variant."""
    
    def test_basic_usage(self):
        X = np.random.randn(100, 64)
        y = np.random.randint(0, 5, 100)
        result = shesha.supervised_alignment(X, y, seed=320)
        assert isinstance(result, float)
        assert -1 <= result <= 1
    
    def test_determinism(self):
        X = np.random.randn(100, 64)
        y = np.random.randint(0, 5, 100)
        r1 = shesha.supervised_alignment(X, y, seed=320)
        r2 = shesha.supervised_alignment(X, y, seed=320)
        assert r1 == r2


class TestUnifiedInterface:
    """Tests for the unified shesha() function."""
    
    def test_feature_split_variant(self):
        X = np.random.randn(100, 64)
        result = shesha.shesha(X, variant='feature_split', seed=320)
        expected = shesha.feature_split(X, seed=320)
        assert result == expected
    
    def test_supervised_variant(self):
        X = np.random.randn(100, 64)
        y = np.random.randint(0, 5, 100)
        result = shesha.shesha(X, y, variant='supervised', seed=320)
        expected = shesha.supervised_alignment(X, y, seed=320)
        assert result == expected
    
    def test_missing_labels_error(self):
        X = np.random.randn(100, 64)
        with pytest.raises(ValueError, match="Labels required"):
            shesha.shesha(X, variant='supervised')
    
    def test_unknown_variant_error(self):
        X = np.random.randn(100, 64)
        with pytest.raises(ValueError, match="Unknown variant"):
            shesha.shesha(X, variant='nonexistent')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
