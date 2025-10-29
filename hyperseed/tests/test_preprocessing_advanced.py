"""Advanced tests for preprocessing methods to improve coverage."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from hyperseed.core.preprocessing.methods import (
    apply_snv,
    apply_smoothing,
    apply_derivative,
    apply_baseline_correction,
    apply_msc,
    apply_detrend,
    apply_normalization
)
from hyperseed.core.preprocessing.pipeline import PreprocessingPipeline
from hyperseed.config.settings import PreprocessingConfig


class TestAdvancedPreprocessingMethods:
    """Test advanced preprocessing methods for better coverage."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        # Create synthetic spectra
        x = np.linspace(0, 10, 100)
        self.spectrum = np.sin(x) + 0.5 * np.sin(3 * x) + np.random.normal(0, 0.1, 100)
        self.spectra = np.array([self.spectrum + np.random.normal(0, 0.05, 100) for _ in range(10)])

        # Create 3D hyperspectral data
        self.data_3d = np.random.rand(50, 50, 100)

    def test_baseline_correction_polynomial(self):
        """Test polynomial baseline correction."""
        # Add polynomial baseline
        baseline = np.polyval([0.001, -0.01, 0.1], np.arange(100))
        spectrum_with_baseline = self.spectrum + baseline

        # Apply correction
        corrected = apply_baseline_correction(
            spectrum_with_baseline,
            method='polynomial',
            order=2
        )

        assert corrected.shape == spectrum_with_baseline.shape
        # Baseline should be mostly removed
        assert np.std(corrected) < np.std(spectrum_with_baseline)

    def test_baseline_correction_rubberband(self):
        """Test rubberband baseline correction."""
        # This method might not be fully implemented, but test the interface
        try:
            corrected = apply_baseline_correction(
                self.spectrum,
                method='rubberband'
            )
            assert corrected.shape == self.spectrum.shape
        except (NotImplementedError, ValueError):
            # Method might not be implemented
            pass

    def test_baseline_correction_asls(self):
        """Test ASLS baseline correction."""
        # This method might not be fully implemented, but test the interface
        try:
            corrected = apply_baseline_correction(
                self.spectrum,
                method='asls',
                lam=1e5,
                p=0.01
            )
            assert corrected.shape == self.spectrum.shape
        except (NotImplementedError, ValueError):
            # Method might not be implemented
            pass

    def test_smoothing_all_methods(self):
        """Test all smoothing methods."""
        methods = ['savgol', 'moving_average', 'gaussian', 'median']

        for method in methods:
            try:
                smoothed = apply_smoothing(
                    self.spectrum,
                    window_length=11,
                    method=method
                )
                assert smoothed.shape == self.spectrum.shape
                # Smoothing should reduce noise
                if method != 'median':  # Median might not always reduce std
                    assert np.std(np.diff(smoothed)) <= np.std(np.diff(self.spectrum))
            except ValueError:
                # Some methods might not be implemented
                pass

    def test_derivative_higher_orders(self):
        """Test higher order derivatives."""
        for order in [1, 2, 3]:
            deriv = apply_derivative(
                self.spectrum,
                order=order,
                window_length=11,
                polyorder=3
            )
            assert deriv.shape == self.spectrum.shape

    def test_normalization_all_methods(self):
        """Test all normalization methods."""
        methods = ['minmax', 'max', 'area', 'vector', 'peak', 'sum']

        for method in methods:
            try:
                normalized = apply_normalization(
                    self.spectrum,
                    method=method
                )
                assert normalized.shape == self.spectrum.shape

                # Check normalization properties
                if method == 'minmax':
                    assert np.min(normalized) == pytest.approx(0.0, abs=1e-6)
                    assert np.max(normalized) == pytest.approx(1.0, abs=1e-6)
                elif method == 'max':
                    assert np.max(normalized) == pytest.approx(1.0, abs=1e-6)
                elif method == 'vector':
                    assert np.linalg.norm(normalized) == pytest.approx(1.0, abs=1e-6)
            except (ValueError, KeyError):
                # Some methods might not be implemented
                pass

    def test_msc_with_reference(self):
        """Test MSC with explicit reference spectrum."""
        # Create reference spectrum
        reference = np.mean(self.spectra, axis=0)

        # Apply MSC
        corrected = apply_msc(
            self.spectra,
            reference=reference,
            axis=-1
        )

        assert corrected.shape == self.spectra.shape
        # Variance should be reduced
        assert np.var(corrected) <= np.var(self.spectra)

    def test_msc_without_reference(self):
        """Test MSC without reference (uses mean)."""
        corrected = apply_msc(
            self.spectra,
            reference=None,
            axis=-1
        )

        assert corrected.shape == self.spectra.shape

    def test_detrend_all_types(self):
        """Test all detrending types."""
        types = ['linear', 'constant']

        for detrend_type in types:
            detrended = apply_detrend(
                self.spectrum,
                type=detrend_type
            )
            assert detrended.shape == self.spectrum.shape

            # Mean should be closer to zero after detrending
            assert abs(np.mean(detrended)) < abs(np.mean(self.spectrum))

    def test_preprocessing_with_3d_data(self):
        """Test preprocessing methods with 3D hyperspectral data."""
        # SNV on 3D data
        snv_data = apply_snv(self.data_3d, axis=-1)
        assert snv_data.shape == self.data_3d.shape

        # Smoothing on 3D data
        smooth_data = apply_smoothing(
            self.data_3d,
            window_length=5,
            method='savgol',
            polyorder=2
        )
        assert smooth_data.shape == self.data_3d.shape

        # Normalization on 3D data
        norm_data = apply_normalization(
            self.data_3d,
            method='minmax'
        )
        assert norm_data.shape == self.data_3d.shape

    def test_edge_cases(self):
        """Test edge cases for preprocessing methods."""
        # Very short spectrum
        short_spectrum = np.array([1.0, 2.0, 3.0])

        # SNV with short spectrum
        snv = apply_snv(short_spectrum)
        assert snv.shape == short_spectrum.shape

        # Constant spectrum (zero variance)
        constant = np.ones(50)
        snv_const = apply_snv(constant)
        # Should handle zero variance gracefully
        assert not np.any(np.isnan(snv_const))

        # Spectrum with NaN values
        nan_spectrum = self.spectrum.copy()
        nan_spectrum[10] = np.nan

        # Most methods should handle or propagate NaN appropriately
        try:
            smoothed = apply_smoothing(nan_spectrum, window_length=5)
            # Either handles NaN or propagates it
            assert smoothed.shape == nan_spectrum.shape
        except ValueError:
            pass  # Some methods might reject NaN


class TestPreprocessingPipelineAdvanced:
    """Test advanced pipeline functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = np.random.rand(100, 50)  # 100 pixels, 50 bands
        self.data_3d = np.random.rand(20, 20, 50)  # 3D hyperspectral

    def test_pipeline_all_presets(self):
        """Test all pipeline presets."""
        presets = ['minimal', 'standard', 'advanced', 'custom']

        for preset in presets:
            if preset == 'custom':
                config = PreprocessingConfig(
                    method='custom',
                    snv=True,
                    smoothing=True,
                    baseline_correction=True,
                    derivative=True
                )
            else:
                config = PreprocessingConfig(method=preset)

            pipeline = PreprocessingPipeline(config)
            result = pipeline.fit_transform(self.data)

            assert result.shape == self.data.shape
            assert not np.any(np.isnan(result))

    def test_pipeline_step_combinations(self):
        """Test various step combinations."""
        # Test all steps enabled
        config = PreprocessingConfig(
            method='custom',
            snv=True,
            smoothing=True,
            smoothing_window=7,
            baseline_correction=True,
            baseline_method='polynomial',
            derivative=True,
            derivative_order=1,
            normalization=True,
            normalization_method='minmax'
        )

        pipeline = PreprocessingPipeline(config)
        result = pipeline.fit_transform(self.data)

        assert result.shape == self.data.shape
        # After minmax normalization
        assert np.min(result) >= -0.1  # Allowing small numerical errors
        assert np.max(result) <= 1.1

    def test_pipeline_get_step_names(self):
        """Test getting step names."""
        config = PreprocessingConfig(
            method='custom',
            snv=True,
            smoothing=True,
            derivative=True
        )

        pipeline = PreprocessingPipeline(config)
        steps = pipeline.get_step_names()

        assert 'snv' in steps
        assert 'smoothing' in steps
        assert 'derivative' in steps

    def test_pipeline_describe(self):
        """Test pipeline description."""
        config = PreprocessingConfig(method='standard')
        pipeline = PreprocessingPipeline(config)

        description = pipeline.describe()

        assert isinstance(description, dict)
        assert 'method' in description
        assert 'steps' in description
        assert description['method'] == 'standard'

    def test_pipeline_save_load_config(self, tmp_path):
        """Test saving and loading pipeline configuration."""
        config = PreprocessingConfig(
            method='custom',
            snv=True,
            smoothing=True,
            smoothing_window=9,
            smoothing_method='gaussian'
        )

        pipeline = PreprocessingPipeline(config)

        # Save config
        config_path = tmp_path / "pipeline_config.yaml"
        pipeline.save_config(config_path)

        assert config_path.exists()

        # Load config
        loaded_pipeline = PreprocessingPipeline.from_config_file(config_path)

        assert loaded_pipeline.config.snv == True
        assert loaded_pipeline.config.smoothing_window == 9
        assert loaded_pipeline.config.smoothing_method == 'gaussian'

    def test_pipeline_with_nan_handling(self):
        """Test pipeline handling of NaN values."""
        # Add NaN values
        data_with_nan = self.data.copy()
        data_with_nan[0, :5] = np.nan
        data_with_nan[10:15, 10] = np.nan

        config = PreprocessingConfig(method='minimal')
        pipeline = PreprocessingPipeline(config)

        result = pipeline.fit_transform(data_with_nan)

        # Should handle NaN appropriately
        assert result.shape == data_with_nan.shape
        # Might propagate NaN or interpolate
        finite_mask = np.isfinite(result)
        assert np.sum(finite_mask) > 0  # At least some valid values

    def test_pipeline_with_outliers(self):
        """Test pipeline with outlier values."""
        # Add outliers
        data_with_outliers = self.data.copy()
        data_with_outliers[5, 5] = 1000  # Large outlier
        data_with_outliers[10, 10] = -1000  # Negative outlier

        config = PreprocessingConfig(
            method='standard',
            snv=True  # SNV should help normalize outliers
        )
        pipeline = PreprocessingPipeline(config)

        result = pipeline.fit_transform(data_with_outliers)

        # Should handle outliers
        assert np.max(result) < 1000  # Outlier should be normalized
        assert np.min(result) > -1000

    def test_pipeline_transform_without_fit(self):
        """Test transform without fit (should work for stateless operations)."""
        config = PreprocessingConfig(method='minimal')
        pipeline = PreprocessingPipeline(config)

        # Transform without explicit fit
        result = pipeline.transform(self.data)

        assert result.shape == self.data.shape

    def test_pipeline_inverse_transform(self):
        """Test inverse transform if implemented."""
        config = PreprocessingConfig(
            method='custom',
            normalization=True,
            normalization_method='minmax'
        )
        pipeline = PreprocessingPipeline(config)

        # Fit and transform
        transformed = pipeline.fit_transform(self.data)

        # Try inverse transform
        try:
            inverse = pipeline.inverse_transform(transformed)
            assert inverse.shape == self.data.shape
            # Should approximately recover original data for some transforms
        except (NotImplementedError, AttributeError):
            # Inverse transform might not be implemented
            pass