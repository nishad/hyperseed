"""Unit tests for preprocessing module."""

import pytest
import numpy as np

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
from hyperseed.tests.fixtures import generate_synthetic_spectrum


class TestPreprocessingMethods:
    """Test suite for individual preprocessing methods."""

    def test_snv_transformation(self):
        """Test Standard Normal Variate transformation."""
        # Generate test spectra
        spectra = np.random.rand(10, 100)

        # Apply SNV
        snv_spectra = apply_snv(spectra, axis=-1)

        # Check that each spectrum has mean ~0 and std ~1
        for i in range(10):
            assert np.abs(np.mean(snv_spectra[i])) < 1e-6
            assert np.abs(np.std(snv_spectra[i]) - 1.0) < 1e-6

    def test_smoothing_methods(self):
        """Test different smoothing methods."""
        # Generate noisy spectrum
        spectrum, _ = generate_synthetic_spectrum(n_bands=100)
        noise = np.random.normal(0, 0.05, 100)
        noisy_spectrum = spectrum + noise

        # Test Savitzky-Golay
        smooth_savgol = apply_smoothing(
            noisy_spectrum, window_length=11, polyorder=3, method="savgol"
        )
        assert smooth_savgol.shape == noisy_spectrum.shape
        # Should reduce noise
        assert np.std(smooth_savgol) < np.std(noisy_spectrum)

        # Test moving average
        smooth_ma = apply_smoothing(
            noisy_spectrum, window_length=11, method="moving_average"
        )
        assert smooth_ma.shape == noisy_spectrum.shape

        # Test gaussian
        smooth_gauss = apply_smoothing(
            noisy_spectrum, window_length=11, method="gaussian"
        )
        assert smooth_gauss.shape == noisy_spectrum.shape

    def test_derivative_calculation(self):
        """Test derivative calculations."""
        # Create a simple quadratic spectrum
        x = np.linspace(0, 10, 100)
        spectrum = x**2

        # First derivative should be ~2x
        first_deriv = apply_derivative(
            spectrum, order=1, window_length=11, polyorder=3
        )
        assert first_deriv.shape == spectrum.shape

        # Second derivative should be ~2 (constant)
        second_deriv = apply_derivative(
            spectrum, order=2, window_length=11, polyorder=3
        )
        assert second_deriv.shape == spectrum.shape

    def test_baseline_correction(self):
        """Test baseline correction methods."""
        # Create spectrum with linear baseline
        spectrum, _ = generate_synthetic_spectrum(n_bands=100)
        baseline = np.linspace(0, 0.5, 100)
        spectrum_with_baseline = spectrum + baseline

        # Apply polynomial baseline correction
        corrected = apply_baseline_correction(
            spectrum_with_baseline, order=1, method="polynomial"
        )

        # Should remove most of the baseline trend
        assert np.std(corrected) < np.std(spectrum_with_baseline)

    def test_msc_correction(self):
        """Test Multiplicative Scatter Correction."""
        # Generate multiple spectra with scatter effects
        n_spectra = 10
        n_bands = 100
        reference = np.random.rand(n_bands) * 0.5 + 0.2

        # Add multiplicative and additive effects
        spectra = np.zeros((n_spectra, n_bands))
        for i in range(n_spectra):
            a = np.random.normal(0, 0.1)  # Additive
            b = np.random.normal(1, 0.2)  # Multiplicative
            spectra[i] = a + b * reference

        # Apply MSC
        corrected = apply_msc(spectra, reference=reference, axis=-1)

        # Should reduce variance across spectra
        assert np.std(corrected) < np.std(spectra)

    def test_detrend_function(self):
        """Test detrending."""
        # Create spectrum with linear trend
        spectrum = np.linspace(0, 1, 100) + np.random.rand(100) * 0.1

        # Apply detrending
        detrended = apply_detrend(spectrum, type="linear")

        # Should remove trend
        assert np.abs(np.mean(detrended)) < np.abs(np.mean(spectrum))

    def test_normalization_methods(self):
        """Test different normalization methods."""
        spectrum = np.random.rand(100) * 10 + 5

        # Min-max normalization
        norm_minmax = apply_normalization(spectrum, method="minmax")
        assert norm_minmax.min() == pytest.approx(0.0)
        assert norm_minmax.max() == pytest.approx(1.0)

        # Max normalization
        norm_max = apply_normalization(spectrum, method="max")
        assert norm_max.max() == pytest.approx(1.0)

        # Area normalization
        norm_area = apply_normalization(spectrum, method="area")
        assert np.sum(np.abs(norm_area)) == pytest.approx(1.0)

        # Vector normalization
        norm_vector = apply_normalization(spectrum, method="vector")
        assert np.linalg.norm(norm_vector) == pytest.approx(1.0)


class TestPreprocessingPipeline:
    """Test suite for preprocessing pipeline."""

    def test_pipeline_creation(self):
        """Test pipeline creation with different configs."""
        # Default config
        pipeline = PreprocessingPipeline()
        assert pipeline.config is not None

        # Custom config
        config = PreprocessingConfig(method="standard")
        pipeline = PreprocessingPipeline(config)
        assert pipeline.config.method == "standard"

    def test_pipeline_presets(self):
        """Test pipeline with different presets."""
        # Generate test data
        data = np.random.rand(100, 50)  # 100 pixels, 50 bands

        # Test minimal preset
        config_minimal = PreprocessingConfig(method="minimal")
        pipeline_minimal = PreprocessingPipeline(config_minimal)
        result_minimal = pipeline_minimal.fit_transform(data)
        assert result_minimal.shape == data.shape

        # Test standard preset
        config_standard = PreprocessingConfig(method="standard")
        pipeline_standard = PreprocessingPipeline(config_standard)
        result_standard = pipeline_standard.fit_transform(data)
        assert result_standard.shape == data.shape

        # Test advanced preset
        config_advanced = PreprocessingConfig(method="advanced")
        pipeline_advanced = PreprocessingPipeline(config_advanced)
        result_advanced = pipeline_advanced.fit_transform(data)
        assert result_advanced.shape == data.shape

    def test_pipeline_with_3d_data(self):
        """Test pipeline with 3D hyperspectral data."""
        # Generate 3D data (lines, samples, bands)
        data = np.random.rand(50, 50, 30)

        config = PreprocessingConfig(method="standard")
        pipeline = PreprocessingPipeline(config)

        # Should handle 3D data
        result = pipeline.fit_transform(data)
        assert result.shape == data.shape

    def test_pipeline_step_ordering(self):
        """Test that pipeline steps are applied in correct order."""
        config = PreprocessingConfig(
            method="custom",
            smoothing=True,
            baseline_correction=True,
            snv=True
        )
        pipeline = PreprocessingPipeline(config)

        steps = pipeline.get_step_names()
        # Smoothing should come before SNV
        assert steps.index("smoothing") < steps.index("snv")

    def test_pipeline_describe(self):
        """Test pipeline describe method."""
        config = PreprocessingConfig(method="standard")
        pipeline = PreprocessingPipeline(config)

        description = pipeline.describe()
        assert "method" in description
        assert "steps" in description
        assert "config" in description

    def test_pipeline_save_load_config(self, tmp_path):
        """Test saving and loading pipeline configuration."""
        # Create pipeline with custom config
        config = PreprocessingConfig(
            method="custom",
            snv=True,
            smoothing=True,
            smoothing_window=7
        )
        pipeline = PreprocessingPipeline(config)

        # Save config
        config_path = tmp_path / "test_config.yaml"
        pipeline.save_config(config_path)
        assert config_path.exists()

        # Load config
        loaded_pipeline = PreprocessingPipeline.from_config_file(config_path)
        assert loaded_pipeline.config.snv == pipeline.config.snv
        assert loaded_pipeline.config.smoothing_window == 7

    def test_nan_inf_handling(self):
        """Test handling of NaN and Inf values."""
        # Create data with NaN and Inf
        data = np.random.rand(10, 50)
        data[0, 0] = np.nan
        data[1, 1] = np.inf

        config = PreprocessingConfig(method="minimal")
        pipeline = PreprocessingPipeline(config)

        result = pipeline.fit_transform(data)

        # Should handle NaN and Inf
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))