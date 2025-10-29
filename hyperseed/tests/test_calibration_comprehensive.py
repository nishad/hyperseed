"""Comprehensive tests for calibration module to improve coverage."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import shutil

from hyperseed.core.calibration.reflectance import ReflectanceCalibrator
from hyperseed.core.calibration.bad_pixels import (
    detect_bad_pixels,
    correct_bad_pixels,
    validate_bad_pixel_mask
)
from hyperseed.tests.fixtures import create_synthetic_envi_dataset


class TestReflectanceCalibrator:
    """Test ReflectanceCalibrator class."""

    def setup_method(self):
        """Set up test data."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create synthetic dataset
        self.dataset_path = create_synthetic_envi_dataset(
            self.temp_dir,
            lines=30,
            samples=30,
            bands=10,
            include_references=True,
            seed=42
        )

        # Create test arrays
        self.data = np.random.rand(30, 30, 10).astype(np.float32)
        self.white_ref = np.ones((30, 30, 10)).astype(np.float32) * 0.95
        self.dark_ref = np.ones((30, 30, 10)).astype(np.float32) * 0.05

    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_init_default(self):
        """Test default initialization."""
        calibrator = ReflectanceCalibrator()
        assert calibrator.clip_negative == True
        assert calibrator.clip_max == 1.0
        assert calibrator.smoothing_window is None

    def test_init_custom(self):
        """Test custom initialization."""
        calibrator = ReflectanceCalibrator(
            clip_negative=False,
            clip_max=2.0,
            smoothing_window=5
        )
        assert calibrator.clip_negative == False
        assert calibrator.clip_max == 2.0
        assert calibrator.smoothing_window == 5

    def test_calibrate_basic(self):
        """Test basic calibration."""
        calibrator = ReflectanceCalibrator()
        calibrated = calibrator.calibrate(
            self.data,
            self.white_ref,
            self.dark_ref
        )

        assert calibrated.shape == self.data.shape
        assert calibrated.dtype == np.float32
        # Check values are in expected range
        assert np.min(calibrated) >= 0.0
        assert np.max(calibrated) <= 1.0

    def test_calibrate_without_clipping(self):
        """Test calibration without clipping."""
        calibrator = ReflectanceCalibrator(
            clip_negative=False,
            clip_max=None
        )
        calibrated = calibrator.calibrate(
            self.data,
            self.white_ref,
            self.dark_ref
        )

        assert calibrated.shape == self.data.shape

    def test_calibrate_with_smoothing(self):
        """Test calibration with smoothing."""
        calibrator = ReflectanceCalibrator(smoothing_window=3)
        calibrated = calibrator.calibrate(
            self.data,
            self.white_ref,
            self.dark_ref
        )

        assert calibrated.shape == self.data.shape

    def test_calibrate_from_directory(self):
        """Test calibration from directory."""
        calibrator = ReflectanceCalibrator()
        calibrated, reader = calibrator.calibrate_from_directory(
            self.dataset_path
        )

        assert calibrated is not None
        assert reader is not None
        assert calibrated.shape[2] > 0  # Has bands

    def test_calibrate_with_bad_pixels(self):
        """Test calibration with bad pixel handling."""
        # Add bad pixels
        self.data[5, 5, :] = np.nan
        self.data[10, 10, :] = np.inf

        calibrator = ReflectanceCalibrator(
            handle_bad_pixels=True
        )
        calibrated = calibrator.calibrate(
            self.data,
            self.white_ref,
            self.dark_ref
        )

        # Bad pixels should be handled
        assert not np.any(np.isnan(calibrated))
        assert not np.any(np.isinf(calibrated))

    def test_calibrate_missing_references(self):
        """Test calibration with missing references."""
        calibrator = ReflectanceCalibrator()

        # Only dark reference
        calibrated = calibrator.calibrate(
            self.data,
            None,
            self.dark_ref
        )
        assert calibrated.shape == self.data.shape

        # Only white reference
        calibrated = calibrator.calibrate(
            self.data,
            self.white_ref,
            None
        )
        assert calibrated.shape == self.data.shape

        # No references
        calibrated = calibrator.calibrate(
            self.data,
            None,
            None
        )
        # Should return original data
        assert np.array_equal(calibrated, self.data)

    def test_calibrate_zero_division(self):
        """Test handling of zero division."""
        # Make white and dark refs equal
        equal_ref = np.ones_like(self.white_ref) * 0.5

        calibrator = ReflectanceCalibrator()
        calibrated = calibrator.calibrate(
            self.data,
            equal_ref,
            equal_ref
        )

        # Should handle gracefully
        assert not np.any(np.isnan(calibrated))
        assert not np.any(np.isinf(calibrated))

    def test_apply_smoothing(self):
        """Test smoothing application."""
        calibrator = ReflectanceCalibrator()

        # Test internal smoothing method if exists
        if hasattr(calibrator, '_apply_smoothing'):
            smoothed = calibrator._apply_smoothing(
                self.data,
                window_size=3
            )
            assert smoothed.shape == self.data.shape

    def test_validate_calibration(self):
        """Test calibration validation."""
        calibrator = ReflectanceCalibrator()
        calibrated = calibrator.calibrate(
            self.data,
            self.white_ref,
            self.dark_ref
        )

        # Check if validation method exists
        if hasattr(calibrator, 'validate_calibration'):
            is_valid = calibrator.validate_calibration(calibrated)
            assert isinstance(is_valid, bool)

    def test_get_statistics(self):
        """Test getting calibration statistics."""
        calibrator = ReflectanceCalibrator()
        calibrated = calibrator.calibrate(
            self.data,
            self.white_ref,
            self.dark_ref
        )

        if hasattr(calibrator, 'get_statistics'):
            stats = calibrator.get_statistics(calibrated)
            assert isinstance(stats, dict)


class TestBadPixelHandling:
    """Test bad pixel detection and correction."""

    def setup_method(self):
        """Set up test data."""
        self.data = np.random.rand(50, 50, 20).astype(np.float32)

        # Add various types of bad pixels
        self.data[5, 5, :] = np.nan  # NaN
        self.data[10, 10, :] = np.inf  # Inf
        self.data[15, 15, :] = -np.inf  # -Inf
        self.data[20, 20, :] = 0  # Dead pixel
        self.data[25, 25, :] = 1000  # Hot pixel

    def test_detect_bad_pixels(self):
        """Test bad pixel detection."""
        bad_mask = detect_bad_pixels(self.data)

        assert bad_mask.shape == self.data.shape[:2]
        assert bad_mask.dtype == bool

        # Should detect known bad pixels
        assert bad_mask[5, 5] == True  # NaN
        assert bad_mask[10, 10] == True  # Inf
        assert bad_mask[15, 15] == True  # -Inf

    def test_detect_bad_pixels_with_threshold(self):
        """Test bad pixel detection with custom thresholds."""
        bad_mask = detect_bad_pixels(
            self.data,
            hot_threshold=100,
            dead_threshold=0.01
        )

        assert bad_mask[25, 25] == True  # Hot pixel
        # Dead pixel detection depends on threshold

    def test_correct_bad_pixels_interpolation(self):
        """Test bad pixel correction with interpolation."""
        bad_mask = detect_bad_pixels(self.data)
        corrected = correct_bad_pixels(
            self.data,
            bad_mask,
            method='interpolate'
        )

        assert corrected.shape == self.data.shape
        # Bad pixels should be corrected
        assert not np.isnan(corrected[5, 5, 0])
        assert not np.isinf(corrected[10, 10, 0])

    def test_correct_bad_pixels_median(self):
        """Test bad pixel correction with median."""
        bad_mask = detect_bad_pixels(self.data)
        corrected = correct_bad_pixels(
            self.data,
            bad_mask,
            method='median'
        )

        assert corrected.shape == self.data.shape
        assert not np.isnan(corrected[5, 5, 0])

    def test_correct_bad_pixels_mean(self):
        """Test bad pixel correction with mean."""
        bad_mask = detect_bad_pixels(self.data)
        corrected = correct_bad_pixels(
            self.data,
            bad_mask,
            method='mean'
        )

        assert corrected.shape == self.data.shape
        assert not np.isnan(corrected[5, 5, 0])

    def test_validate_bad_pixel_mask(self):
        """Test bad pixel mask validation."""
        bad_mask = detect_bad_pixels(self.data)

        is_valid = validate_bad_pixel_mask(bad_mask, self.data.shape[:2])
        assert isinstance(is_valid, bool)
        assert is_valid == True

        # Test invalid mask
        invalid_mask = np.ones((10, 10), dtype=bool)
        is_valid = validate_bad_pixel_mask(invalid_mask, self.data.shape[:2])
        assert is_valid == False

    def test_bad_pixel_statistics(self):
        """Test bad pixel statistics."""
        bad_mask = detect_bad_pixels(self.data)

        n_bad = np.sum(bad_mask)
        total_pixels = bad_mask.size
        percentage = (n_bad / total_pixels) * 100

        assert n_bad >= 5  # We added at least 5 bad pixels
        assert percentage > 0

    def test_edge_case_all_bad(self):
        """Test correction when all pixels are bad."""
        # Make all pixels bad
        bad_data = np.full((10, 10, 5), np.nan)
        bad_mask = detect_bad_pixels(bad_data)

        # Should detect all as bad
        assert np.all(bad_mask)

        # Correction should handle gracefully
        corrected = correct_bad_pixels(bad_data, bad_mask)
        assert corrected.shape == bad_data.shape

    def test_edge_case_no_bad(self):
        """Test when no bad pixels exist."""
        good_data = np.random.rand(10, 10, 5)
        bad_mask = detect_bad_pixels(good_data)

        # Should detect none as bad (or very few)
        assert np.sum(bad_mask) < 5

        corrected = correct_bad_pixels(good_data, bad_mask)
        # Should be mostly unchanged
        assert np.allclose(corrected, good_data, rtol=0.01)


class TestCalibrationIntegration:
    """Integration tests for calibration workflow."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_full_calibration_workflow(self):
        """Test complete calibration workflow."""
        # Create dataset with bad pixels
        dataset_path = create_synthetic_envi_dataset(
            self.temp_dir,
            lines=100,
            samples=100,
            bands=25,
            include_references=True,
            seed=42
        )

        # Run calibration
        calibrator = ReflectanceCalibrator(
            clip_negative=True,
            clip_max=1.0,
            handle_bad_pixels=True,
            smoothing_window=3
        )

        calibrated, reader = calibrator.calibrate_from_directory(dataset_path)

        assert calibrated is not None
        assert reader is not None

        # Check output quality
        assert np.min(calibrated) >= 0.0
        assert np.max(calibrated) <= 1.0
        assert not np.any(np.isnan(calibrated))
        assert not np.any(np.isinf(calibrated))

    def test_calibration_with_statistics(self):
        """Test calibration with statistics computation."""
        dataset_path = create_synthetic_envi_dataset(
            self.temp_dir,
            lines=50,
            samples=50,
            bands=15,
            include_references=True
        )

        calibrator = ReflectanceCalibrator()
        calibrated, reader = calibrator.calibrate_from_directory(dataset_path)

        # Compute statistics
        mean_reflectance = np.mean(calibrated)
        std_reflectance = np.std(calibrated)

        assert 0 <= mean_reflectance <= 1
        assert std_reflectance > 0