"""Unit tests for calibration module."""

import pytest
import numpy as np

from hyperseed.core.calibration.reflectance import ReflectanceCalibrator
from hyperseed.tests.fixtures import SyntheticDataFixture, generate_synthetic_hypercube


class TestReflectanceCalibrator:
    """Test suite for reflectance calibration."""

    def test_calibrator_initialization(self):
        """Test calibrator initialization."""
        calibrator = ReflectanceCalibrator(
            clip_negative=True,
            clip_max=1.0
        )

        assert calibrator.clip_negative is True
        assert calibrator.clip_max == 1.0
        assert calibrator.white_ref_data is None
        assert calibrator.dark_ref_data is None

    def test_calibration_with_synthetic_references(self):
        """Test calibration with synthetic white and dark references."""
        # Generate synthetic data
        data, _, _ = generate_synthetic_hypercube(
            lines=100, samples=100, bands=50, n_seeds=10
        )

        # Create synthetic references
        white_ref = np.ones_like(data[0:1, :, :]) * 0.9
        dark_ref = np.ones_like(data[0:1, :, :]) * 0.1

        # Create calibrator
        calibrator = ReflectanceCalibrator(
            white_ref=white_ref,
            dark_ref=dark_ref,
            clip_negative=True,
            clip_max=1.0
        )

        # Calibrate
        calibrated = calibrator.calibrate(data)

        # Check output
        assert calibrated.shape == data.shape
        assert calibrated.dtype == np.float32
        assert calibrated.min() >= 0  # No negative values
        assert calibrated.max() <= 1.0  # Clipped to max

    def test_calibration_from_directory(self):
        """Test automatic calibration from directory structure."""
        with SyntheticDataFixture() as dataset_path:
            calibrator = ReflectanceCalibrator(
                clip_negative=True,
                clip_max=1.0
            )

            calibrated, reader = calibrator.calibrate_from_directory(dataset_path)

            # Check output
            assert calibrated.shape == (200, 200, 100)
            assert calibrated.dtype == np.float32
            assert reader is not None

    def test_calibration_without_references(self):
        """Test behavior when references are not available."""
        # Generate synthetic data
        data, _, _ = generate_synthetic_hypercube(
            lines=100, samples=100, bands=50, n_seeds=10
        )

        # Calibrator without references
        calibrator = ReflectanceCalibrator()

        # Should return data as float32
        result = calibrator.calibrate(data)
        assert result.dtype == np.float32
        assert result.shape == data.shape

    def test_calibration_statistics(self):
        """Test calculation of calibration statistics."""
        # Generate synthetic data
        data, _, _ = generate_synthetic_hypercube(
            lines=100, samples=100, bands=50, n_seeds=10
        )

        calibrator = ReflectanceCalibrator()
        calibrated = calibrator.calibrate(data)

        # Get statistics
        stats = calibrator.get_statistics(calibrated)

        assert 'min' in stats
        assert 'max' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'negative_pixels' in stats
        assert 'saturated_pixels' in stats
        assert stats['shape'] == calibrated.shape

    def test_clipping_behavior(self):
        """Test clipping of reflectance values."""
        # Create data with values outside [0, 1]
        data = np.array([[[2.0, -0.5, 0.5]]])
        white_ref = np.array([[[1.0, 1.0, 1.0]]])
        dark_ref = np.array([[[0.0, 0.0, 0.0]]])

        # Test with clipping
        calibrator = ReflectanceCalibrator(
            white_ref=white_ref,
            dark_ref=dark_ref,
            clip_negative=True,
            clip_max=1.0
        )
        calibrated = calibrator.calibrate(data)

        assert calibrated[0, 0, 0] <= 1.0  # Clipped to max
        assert calibrated[0, 0, 1] >= 0.0  # Clipped to 0
        assert 0.0 <= calibrated[0, 0, 2] <= 1.0  # Within range

        # Test without clipping
        calibrator_no_clip = ReflectanceCalibrator(
            white_ref=white_ref,
            dark_ref=dark_ref,
            clip_negative=False,
            clip_max=None
        )
        calibrated_no_clip = calibrator_no_clip.calibrate(data)

        assert calibrated_no_clip[0, 0, 0] > 1.0  # Not clipped
        assert calibrated_no_clip[0, 0, 1] < 0.0  # Not clipped

    def test_division_by_zero_handling(self):
        """Test handling of division by zero in calibration."""
        # Create data where white == dark (would cause division by zero)
        data = np.ones((10, 10, 5)) * 0.5
        white_ref = np.ones((1, 10, 5)) * 0.5
        dark_ref = np.ones((1, 10, 5)) * 0.5  # Same as white

        calibrator = ReflectanceCalibrator(
            white_ref=white_ref,
            dark_ref=dark_ref
        )

        # Should not raise error
        calibrated = calibrator.calibrate(data)
        assert not np.any(np.isnan(calibrated))
        assert not np.any(np.isinf(calibrated))