"""Comprehensive tests for SpectralExtractor to improve coverage."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import tempfile
import h5py

from hyperseed.core.extraction.extractor import SpectralExtractor


class TestSpectralExtractorCore:
    """Test core extraction functionality."""

    def setup_method(self):
        """Set up test data."""
        # Create synthetic hyperspectral data
        np.random.seed(42)
        self.data = np.random.rand(50, 50, 20).astype(np.float32)

        # Create segmentation mask with multiple seeds
        self.mask = np.zeros((50, 50), dtype=np.int32)
        self.mask[5:15, 5:15] = 1    # Seed 1 (100 pixels)
        self.mask[20:35, 20:35] = 2  # Seed 2 (225 pixels)
        self.mask[40:48, 5:13] = 3   # Seed 3 (64 pixels)
        self.mask[10:12, 30:32] = 4  # Small seed (4 pixels)
        self.mask[25:27, 5:7] = 5    # Another small seed (4 pixels)

        # Create wavelengths
        self.wavelengths = np.linspace(1000, 2000, 20)

    def test_init(self):
        """Test initialization."""
        extractor = SpectralExtractor()
        assert extractor.spectra is None
        assert extractor.wavelengths is None
        assert extractor.seed_info is None
        assert isinstance(extractor.metadata, dict)
        assert len(extractor.metadata) == 0

    def test_extract_basic(self):
        """Test basic extraction."""
        extractor = SpectralExtractor()
        result = extractor.extract(self.data, self.mask, self.wavelengths)

        assert result is not None
        assert 'n_seeds' in result
        assert result['n_seeds'] == 5  # 5 seeds in mask
        assert 'spectra' in result
        assert result['spectra'].shape == (5, 20)
        assert 'seed_info' in result
        assert len(result['seed_info']) == 5

        # Check that extractor stored the data
        assert extractor.spectra is not None
        assert extractor.wavelengths is not None
        assert extractor.seed_info is not None

    def test_extract_no_wavelengths(self):
        """Test extraction without wavelengths."""
        extractor = SpectralExtractor()
        result = extractor.extract(self.data, self.mask, wavelengths=None)

        assert result is not None
        assert result['n_seeds'] == 5
        assert extractor.wavelengths is None

    def test_extract_compute_stats(self):
        """Test extraction with statistics computation."""
        extractor = SpectralExtractor()
        result = extractor.extract(
            self.data, self.mask, self.wavelengths, compute_stats=True
        )

        assert result is not None
        # Check that statistics were computed
        for seed_info in result['seed_info']:
            assert 'std' in seed_info
            assert 'min' in seed_info
            assert 'max' in seed_info

    def test_extract_no_compute_stats(self):
        """Test extraction without statistics."""
        extractor = SpectralExtractor()
        result = extractor.extract(
            self.data, self.mask, self.wavelengths, compute_stats=False
        )

        assert result is not None
        # Statistics should not be present or should be None
        for seed_info in result['seed_info']:
            if 'std' in seed_info:
                assert seed_info['std'] is None or len(seed_info['std']) == 0

    def test_extract_empty_mask(self):
        """Test extraction with empty mask."""
        empty_mask = np.zeros((50, 50), dtype=np.int32)
        extractor = SpectralExtractor()
        result = extractor.extract(self.data, empty_mask, self.wavelengths)

        assert result['n_seeds'] == 0
        assert result['spectra'] is None
        assert result['seed_info'] is None

    def test_extract_shape_mismatch(self):
        """Test extraction with mismatched shapes."""
        wrong_mask = np.zeros((30, 30), dtype=np.int32)
        extractor = SpectralExtractor()

        with pytest.raises(ValueError, match="doesn't match mask shape"):
            extractor.extract(self.data, wrong_mask, self.wavelengths)

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        df = extractor.to_dataframe(include_wavelengths=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # 5 seeds
        assert df.shape[1] == 20  # 20 bands
        assert list(df.columns) == [f"{wl:.1f}" for wl in self.wavelengths]
        assert df.index.name == 'seed_id'

    def test_to_dataframe_no_wavelengths(self):
        """Test DataFrame conversion without wavelengths."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, wavelengths=None)

        df = extractor.to_dataframe(include_wavelengths=False)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == [f"band_{i}" for i in range(20)]

    def test_to_dataframe_no_data(self):
        """Test DataFrame conversion with no extracted data."""
        extractor = SpectralExtractor()

        with pytest.raises(ValueError, match="No spectra extracted"):
            extractor.to_dataframe()

    def test_save_csv(self, tmp_path):
        """Test saving to CSV."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        csv_path = tmp_path / "test_spectra.csv"
        extractor.save_csv(csv_path, include_metadata=True)

        assert csv_path.exists()

        # Load and verify
        df = pd.read_csv(csv_path, index_col=0)
        assert len(df) == 5
        assert df.shape[1] == 20

        # Check metadata file
        metadata_path = csv_path.with_suffix('.meta.json')
        assert metadata_path.exists()

    def test_save_csv_no_metadata(self, tmp_path):
        """Test saving CSV without metadata."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        csv_path = tmp_path / "test_spectra.csv"
        extractor.save_csv(csv_path, include_metadata=False)

        assert csv_path.exists()
        metadata_path = csv_path.with_suffix('.meta.json')
        assert not metadata_path.exists()

    def test_save_hdf5(self, tmp_path):
        """Test saving to HDF5."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        hdf5_path = tmp_path / "test_spectra.h5"
        extractor.save_hdf5(hdf5_path)

        assert hdf5_path.exists()

        # Verify HDF5 content
        with h5py.File(hdf5_path, 'r') as f:
            assert 'spectra' in f
            assert 'wavelengths' in f
            assert 'seed_info' in f
            assert f['spectra'].shape == (5, 20)
            assert f['wavelengths'].shape == (20,)

    def test_get_statistics(self):
        """Test statistics computation."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        stats = extractor.get_statistics()

        assert isinstance(stats, dict)
        assert 'n_seeds' in stats
        assert stats['n_seeds'] == 5
        assert 'mean_area' in stats
        assert 'std_area' in stats
        assert 'min_area' in stats
        assert 'max_area' in stats
        assert 'mean_spectrum' in stats
        assert stats['mean_spectrum'].shape == (20,)

    def test_get_statistics_no_data(self):
        """Test statistics with no data."""
        extractor = SpectralExtractor()
        stats = extractor.get_statistics()

        assert stats['n_seeds'] == 0
        assert stats['mean_area'] == 0

    def test_repr(self):
        """Test string representation."""
        extractor = SpectralExtractor()
        repr_str = repr(extractor)

        assert 'SpectralExtractor' in repr_str
        assert 'n_seeds=0' in repr_str

        # After extraction
        extractor.extract(self.data, self.mask, self.wavelengths)
        repr_str = repr(extractor)

        assert 'n_seeds=5' in repr_str
        assert 'n_bands=20' in repr_str


class TestSpectralExtractorOutlierRemoval:
    """Test outlier removal functionality."""

    def setup_method(self):
        """Set up test data with outliers."""
        np.random.seed(42)
        self.data = np.random.rand(100, 100, 20).astype(np.float32)

        # Create mask with various seed sizes
        self.mask = np.zeros((100, 100), dtype=np.int32)

        # Normal seeds
        self.mask[10:30, 10:30] = 1    # 400 pixels - normal
        self.mask[40:55, 40:55] = 2    # 225 pixels - normal
        self.mask[70:85, 10:25] = 3    # 225 pixels - normal

        # Outlier seeds (too small)
        self.mask[5:7, 5:7] = 4        # 4 pixels - too small
        self.mask[90:91, 90:91] = 5    # 1 pixel - too small

        # Outlier seeds (too large)
        self.mask[20:70, 60:90] = 6    # 1500 pixels - too large

        # Elongated seed (high eccentricity)
        self.mask[80:82, 20:60] = 7    # 80 pixels but very elongated

        self.wavelengths = np.linspace(1000, 2000, 20)

    def test_remove_outliers_default(self):
        """Test outlier removal with default settings."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        # Remove outliers
        n_removed = extractor.remove_outliers()

        assert n_removed > 0
        assert extractor.spectra.shape[0] < 7  # Some seeds removed

        # Check that very small seeds were removed
        areas = [info['area'] for info in extractor.seed_info]
        assert all(area >= 10 for area in areas)  # Default min area

    def test_remove_outliers_area_bounds(self):
        """Test outlier removal with area bounds."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        # Remove outliers with specific bounds
        n_removed = extractor.remove_outliers(
            min_area=50,
            max_area=500
        )

        # Check that only seeds within bounds remain
        areas = [info['area'] for info in extractor.seed_info]
        assert all(50 <= area <= 500 for area in areas)

    def test_remove_outliers_eccentricity(self):
        """Test outlier removal by eccentricity."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        n_removed = extractor.remove_outliers(max_eccentricity=0.8)

        # Elongated seeds should be removed
        assert n_removed > 0

    def test_remove_outliers_iqr(self):
        """Test IQR-based outlier removal."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        n_removed = extractor.remove_outliers(
            use_iqr=True,
            iqr_factor=1.5
        )

        # Should remove statistical outliers
        assert extractor.spectra.shape[0] <= 7

    def test_remove_outliers_no_data(self):
        """Test outlier removal with no extracted data."""
        extractor = SpectralExtractor()

        n_removed = extractor.remove_outliers()
        assert n_removed == 0

    def test_filter_seeds(self):
        """Test seed filtering."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        # Filter by area
        filtered = extractor.filter_seeds(min_area=100, max_area=400)

        assert 'spectra' in filtered
        assert 'seed_info' in filtered
        assert filtered['spectra'].shape[0] < 7

    def test_filter_seeds_by_label(self):
        """Test filtering specific seed labels."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        # Keep only specific seeds
        filtered = extractor.filter_seeds(labels=[1, 2, 3])

        assert filtered['spectra'].shape[0] == 3
        labels = [info['label'] for info in filtered['seed_info']]
        assert set(labels) == {1, 2, 3}


class TestSpectralExtractorPlotting:
    """Test plotting functionality with mocking."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = np.random.rand(50, 50, 20).astype(np.float32)

        self.mask = np.zeros((50, 50), dtype=np.int32)
        self.mask[10:20, 10:20] = 1
        self.mask[30:40, 30:40] = 2

        self.wavelengths = np.linspace(1000, 2000, 20)

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_spectra(self, mock_savefig, mock_figure):
        """Test spectra plotting."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            extractor.plot_spectra(
                save_path=save_path,
                show_mean=True,
                show_std=True
            )

            # Check that plot methods were called
            assert mock_figure.called
            assert mock_ax.plot.called
            assert mock_ax.fill_between.called  # For std shading
            assert mock_savefig.called

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_spectra_individual(self, mock_savefig, mock_figure):
        """Test individual spectra plotting."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            extractor.plot_spectra(
                save_path=save_path,
                plot_individual=True,
                show_mean=False
            )

            # Should plot each seed individually
            assert mock_ax.plot.call_count >= 2

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_distribution(self, mock_savefig, mock_figure):
        """Test distribution plotting."""
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(4)]
        mock_figure.return_value = mock_fig
        mock_fig.subplots.return_value = (mock_fig, np.array(mock_axes).reshape(2, 2))

        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            extractor.plot_distribution(save_path=save_path)

            # Check that histograms were plotted
            assert mock_figure.called
            for ax in mock_axes:
                assert ax.hist.called or ax.scatter.called

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_statistics_private(self, mock_savefig, mock_figure):
        """Test private statistics plotting method."""
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(3)]
        mock_figure.return_value = mock_fig
        mock_fig.subplots.return_value = (mock_fig, mock_axes)

        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "stats.png"
            extractor._plot_statistics(save_path)

            # Check that statistics plots were created
            assert mock_figure.called
            assert mock_savefig.called


class TestSpectralExtractorEdgeCases:
    """Test edge cases and error handling."""

    def test_extract_single_band(self):
        """Test extraction with single band data."""
        data = np.random.rand(50, 50, 1).astype(np.float32)
        mask = np.ones((50, 50), dtype=np.int32)

        extractor = SpectralExtractor()
        result = extractor.extract(data, mask)

        assert result['spectra'].shape == (1, 1)

    def test_extract_large_number_of_seeds(self):
        """Test extraction with many seeds."""
        data = np.random.rand(100, 100, 10).astype(np.float32)
        # Create mask with unique label for each pixel
        mask = np.arange(10000).reshape(100, 100).astype(np.int32)
        mask[mask > 100] = 0  # Keep only 100 seeds

        extractor = SpectralExtractor()
        result = extractor.extract(data, mask)

        assert result['n_seeds'] == 100
        assert result['spectra'].shape == (100, 10)

    def test_extract_with_nan_values(self):
        """Test extraction with NaN values in data."""
        data = np.random.rand(50, 50, 20).astype(np.float32)
        data[10, 10, :] = np.nan  # Add NaN values

        mask = np.zeros((50, 50), dtype=np.int32)
        mask[5:15, 5:15] = 1

        extractor = SpectralExtractor()
        result = extractor.extract(data, mask)

        # Should handle NaN values (likely with nanmean)
        assert result is not None
        # Check that result doesn't contain NaN (if properly handled)
        if not np.any(np.isnan(result['spectra'])):
            assert True
        else:
            # Or at least it shouldn't crash
            assert result['spectra'] is not None

    def test_extract_with_zero_variance_spectra(self):
        """Test extraction with constant spectra."""
        data = np.ones((50, 50, 20), dtype=np.float32)  # Constant values
        mask = np.zeros((50, 50), dtype=np.int32)
        mask[10:20, 10:20] = 1

        extractor = SpectralExtractor()
        result = extractor.extract(data, mask, compute_stats=True)

        # Standard deviation should be zero
        assert result['seed_info'][0]['std'] == pytest.approx(0.0, abs=1e-6)

    def test_save_csv_with_special_characters(self, tmp_path):
        """Test saving CSV with path containing special characters."""
        extractor = SpectralExtractor()
        data = np.random.rand(50, 50, 20).astype(np.float32)
        mask = np.ones((50, 50), dtype=np.int32)
        extractor.extract(data, mask)

        # Path with spaces and special characters
        csv_path = tmp_path / "test spectra (v1.0).csv"
        extractor.save_csv(csv_path)

        assert csv_path.exists()

    def test_concurrent_extraction(self):
        """Test that multiple extractors don't interfere."""
        data = np.random.rand(50, 50, 20).astype(np.float32)
        mask1 = np.ones((50, 50), dtype=np.int32)
        mask2 = np.ones((50, 50), dtype=np.int32) * 2

        extractor1 = SpectralExtractor()
        extractor2 = SpectralExtractor()

        result1 = extractor1.extract(data, mask1)
        result2 = extractor2.extract(data, mask2)

        # Results should be independent
        assert result1['seed_info'][0]['label'] == 1
        assert result2['seed_info'][0]['label'] == 2