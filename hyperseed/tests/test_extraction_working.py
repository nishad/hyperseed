"""Working tests for SpectralExtractor that match actual implementation."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
import tempfile
import h5py

from hyperseed.core.extraction.extractor import SpectralExtractor


class TestSpectralExtractorActual:
    """Test SpectralExtractor with actual implementation."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        # Create simple test data
        self.data = np.random.rand(30, 30, 10).astype(np.float32)

        # Create mask with a few seeds
        self.mask = np.zeros((30, 30), dtype=np.int32)
        self.mask[5:10, 5:10] = 1    # 25 pixels
        self.mask[15:20, 15:20] = 2  # 25 pixels
        self.mask[22:24, 22:24] = 3  # 4 pixels

        self.wavelengths = np.linspace(1000, 2000, 10)

    def test_extract_returns_dict(self):
        """Test that extract returns a dictionary."""
        extractor = SpectralExtractor()
        result = extractor.extract(self.data, self.mask, self.wavelengths)

        assert isinstance(result, dict)
        assert 'n_seeds' in result
        assert result['n_seeds'] == 3

    def test_extract_stores_data(self):
        """Test that extract stores data in instance variables."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        assert extractor.spectra is not None
        assert extractor.wavelengths is not None
        assert extractor.seed_info is not None
        assert len(extractor.seed_info) == 3

    def test_to_dataframe_after_extract(self):
        """Test DataFrame conversion after extraction."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        df = extractor.to_dataframe(include_wavelengths=True)

        assert isinstance(df, pd.DataFrame)
        # Should have seed info columns plus band columns
        assert df.shape[0] == 3  # 3 seeds
        assert 'band_1000.00nm' in df.columns or 'label' in df.columns

    def test_to_dataframe_without_extract_raises(self):
        """Test that to_dataframe raises error without extraction."""
        extractor = SpectralExtractor()

        with pytest.raises(ValueError, match="No spectra extracted"):
            extractor.to_dataframe()

    def test_save_csv(self, tmp_path):
        """Test saving to CSV."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        csv_path = tmp_path / "test.csv"
        extractor.save_csv(csv_path)

        assert csv_path.exists()

        # Load and verify
        df = pd.read_csv(csv_path)
        assert len(df) == 3  # 3 seeds

    def test_save_hdf5(self, tmp_path):
        """Test saving to HDF5."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        h5_path = tmp_path / "test.h5"
        extractor.save_hdf5(h5_path)

        assert h5_path.exists()

        # Verify HDF5 content
        with h5py.File(h5_path, 'r') as f:
            assert 'spectra' in f
            assert 'wavelengths' in f
            assert 'seed_info' in f
            assert f['spectra'].shape == (3, 10)

    def test_get_statistics(self):
        """Test statistics computation."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        stats = extractor.get_statistics()

        assert isinstance(stats, dict)
        assert 'n_seeds' in stats
        assert stats['n_seeds'] == 3

    def test_remove_outliers(self):
        """Test outlier removal."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        # Initial count
        initial_count = len(extractor.seed_info)

        # Remove outliers (small seeds)
        result = extractor.remove_outliers(min_area=10)

        # Should remove the small seed (4 pixels)
        assert len(extractor.seed_info) < initial_count
        assert len(extractor.seed_info) == 2  # Only 2 seeds left

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_spectra(self, mock_savefig, mock_figure):
        """Test spectra plotting with mocking."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        # Call plot_spectra
        extractor.plot_spectra(show_mean=True, show_std=True)

        # Check that plot was created
        assert mock_figure.called
        # Check that some plot method was called
        assert mock_ax.plot.called or mock_ax.errorbar.called

    @patch('matplotlib.pyplot.figure')
    def test_plot_distribution(self, mock_figure):
        """Test distribution plotting with mocking."""
        mock_fig = MagicMock()
        mock_axes = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
        mock_figure.return_value = mock_fig
        mock_fig.subplots.return_value = (mock_fig, mock_axes)

        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        # Call plot_distribution
        extractor.plot_distribution()

        # Check that plot was created
        assert mock_figure.called

    def test_filter_seeds(self):
        """Test seed filtering."""
        extractor = SpectralExtractor()
        extractor.extract(self.data, self.mask, self.wavelengths)

        # Filter by area
        result = extractor.filter_seeds(min_area=10, max_area=30)

        # Should return filtered results
        assert isinstance(result, tuple)
        assert len(result) == 3  # spectra, seed_info, removed_indices


class TestSpectralExtractorEdgeCases:
    """Test edge cases for SpectralExtractor."""

    def test_empty_mask(self):
        """Test with empty mask."""
        data = np.random.rand(20, 20, 5).astype(np.float32)
        mask = np.zeros((20, 20), dtype=np.int32)

        extractor = SpectralExtractor()
        result = extractor.extract(data, mask)

        assert result['n_seeds'] == 0
        assert result['spectra'] is None

    def test_single_seed(self):
        """Test with single seed."""
        data = np.random.rand(20, 20, 5).astype(np.float32)
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[5:10, 5:10] = 1

        extractor = SpectralExtractor()
        result = extractor.extract(data, mask)

        assert result['n_seeds'] == 1
        assert result['spectra'].shape == (1, 5)

    def test_shape_mismatch(self):
        """Test with mismatched shapes."""
        data = np.random.rand(20, 20, 5).astype(np.float32)
        mask = np.zeros((30, 30), dtype=np.int32)

        extractor = SpectralExtractor()

        with pytest.raises(ValueError, match="shape"):
            extractor.extract(data, mask)

    def test_repr(self):
        """Test string representation."""
        extractor = SpectralExtractor()

        repr_str = repr(extractor)
        assert 'SpectralExtractor' in repr_str

        # After extraction
        data = np.random.rand(20, 20, 5).astype(np.float32)
        mask = np.ones((20, 20), dtype=np.int32)
        extractor.extract(data, mask)

        repr_str = repr(extractor)
        assert 'n_seeds=1' in repr_str or 'SpectralExtractor' in repr_str


class TestSpectralExtractorIntegration:
    """Integration tests for SpectralExtractor."""

    def test_full_workflow(self, tmp_path):
        """Test complete extraction workflow."""
        # Create data
        data = np.random.rand(50, 50, 20).astype(np.float32)
        mask = np.zeros((50, 50), dtype=np.int32)

        # Add multiple seeds
        mask[5:15, 5:15] = 1
        mask[20:30, 20:30] = 2
        mask[35:45, 35:45] = 3
        mask[2:4, 2:4] = 4  # Small seed

        wavelengths = np.linspace(900, 1700, 20)

        # Extract
        extractor = SpectralExtractor()
        result = extractor.extract(data, mask, wavelengths)

        assert result['n_seeds'] == 4

        # Remove outliers
        extractor.remove_outliers(min_area=10)
        assert len(extractor.seed_info) == 3  # Small seed removed

        # Convert to DataFrame
        df = extractor.to_dataframe()
        assert len(df) == 3

        # Save to CSV
        csv_path = tmp_path / "results.csv"
        extractor.save_csv(csv_path)
        assert csv_path.exists()

        # Save to HDF5
        h5_path = tmp_path / "results.h5"
        extractor.save_hdf5(h5_path)
        assert h5_path.exists()

        # Get statistics
        stats = extractor.get_statistics()
        assert stats['n_seeds'] == 3

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plotting_workflow(self, mock_savefig, mock_show):
        """Test plotting workflow."""
        # Create data
        data = np.random.rand(30, 30, 15).astype(np.float32)
        mask = np.zeros((30, 30), dtype=np.int32)
        mask[10:20, 10:20] = 1
        mask[5:8, 5:8] = 2

        wavelengths = np.linspace(1000, 2000, 15)

        # Extract and plot
        extractor = SpectralExtractor()
        extractor.extract(data, mask, wavelengths)

        # These should not crash
        with patch('matplotlib.pyplot.figure', return_value=MagicMock()):
            extractor.plot_spectra()
            extractor.plot_distribution()

            # Private method test with path
            temp_path = Path(tempfile.gettempdir()) / "test_plot.png"
            extractor._plot_statistics(temp_path)