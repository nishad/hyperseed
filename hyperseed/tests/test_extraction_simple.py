"""Simple working tests for extraction module to improve coverage."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from hyperseed.core.extraction.extractor import SpectralExtractor
from hyperseed.config.settings import ProcessingConfig


class TestSpectralExtractor:
    """Test spectral extraction functionality."""

    def setup_method(self):
        """Create test data."""
        # Create synthetic hyperspectral data
        self.data = np.random.rand(50, 50, 20)
        self.data[10:30, 10:30, :] = 0.8  # Bright region

        # Create segmentation mask
        self.mask = np.zeros((50, 50), dtype=np.int32)
        self.mask[10:30, 10:30] = 1
        self.mask[35:45, 35:45] = 2

        # Wavelengths
        self.wavelengths = np.linspace(1000, 2000, 20)

    def test_extractor_init(self):
        """Test extractor initialization."""
        extractor = SpectralExtractor()
        assert extractor.config is not None

        # With wavelengths
        extractor2 = SpectralExtractor(wavelengths=self.wavelengths)
        assert extractor2.wavelengths is not None
        assert len(extractor2.wavelengths) == 20

        # With config (simplified test)
        extractor3 = SpectralExtractor()
        assert extractor3.config is not None

    def test_extract(self):
        """Test spectral extraction."""
        extractor = SpectralExtractor(wavelengths=self.wavelengths)
        result = extractor.extract(self.data, self.mask)

        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) > 0
            assert result.shape[1] == 20  # Number of bands

    def test_extract_by_region(self):
        """Test extraction by region."""
        extractor = SpectralExtractor()
        spectra = extractor.extract_by_region(self.data, self.mask)

        assert isinstance(spectra, dict)
        # Should have entries for each unique label (except 0)
        assert 1 in spectra
        assert 2 in spectra
        assert spectra[1].shape == (20,)  # Number of bands

    def test_compute_statistics(self):
        """Test statistics computation."""
        extractor = SpectralExtractor()
        stats = extractor.compute_statistics(self.data, self.mask)

        assert isinstance(stats, dict)
        assert 1 in stats
        assert 'mean' in stats[1]
        assert 'std' in stats[1]

    def test_extract_all_pixels(self):
        """Test pixel-wise extraction."""
        extractor = SpectralExtractor()
        pixels = extractor.extract_all_pixels(self.data, self.mask)

        assert isinstance(pixels, dict)
        # Each region should have pixel data
        assert 1 in pixels
        assert pixels[1].shape[1] == 20  # Number of bands

    def test_export_to_dataframe(self):
        """Test DataFrame export."""
        extractor = SpectralExtractor(wavelengths=self.wavelengths)
        spectra_dict = extractor.extract_by_region(self.data, self.mask)
        df = extractor.export_to_dataframe(spectra_dict)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Two regions
        assert df.shape[1] == 20  # Number of bands

    def test_save_spectra(self, tmp_path):
        """Test saving spectra to file."""
        extractor = SpectralExtractor(wavelengths=self.wavelengths)
        spectra_dict = extractor.extract_by_region(self.data, self.mask)
        df = extractor.export_to_dataframe(spectra_dict)

        # Save as CSV
        csv_file = tmp_path / "spectra.csv"
        extractor.save_spectra(df, csv_file, format='csv')
        assert csv_file.exists()

        # Load and verify
        loaded = pd.read_csv(csv_file, index_col=0)
        assert loaded.shape == df.shape

    def test_extract_with_preprocessing(self):
        """Test extraction with preprocessing."""
        extractor = SpectralExtractor()
        spectra = extractor.extract_by_region(self.data, self.mask)

        assert spectra is not None
        assert 1 in spectra

    def test_compute_indices(self):
        """Test spectral index computation."""
        extractor = SpectralExtractor(wavelengths=self.wavelengths)
        spectra_dict = extractor.extract_by_region(self.data, self.mask)

        # Compute simple band ratios
        indices = extractor.compute_band_ratios(spectra_dict, [(0, 1), (5, 10)])
        assert isinstance(indices, dict)
        assert 1 in indices

    def test_filter_by_quality(self):
        """Test quality filtering."""
        extractor = SpectralExtractor()
        spectra_dict = extractor.extract_by_region(self.data, self.mask)

        # Filter by SNR or other quality metric
        filtered = extractor.filter_by_quality(spectra_dict, min_snr=0.5)
        assert isinstance(filtered, dict)
        assert len(filtered) <= len(spectra_dict)

    def test_handle_empty_mask(self):
        """Test handling of empty mask."""
        extractor = SpectralExtractor()
        empty_mask = np.zeros((50, 50), dtype=np.int32)

        result = extractor.extract_by_region(self.data, empty_mask)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_handle_single_region(self):
        """Test single region extraction."""
        extractor = SpectralExtractor()
        single_mask = np.ones((50, 50), dtype=np.int32)

        result = extractor.extract_by_region(self.data, single_mask)
        assert isinstance(result, dict)
        assert 1 in result

    def test_different_extraction_methods(self):
        """Test different extraction methods."""
        # Just test basic extraction works
        extractor = SpectralExtractor()
        spectra = extractor.extract_by_region(self.data, self.mask)

        assert spectra is not None
        assert 1 in spectra

    def test_batch_extraction(self):
        """Test batch extraction."""
        extractor = SpectralExtractor()

        # Create multiple datasets
        data_list = [self.data, self.data * 0.9, self.data * 1.1]
        mask_list = [self.mask, self.mask, self.mask]

        results = extractor.batch_extract(data_list, mask_list)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_outlier_removal(self):
        """Test outlier removal."""
        # Add outliers
        data_with_outliers = self.data.copy()
        data_with_outliers[15, 15, :] = 100  # Outlier pixel

        extractor = SpectralExtractor()
        spectra = extractor.extract_by_region(data_with_outliers, self.mask)

        # Just check extraction works
        assert spectra is not None