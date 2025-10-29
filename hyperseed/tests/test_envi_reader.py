"""Unit tests for ENVI reader module."""

import pytest
import numpy as np
from pathlib import Path

from hyperseed.core.io.envi_reader import ENVIReader, ENVIHeader
from hyperseed.tests.fixtures import SyntheticDataFixture


class TestENVIReader:
    """Test suite for ENVI reader functionality."""

    def test_envi_header_creation(self):
        """Test ENVIHeader dataclass creation."""
        header = ENVIHeader(
            samples=384,
            lines=500,
            bands=272,
            header_offset=0,
            data_type=4,
            interleave="bsq",
            byte_order=0
        )

        assert header.samples == 384
        assert header.lines == 500
        assert header.bands == 272
        assert header.interleave == "bsq"
        assert header.metadata == {}

    def test_read_synthetic_data(self):
        """Test reading synthetic ENVI data."""
        with SyntheticDataFixture() as dataset_path:
            # Find header file
            hdr_file = dataset_path / "capture" / "synthetic_data.hdr"
            assert hdr_file.exists()

            # Create reader
            reader = ENVIReader(hdr_file)

            # Check header parsing
            assert reader.header.samples == 200
            assert reader.header.lines == 200
            assert reader.header.bands == 100

            # Read data
            data = reader.read_data()
            assert data.shape == (200, 200, 100)
            assert data.dtype == np.float32

            # Check wavelengths
            wavelengths = reader.get_wavelengths()
            assert wavelengths is not None
            assert len(wavelengths) == 100

    def test_read_single_band(self):
        """Test reading a single band."""
        with SyntheticDataFixture() as dataset_path:
            hdr_file = dataset_path / "capture" / "synthetic_data.hdr"
            reader = ENVIReader(hdr_file)

            # Read first band
            band_0 = reader.read_band(0)
            assert band_0.shape == (200, 200)

            # Read last band
            band_last = reader.read_band(99)
            assert band_last.shape == (200, 200)

            # Test invalid band index
            with pytest.raises(ValueError):
                reader.read_band(100)

    def test_data_subsetting(self):
        """Test reading subsets of data."""
        with SyntheticDataFixture() as dataset_path:
            hdr_file = dataset_path / "capture" / "synthetic_data.hdr"
            reader = ENVIReader(hdr_file)

            # Read subset of lines
            subset = reader.read_data(lines=slice(0, 10))
            assert subset.shape == (10, 200, 100)

            # Read subset of samples
            subset = reader.read_data(samples=slice(50, 150))
            assert subset.shape == (200, 100, 100)

            # Read subset of bands
            subset = reader.read_data(bands=slice(0, 50))
            assert subset.shape == (200, 200, 50)

            # Read combined subset
            subset = reader.read_data(
                lines=slice(0, 50),
                samples=slice(0, 50),
                bands=slice(0, 10)
            )
            assert subset.shape == (50, 50, 10)

    def test_metadata_extraction(self):
        """Test metadata extraction from header."""
        with SyntheticDataFixture() as dataset_path:
            hdr_file = dataset_path / "capture" / "synthetic_data.hdr"
            reader = ENVIReader(hdr_file)

            metadata = reader.get_metadata()
            assert isinstance(metadata, dict)
            assert 'samples' in metadata
            assert 'lines' in metadata
            assert 'bands' in metadata

    def test_shape_property(self):
        """Test get_shape method."""
        with SyntheticDataFixture() as dataset_path:
            hdr_file = dataset_path / "capture" / "synthetic_data.hdr"
            reader = ENVIReader(hdr_file)

            shape = reader.get_shape()
            assert shape == (200, 200, 100)

    def test_repr_string(self):
        """Test string representation."""
        with SyntheticDataFixture() as dataset_path:
            hdr_file = dataset_path / "capture" / "synthetic_data.hdr"
            reader = ENVIReader(hdr_file)

            repr_str = repr(reader)
            assert "ENVIReader" in repr_str
            assert "200" in repr_str  # dimensions
            assert "100" in repr_str  # bands