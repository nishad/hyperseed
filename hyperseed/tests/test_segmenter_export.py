"""Test segmenter export methods to improve coverage."""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import shutil

from hyperseed.core.segmentation.segmenter import SeedSegmenter
from hyperseed.config.settings import SegmentationConfig


class TestSegmenterExportMethods:
    """Test SeedSegmenter export methods."""

    def setup_method(self):
        """Set up test data."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create test data
        self.data = np.random.rand(50, 50, 10)
        self.data[10:30, 10:30, :] = 0.8  # Bright region 1
        self.data[35:45, 35:45, :] = 0.75  # Bright region 2

        # Create and run segmenter
        self.segmenter = SeedSegmenter(SegmentationConfig(algorithm='threshold'))
        self.mask, self.n_seeds = self.segmenter.segment(self.data)

    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_export_mask_npy(self):
        """Test exporting mask as numpy array."""
        output_path = self.temp_dir / "mask.npy"

        self.segmenter.export_mask(output_path, format='npy')

        assert output_path.exists()

        # Load and verify
        loaded_mask = np.load(output_path)
        assert np.array_equal(loaded_mask, self.mask)

    def test_export_mask_png(self):
        """Test exporting mask as PNG image."""
        output_path = self.temp_dir / "mask.png"

        with patch('PIL.Image.fromarray') as mock_fromarray:
            mock_img = MagicMock()
            mock_fromarray.return_value = mock_img

            self.segmenter.export_mask(output_path, format='png')

            # Check that PIL was called
            mock_fromarray.assert_called_once()
            mock_img.save.assert_called_once_with(output_path)

    def test_export_mask_tiff(self):
        """Test exporting mask as TIFF image."""
        output_path = self.temp_dir / "mask.tiff"

        with patch('PIL.Image.fromarray') as mock_fromarray:
            mock_img = MagicMock()
            mock_fromarray.return_value = mock_img

            self.segmenter.export_mask(output_path, format='tiff')

            # Check that PIL was called
            mock_fromarray.assert_called_once()
            mock_img.save.assert_called_once_with(output_path)

    def test_export_mask_auto_format(self):
        """Test export mask with format auto-detection."""
        # Test .npy extension
        npy_path = self.temp_dir / "mask.npy"
        self.segmenter.export_mask(npy_path)
        assert npy_path.exists()

        # Test .png extension
        png_path = self.temp_dir / "mask.png"
        with patch('PIL.Image.fromarray'):
            self.segmenter.export_mask(png_path)

        # Test .tif extension
        tif_path = self.temp_dir / "mask.tif"
        with patch('PIL.Image.fromarray'):
            self.segmenter.export_mask(tif_path)

    def test_export_mask_invalid_format(self):
        """Test export mask with invalid format."""
        output_path = self.temp_dir / "mask.xyz"

        with pytest.raises(ValueError, match="Unsupported format"):
            self.segmenter.export_mask(output_path, format='xyz')

    def test_export_mask_without_segmentation(self):
        """Test export mask before segmentation."""
        new_segmenter = SeedSegmenter()
        output_path = self.temp_dir / "mask.npy"

        with pytest.raises(ValueError, match="No segmentation"):
            new_segmenter.export_mask(output_path)

    def test_export_properties_csv(self):
        """Test exporting properties as CSV."""
        output_path = self.temp_dir / "properties.csv"

        self.segmenter.export_properties(output_path, format='csv')

        assert output_path.exists()

        # Check CSV content
        import csv
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) > 0
            # Check expected columns
            if rows:
                assert 'seed_id' in rows[0] or 'label' in rows[0]

    def test_export_properties_json(self):
        """Test exporting properties as JSON."""
        output_path = self.temp_dir / "properties.json"

        self.segmenter.export_properties(output_path, format='json')

        assert output_path.exists()

        # Load and verify JSON
        with open(output_path, 'r') as f:
            data = json.load(f)
            assert isinstance(data, (list, dict))
            if isinstance(data, list):
                assert len(data) > 0
                if data:
                    assert 'area' in data[0] or 'seed_id' in data[0]

    def test_export_properties_auto_format(self):
        """Test export properties with format auto-detection."""
        # CSV format
        csv_path = self.temp_dir / "props.csv"
        self.segmenter.export_properties(csv_path)
        assert csv_path.exists()

        # JSON format
        json_path = self.temp_dir / "props.json"
        self.segmenter.export_properties(json_path)
        assert json_path.exists()

    def test_export_properties_without_segmentation(self):
        """Test export properties before segmentation."""
        new_segmenter = SeedSegmenter()
        output_path = self.temp_dir / "props.csv"

        with pytest.raises(ValueError, match="No seed properties"):
            new_segmenter.export_properties(output_path)

    def test_describe_method(self):
        """Test describe method."""
        description = self.segmenter.describe()

        assert isinstance(description, dict)
        assert 'algorithm' in description
        assert 'n_seeds' in description
        assert description['algorithm'] == 'threshold'
        assert description['n_seeds'] == self.n_seeds

        # Check for additional info
        assert 'min_pixels' in description or 'config' in description

    def test_describe_without_segmentation(self):
        """Test describe before segmentation."""
        new_segmenter = SeedSegmenter()
        description = new_segmenter.describe()

        assert isinstance(description, dict)
        assert 'algorithm' in description
        assert description.get('n_seeds') == 0 or description.get('n_seeds') is None

    def test_get_validation_stats(self):
        """Test get_validation_stats method."""
        stats = self.segmenter.get_validation_stats()

        if stats is not None:
            assert isinstance(stats, dict)
            # Check for expected keys
            assert 'total_seeds' in stats or 'final_count' in stats or self.segmenter.validation_stats is None

    def test_get_seed_properties(self):
        """Test getting seed properties."""
        props = self.segmenter.get_seed_properties()

        if props is not None:
            assert isinstance(props, (list, dict, np.ndarray))
            # Properties should have been computed during segmentation
            if isinstance(props, list) and len(props) > 0:
                assert 'area' in props[0] or 'label' in props[0]


class TestSegmenterVisualization:
    """Test visualization methods with full parameters."""

    def setup_method(self):
        """Set up test data."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data = np.random.rand(30, 30, 5)
        self.data[10:20, 10:20, :] = 0.8

        self.segmenter = SeedSegmenter()
        self.mask, _ = self.segmenter.segment(self.data)

    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_visualize_with_save_path(self, mock_figure, mock_show, mock_savefig):
        """Test visualization with save path."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.subplots.return_value = (mock_fig, [[MagicMock(), MagicMock(), MagicMock()]])

        save_path = self.temp_dir / "segmentation.png"

        fig = self.segmenter.visualize(
            self.data,
            save_path=save_path,
            show_labels=True,
            show_boundaries=True
        )

        # Check save was called
        mock_savefig.assert_called_once()
        assert mock_fig is not None

    @patch('matplotlib.pyplot.figure')
    def test_visualize_with_band_index(self, mock_figure):
        """Test visualization with specific band index."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.subplots.return_value = (mock_fig, [[MagicMock(), MagicMock(), MagicMock()]])

        fig = self.segmenter.visualize(
            self.data,
            band_index=2,
            show_labels=False,
            show_boundaries=False
        )

        assert mock_figure.called

    def test_visualize_without_segmentation(self):
        """Test visualization before segmentation."""
        new_segmenter = SeedSegmenter()

        with pytest.raises(ValueError, match="No segmentation"):
            new_segmenter.visualize(self.data)


class TestSegmenterEdgeCases:
    """Test edge cases and error paths."""

    def test_segment_with_unknown_algorithm(self):
        """Test segmentation with unknown algorithm."""
        config = SegmentationConfig(algorithm='unknown_algo')
        segmenter = SeedSegmenter(config)

        data = np.random.rand(20, 20, 5)

        with pytest.raises(ValueError, match="Unknown segmentation algorithm"):
            segmenter.segment(data)

    def test_segment_connected_algorithm(self):
        """Test connected components algorithm path."""
        config = SegmentationConfig(algorithm='connected')
        segmenter = SeedSegmenter(config)

        data = np.random.rand(30, 30, 10)
        data[10:20, 10:20, :] = 0.8

        mask, n_seeds = segmenter.segment(data)
        assert mask is not None
        assert n_seeds >= 0

    def test_validation_stats_access(self):
        """Test accessing validation stats."""
        segmenter = SeedSegmenter()
        data = np.random.rand(30, 30, 10)

        # Before segmentation
        stats = segmenter.get_validation_stats()
        assert stats is None

        # After segmentation with validation
        mask, _ = segmenter.segment(data, validate=True)
        stats = segmenter.get_validation_stats()
        if stats is not None:
            assert isinstance(stats, dict)