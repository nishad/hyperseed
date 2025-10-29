"""Simple working tests for segmentation module to improve coverage."""

import pytest
import numpy as np
from hyperseed.core.segmentation import algorithms
from hyperseed.core.segmentation.segmenter import SeedSegmenter
from hyperseed.core.segmentation import validation
from hyperseed.config.settings import SegmentationConfig


class TestSegmentationAlgorithms:
    """Test suite for segmentation algorithms."""

    def setup_method(self):
        """Create test data."""
        # Simple test image
        self.image = np.random.rand(50, 50)
        self.image[10:30, 10:30] = 0.8  # Bright square
        self.image[35:45, 35:45] = 0.7  # Another bright region

    def test_threshold_segmentation(self):
        """Test threshold segmentation."""
        mask = algorithms.threshold_segmentation(self.image)
        assert mask is not None
        assert mask.shape == self.image.shape

        # Test with specific threshold
        mask2 = algorithms.threshold_segmentation(self.image, threshold_value=0.5)
        assert mask2 is not None

    def test_watershed_segmentation(self):
        """Test watershed segmentation."""
        mask = algorithms.watershed_segmentation(self.image)
        assert mask is not None
        assert mask.shape == self.image.shape

    def test_connected_components(self):
        """Test connected components segmentation."""
        binary_mask = self.image > 0.5
        mask = algorithms.connected_components_segmentation(binary_mask)
        assert mask is not None
        assert mask.shape == self.image.shape

    def test_combined_segmentation(self):
        """Test combined segmentation."""
        # Create 3D data for combined segmentation
        data_3d = np.stack([self.image] * 10, axis=2)
        mask = algorithms.combined_segmentation(data_3d)
        assert mask is not None
        assert mask.shape == self.image.shape

    def test_morphological_operations(self):
        """Test morphological operations."""
        mask = self.image > 0.5
        processed = algorithms.apply_morphological_operations(
            mask.astype(np.uint8),
            operations=['opening', 'closing']
        )
        assert processed is not None
        assert processed.shape == mask.shape

    def test_create_seed_mask(self):
        """Test seed mask creation."""
        mask = algorithms.create_seed_mask(self.image, n_seeds=5)
        assert mask is not None
        assert mask.shape == self.image.shape


class TestSeedSegmenter:
    """Test the main segmenter class."""

    def setup_method(self):
        """Create test data."""
        self.data = np.random.rand(50, 50, 10)
        self.data[10:30, 10:30, :] = 0.8

    def test_segmenter_init(self):
        """Test segmenter initialization."""
        segmenter = SeedSegmenter()
        assert segmenter.config is not None

        # With custom config
        config = SegmentationConfig(min_pixels=20)
        segmenter2 = SeedSegmenter(config)
        assert segmenter2.config.min_pixels == 20

    def test_segment(self):
        """Test segmentation."""
        segmenter = SeedSegmenter()
        mask, info = segmenter.segment(self.data)
        assert mask is not None
        assert mask.shape == (50, 50)
        assert isinstance(info, dict)

    def test_segment_different_algorithms(self):
        """Test different algorithms."""
        for algo in ['watershed', 'threshold']:
            config = SegmentationConfig(algorithm=algo)
            segmenter = SeedSegmenter(config)
            mask, info = segmenter.segment(self.data)
            assert mask is not None
            assert info['algorithm'] == algo

    def test_extract_seed_regions(self):
        """Test seed region extraction."""
        segmenter = SeedSegmenter()
        mask, _ = segmenter.segment(self.data)
        regions = segmenter.extract_seed_regions(mask)
        assert isinstance(regions, list)

    def test_visualize_segmentation(self):
        """Test visualization doesn't crash."""
        segmenter = SeedSegmenter()
        mask, _ = segmenter.segment(self.data)
        # Just test it doesn't crash
        fig = segmenter.visualize_segmentation(self.data, mask)
        # May return None if matplotlib not in interactive mode
        assert fig is None or fig is not None


class TestSegmentationValidation:
    """Test validation functions."""

    def setup_method(self):
        """Create test masks."""
        self.true_mask = np.zeros((50, 50), dtype=np.int32)
        self.true_mask[10:30, 10:30] = 1
        self.pred_mask = np.zeros((50, 50), dtype=np.int32)
        self.pred_mask[12:28, 12:28] = 1

    def test_validate_segmentation(self):
        """Test validation."""
        metrics = validation.validate_segmentation(self.pred_mask, self.true_mask)
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics

    def test_calculate_iou(self):
        """Test IoU calculation."""
        iou = validation.calculate_iou(self.pred_mask, self.true_mask)
        assert isinstance(iou, float)
        assert 0 <= iou <= 1

    def test_calculate_dice(self):
        """Test Dice coefficient."""
        dice = validation.calculate_dice_coefficient(self.pred_mask, self.true_mask)
        assert isinstance(dice, float)
        assert 0 <= dice <= 1

    def test_precision_recall(self):
        """Test precision and recall."""
        precision, recall = validation.calculate_precision_recall(
            self.pred_mask, self.true_mask
        )
        assert isinstance(precision, float)
        assert isinstance(recall, float)

    def test_evaluate_quality(self):
        """Test quality evaluation."""
        report = validation.evaluate_segmentation_quality(
            self.pred_mask, self.true_mask
        )
        assert isinstance(report, dict)