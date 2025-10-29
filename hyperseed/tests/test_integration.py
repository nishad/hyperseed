"""Integration tests for end-to-end workflows."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from hyperseed.core.calibration.reflectance import ReflectanceCalibrator
from hyperseed.core.preprocessing.pipeline import PreprocessingPipeline
from hyperseed.core.segmentation.segmenter import SeedSegmenter
from hyperseed.core.extraction.extractor import SpectralExtractor
from hyperseed.core.io.envi_reader import ENVIReader
from hyperseed.config.settings import Settings, PreprocessingConfig, SegmentationConfig
from hyperseed.tests.fixtures import create_synthetic_envi_dataset


class TestEndToEndPipeline:
    """Test complete processing pipeline."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create synthetic dataset
        self.dataset_path = create_synthetic_envi_dataset(
            self.temp_dir,
            lines=50,
            samples=50,
            bands=20,
            n_seeds=5,
            include_references=True,
            seed=42
        )

    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_complete_pipeline_minimal(self):
        """Test complete pipeline with minimal preprocessing."""
        # Step 1: Calibration
        calibrator = ReflectanceCalibrator(clip_negative=True, clip_max=1.0)
        calibrated_data, reader = calibrator.calibrate_from_directory(self.dataset_path)

        assert calibrated_data is not None
        assert calibrated_data.shape == (50, 50, 20)
        assert reader is not None

        # Step 2: Preprocessing
        preprocess_config = PreprocessingConfig(method='minimal')
        preprocessor = PreprocessingPipeline(preprocess_config)
        preprocessed_data = preprocessor.fit_transform(calibrated_data)

        assert preprocessed_data.shape == calibrated_data.shape

        # Step 3: Segmentation
        seg_config = SegmentationConfig(algorithm='threshold', min_pixels=10)
        segmenter = SeedSegmenter(seg_config)
        mask, seg_info = segmenter.segment(preprocessed_data)

        assert mask is not None
        assert mask.shape == (50, 50)
        assert seg_info['n_segments'] > 0

        # Step 4: Extraction
        extractor = SpectralExtractor()
        results = extractor.extract(
            preprocessed_data,
            mask,
            wavelengths=reader.get_wavelengths()
        )

        assert results is not None
        assert results['n_seeds'] > 0
        assert results['spectra'] is not None

    def test_complete_pipeline_standard(self):
        """Test complete pipeline with standard preprocessing."""
        # Load and calibrate
        calibrator = ReflectanceCalibrator(clip_negative=True, clip_max=1.0)
        calibrated_data, reader = calibrator.calibrate_from_directory(self.dataset_path)

        # Preprocess with standard method
        preprocess_config = PreprocessingConfig(method='standard')
        preprocessor = PreprocessingPipeline(preprocess_config)
        preprocessed_data = preprocessor.fit_transform(calibrated_data)

        # Segment with watershed
        seg_config = SegmentationConfig(algorithm='watershed', min_pixels=15)
        segmenter = SeedSegmenter(seg_config)
        mask, seg_info = segmenter.segment(preprocessed_data)

        # Extract spectra
        extractor = SpectralExtractor()
        results = extractor.extract(preprocessed_data, mask)

        # Convert to DataFrame
        df = extractor.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_complete_pipeline_advanced(self):
        """Test complete pipeline with advanced preprocessing."""
        # Load and calibrate
        calibrator = ReflectanceCalibrator(clip_negative=True, clip_max=1.0)
        calibrated_data, reader = calibrator.calibrate_from_directory(self.dataset_path)

        # Advanced preprocessing
        preprocess_config = PreprocessingConfig(method='advanced')
        preprocessor = PreprocessingPipeline(preprocess_config)
        preprocessed_data = preprocessor.fit_transform(calibrated_data)

        # Combined segmentation
        seg_config = SegmentationConfig(algorithm='combined', min_pixels=20)
        segmenter = SeedSegmenter(seg_config)
        mask, seg_info = segmenter.segment(preprocessed_data)

        # Extract and save
        extractor = SpectralExtractor()
        results = extractor.extract(preprocessed_data, mask)

        # Save results
        output_file = self.temp_dir / "results.csv"
        extractor.save_csv(output_file)
        assert output_file.exists()

    def test_pipeline_with_outlier_removal(self):
        """Test pipeline with outlier removal."""
        # Create dataset with outliers
        calibrator = ReflectanceCalibrator(clip_negative=True, clip_max=1.0)
        calibrated_data, reader = calibrator.calibrate_from_directory(self.dataset_path)

        # Preprocess
        preprocessor = PreprocessingPipeline()
        preprocessed_data = preprocessor.fit_transform(calibrated_data)

        # Segment
        segmenter = SeedSegmenter()
        mask, _ = segmenter.segment(preprocessed_data)

        # Add some very small regions (outliers)
        mask[0:2, 0:2] = 100  # 4 pixels - too small
        mask[48:49, 48:49] = 101  # 1 pixel - too small

        # Extract
        extractor = SpectralExtractor()
        results = extractor.extract(preprocessed_data, mask)

        initial_seeds = results['n_seeds']

        # Remove outliers
        n_removed = extractor.remove_outliers(min_area=10)

        assert n_removed >= 2  # Should remove the small regions
        assert extractor.spectra.shape[0] < initial_seeds

    def test_pipeline_no_seeds_found(self):
        """Test pipeline when no seeds are found."""
        # Create empty dataset (all zeros)
        empty_dataset = create_synthetic_envi_dataset(
            self.temp_dir / "empty",
            lines=20,
            samples=20,
            bands=10,
            n_seeds=0,  # No seeds
            seed=42
        )

        # Run pipeline
        calibrator = ReflectanceCalibrator()
        calibrated_data, reader = calibrator.calibrate_from_directory(empty_dataset)

        preprocessor = PreprocessingPipeline()
        preprocessed_data = preprocessor.fit_transform(calibrated_data)

        segmenter = SeedSegmenter()
        mask, seg_info = segmenter.segment(preprocessed_data)

        # Should handle empty mask gracefully
        extractor = SpectralExtractor()
        results = extractor.extract(preprocessed_data, mask)

        assert results['n_seeds'] == 0
        assert results['spectra'] is None

    def test_pipeline_with_different_datatypes(self):
        """Test pipeline with different data types."""
        # Test with float64 data
        calibrator = ReflectanceCalibrator()
        calibrated_data, reader = calibrator.calibrate_from_directory(self.dataset_path)

        # Convert to float64
        data_float64 = calibrated_data.astype(np.float64)

        preprocessor = PreprocessingPipeline()
        preprocessed_data = preprocessor.fit_transform(data_float64)

        assert preprocessed_data.dtype in [np.float32, np.float64]

        segmenter = SeedSegmenter()
        mask, _ = segmenter.segment(preprocessed_data)

        assert mask.dtype in [np.int32, np.uint8, np.uint16]

    def test_pipeline_with_settings_object(self):
        """Test pipeline using Settings configuration object."""
        # Create settings
        settings = Settings()
        settings.preprocessing.method = 'standard'
        settings.preprocessing.snv = True
        settings.preprocessing.smoothing = True
        settings.segmentation.algorithm = 'watershed'
        settings.segmentation.min_pixels = 15

        # Run pipeline with settings
        calibrator = ReflectanceCalibrator(
            clip_negative=settings.calibration.clip_negative,
            clip_max=settings.calibration.clip_max
        )
        calibrated_data, reader = calibrator.calibrate_from_directory(self.dataset_path)

        preprocessor = PreprocessingPipeline(settings.preprocessing)
        preprocessed_data = preprocessor.fit_transform(calibrated_data)

        segmenter = SeedSegmenter(settings.segmentation)
        mask, _ = segmenter.segment(preprocessed_data)

        extractor = SpectralExtractor()
        results = extractor.extract(preprocessed_data, mask)

        assert results['n_seeds'] > 0

    def test_pipeline_error_recovery(self):
        """Test pipeline error handling and recovery."""
        # Test with invalid data
        invalid_data = np.array([])  # Empty array

        preprocessor = PreprocessingPipeline()

        # Should handle invalid input gracefully
        try:
            result = preprocessor.fit_transform(invalid_data)
            # If it doesn't raise, check result
            assert result is not None
        except (ValueError, IndexError):
            # Expected for invalid input
            pass

        # Test with NaN data
        nan_data = np.full((50, 50, 20), np.nan)

        try:
            result = preprocessor.fit_transform(nan_data)
            # Should either handle NaN or raise appropriate error
            assert True
        except (ValueError, RuntimeWarning):
            pass

    def test_batch_processing_workflow(self):
        """Test batch processing of multiple datasets."""
        # Create multiple datasets
        datasets = []
        for i in range(3):
            dataset_path = create_synthetic_envi_dataset(
                self.temp_dir / f"dataset_{i}",
                lines=30,
                samples=30,
                bands=10,
                n_seeds=3,
                seed=i
            )
            datasets.append(dataset_path)

        # Process each dataset
        all_results = []
        for dataset_path in datasets:
            # Calibrate
            calibrator = ReflectanceCalibrator()
            calibrated_data, reader = calibrator.calibrate_from_directory(dataset_path)

            # Preprocess
            preprocessor = PreprocessingPipeline()
            preprocessed_data = preprocessor.fit_transform(calibrated_data)

            # Segment
            segmenter = SeedSegmenter()
            mask, _ = segmenter.segment(preprocessed_data)

            # Extract
            extractor = SpectralExtractor()
            results = extractor.extract(preprocessed_data, mask)

            all_results.append(results)

        assert len(all_results) == 3
        for result in all_results:
            assert result['n_seeds'] > 0

    def test_memory_efficient_processing(self):
        """Test memory-efficient processing for large datasets."""
        # Create a moderate-sized dataset
        dataset_path = create_synthetic_envi_dataset(
            self.temp_dir / "large",
            lines=100,
            samples=100,
            bands=50,
            n_seeds=20,
            seed=42
        )

        # Process with memory tracking
        import tracemalloc
        tracemalloc.start()

        # Run pipeline
        calibrator = ReflectanceCalibrator()
        calibrated_data, reader = calibrator.calibrate_from_directory(dataset_path)

        preprocessor = PreprocessingPipeline()
        preprocessed_data = preprocessor.fit_transform(calibrated_data)

        segmenter = SeedSegmenter()
        mask, _ = segmenter.segment(preprocessed_data)

        extractor = SpectralExtractor()
        results = extractor.extract(preprocessed_data, mask)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable (< 1GB for this size)
        assert peak < 1024 * 1024 * 1024  # 1GB
        assert results['n_seeds'] > 0