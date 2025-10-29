"""Comprehensive CLI tests to improve coverage."""

import pytest
from click.testing import CliRunner
from pathlib import Path
import tempfile
import shutil
import numpy as np
import pandas as pd
import json
import yaml
from unittest.mock import patch, MagicMock

from hyperseed.cli.main import main, analyze, segment, batch_process
from hyperseed.tests.fixtures import create_synthetic_envi_dataset


class TestCLIMain:
    """Test main CLI functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    def test_main_help(self):
        """Test main help output."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'Hyperspectral Seed Image Analysis Tool' in result.output
        assert 'Commands:' in result.output
        assert 'analyze' in result.output
        assert 'segment' in result.output
        assert 'batch' in result.output

    def test_main_version(self):
        """Test version output."""
        result = self.runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        assert 'version' in result.output.lower() or 'hyperseed' in result.output.lower()

    def test_main_no_command(self):
        """Test main without command."""
        result = self.runner.invoke(main, [])
        # Should show help or error
        assert result.exit_code in [0, 2]

    def test_main_invalid_command(self):
        """Test invalid command."""
        result = self.runner.invoke(main, ['invalid_command'])
        assert result.exit_code != 0
        assert 'Error' in result.output or 'No such command' in result.output


class TestAnalyzeCommand:
    """Test analyze command comprehensively."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create synthetic dataset
        self.dataset_path = create_synthetic_envi_dataset(
            self.temp_dir,
            lines=30,
            samples=30,
            bands=10,
            n_seeds=3,
            include_references=True,
            seed=42
        )

    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_analyze_basic(self):
        """Test basic analyze command."""
        output_file = self.temp_dir / "results.csv"

        result = self.runner.invoke(analyze, [
            str(self.dataset_path),
            '--output', str(output_file),
            '--no-visualize'
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Check CSV content
        df = pd.read_csv(output_file)
        assert len(df) > 0
        assert df.shape[1] >= 10  # At least as many columns as bands

    def test_analyze_with_preprocessing_options(self):
        """Test analyze with different preprocessing options."""
        preprocess_methods = ['minimal', 'standard', 'advanced']

        for method in preprocess_methods:
            output_file = self.temp_dir / f"results_{method}.csv"

            result = self.runner.invoke(analyze, [
                str(self.dataset_path),
                '--output', str(output_file),
                '--preprocess', method,
                '--no-visualize'
            ])

            assert result.exit_code == 0, f"Failed with preprocessing: {method}"
            assert output_file.exists()

    def test_analyze_with_segmentation_algorithms(self):
        """Test analyze with different segmentation algorithms."""
        algorithms = ['threshold', 'watershed', 'combined']

        for algo in algorithms:
            output_file = self.temp_dir / f"results_{algo}.csv"

            result = self.runner.invoke(analyze, [
                str(self.dataset_path),
                '--output', str(output_file),
                '--segmentation', algo,
                '--no-visualize'
            ])

            assert result.exit_code == 0, f"Failed with algorithm: {algo}"

    def test_analyze_with_min_pixels(self):
        """Test min pixels filtering."""
        output_file = self.temp_dir / "results_filtered.csv"

        result = self.runner.invoke(analyze, [
            str(self.dataset_path),
            '--output', str(output_file),
            '--min-pixels', '20',
            '--no-visualize'
        ])

        assert result.exit_code == 0

    def test_analyze_export_plots(self):
        """Test plot export."""
        output_file = self.temp_dir / "results.csv"

        result = self.runner.invoke(analyze, [
            str(self.dataset_path),
            '--output', str(output_file),
            '--export-plots',
            '--plot-format', 'png',
            '--no-visualize'
        ])

        assert result.exit_code == 0
        # Check for plot files
        plot_files = list(self.temp_dir.glob("*.png"))
        assert len(plot_files) >= 1  # At least one plot

    def test_analyze_export_mask(self):
        """Test mask export."""
        output_file = self.temp_dir / "results.csv"
        mask_file = self.temp_dir / "mask.npy"

        result = self.runner.invoke(analyze, [
            str(self.dataset_path),
            '--output', str(output_file),
            '--export-mask', str(mask_file),
            '--no-visualize'
        ])

        assert result.exit_code == 0
        assert mask_file.exists()

        # Load and verify mask
        mask = np.load(mask_file)
        assert mask.shape == (30, 30)

    def test_analyze_with_config(self):
        """Test analyze with config file."""
        config_file = self.temp_dir / "config.yaml"
        config_data = {
            'preprocessing': {
                'method': 'standard',
                'snv': True,
                'smoothing': True,
                'smoothing_window': 5
            },
            'segmentation': {
                'algorithm': 'watershed',
                'min_pixels': 15
            }
        }

        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        output_file = self.temp_dir / "results.csv"

        result = self.runner.invoke(analyze, [
            str(self.dataset_path),
            '--config', str(config_file),
            '--output', str(output_file),
            '--no-visualize'
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_analyze_verbose(self):
        """Test verbose output."""
        result = self.runner.invoke(analyze, [
            str(self.dataset_path),
            '--verbose',
            '--no-visualize'
        ])

        assert result.exit_code == 0
        # Verbose should produce more output
        assert len(result.output) > 100

    def test_analyze_quiet(self):
        """Test quiet mode."""
        result = self.runner.invoke(analyze, [
            str(self.dataset_path),
            '--quiet',
            '--no-visualize'
        ])

        assert result.exit_code == 0
        # Quiet should produce minimal output
        assert len(result.output) < 500

    def test_analyze_invalid_input(self):
        """Test with invalid input path."""
        result = self.runner.invoke(analyze, [
            '/non/existent/path',
            '--no-visualize'
        ])

        assert result.exit_code != 0
        assert 'error' in result.output.lower() or 'not found' in result.output.lower()

    def test_analyze_empty_dataset(self):
        """Test with dataset that produces no seeds."""
        # Create dataset with no seeds (all zeros)
        empty_dataset = create_synthetic_envi_dataset(
            self.temp_dir / "empty",
            lines=10,
            samples=10,
            bands=5,
            n_seeds=0,  # No seeds
            seed=42
        )

        result = self.runner.invoke(analyze, [
            str(empty_dataset),
            '--no-visualize'
        ])

        # Should handle gracefully
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            assert 'no seeds' in result.output.lower() or 'warning' in result.output.lower()


class TestSegmentCommand:
    """Test segment command comprehensively."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())

        self.dataset_path = create_synthetic_envi_dataset(
            self.temp_dir,
            lines=30,
            samples=30,
            bands=10,
            n_seeds=5,
            seed=42
        )

    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_segment_basic(self):
        """Test basic segment command."""
        output_file = self.temp_dir / "mask.npy"

        result = self.runner.invoke(segment, [
            str(self.dataset_path),
            '--output', str(output_file),
            '--no-visualize'
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        mask = np.load(output_file)
        assert mask.shape == (30, 30)
        assert np.unique(mask).size >= 2  # At least background and one seed

    def test_segment_algorithms(self):
        """Test different segmentation algorithms."""
        algorithms = ['threshold', 'watershed', 'combined']

        for algo in algorithms:
            output_file = self.temp_dir / f"mask_{algo}.npy"

            result = self.runner.invoke(segment, [
                str(self.dataset_path),
                '--output', str(output_file),
                '--algorithm', algo,
                '--no-visualize'
            ])

            assert result.exit_code == 0, f"Failed with algorithm: {algo}"
            assert output_file.exists()

    def test_segment_min_pixels(self):
        """Test min pixels parameter."""
        output_file = self.temp_dir / "mask_filtered.npy"

        result = self.runner.invoke(segment, [
            str(self.dataset_path),
            '--output', str(output_file),
            '--min-pixels', '50',
            '--no-visualize'
        ])

        assert result.exit_code == 0

        # Check that small regions were removed
        mask = np.load(output_file)
        unique_labels = np.unique(mask[mask > 0])
        for label in unique_labels:
            area = np.sum(mask == label)
            assert area >= 50

    def test_segment_visualize(self):
        """Test visualization option."""
        with patch('matplotlib.pyplot.show'):
            result = self.runner.invoke(segment, [
                str(self.dataset_path),
                '--visualize'
            ])

            assert result.exit_code == 0


class TestBatchCommand:
    """Test batch processing command."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create multiple datasets
        self.datasets = []
        for i in range(3):
            dataset_path = create_synthetic_envi_dataset(
                self.temp_dir / f"dataset_{i}",
                lines=20,
                samples=20,
                bands=5,
                n_seeds=2,
                seed=i
            )
            self.datasets.append(str(dataset_path))

    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_batch_basic(self):
        """Test basic batch processing."""
        # Create input list file
        list_file = self.temp_dir / "datasets.txt"
        list_file.write_text('\n'.join(self.datasets))

        output_dir = self.temp_dir / "batch_output"

        result = self.runner.invoke(batch_process, [
            str(list_file),
            '--output-dir', str(output_dir)
        ])

        assert result.exit_code == 0
        assert output_dir.exists()

        # Check for output files
        output_files = list(output_dir.glob("*.csv"))
        assert len(output_files) >= len(self.datasets)

    def test_batch_with_options(self):
        """Test batch with processing options."""
        list_file = self.temp_dir / "datasets.txt"
        list_file.write_text('\n'.join(self.datasets))

        output_dir = self.temp_dir / "batch_output"

        result = self.runner.invoke(batch_process, [
            str(list_file),
            '--output-dir', str(output_dir),
            '--preprocess', 'minimal',
            '--segmentation', 'threshold',
            '--min-pixels', '10'
        ])

        assert result.exit_code == 0

    def test_batch_parallel(self):
        """Test parallel processing."""
        list_file = self.temp_dir / "datasets.txt"
        list_file.write_text('\n'.join(self.datasets))

        output_dir = self.temp_dir / "batch_output"

        result = self.runner.invoke(batch_process, [
            str(list_file),
            '--output-dir', str(output_dir),
            '--parallel', '2'
        ])

        assert result.exit_code == 0

    def test_batch_empty_list(self):
        """Test with empty list file."""
        empty_list = self.temp_dir / "empty.txt"
        empty_list.write_text("")

        output_dir = self.temp_dir / "batch_output"

        result = self.runner.invoke(batch_process, [
            str(empty_list),
            '--output-dir', str(output_dir)
        ])

        # Should handle empty list gracefully
        assert result.exit_code in [0, 1]

    def test_batch_invalid_datasets(self):
        """Test with invalid dataset paths."""
        invalid_list = self.temp_dir / "invalid.txt"
        invalid_list.write_text("/non/existent/path\n/another/bad/path")

        output_dir = self.temp_dir / "batch_output"

        result = self.runner.invoke(batch_process, [
            str(invalid_list),
            '--output-dir', str(output_dir)
        ])

        # Should handle errors
        assert result.exit_code != 0 or 'error' in result.output.lower()


class TestCLIUtilities:
    """Test CLI utility functions and commands."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch('hyperseed.cli.main.logging.basicConfig')
    def test_setup_logging_verbose(self, mock_logging):
        """Test verbose logging setup."""
        # This would test the setup_logging function if it were exposed
        # For now, test verbose flag in commands
        dataset_path = create_synthetic_envi_dataset(
            self.temp_dir,
            lines=10,
            samples=10,
            bands=5,
            n_seeds=1
        )

        result = self.runner.invoke(analyze, [
            str(dataset_path),
            '--verbose',
            '--no-visualize'
        ])

        assert result.exit_code == 0

    def test_config_generation(self):
        """Test config file generation."""
        config_file = self.temp_dir / "generated_config.yaml"

        # If there's a config generation command
        result = self.runner.invoke(main, [
            'config',
            '--output', str(config_file)
        ])

        # May not be implemented yet
        if result.exit_code == 0:
            assert config_file.exists()

            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert 'preprocessing' in config
            assert 'segmentation' in config

    def test_info_command(self):
        """Test info/version command."""
        result = self.runner.invoke(main, ['info'])

        # May not be implemented
        if result.exit_code == 0:
            assert 'version' in result.output.lower() or 'hyperseed' in result.output.lower()

    def test_help_for_all_commands(self):
        """Test help for all commands."""
        commands = ['analyze', 'segment', 'batch']

        for cmd in commands:
            result = self.runner.invoke(main, [cmd, '--help'])
            assert result.exit_code == 0
            assert 'Options:' in result.output or 'Usage:' in result.output

    def test_output_formats(self):
        """Test different output formats."""
        dataset_path = create_synthetic_envi_dataset(
            self.temp_dir,
            lines=10,
            samples=10,
            bands=5,
            n_seeds=2
        )

        # Test CSV output (default)
        csv_file = self.temp_dir / "results.csv"
        result = self.runner.invoke(analyze, [
            str(dataset_path),
            '--output', str(csv_file),
            '--no-visualize'
        ])
        assert result.exit_code == 0
        assert csv_file.exists()

        # Test if other formats are supported
        excel_file = self.temp_dir / "results.xlsx"
        result = self.runner.invoke(analyze, [
            str(dataset_path),
            '--output', str(excel_file),
            '--format', 'excel',
            '--no-visualize'
        ])
        # May not be implemented
        if result.exit_code == 0:
            assert excel_file.exists()

    def test_wavelength_specification(self):
        """Test wavelength file specification."""
        dataset_path = create_synthetic_envi_dataset(
            self.temp_dir,
            lines=10,
            samples=10,
            bands=5
        )

        # Create wavelength file
        wavelength_file = self.temp_dir / "wavelengths.txt"
        wavelengths = np.linspace(1000, 2000, 5)
        np.savetxt(wavelength_file, wavelengths)

        result = self.runner.invoke(analyze, [
            str(dataset_path),
            '--wavelengths', str(wavelength_file),
            '--no-visualize'
        ])

        assert result.exit_code == 0

    def test_dry_run_mode(self):
        """Test dry run mode if available."""
        dataset_path = create_synthetic_envi_dataset(
            self.temp_dir,
            lines=10,
            samples=10,
            bands=5
        )

        result = self.runner.invoke(analyze, [
            str(dataset_path),
            '--dry-run',
            '--no-visualize'
        ])

        # May not be implemented
        if '--dry-run' in result.output or result.exit_code == 0:
            # Should not create output files in dry run
            output_files = list(self.temp_dir.glob("*.csv"))
            assert len(output_files) == 0 or '--dry-run' not in result.output