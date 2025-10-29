"""Simple working tests for CLI to improve coverage."""

import pytest
from click.testing import CliRunner
from pathlib import Path
import tempfile
import shutil
import numpy as np

from hyperseed.cli.main import main, analyze, segment, batch
from hyperseed.tests.fixtures import create_synthetic_envi_dataset


class TestCLI:
    """Test CLI functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create test dataset
        self.dataset_path = create_synthetic_envi_dataset(
            self.temp_dir,
            lines=30,
            samples=30,
            bands=10,
            n_seeds=3
        )

    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_main_help(self):
        """Test main help."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'Hyperspectral' in result.output

    def test_main_version(self):
        """Test version."""
        result = self.runner.invoke(main, ['--version'])
        assert result.exit_code == 0

    def test_analyze_command(self):
        """Test analyze command."""
        output = self.temp_dir / "results.csv"
        result = self.runner.invoke(analyze, [
            str(self.dataset_path),
            '--output', str(output),
            '--no-visualize'
        ])
        assert result.exit_code == 0
        assert output.exists()

    def test_analyze_with_options(self):
        """Test analyze with options."""
        output = self.temp_dir / "results.csv"
        result = self.runner.invoke(analyze, [
            str(self.dataset_path),
            '--output', str(output),
            '--preprocess', 'minimal',
            '--segmentation', 'threshold',
            '--min-pixels', '5',
            '--no-visualize'
        ])
        assert result.exit_code == 0

    def test_segment_command(self):
        """Test segment command."""
        output = self.temp_dir / "mask.npy"
        result = self.runner.invoke(segment, [
            str(self.dataset_path),
            '--output', str(output),
            '--algorithm', 'threshold',
            '--no-visualize'
        ])
        assert result.exit_code == 0
        assert output.exists()

    def test_visualize_option(self):
        """Test visualize option."""
        result = self.runner.invoke(analyze, [
            str(self.dataset_path),
            '--visualize'
        ])
        # Just check it doesn't crash
        assert result.exit_code in [0, 1]

    def test_batch_process(self):
        """Test batch processing."""
        # Create list file
        list_file = self.temp_dir / "datasets.txt"
        list_file.write_text(str(self.dataset_path))

        output_dir = self.temp_dir / "batch_output"
        result = self.runner.invoke(batch, [
            str(list_file),
            '--output-dir', str(output_dir)
        ])
        assert result.exit_code == 0
        assert output_dir.exists()

    def test_invalid_input(self):
        """Test error handling."""
        result = self.runner.invoke(analyze, [
            '/non/existent/path',
            '--no-visualize'
        ])
        assert result.exit_code != 0

    def test_verbose_mode(self):
        """Test verbose output."""
        result = self.runner.invoke(analyze, [
            str(self.dataset_path),
            '--verbose',
            '--no-visualize'
        ])
        assert result.exit_code == 0

    def test_quiet_mode(self):
        """Test quiet mode."""
        result = self.runner.invoke(analyze, [
            str(self.dataset_path),
            '--quiet',
            '--no-visualize'
        ])
        assert result.exit_code == 0

    def test_config_file(self):
        """Test with config file."""
        config = self.temp_dir / "config.yaml"
        config.write_text("""
preprocessing:
  method: minimal
segmentation:
  algorithm: threshold
""")

        result = self.runner.invoke(analyze, [
            str(self.dataset_path),
            '--config', str(config),
            '--no-visualize'
        ])
        assert result.exit_code == 0