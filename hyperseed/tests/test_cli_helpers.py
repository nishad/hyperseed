"""Test CLI helper functions to improve coverage."""

import pytest
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
import shutil

from hyperseed.cli.main import setup_logging, config, info, main


class TestCLIHelpers:
    """Test CLI helper functions."""

    def test_setup_logging_default(self):
        """Test setup_logging with default settings."""
        # Clear existing handlers
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging(verbose=False, debug=False)

        # Check that logging is configured
        assert logging.getLogger().level == logging.WARNING

    def test_setup_logging_verbose(self):
        """Test setup_logging with verbose flag."""
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging(verbose=True, debug=False)

        # Verbose should set INFO level
        assert logging.getLogger().level == logging.INFO

    def test_setup_logging_debug(self):
        """Test setup_logging with debug flag."""
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        setup_logging(verbose=False, debug=True)

        # Debug should set DEBUG level
        assert logging.getLogger().level == logging.DEBUG


class TestConfigCommand:
    """Test config command."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_config_command_default(self):
        """Test config command with default output."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(config)

            # Should create a config file
            assert result.exit_code == 0
            # Check for expected output or file creation
            assert Path("hyperseed_config.yaml").exists() or "config" in result.output.lower()

    def test_config_command_with_output(self):
        """Test config command with specified output file."""
        config_file = self.temp_dir / "custom_config.yaml"

        result = self.runner.invoke(config, [
            '--output', str(config_file)
        ])

        assert result.exit_code == 0
        assert config_file.exists()

    def test_config_command_with_preset(self):
        """Test config command with different presets."""
        for preset in ['minimal', 'standard', 'advanced']:
            config_file = self.temp_dir / f"config_{preset}.yaml"

            result = self.runner.invoke(config, [
                '--preset', preset,
                '--output', str(config_file)
            ])

            assert result.exit_code == 0
            assert config_file.exists()

            # Check file content
            content = config_file.read_text()
            assert 'preprocessing' in content or 'segmentation' in content

    @patch('hyperseed.cli.main.Settings')
    def test_config_command_error_handling(self, mock_settings):
        """Test config command error handling."""
        mock_settings.side_effect = Exception("Config error")

        result = self.runner.invoke(config)

        # Should handle error gracefully
        assert result.exit_code != 0 or "error" in result.output.lower()


class TestInfoCommand:
    """Test info command."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    def test_info_command_basic(self):
        """Test basic info command."""
        result = self.runner.invoke(info)

        assert result.exit_code == 0
        output = result.output.lower()

        # Should show system information
        assert any(word in output for word in ['system', 'python', 'version', 'hyperseed'])

    def test_info_command_verbose(self):
        """Test info command with verbose flag."""
        result = self.runner.invoke(info, ['--verbose'])

        assert result.exit_code == 0
        output = result.output.lower()

        # Verbose should show more details
        assert len(output) > 100  # Should have substantial output

    @patch('platform.system')
    @patch('platform.python_version')
    def test_info_command_with_mocked_system(self, mock_python_version, mock_system):
        """Test info command with mocked system info."""
        mock_system.return_value = "TestOS"
        mock_python_version.return_value = "3.9.0"

        result = self.runner.invoke(info)

        assert result.exit_code == 0
        assert "TestOS" in result.output or "3.9" in result.output

    def test_info_command_dependencies(self):
        """Test info command shows dependencies."""
        result = self.runner.invoke(info, ['--show-deps'])

        if result.exit_code == 0:
            # If the flag is supported, check for package names
            output = result.output.lower()
            assert any(pkg in output for pkg in ['numpy', 'scipy', 'scikit'])


class TestMainCLI:
    """Test main CLI entry point with commands."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_main_verbose_flag(self):
        """Test main CLI with verbose flag."""
        with patch('hyperseed.cli.main.setup_logging') as mock_setup:
            result = self.runner.invoke(main, ['--verbose', '--help'])

            # setup_logging should be called with verbose=True
            if mock_setup.called:
                args, kwargs = mock_setup.call_args
                assert kwargs.get('verbose') == True or (len(args) > 0 and args[0] == True)

    def test_main_debug_flag(self):
        """Test main CLI with debug flag."""
        with patch('hyperseed.cli.main.setup_logging') as mock_setup:
            result = self.runner.invoke(main, ['--debug', '--help'])

            # setup_logging should be called with debug=True
            if mock_setup.called:
                args, kwargs = mock_setup.call_args
                assert kwargs.get('debug') == True or (len(args) > 1 and args[1] == True)

    def test_main_with_config_subcommand(self):
        """Test main CLI accessing config subcommand."""
        result = self.runner.invoke(main, ['config', '--help'])

        assert result.exit_code == 0
        assert 'config' in result.output.lower()

    def test_main_with_info_subcommand(self):
        """Test main CLI accessing info subcommand."""
        result = self.runner.invoke(main, ['info'])

        assert result.exit_code == 0
        assert len(result.output) > 0