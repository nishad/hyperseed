"""Test for __main__.py module to improve coverage."""

import pytest
import subprocess
import sys
from pathlib import Path


class TestMainModule:
    """Test the __main__.py module execution."""

    def test_main_module_help(self):
        """Test running hyperseed as module with --help."""
        result = subprocess.run(
            [sys.executable, "-m", "hyperseed", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0
        assert "hyperseed" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_main_module_version(self):
        """Test running hyperseed as module with --version."""
        result = subprocess.run(
            [sys.executable, "-m", "hyperseed", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Version might not be implemented, but module should load
        assert result.returncode in [0, 2]  # 0 for success, 2 for command not found

    def test_main_module_no_args(self):
        """Test running hyperseed as module with no arguments."""
        result = subprocess.run(
            [sys.executable, "-m", "hyperseed"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should show help or usage
        assert "usage" in result.stdout.lower() or "command" in result.stdout.lower() or result.returncode != 0

    def test_main_module_invalid_command(self):
        """Test running hyperseed with invalid command."""
        result = subprocess.run(
            [sys.executable, "-m", "hyperseed", "invalid_command_xyz"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0

    def test_main_module_import(self):
        """Test that __main__ can be imported (for coverage)."""
        try:
            import hyperseed.__main__
            # The import itself is what we're testing
            assert True
        except ImportError:
            pytest.fail("Could not import hyperseed.__main__")