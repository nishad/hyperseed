"""Hyperseed - Professional Hyperspectral Seed Image Analysis Tool.

A comprehensive Python package for analyzing hyperspectral imagery of plant seeds,
designed for scientific research with emphasis on accuracy and reproducibility.
"""

__version__ = "0.1.0a3"
__author__ = "Hyperseed Development Team"

# Package-level imports for convenience
from hyperseed.core.io.envi_reader import ENVIReader
from hyperseed.config.settings import Settings
from hyperseed.cli.main import main

__all__ = [
    "ENVIReader",
    "Settings",
    "main",
    "__version__",
]