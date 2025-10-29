"""Synthetic data generation for testing.

This module provides functions to generate synthetic hyperspectral data
for testing purposes without using any real research data.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import tempfile
import shutil


def generate_synthetic_spectrum(
    n_bands: int = 272,
    wavelength_range: Tuple[float, float] = (1000.0, 2500.0),
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic spectral signature.

    Args:
        n_bands: Number of spectral bands.
        wavelength_range: Range of wavelengths in nm.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (spectrum, wavelengths).
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate wavelengths
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_bands)

    # Generate base spectrum with some realistic features
    # Simulate absorption features at certain wavelengths
    spectrum = np.ones(n_bands) * 0.5

    # Add some gaussian absorption features
    absorption_centers = [1200, 1450, 1940, 2100]  # Common absorption bands
    for center in absorption_centers:
        if wavelength_range[0] <= center <= wavelength_range[1]:
            idx = np.argmin(np.abs(wavelengths - center))
            width = 20  # Width of absorption feature
            depth = 0.1 + np.random.random() * 0.1
            for i in range(max(0, idx - width), min(n_bands, idx + width)):
                dist = abs(i - idx)
                spectrum[i] -= depth * np.exp(-dist**2 / (2 * (width/3)**2))

    # Add some noise
    spectrum += np.random.normal(0, 0.01, n_bands)

    # Ensure values are positive
    spectrum = np.maximum(spectrum, 0.01)

    return spectrum, wavelengths


def generate_synthetic_hypercube(
    lines: int = 500,
    samples: int = 384,
    bands: int = 272,
    n_seeds: int = 50,
    seed_size_range: Tuple[int, int] = (150, 500),
    background_value: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic hyperspectral data cube with seeds.

    Args:
        lines: Number of lines (rows).
        samples: Number of samples (columns).
        bands: Number of spectral bands.
        n_seeds: Number of seeds to generate.
        seed_size_range: Range of seed sizes in pixels.
        background_value: Background reflectance value.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (data_cube, segmentation_mask, wavelengths).
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize data cube with background
    data_cube = np.ones((lines, samples, bands), dtype=np.float32) * background_value
    data_cube += np.random.normal(0, 0.01, (lines, samples, bands))

    # Initialize segmentation mask
    mask = np.zeros((lines, samples), dtype=np.int32)

    # Generate wavelengths
    wavelengths = np.linspace(1000.0, 2500.0, bands)

    # Generate seeds
    seed_id = 1
    for _ in range(n_seeds):
        # Random position
        center_y = np.random.randint(50, lines - 50)
        center_x = np.random.randint(50, samples - 50)

        # Random size
        size = np.random.randint(seed_size_range[0], seed_size_range[1])
        radius = int(np.sqrt(size / np.pi))

        # Create elliptical seed shape
        angle = np.random.random() * np.pi
        eccentricity = 0.3 + np.random.random() * 0.4

        # Generate seed mask
        y, x = np.ogrid[:lines, :samples]
        y_rot = (y - center_y) * np.cos(angle) - (x - center_x) * np.sin(angle)
        x_rot = (y - center_y) * np.sin(angle) + (x - center_x) * np.cos(angle)
        seed_mask = (x_rot**2 / radius**2 + y_rot**2 / (radius * eccentricity)**2) <= 1

        # Check for overlap
        if np.any(mask[seed_mask] > 0):
            continue  # Skip overlapping seeds

        # Generate seed spectrum
        seed_spectrum, _ = generate_synthetic_spectrum(bands, seed=seed_id)

        # Add spatial variation within seed
        spatial_variation = np.random.normal(1.0, 0.05, np.sum(seed_mask))

        # Assign spectrum to seed pixels
        for idx, (y_idx, x_idx) in enumerate(zip(*np.where(seed_mask))):
            data_cube[y_idx, x_idx, :] = seed_spectrum * spatial_variation[idx]

        # Update mask
        mask[seed_mask] = seed_id
        seed_id += 1

        if seed_id > n_seeds:
            break

    # Ensure positive values
    data_cube = np.maximum(data_cube, 0.01)

    return data_cube, mask, wavelengths


def create_synthetic_envi_dataset(
    output_dir: Path,
    lines: int = 500,
    samples: int = 384,
    bands: int = 272,
    n_seeds: int = 50,
    include_references: bool = True,
    seed: Optional[int] = None
) -> Path:
    """Create a complete synthetic ENVI dataset directory.

    Args:
        output_dir: Directory to create dataset in.
        lines: Number of lines.
        samples: Number of samples.
        bands: Number of bands.
        n_seeds: Number of seeds.
        include_references: Include white/dark references.
        seed: Random seed.

    Returns:
        Path to created dataset directory.
    """
    if seed is not None:
        np.random.seed(seed)

    # Create directory structure
    dataset_dir = output_dir / "synthetic_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = dataset_dir / "capture"
    capture_dir.mkdir(exist_ok=True)

    # Generate main data
    data_cube, mask, wavelengths = generate_synthetic_hypercube(
        lines, samples, bands, n_seeds, seed=seed
    )

    # Save main data
    main_file = capture_dir / "synthetic_data.raw"
    data_cube.astype(np.float32).tofile(main_file)

    # Create header file
    header_file = capture_dir / "synthetic_data.hdr"
    with open(header_file, 'w') as f:
        f.write("ENVI\n")
        f.write(f"samples = {samples}\n")
        f.write(f"lines = {lines}\n")
        f.write(f"bands = {bands}\n")
        f.write("header offset = 0\n")
        f.write("file type = ENVI Standard\n")
        f.write("data type = 4\n")  # float32
        f.write("interleave = bsq\n")
        f.write("byte order = 0\n")
        f.write(f"wavelength = {{{', '.join(map(str, wavelengths))}}}\n")

    if include_references:
        # Generate white reference (high values)
        white_ref = np.ones((10, samples, bands), dtype=np.float32) * 0.9
        white_ref += np.random.normal(0, 0.01, white_ref.shape)
        white_file = capture_dir / "WHITEREF_synthetic_data.raw"
        white_ref.astype(np.float32).tofile(white_file)

        # White reference header
        white_header = capture_dir / "WHITEREF_synthetic_data.hdr"
        with open(white_header, 'w') as f:
            f.write("ENVI\n")
            f.write(f"samples = {samples}\n")
            f.write(f"lines = 10\n")
            f.write(f"bands = {bands}\n")
            f.write("header offset = 0\n")
            f.write("file type = ENVI Standard\n")
            f.write("data type = 4\n")
            f.write("interleave = bsq\n")
            f.write("byte order = 0\n")

        # Generate dark reference (low values)
        dark_ref = np.ones((10, samples, bands), dtype=np.float32) * 0.05
        dark_ref += np.random.normal(0, 0.005, dark_ref.shape)
        dark_file = capture_dir / "DARKREF_synthetic_data.raw"
        dark_ref.astype(np.float32).tofile(dark_file)

        # Dark reference header
        dark_header = capture_dir / "DARKREF_synthetic_data.hdr"
        with open(dark_header, 'w') as f:
            f.write("ENVI\n")
            f.write(f"samples = {samples}\n")
            f.write(f"lines = 10\n")
            f.write(f"bands = {bands}\n")
            f.write("header offset = 0\n")
            f.write("file type = ENVI Standard\n")
            f.write("data type = 4\n")
            f.write("interleave = bsq\n")
            f.write("byte order = 0\n")

    return dataset_dir


class SyntheticDataFixture:
    """Fixture class for managing synthetic test data."""

    def __init__(self):
        """Initialize fixture."""
        self.temp_dir = None
        self.dataset_path = None

    def setup(self, seed: int = 42) -> Path:
        """Set up synthetic test data.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Path to synthetic dataset directory.
        """
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="hyperseed_test_"))

        # Create synthetic dataset
        self.dataset_path = create_synthetic_envi_dataset(
            self.temp_dir,
            lines=200,  # Smaller for faster tests
            samples=200,
            bands=100,  # Fewer bands for faster tests
            n_seeds=20,
            include_references=True,
            seed=seed
        )

        return self.dataset_path

    def teardown(self):
        """Clean up temporary test data."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            self.dataset_path = None

    def __enter__(self):
        """Context manager entry."""
        return self.setup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.teardown()