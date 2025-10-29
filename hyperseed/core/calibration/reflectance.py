"""Reflectance calibration for hyperspectral data.

This module provides functionality for calibrating raw hyperspectral data
using white and dark reference measurements to compute reflectance values.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from hyperseed.core.io.envi_reader import ENVIReader


logger = logging.getLogger(__name__)


class ReflectanceCalibrator:
    """Calibrator for converting raw hyperspectral data to reflectance.

    Applies white and dark reference calibration to compute reflectance values
    using the formula: R = (Raw - Dark) / (White - Dark)

    Optionally handles bad pixel replacement and reflectance value clipping.

    Example:
        >>> calibrator = ReflectanceCalibrator(
        ...     white_ref="white_ref.hdr",
        ...     dark_ref="dark_ref.hdr"
        ... )
        >>> reflectance = calibrator.calibrate(raw_data)
    """

    def __init__(
        self,
        white_ref: Optional[Union[str, Path, np.ndarray]] = None,
        dark_ref: Optional[Union[str, Path, np.ndarray]] = None,
        bad_pixel_map: Optional[Union[str, Path, np.ndarray]] = None,
        clip_negative: bool = True,
        clip_max: Optional[float] = 1.0,
    ):
        """Initialize reflectance calibrator.

        Args:
            white_ref: White reference data or path to white reference file.
            dark_ref: Dark reference data or path to dark reference file.
            bad_pixel_map: Bad pixel map data or path to bad pixel file.
            clip_negative: Whether to clip negative reflectance values to 0.
            clip_max: Maximum reflectance value to clip to (None for no clipping).
        """
        self.white_ref_data = None
        self.dark_ref_data = None
        self.bad_pixel_map_data = None
        self.clip_negative = clip_negative
        self.clip_max = clip_max

        # Load reference data
        if white_ref is not None:
            self.white_ref_data = self._load_reference(white_ref, "white")

        if dark_ref is not None:
            self.dark_ref_data = self._load_reference(dark_ref, "dark")

        if bad_pixel_map is not None:
            self.bad_pixel_map_data = self._load_bad_pixel_map(bad_pixel_map)

    def _load_reference(
        self, reference: Union[str, Path, np.ndarray], ref_type: str
    ) -> np.ndarray:
        """Load reference data from file or array.

        Args:
            reference: Reference data or path to reference file.
            ref_type: Type of reference ('white' or 'dark').

        Returns:
            Reference data as numpy array.

        Raises:
            ValueError: If reference data cannot be loaded.
        """
        if isinstance(reference, np.ndarray):
            logger.info(f"Using provided {ref_type} reference array")
            return reference

        try:
            ref_path = Path(reference)
            if not ref_path.exists():
                raise FileNotFoundError(f"{ref_type} reference file not found: {ref_path}")

            logger.info(f"Loading {ref_type} reference from: {ref_path}")
            reader = ENVIReader(ref_path)
            ref_data = reader.read_data()

            # If reference has multiple frames, average them
            if ref_data.ndim == 3 and ref_data.shape[0] > 1:
                logger.info(
                    f"Averaging {ref_data.shape[0]} frames for {ref_type} reference"
                )
                ref_data = np.mean(ref_data, axis=0, keepdims=True)

            return ref_data

        except Exception as e:
            raise ValueError(f"Failed to load {ref_type} reference: {e}")

    def _load_bad_pixel_map(
        self, bad_pixel_map: Union[str, Path, np.ndarray]
    ) -> np.ndarray:
        """Load bad pixel map from file or array.

        Args:
            bad_pixel_map: Bad pixel map data or path to file.

        Returns:
            Bad pixel map as numpy array.

        Raises:
            ValueError: If bad pixel map cannot be loaded.
        """
        if isinstance(bad_pixel_map, np.ndarray):
            logger.info("Using provided bad pixel map array")
            return bad_pixel_map

        try:
            map_path = Path(bad_pixel_map)
            if not map_path.exists():
                raise FileNotFoundError(f"Bad pixel map file not found: {map_path}")

            logger.info(f"Loading bad pixel map from: {map_path}")

            # Check if it's an ENVI file
            if map_path.suffix == '.bpr' or map_path.suffix == '.hdr':
                # Try to find the header file
                if map_path.suffix == '.bpr':
                    hdr_path = map_path.with_suffix('.hdr')
                else:
                    hdr_path = map_path

                reader = ENVIReader(hdr_path)
                bad_pixel_map_data = reader.read_data()
            else:
                # Try to load as numpy array
                bad_pixel_map_data = np.load(map_path)

            return bad_pixel_map_data

        except Exception as e:
            raise ValueError(f"Failed to load bad pixel map: {e}")

    def calibrate(
        self,
        raw_data: np.ndarray,
        white_ref: Optional[np.ndarray] = None,
        dark_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calibrate raw hyperspectral data to reflectance.

        Args:
            raw_data: Raw hyperspectral data array (lines, samples, bands).
            white_ref: Optional white reference to use instead of stored.
            dark_ref: Optional dark reference to use instead of stored.

        Returns:
            Calibrated reflectance data array.

        Raises:
            ValueError: If required references are not available or shapes mismatch.
        """
        # Use provided references or stored ones
        white = white_ref if white_ref is not None else self.white_ref_data
        dark = dark_ref if dark_ref is not None else self.dark_ref_data

        if white is None or dark is None:
            logger.warning(
                "White or dark reference not available, returning raw data"
            )
            return raw_data.astype(np.float32)

        # Ensure all data is float32 for calculation
        raw_float = raw_data.astype(np.float32)
        white_float = white.astype(np.float32)
        dark_float = dark.astype(np.float32)

        # Handle shape differences
        raw_shape = raw_float.shape
        white_shape = white_float.shape
        dark_shape = dark_float.shape

        # If references have fewer lines (averaged), broadcast them
        # But only for the lines dimension, not samples or bands
        if white_shape[0] == 1 and raw_shape[0] > 1:
            # Check if samples and bands match
            if white_shape[1:] == raw_shape[1:]:
                white_float = np.broadcast_to(white_float, raw_shape)
            else:
                # Extract matching region from references
                white_float = white_float[:, :raw_shape[1], :raw_shape[2]]

        if dark_shape[0] == 1 and raw_shape[0] > 1:
            # Check if samples and bands match
            if dark_shape[1:] == raw_shape[1:]:
                dark_float = np.broadcast_to(dark_float, raw_shape)
            else:
                # Extract matching region from references
                dark_float = dark_float[:, :raw_shape[1], :raw_shape[2]]

        # Ensure references match raw data dimensions
        if white_float.shape[0] == 1:
            white_float = np.broadcast_to(white_float,
                                         (raw_shape[0], white_float.shape[1], white_float.shape[2]))
        if dark_float.shape[0] == 1:
            dark_float = np.broadcast_to(dark_float,
                                        (raw_shape[0], dark_float.shape[1], dark_float.shape[2]))

        # Now extract matching region if needed
        white_float = white_float[:raw_shape[0], :raw_shape[1], :raw_shape[2]]
        dark_float = dark_float[:raw_shape[0], :raw_shape[1], :raw_shape[2]]

        # Calculate reflectance: R = (Raw - Dark) / (White - Dark)
        denominator = white_float - dark_float

        # Avoid division by zero
        # Set a minimum denominator value
        min_denominator = 1.0
        denominator = np.where(
            np.abs(denominator) < min_denominator,
            min_denominator,
            denominator
        )

        reflectance = (raw_float - dark_float) / denominator

        # Apply bad pixel replacement if available
        if self.bad_pixel_map_data is not None:
            reflectance = self._apply_bad_pixel_replacement(
                reflectance, self.bad_pixel_map_data
            )

        # Clip values if requested
        if self.clip_negative:
            reflectance = np.maximum(reflectance, 0)

        if self.clip_max is not None:
            reflectance = np.minimum(reflectance, self.clip_max)

        return reflectance

    def _apply_bad_pixel_replacement(
        self, data: np.ndarray, bad_pixel_map: np.ndarray
    ) -> np.ndarray:
        """Apply bad pixel replacement to data.

        Args:
            data: Data array to correct.
            bad_pixel_map: Bad pixel map (0=good, 1=bad).

        Returns:
            Corrected data array.
        """
        logger.info("Applying bad pixel replacement")

        # Ensure bad pixel map has correct shape
        if bad_pixel_map.shape != data.shape:
            # Case 1: Bad pixel map is 2D and data is 3D, broadcast
            if bad_pixel_map.ndim == 2 and data.ndim == 3:
                bad_pixel_map = np.broadcast_to(
                    bad_pixel_map[..., np.newaxis], data.shape
                )
            # Case 2: Bad pixel map is 3D with 1 line (Specim SWIR format)
            elif (bad_pixel_map.ndim == 3 and bad_pixel_map.shape[0] == 1 and
                  data.ndim == 3 and bad_pixel_map.shape[1:] == data.shape[1:]):
                logger.info(
                    f"Broadcasting bad pixel map from {bad_pixel_map.shape} to {data.shape}"
                )
                bad_pixel_map = np.broadcast_to(bad_pixel_map, data.shape)
            else:
                logger.warning(
                    f"Bad pixel map shape {bad_pixel_map.shape} doesn't match "
                    f"data shape {data.shape}, skipping replacement"
                )
                return data

        # Find bad pixels
        bad_pixels = bad_pixel_map > 0

        if not np.any(bad_pixels):
            logger.info("No bad pixels found")
            return data

        # Replace bad pixels with interpolated values
        # For simplicity, use nearest neighbor interpolation
        from scipy import ndimage

        for band in range(data.shape[2]):
            band_data = data[:, :, band]
            band_bad = bad_pixels[:, :, band] if bad_pixels.ndim == 3 else bad_pixels

            if np.any(band_bad):
                # Create mask of good pixels
                good_pixels = ~band_bad

                # Use nearest neighbor interpolation
                indices = ndimage.distance_transform_edt(
                    band_bad, return_indices=True, return_distances=False
                )
                band_data[band_bad] = band_data[tuple(indices[:, band_bad])]
                data[:, :, band] = band_data

        logger.info(f"Replaced {np.sum(bad_pixels)} bad pixels")
        return data

    def calibrate_from_directory(
        self, directory: Union[str, Path], data_pattern: str = "*.hdr"
    ) -> Tuple[np.ndarray, ENVIReader]:
        """Calibrate hyperspectral data from a directory structure.

        Automatically finds main data, white reference, and dark reference files.

        Args:
            directory: Directory containing hyperspectral data files.
            data_pattern: Pattern to match main data files.

        Returns:
            Tuple of (calibrated data, ENVI reader for main data).

        Raises:
            ValueError: If required files are not found.
        """
        directory = Path(directory)

        # Find capture directory
        capture_dir = directory / "capture"
        if not capture_dir.exists():
            capture_dir = directory

        # Find main data file
        main_files = list(capture_dir.glob(data_pattern))
        main_files = [
            f for f in main_files
            if not any(ref in f.name.upper() for ref in ["DARKREF", "WHITEREF"])
        ]

        if not main_files:
            raise ValueError(f"No main data files found in {capture_dir}")

        main_file = main_files[0]
        logger.info(f"Found main data file: {main_file}")

        # Find reference files
        white_files = list(
            capture_dir.glob("WHITEREF*.hdr")
        ) + list(capture_dir.glob("*WHITE*.hdr"))
        dark_files = list(
            capture_dir.glob("DARKREF*.hdr")
        ) + list(capture_dir.glob("*DARK*.hdr"))

        # Find bad pixel map
        bpr_dir = directory / "calibrations" / "bpr"
        bad_pixel_files = []
        if bpr_dir.exists():
            bad_pixel_files = list(bpr_dir.glob("*.hdr")) + list(bpr_dir.glob("*.bpr"))

        # Load main data
        main_reader = ENVIReader(main_file)
        raw_data = main_reader.read_data()

        # Load references if found
        white_ref = None
        dark_ref = None
        bad_pixel_map = None

        if white_files:
            logger.info(f"Found white reference: {white_files[0]}")
            white_reader = ENVIReader(white_files[0])
            white_ref = white_reader.read_data()
            # Average frames if multiple
            if white_ref.shape[0] > 1:
                white_ref = np.mean(white_ref, axis=0, keepdims=True)

        if dark_files:
            logger.info(f"Found dark reference: {dark_files[0]}")
            dark_reader = ENVIReader(dark_files[0])
            dark_ref = dark_reader.read_data()
            # Average frames if multiple
            if dark_ref.shape[0] > 1:
                dark_ref = np.mean(dark_ref, axis=0, keepdims=True)

        if bad_pixel_files:
            logger.info(f"Found bad pixel map: {bad_pixel_files[0]}")
            if bad_pixel_files[0].suffix == '.hdr':
                bad_pixel_map = self._load_bad_pixel_map(bad_pixel_files[0])
            else:
                # For .bpr files, check for corresponding .hdr
                hdr_file = bad_pixel_files[0].with_suffix('.hdr')
                if hdr_file.exists():
                    bad_pixel_map = self._load_bad_pixel_map(hdr_file)

        # Update calibrator with found references
        if white_ref is not None:
            self.white_ref_data = white_ref
        if dark_ref is not None:
            self.dark_ref_data = dark_ref
        if bad_pixel_map is not None:
            self.bad_pixel_map_data = bad_pixel_map

        # Calibrate data
        calibrated = self.calibrate(raw_data)

        return calibrated, main_reader

    def get_statistics(self, reflectance: np.ndarray) -> dict:
        """Calculate statistics for calibrated reflectance data.

        Args:
            reflectance: Calibrated reflectance data.

        Returns:
            Dictionary containing statistics.
        """
        stats = {
            "min": float(np.min(reflectance)),
            "max": float(np.max(reflectance)),
            "mean": float(np.mean(reflectance)),
            "std": float(np.std(reflectance)),
            "negative_pixels": int(np.sum(reflectance < 0)),
            "saturated_pixels": int(np.sum(reflectance > 1.0)),
            "shape": reflectance.shape,
        }

        return stats