"""ENVI format reader for hyperspectral data.

This module provides functionality to read ENVI format hyperspectral data files
commonly produced by Specim SWIR cameras and other hyperspectral imaging systems.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from dataclasses import dataclass


@dataclass
class ENVIHeader:
    """Container for ENVI header information.

    Attributes:
        samples: Number of samples (columns) in the image.
        lines: Number of lines (rows) in the image.
        bands: Number of spectral bands.
        header_offset: Offset in bytes before data begins.
        data_type: ENVI data type code.
        interleave: Data interleave format (bil, bip, bsq).
        byte_order: Byte order (0=little endian, 1=big endian).
        wavelength: Array of wavelength values for each band.
        fwhm: Full width at half maximum for each band.
        metadata: Dictionary of additional metadata fields.
    """

    samples: int
    lines: int
    bands: int
    header_offset: int
    data_type: int
    interleave: str
    byte_order: int
    wavelength: Optional[np.ndarray] = None
    fwhm: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata dictionary if not provided."""
        if self.metadata is None:
            self.metadata = {}


class ENVIReader:
    """Reader for ENVI format hyperspectral data files.

    Supports reading of ENVI standard format with .hdr header files and
    .raw binary data files. Handles different data types and interleave formats.

    Example:
        >>> reader = ENVIReader("path/to/data.hdr")
        >>> data = reader.read_data()
        >>> header = reader.header
        >>> wavelengths = header.wavelength
    """

    # ENVI data type mapping to numpy dtypes
    DTYPE_MAP = {
        1: np.uint8,    # 8-bit unsigned integer
        2: np.int16,    # 16-bit signed integer
        3: np.int32,    # 32-bit signed integer
        4: np.float32,  # 32-bit floating point
        5: np.float64,  # 64-bit floating point
        6: np.complex64,  # 2x32-bit complex
        9: np.complex128,  # 2x64-bit complex
        12: np.uint16,   # 16-bit unsigned integer
        13: np.uint32,   # 32-bit unsigned integer
        14: np.int64,    # 64-bit signed integer
        15: np.uint64    # 64-bit unsigned integer
    }

    def __init__(self, header_path: Union[str, Path]):
        """Initialize ENVI reader with header file path.

        Args:
            header_path: Path to the ENVI header file (.hdr).

        Raises:
            FileNotFoundError: If header file doesn't exist.
            ValueError: If header file is invalid or data file not found.
        """
        self.header_path = Path(header_path)

        if not self.header_path.exists():
            raise FileNotFoundError(f"Header file not found: {self.header_path}")

        # Parse header file
        self.header = self._parse_header()

        # Find and validate data file
        self.data_path = self._find_data_file()

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def _find_data_file(self) -> Path:
        """Find the corresponding data file for the header.

        Returns:
            Path to the data file.

        Raises:
            FileNotFoundError: If no matching data file is found.
        """
        # Common data file extensions
        possible_extensions = ['.raw', '.dat', '.img', '.bil', '.bip', '.bsq', '.bpr']

        # Try replacing .hdr extension
        base = self.header_path.with_suffix('')

        for ext in possible_extensions:
            data_path = base.with_suffix(ext)
            if data_path.exists():
                return data_path

        # Try without changing extension (sometimes .hdr paired with no extension)
        if base.exists() and base != self.header_path:
            return base

        raise FileNotFoundError(
            f"Could not find data file for header: {self.header_path}"
        )

    def _parse_header(self) -> ENVIHeader:
        """Parse ENVI header file.

        Returns:
            ENVIHeader object with parsed information.

        Raises:
            ValueError: If header format is invalid.
        """
        header_dict = {}

        with open(self.header_path, 'r') as f:
            content = f.read()

        # Check for ENVI signature
        if not content.strip().startswith('ENVI'):
            raise ValueError(f"Invalid ENVI header file: {self.header_path}")

        # Parse key-value pairs
        # Handle multi-line values in braces
        pattern = r'(\w+(?:\s+\w+)*)\s*=\s*({[^}]*}|[^\n]*)'
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

        for key, value in matches:
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()

            # Handle bracketed lists
            if value.startswith('{') and value.endswith('}'):
                value = value[1:-1].strip()
                # Try to parse as numeric list
                try:
                    value_list = [float(x.strip()) for x in value.split(',') if x.strip()]
                    header_dict[key] = value_list
                except ValueError:
                    # Keep as string list if not numeric
                    value_list = [x.strip() for x in value.split(',') if x.strip()]
                    header_dict[key] = value_list
            else:
                # Try to parse as number
                try:
                    if '.' in value:
                        header_dict[key] = float(value)
                    else:
                        header_dict[key] = int(value)
                except ValueError:
                    header_dict[key] = value

        # Extract required fields
        try:
            header = ENVIHeader(
                samples=int(header_dict.get('samples', 0)),
                lines=int(header_dict.get('lines', 0)),
                bands=int(header_dict.get('bands', 0)),
                header_offset=int(header_dict.get('header_offset', 0)),
                data_type=int(header_dict.get('data_type', 4)),
                interleave=header_dict.get('interleave', 'bsq').lower(),
                byte_order=int(header_dict.get('byte_order', 0)),
                metadata=header_dict
            )

            # Add wavelength information if available
            if 'wavelength' in header_dict:
                header.wavelength = np.array(header_dict['wavelength'])

            # Add FWHM information if available
            if 'fwhm' in header_dict:
                header.fwhm = np.array(header_dict['fwhm'])

            return header

        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid or incomplete ENVI header: {e}")

    def read_data(self,
                  bands: Optional[Union[int, List[int], slice]] = None,
                  lines: Optional[Union[int, List[int], slice]] = None,
                  samples: Optional[Union[int, List[int], slice]] = None,
                  mmap: bool = True) -> np.ndarray:
        """Read hyperspectral data from the ENVI file.

        Args:
            bands: Band indices to read (None for all bands).
            lines: Line indices to read (None for all lines).
            samples: Sample indices to read (None for all samples).
            mmap: Use memory mapping for large files.

        Returns:
            Numpy array with shape (lines, samples, bands) in BSQ format.

        Raises:
            ValueError: If indices are out of bounds or data type unsupported.
        """
        # Get data type
        if self.header.data_type not in self.DTYPE_MAP:
            raise ValueError(f"Unsupported ENVI data type: {self.header.data_type}")

        dtype = self.DTYPE_MAP[self.header.data_type]

        # Handle byte order
        if self.header.byte_order == 1:  # Big endian
            dtype = np.dtype(dtype).newbyteorder('>')
        else:  # Little endian
            dtype = np.dtype(dtype).newbyteorder('<')

        # Calculate shape based on interleave
        if self.header.interleave == 'bsq':
            shape = (self.header.bands, self.header.lines, self.header.samples)
        elif self.header.interleave == 'bil':
            shape = (self.header.lines, self.header.bands, self.header.samples)
        elif self.header.interleave == 'bip':
            shape = (self.header.lines, self.header.samples, self.header.bands)
        else:
            raise ValueError(f"Unsupported interleave format: {self.header.interleave}")

        # Read data
        if mmap:
            # Use memory mapping for efficient access
            data = np.memmap(
                self.data_path,
                dtype=dtype,
                mode='r',
                offset=self.header.header_offset,
                shape=shape
            )
        else:
            # Read entire file into memory
            with open(self.data_path, 'rb') as f:
                f.seek(self.header.header_offset)
                data = np.fromfile(f, dtype=dtype)
                data = data.reshape(shape)

        # Reorganize to standard format (lines, samples, bands)
        if self.header.interleave == 'bsq':
            data = np.transpose(data, (1, 2, 0))
        elif self.header.interleave == 'bip':
            pass  # Already in correct format
        elif self.header.interleave == 'bil':
            data = np.transpose(data, (0, 2, 1))

        # Apply subsetting if requested
        if bands is not None or lines is not None or samples is not None:
            lines_idx = lines if lines is not None else slice(None)
            samples_idx = samples if samples is not None else slice(None)

            # Handle different types of band indexing
            if bands is not None:
                if isinstance(bands, int):
                    # Single band - add as slice to maintain 3D shape
                    bands_idx = slice(bands, bands + 1)
                elif isinstance(bands, (list, tuple)):
                    bands_idx = list(bands)
                else:
                    bands_idx = bands
            else:
                bands_idx = slice(None)

            data = data[lines_idx, samples_idx, bands_idx]

        return np.array(data)  # Convert from memmap if needed

    def read_band(self, band_index: int) -> np.ndarray:
        """Read a single band from the hyperspectral data.

        Args:
            band_index: Index of the band to read (0-based).

        Returns:
            2D numpy array of shape (lines, samples).

        Raises:
            ValueError: If band index is out of range.
        """
        if band_index < 0 or band_index >= self.header.bands:
            raise ValueError(
                f"Band index {band_index} out of range [0, {self.header.bands})"
            )

        return self.read_data(bands=band_index)[:, :, 0]

    def get_wavelengths(self) -> Optional[np.ndarray]:
        """Get wavelength values for all bands.

        Returns:
            Array of wavelength values or None if not available.
        """
        return self.header.wavelength

    def get_metadata(self) -> Dict[str, Any]:
        """Get all metadata from the header file.

        Returns:
            Dictionary containing all header metadata.
        """
        return self.header.metadata

    def get_shape(self) -> Tuple[int, int, int]:
        """Get the shape of the hyperspectral data.

        Returns:
            Tuple of (lines, samples, bands).
        """
        return (self.header.lines, self.header.samples, self.header.bands)

    def __repr__(self) -> str:
        """String representation of the ENVI reader."""
        return (
            f"ENVIReader(file='{self.header_path.name}', "
            f"shape=({self.header.lines}, {self.header.samples}, {self.header.bands}), "
            f"dtype={self.DTYPE_MAP.get(self.header.data_type, 'unknown')}, "
            f"interleave='{self.header.interleave}')"
        )