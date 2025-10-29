"""Segmentation algorithms for seed detection.

This module provides various segmentation algorithms for detecting
and isolating individual seeds in hyperspectral images.
"""

import logging
from typing import Tuple, Optional, Literal

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage import morphology, measure, segmentation
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max


logger = logging.getLogger(__name__)


def create_seed_mask(
    data: np.ndarray,
    band_index: Optional[int] = None,
    percentile: float = 90
) -> np.ndarray:
    """Create a binary mask highlighting potential seed regions.

    Args:
        data: Hyperspectral data array (lines, samples, bands).
        band_index: Specific band to use (None for average).
        percentile: Percentile for high-reflectance band selection.

    Returns:
        2D binary mask array.
    """
    if data.ndim == 3:
        if band_index is not None:
            # Use specific band
            image = data[:, :, band_index]
        else:
            # Use band with high variance (likely contains seeds)
            band_vars = np.var(data, axis=(0, 1))
            high_var_bands = np.where(band_vars > np.percentile(band_vars, percentile))[0]

            if len(high_var_bands) > 0:
                # Average high-variance bands
                image = np.mean(data[:, :, high_var_bands], axis=2)
            else:
                # Fallback to mean of all bands
                image = np.mean(data, axis=2)
    else:
        image = data

    return image


def threshold_segmentation(
    data: np.ndarray,
    method: Literal["otsu", "adaptive", "manual"] = "otsu",
    threshold_value: Optional[float] = None,
    min_seed_size: int = 200,
    max_seed_size: Optional[int] = None,
    band_index: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """Perform threshold-based segmentation.

    Args:
        data: Hyperspectral data array.
        method: Thresholding method to use.
        threshold_value: Manual threshold value (for 'manual' method).
        min_seed_size: Minimum number of pixels for valid seed.
        max_seed_size: Maximum number of pixels for valid seed.
        band_index: Specific band to use for segmentation.

    Returns:
        Tuple of (labeled mask array, number of seeds found).

    Example:
        >>> mask, n_seeds = threshold_segmentation(data, method="otsu")
    """
    # Create initial mask
    image = create_seed_mask(data, band_index)

    # Apply thresholding
    if method == "otsu":
        threshold = threshold_otsu(image)
        binary = image > threshold
        logger.debug(f"Otsu threshold: {threshold:.4f}")

    elif method == "adaptive":
        # Use local adaptive thresholding
        from skimage.filters import threshold_local
        block_size = 35  # Should be odd
        threshold = threshold_local(image, block_size, method='gaussian')
        binary = image > threshold

    elif method == "manual":
        if threshold_value is None:
            raise ValueError("Manual method requires threshold_value")
        binary = image > threshold_value

    else:
        raise ValueError(f"Unknown thresholding method: {method}")

    # Apply morphological operations to clean up
    binary = morphology.remove_small_objects(binary, min_size=min_seed_size)
    binary = morphology.remove_small_holes(binary, area_threshold=min_seed_size // 4)

    # Erosion and dilation to separate touching seeds
    binary = morphology.binary_erosion(binary, morphology.disk(2))
    binary = morphology.binary_dilation(binary, morphology.disk(2))

    # Label connected components
    labeled = measure.label(binary, connectivity=2)

    # Filter by size
    props = measure.regionprops(labeled)
    valid_labels = []

    for prop in props:
        area = prop.area

        # Check size constraints
        if area >= min_seed_size:
            if max_seed_size is None or area <= max_seed_size:
                valid_labels.append(prop.label)

    # Create final mask with only valid seeds
    final_mask = np.zeros_like(labeled)
    for i, label in enumerate(valid_labels, 1):
        final_mask[labeled == label] = i

    n_seeds = len(valid_labels)
    logger.info(f"Threshold segmentation found {n_seeds} seeds")

    return final_mask, n_seeds


def watershed_segmentation(
    data: np.ndarray,
    min_seed_size: int = 200,
    max_seed_size: Optional[int] = None,
    band_index: Optional[int] = None,
    min_distance: int = 20
) -> Tuple[np.ndarray, int]:
    """Perform watershed segmentation.

    Watershed segmentation is effective for separating touching seeds.

    Args:
        data: Hyperspectral data array.
        min_seed_size: Minimum number of pixels for valid seed.
        max_seed_size: Maximum number of pixels for valid seed.
        band_index: Specific band to use for segmentation.
        min_distance: Minimum distance between seed centers.

    Returns:
        Tuple of (labeled mask array, number of seeds found).

    Example:
        >>> mask, n_seeds = watershed_segmentation(data)
    """
    # Create initial mask
    image = create_seed_mask(data, band_index)

    # Initial thresholding
    threshold = threshold_otsu(image)
    binary = image > threshold

    # Clean up binary image
    binary = morphology.remove_small_objects(binary, min_size=min_seed_size // 2)
    binary = morphology.binary_closing(binary, morphology.disk(3))

    # Compute distance transform
    distance = distance_transform_edt(binary)

    # Find local maxima (seed centers)
    # Get coordinates of local maxima
    coordinates = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary
    )

    # Create boolean mask from coordinates
    local_maxima = np.zeros_like(binary, dtype=bool)
    if len(coordinates) > 0:
        local_maxima[coordinates[:, 0], coordinates[:, 1]] = True

    # Create markers for watershed
    markers = measure.label(local_maxima)

    # Apply watershed
    labels = segmentation.watershed(-distance, markers, mask=binary)

    # Filter by size
    props = measure.regionprops(labels)
    valid_labels = []

    for prop in props:
        area = prop.area

        # Check size constraints
        if area >= min_seed_size:
            if max_seed_size is None or area <= max_seed_size:
                valid_labels.append(prop.label)

    # Create final mask with only valid seeds
    final_mask = np.zeros_like(labels)
    for i, label in enumerate(valid_labels, 1):
        final_mask[labels == label] = i

    n_seeds = len(valid_labels)
    logger.info(f"Watershed segmentation found {n_seeds} seeds")

    return final_mask, n_seeds


def connected_components_segmentation(
    data: np.ndarray,
    min_seed_size: int = 200,
    max_seed_size: Optional[int] = None,
    band_index: Optional[int] = None,
    connectivity: int = 2
) -> Tuple[np.ndarray, int]:
    """Perform connected components segmentation.

    Simple segmentation based on connected component analysis.

    Args:
        data: Hyperspectral data array.
        min_seed_size: Minimum number of pixels for valid seed.
        max_seed_size: Maximum number of pixels for valid seed.
        band_index: Specific band to use for segmentation.
        connectivity: Connectivity for labeling (1 or 2).

    Returns:
        Tuple of (labeled mask array, number of seeds found).

    Example:
        >>> mask, n_seeds = connected_components_segmentation(data)
    """
    # Create initial mask
    image = create_seed_mask(data, band_index)

    # Apply thresholding
    threshold = threshold_otsu(image)
    binary = image > threshold

    # Morphological operations
    binary = morphology.remove_small_objects(binary, min_size=min_seed_size // 2)
    binary = morphology.binary_closing(binary, morphology.disk(2))
    binary = morphology.binary_opening(binary, morphology.disk(2))

    # Label connected components
    labeled = measure.label(binary, connectivity=connectivity)

    # Filter by size and properties
    props = measure.regionprops(labeled)
    valid_labels = []

    for prop in props:
        area = prop.area

        # Check size constraints
        if area >= min_seed_size:
            if max_seed_size is None or area <= max_seed_size:
                # Additional shape checks
                eccentricity = prop.eccentricity
                solidity = prop.solidity

                # Seeds should not be too elongated or irregular
                if eccentricity < 0.95 and solidity > 0.7:
                    valid_labels.append(prop.label)

    # Create final mask with only valid seeds
    final_mask = np.zeros_like(labeled)
    for i, label in enumerate(valid_labels, 1):
        final_mask[labeled == label] = i

    n_seeds = len(valid_labels)
    logger.info(f"Connected components segmentation found {n_seeds} seeds")

    return final_mask, n_seeds


def combined_segmentation(
    data: np.ndarray,
    min_seed_size: int = 200,
    max_seed_size: Optional[int] = None,
    band_index: Optional[int] = None,
    methods: list = ["threshold", "watershed"]
) -> Tuple[np.ndarray, int]:
    """Perform combined segmentation using multiple methods.

    Combines results from multiple segmentation methods for robustness.

    Args:
        data: Hyperspectral data array.
        min_seed_size: Minimum number of pixels for valid seed.
        max_seed_size: Maximum number of pixels for valid seed.
        band_index: Specific band to use for segmentation.
        methods: List of methods to combine.

    Returns:
        Tuple of (labeled mask array, number of seeds found).

    Example:
        >>> mask, n_seeds = combined_segmentation(
        ...     data, methods=["threshold", "watershed"]
        ... )
    """
    results = []

    # Apply each method
    if "threshold" in methods:
        mask, _ = threshold_segmentation(
            data, min_seed_size=min_seed_size,
            max_seed_size=max_seed_size, band_index=band_index
        )
        results.append(mask)

    if "watershed" in methods:
        mask, _ = watershed_segmentation(
            data, min_seed_size=min_seed_size,
            max_seed_size=max_seed_size, band_index=band_index
        )
        results.append(mask)

    if "connected" in methods:
        mask, _ = connected_components_segmentation(
            data, min_seed_size=min_seed_size,
            max_seed_size=max_seed_size, band_index=band_index
        )
        results.append(mask)

    if not results:
        raise ValueError("No valid segmentation methods specified")

    # Combine results using consensus
    # Convert all masks to binary
    binary_masks = [(mask > 0).astype(np.uint8) for mask in results]

    # Sum binary masks
    consensus = np.sum(binary_masks, axis=0)

    # Threshold for consensus (majority vote)
    threshold = len(results) // 2 + 1
    final_binary = consensus >= threshold

    # Apply size filtering again
    final_binary = morphology.remove_small_objects(final_binary, min_size=min_seed_size)

    # Label final result
    labeled = measure.label(final_binary, connectivity=2)

    # Filter by size one more time
    props = measure.regionprops(labeled)
    valid_labels = []

    for prop in props:
        area = prop.area
        if area >= min_seed_size:
            if max_seed_size is None or area <= max_seed_size:
                valid_labels.append(prop.label)

    # Create final mask
    final_mask = np.zeros_like(labeled)
    for i, label in enumerate(valid_labels, 1):
        final_mask[labeled == label] = i

    n_seeds = len(valid_labels)
    logger.info(f"Combined segmentation found {n_seeds} seeds")

    return final_mask, n_seeds


def apply_morphological_operations(
    mask: np.ndarray,
    operations: list = ["closing", "opening"],
    kernel_size: int = 3
) -> np.ndarray:
    """Apply morphological operations to clean up segmentation.

    Args:
        mask: Binary or labeled mask array.
        operations: List of operations to apply.
        kernel_size: Size of the morphological kernel.

    Returns:
        Processed mask array.
    """
    # Convert to binary if labeled
    binary = mask > 0

    # Create kernel
    kernel = morphology.disk(kernel_size)

    # Apply operations
    for op in operations:
        if op == "closing":
            binary = morphology.binary_closing(binary, kernel)
        elif op == "opening":
            binary = morphology.binary_opening(binary, kernel)
        elif op == "erosion":
            binary = morphology.binary_erosion(binary, kernel)
        elif op == "dilation":
            binary = morphology.binary_dilation(binary, kernel)
        else:
            logger.warning(f"Unknown morphological operation: {op}")

    # Re-label if original was labeled
    if mask.max() > 1:
        return measure.label(binary, connectivity=2)
    else:
        return binary.astype(mask.dtype)