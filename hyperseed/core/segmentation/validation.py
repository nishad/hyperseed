"""Validation functions for seed segmentation.

This module provides functions to validate segmented seeds based on
various criteria like size, overlap, and shape properties.
"""

import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy import ndimage
from skimage import measure


logger = logging.getLogger(__name__)


def check_overlap(mask: np.ndarray, threshold: float = 0.1) -> List[Tuple[int, int]]:
    """Check for overlapping seeds in segmentation mask.

    Args:
        mask: Labeled segmentation mask.
        threshold: Maximum allowed overlap ratio.

    Returns:
        List of overlapping seed pairs (label1, label2).
    """
    overlapping_pairs = []

    # Get unique labels (excluding background)
    labels = np.unique(mask[mask > 0])

    if len(labels) < 2:
        return overlapping_pairs

    # Check each pair of seeds
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label1, label2 = labels[i], labels[j]

            # Get masks for each seed
            mask1 = (mask == label1).astype(np.uint8)
            mask2 = (mask == label2).astype(np.uint8)

            # Dilate masks to check for proximity
            dilated1 = ndimage.binary_dilation(mask1, iterations=2)
            dilated2 = ndimage.binary_dilation(mask2, iterations=2)

            # Check for overlap
            overlap = np.logical_and(dilated1, dilated2)

            if np.any(overlap):
                # Calculate overlap ratio
                overlap_ratio = np.sum(overlap) / min(np.sum(mask1), np.sum(mask2))

                if overlap_ratio > threshold:
                    overlapping_pairs.append((int(label1), int(label2)))
                    logger.debug(f"Seeds {label1} and {label2} overlap by {overlap_ratio:.2%}")

    return overlapping_pairs


def validate_seed_size(
    mask: np.ndarray,
    min_size: int = 200,
    max_size: Optional[int] = None
) -> Tuple[np.ndarray, List[int]]:
    """Validate seeds based on size constraints.

    Args:
        mask: Labeled segmentation mask.
        min_size: Minimum number of pixels for valid seed.
        max_size: Maximum number of pixels for valid seed.

    Returns:
        Tuple of (validated mask, list of invalid labels).
    """
    validated_mask = np.zeros_like(mask)
    invalid_labels = []

    # Get region properties
    props = measure.regionprops(mask)

    new_label = 1
    for prop in props:
        area = prop.area

        # Check size constraints
        if area < min_size:
            invalid_labels.append(prop.label)
            logger.debug(f"Seed {prop.label} too small: {area} pixels")
        elif max_size is not None and area > max_size:
            invalid_labels.append(prop.label)
            logger.debug(f"Seed {prop.label} too large: {area} pixels")
        else:
            # Valid seed - add to validated mask with new label
            validated_mask[mask == prop.label] = new_label
            new_label += 1

    logger.info(f"Size validation: {new_label - 1} valid seeds, {len(invalid_labels)} rejected")

    return validated_mask, invalid_labels


def validate_seed_shape(
    mask: np.ndarray,
    max_eccentricity: float = 0.95,
    min_solidity: float = 0.7,
    min_extent: float = 0.3
) -> Tuple[np.ndarray, List[int]]:
    """Validate seeds based on shape properties.

    Args:
        mask: Labeled segmentation mask.
        max_eccentricity: Maximum allowed eccentricity (elongation).
        min_solidity: Minimum allowed solidity (convexity).
        min_extent: Minimum allowed extent (fill ratio).

    Returns:
        Tuple of (validated mask, list of invalid labels).
    """
    validated_mask = np.zeros_like(mask)
    invalid_labels = []

    # Get region properties
    props = measure.regionprops(mask)

    new_label = 1
    for prop in props:
        # Check shape properties
        valid = True
        reasons = []

        if prop.eccentricity > max_eccentricity:
            valid = False
            reasons.append(f"eccentricity={prop.eccentricity:.2f}")

        if prop.solidity < min_solidity:
            valid = False
            reasons.append(f"solidity={prop.solidity:.2f}")

        if prop.extent < min_extent:
            valid = False
            reasons.append(f"extent={prop.extent:.2f}")

        if not valid:
            invalid_labels.append(prop.label)
            logger.debug(f"Seed {prop.label} invalid shape: {', '.join(reasons)}")
        else:
            # Valid seed
            validated_mask[mask == prop.label] = new_label
            new_label += 1

    logger.info(f"Shape validation: {new_label - 1} valid seeds, {len(invalid_labels)} rejected")

    return validated_mask, invalid_labels


def validate_seeds(
    mask: np.ndarray,
    min_size: int = 200,
    max_size: Optional[int] = None,
    reject_overlapping: bool = True,
    check_shape: bool = True,
    max_eccentricity: float = 0.95,
    min_solidity: float = 0.7
) -> Tuple[np.ndarray, Dict[str, any]]:
    """Comprehensive seed validation.

    Args:
        mask: Labeled segmentation mask.
        min_size: Minimum number of pixels for valid seed.
        max_size: Maximum number of pixels for valid seed.
        reject_overlapping: Whether to reject overlapping seeds.
        check_shape: Whether to validate shape properties.
        max_eccentricity: Maximum allowed eccentricity.
        min_solidity: Minimum allowed solidity.

    Returns:
        Tuple of (validated mask, validation statistics).
    """
    initial_count = len(np.unique(mask[mask > 0]))
    stats = {
        "initial_count": initial_count,
        "rejected_size": 0,
        "rejected_overlap": 0,
        "rejected_shape": 0,
        "final_count": 0,
        "overlapping_pairs": [],
        "size_invalid": [],
        "shape_invalid": []
    }

    # Validate size
    validated_mask, size_invalid = validate_seed_size(mask, min_size, max_size)
    stats["rejected_size"] = len(size_invalid)
    stats["size_invalid"] = size_invalid

    # Check overlaps
    if reject_overlapping:
        overlapping_pairs = check_overlap(validated_mask)
        stats["overlapping_pairs"] = overlapping_pairs

        # Remove overlapping seeds (keep larger one)
        for label1, label2 in overlapping_pairs:
            size1 = np.sum(validated_mask == label1)
            size2 = np.sum(validated_mask == label2)

            # Remove smaller seed
            if size1 < size2:
                validated_mask[validated_mask == label1] = 0
                stats["rejected_overlap"] += 1
            else:
                validated_mask[validated_mask == label2] = 0
                stats["rejected_overlap"] += 1

    # Validate shape
    if check_shape:
        validated_mask, shape_invalid = validate_seed_shape(
            validated_mask, max_eccentricity, min_solidity
        )
        stats["rejected_shape"] = len(shape_invalid)
        stats["shape_invalid"] = shape_invalid

    # Re-label to ensure continuous labels
    validated_mask = measure.label(validated_mask > 0, connectivity=2)

    stats["final_count"] = len(np.unique(validated_mask[validated_mask > 0]))

    logger.info(
        f"Validation complete: {stats['initial_count']} → {stats['final_count']} seeds "
        f"(rejected: {stats['rejected_size']} size, {stats['rejected_overlap']} overlap, "
        f"{stats['rejected_shape']} shape)"
    )

    return validated_mask, stats


def get_seed_properties(mask: np.ndarray) -> List[Dict[str, any]]:
    """Extract properties of each segmented seed.

    Args:
        mask: Labeled segmentation mask.

    Returns:
        List of dictionaries containing seed properties.
    """
    props = measure.regionprops(mask)
    seed_properties = []

    for prop in props:
        properties = {
            "label": prop.label,
            "area": prop.area,
            "centroid": prop.centroid,
            "bbox": prop.bbox,
            "eccentricity": prop.eccentricity,
            "solidity": prop.solidity,
            "extent": prop.extent,
            "major_axis_length": prop.major_axis_length,
            "minor_axis_length": prop.minor_axis_length,
            "perimeter": prop.perimeter,
            "coords": prop.coords  # Pixel coordinates
        }
        seed_properties.append(properties)

    return seed_properties


def filter_border_seeds(
    mask: np.ndarray,
    border_width: int = 5
) -> Tuple[np.ndarray, List[int]]:
    """Filter out seeds touching image borders.

    Args:
        mask: Labeled segmentation mask.
        border_width: Width of border region to check.

    Returns:
        Tuple of (filtered mask, list of removed labels).
    """
    filtered_mask = mask.copy()
    removed_labels = []

    height, width = mask.shape

    # Get region properties
    props = measure.regionprops(mask)

    for prop in props:
        # Check if bbox touches borders
        min_row, min_col, max_row, max_col = prop.bbox

        if (min_row < border_width or
            min_col < border_width or
            max_row > height - border_width or
            max_col > width - border_width):

            filtered_mask[mask == prop.label] = 0
            removed_labels.append(prop.label)
            logger.debug(f"Removed border seed {prop.label}")

    # Re-label to ensure continuous labels
    if len(removed_labels) > 0:
        filtered_mask = measure.label(filtered_mask > 0, connectivity=2)
        logger.info(f"Removed {len(removed_labels)} border seeds")

    return filtered_mask, removed_labels