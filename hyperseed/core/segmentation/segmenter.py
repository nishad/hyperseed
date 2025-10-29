"""Main segmenter class for seed detection in hyperspectral images.

This module provides the SeedSegmenter class which combines various
segmentation algorithms with validation and post-processing.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color

from hyperseed.config.settings import SegmentationConfig
from hyperseed.core.segmentation.algorithms import (
    threshold_segmentation,
    watershed_segmentation,
    connected_components_segmentation,
    combined_segmentation,
    apply_morphological_operations
)
from hyperseed.core.segmentation.validation import (
    validate_seeds,
    get_seed_properties,
    filter_border_seeds
)


logger = logging.getLogger(__name__)


class SeedSegmenter:
    """Segmenter for detecting seeds in hyperspectral images.

    Provides a unified interface for applying various segmentation
    algorithms with validation and visualization capabilities.

    Example:
        >>> from hyperseed.config import Settings
        >>> settings = Settings()
        >>> segmenter = SeedSegmenter(settings.segmentation)
        >>> mask, n_seeds = segmenter.segment(hyperspectral_data)
        >>> properties = segmenter.get_seed_properties()
    """

    def __init__(self, config: Optional[SegmentationConfig] = None):
        """Initialize seed segmenter.

        Args:
            config: Segmentation configuration object.
        """
        self.config = config or SegmentationConfig()
        self.mask = None
        self.validation_stats = None
        self.seed_properties = None

    def segment(
        self,
        data: np.ndarray,
        band_index: Optional[int] = None,
        validate: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Perform seed segmentation on hyperspectral data.

        Args:
            data: Hyperspectral data array (lines, samples, bands).
            band_index: Specific band to use for segmentation.
            validate: Whether to apply validation after segmentation.

        Returns:
            Tuple of (labeled mask, number of seeds).
        """
        logger.info(f"Starting segmentation with algorithm: {self.config.algorithm}")

        # Apply selected segmentation algorithm
        if self.config.algorithm == "threshold":
            mask, n_seeds = threshold_segmentation(
                data,
                method=self.config.threshold_method,
                min_seed_size=self.config.min_pixels,
                max_seed_size=self.config.max_pixels,
                band_index=band_index
            )

        elif self.config.algorithm == "watershed":
            mask, n_seeds = watershed_segmentation(
                data,
                min_seed_size=self.config.min_pixels,
                max_seed_size=self.config.max_pixels,
                band_index=band_index
            )

        elif self.config.algorithm == "connected":
            mask, n_seeds = connected_components_segmentation(
                data,
                min_seed_size=self.config.min_pixels,
                max_seed_size=self.config.max_pixels,
                band_index=band_index
            )

        elif self.config.algorithm == "combined":
            mask, n_seeds = combined_segmentation(
                data,
                min_seed_size=self.config.min_pixels,
                max_seed_size=self.config.max_pixels,
                band_index=band_index,
                methods=["threshold", "watershed"]
            )

        else:
            raise ValueError(f"Unknown segmentation algorithm: {self.config.algorithm}")

        # Apply morphological operations if requested
        if self.config.morphology_operations:
            mask = apply_morphological_operations(
                mask,
                operations=["closing", "opening"],
                kernel_size=self.config.morphology_kernel_size
            )

        # Validate seeds if requested
        if validate:
            mask, self.validation_stats = validate_seeds(
                mask,
                min_size=self.config.min_pixels,
                max_size=self.config.max_pixels,
                reject_overlapping=self.config.reject_overlapping,
                check_shape=True
            )
            n_seeds = self.validation_stats["final_count"]

        # Filter border seeds if configured
        if self.config.filter_border_seeds:
            mask, removed_border = filter_border_seeds(mask, border_width=self.config.border_width)
            if removed_border:
                n_seeds = len(np.unique(mask[mask > 0]))
                logger.info(f"Removed {len(removed_border)} border seeds")

        # Store results
        self.mask = mask
        self.seed_properties = get_seed_properties(mask)

        logger.info(f"Segmentation complete: {n_seeds} seeds detected")

        return mask, n_seeds

    def visualize(
        self,
        data: np.ndarray,
        band_index: Optional[int] = None,
        save_path: Optional[Union[str, Path]] = None,
        show_labels: bool = True,
        show_boundaries: bool = True
    ) -> plt.Figure:
        """Visualize segmentation results.

        Args:
            data: Original hyperspectral data.
            band_index: Band to use for background image.
            save_path: Path to save the figure.
            show_labels: Whether to show seed labels.
            show_boundaries: Whether to show seed boundaries.

        Returns:
            Matplotlib figure object.
        """
        if self.mask is None:
            raise ValueError("No segmentation results to visualize. Run segment() first.")

        # Create background image
        if data.ndim == 3:
            if band_index is not None:
                bg_image = data[:, :, band_index]
            else:
                # Use mean of middle bands
                mid_band = data.shape[2] // 2
                bg_image = np.mean(
                    data[:, :, max(0, mid_band - 5):min(data.shape[2], mid_band + 5)],
                    axis=2
                )
        else:
            bg_image = data

        # Normalize background image
        bg_image = (bg_image - bg_image.min()) / (bg_image.max() - bg_image.min())

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(bg_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Segmentation with numbered labels
        # Create a dark background for better visibility of labels
        labeled_img = np.zeros_like(bg_image)
        labeled_img = np.stack([labeled_img] * 3, axis=-1)

        # Add colored regions for each seed
        from matplotlib import cm
        cmap = cm.get_cmap('tab20')

        if self.seed_properties:
            for prop in self.seed_properties:
                mask_region = self.mask == prop["label"]
                color_idx = (prop["label"] - 1) % 20
                color = cmap(color_idx / 20.0)[:3]
                for i in range(3):
                    labeled_img[:, :, i][mask_region] = color[i] * 0.7

        axes[1].imshow(labeled_img)
        axes[1].set_title(f'Segmentation ({len(self.seed_properties)} seeds)')
        axes[1].axis('off')

        # Add labels with seed numbers
        if show_labels and self.seed_properties:
            for prop in self.seed_properties:
                y, x = prop["centroid"]
                axes[1].text(
                    x, y, str(prop["label"]),
                    color='yellow', fontsize=10, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7)
                )

        # Boundaries
        if show_boundaries:
            from skimage.segmentation import find_boundaries
            boundaries = find_boundaries(self.mask, mode='thick')

            # Show boundaries on original
            boundary_overlay = bg_image.copy()
            boundary_overlay = np.stack([boundary_overlay] * 3, axis=-1)
            boundary_overlay[boundaries] = [1, 0, 0]  # Red boundaries

            axes[2].imshow(boundary_overlay)
            axes[2].set_title('Seed Boundaries')
            axes[2].axis('off')
        else:
            # Just show mask
            axes[2].imshow(self.mask, cmap='tab20')
            axes[2].set_title('Labeled Mask')
            axes[2].axis('off')

        plt.suptitle(f'Seed Segmentation - Algorithm: {self.config.algorithm}')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved segmentation visualization to {save_path}")

        return fig

    def get_seed_properties(self) -> list:
        """Get properties of all segmented seeds.

        Returns:
            List of seed property dictionaries.
        """
        if self.seed_properties is None:
            if self.mask is None:
                raise ValueError("No segmentation results. Run segment() first.")
            self.seed_properties = get_seed_properties(self.mask)

        return self.seed_properties

    def get_validation_stats(self) -> Optional[Dict[str, Any]]:
        """Get validation statistics from last segmentation.

        Returns:
            Dictionary of validation statistics or None.
        """
        return self.validation_stats

    def export_mask(
        self,
        path: Union[str, Path],
        format: str = "npy"
    ) -> None:
        """Export segmentation mask to file.

        Args:
            path: Path to save the mask.
            format: Export format ('npy', 'png', 'tiff').
        """
        if self.mask is None:
            raise ValueError("No segmentation mask to export. Run segment() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "npy":
            np.save(path, self.mask)
        elif format == "png":
            from skimage import io
            # Scale to 0-255 for PNG
            scaled = (self.mask * 255.0 / self.mask.max()).astype(np.uint8)
            io.imsave(path, scaled)
        elif format == "tiff":
            from skimage import io
            io.imsave(path, self.mask)
        else:
            raise ValueError(f"Unknown export format: {format}")

        logger.info(f"Exported segmentation mask to {path}")

    def export_properties(
        self,
        path: Union[str, Path],
        format: str = "csv"
    ) -> None:
        """Export seed properties to file.

        Args:
            path: Path to save the properties.
            format: Export format ('csv', 'json').
        """
        if self.seed_properties is None:
            self.get_seed_properties()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            import pandas as pd

            # Convert to DataFrame (exclude coords for CSV)
            props_for_df = []
            for prop in self.seed_properties:
                prop_copy = prop.copy()
                # Convert centroid and bbox to separate columns
                prop_copy['centroid_y'] = prop_copy['centroid'][0]
                prop_copy['centroid_x'] = prop_copy['centroid'][1]
                prop_copy['bbox_min_row'] = prop_copy['bbox'][0]
                prop_copy['bbox_min_col'] = prop_copy['bbox'][1]
                prop_copy['bbox_max_row'] = prop_copy['bbox'][2]
                prop_copy['bbox_max_col'] = prop_copy['bbox'][3]
                # Remove original complex fields
                del prop_copy['centroid']
                del prop_copy['bbox']
                del prop_copy['coords']  # Too large for CSV
                props_for_df.append(prop_copy)

            df = pd.DataFrame(props_for_df)
            df.to_csv(path, index=False)

        elif format == "json":
            import json

            # Convert numpy types to Python types
            props_json = []
            for prop in self.seed_properties:
                prop_json = {}
                for key, value in prop.items():
                    if key == 'coords':
                        # Skip coords (too large)
                        continue
                    elif isinstance(value, np.ndarray):
                        prop_json[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        prop_json[key] = float(value)
                    elif isinstance(value, tuple):
                        prop_json[key] = list(value)
                    else:
                        prop_json[key] = value
                props_json.append(prop_json)

            with open(path, 'w') as f:
                json.dump(props_json, f, indent=2)

        else:
            raise ValueError(f"Unknown export format: {format}")

        logger.info(f"Exported {len(self.seed_properties)} seed properties to {path}")

    def describe(self) -> Dict[str, Any]:
        """Get description of segmentation results.

        Returns:
            Dictionary describing the segmentation.
        """
        if self.mask is None:
            return {"status": "No segmentation performed"}

        description = {
            "algorithm": self.config.algorithm,
            "n_seeds": len(self.seed_properties) if self.seed_properties else 0,
            "mask_shape": self.mask.shape,
            "config": self.config.model_dump(),
        }

        if self.validation_stats:
            description["validation"] = self.validation_stats

        if self.seed_properties:
            areas = [p["area"] for p in self.seed_properties]
            description["seed_statistics"] = {
                "min_area": min(areas),
                "max_area": max(areas),
                "mean_area": np.mean(areas),
                "std_area": np.std(areas)
            }

        return description

    def __repr__(self) -> str:
        """String representation of the segmenter."""
        n_seeds = len(self.seed_properties) if self.seed_properties else 0
        return f"SeedSegmenter(algorithm='{self.config.algorithm}', seeds={n_seeds})"