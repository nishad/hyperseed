"""Spectral extraction module for extracting seed spectra from hyperspectral data.

This module provides functionality to extract average spectral signatures
from individual seeds based on segmentation masks.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from hyperseed.core.segmentation.validation import get_seed_properties


logger = logging.getLogger(__name__)


class SpectralExtractor:
    """Extract spectral signatures from segmented seeds.

    This class provides methods to extract average spectra from individual
    seeds identified by segmentation masks, along with spatial and statistical
    information.

    Example:
        >>> extractor = SpectralExtractor()
        >>> results = extractor.extract(data, mask, wavelengths)
        >>> df = extractor.to_dataframe()
        >>> extractor.save_csv("seed_spectra.csv")
    """

    def __init__(self):
        """Initialize spectral extractor."""
        self.spectra = None
        self.wavelengths = None
        self.seed_info = None
        self.metadata = {}

    def extract(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
        compute_stats: bool = True
    ) -> Dict[str, Any]:
        """Extract average spectra from segmented seeds.

        Args:
            data: Hyperspectral data array (lines, samples, bands).
            mask: Labeled segmentation mask (lines, samples).
            wavelengths: Array of wavelength values for each band.
            compute_stats: Whether to compute statistical measures.

        Returns:
            Dictionary containing extracted spectra and seed information.

        Raises:
            ValueError: If data and mask shapes don't match.
        """
        # Validate inputs
        if data.shape[:2] != mask.shape:
            raise ValueError(
                f"Data shape {data.shape[:2]} doesn't match mask shape {mask.shape}"
            )

        # Get unique seed labels (excluding background)
        seed_labels = np.unique(mask[mask > 0])
        n_seeds = len(seed_labels)
        n_bands = data.shape[2]

        if n_seeds == 0:
            logger.warning("No seeds found in mask")
            return {"n_seeds": 0, "spectra": None, "seed_info": None}

        logger.info(f"Extracting spectra from {n_seeds} seeds")

        # Initialize arrays for results
        spectra = np.zeros((n_seeds, n_bands), dtype=np.float32)
        seed_info = []

        # Get seed properties from mask
        seed_properties = get_seed_properties(mask)
        properties_dict = {prop['label']: prop for prop in seed_properties}

        # Extract spectrum for each seed
        for idx, label in enumerate(seed_labels):
            # Get mask for current seed
            seed_mask = (mask == label)

            # Extract spectra for all pixels in seed
            seed_pixels = data[seed_mask]  # Shape: (n_pixels, n_bands)

            # Calculate average spectrum
            avg_spectrum = np.mean(seed_pixels, axis=0)
            spectra[idx] = avg_spectrum

            # Get seed information
            if label in properties_dict:
                prop = properties_dict[label]
                info = {
                    'seed_id': int(label),
                    'index': idx,
                    'centroid_y': prop['centroid'][0],
                    'centroid_x': prop['centroid'][1],
                    'area': prop['area'],
                    'bbox_min_row': prop['bbox'][0],
                    'bbox_min_col': prop['bbox'][1],
                    'bbox_max_row': prop['bbox'][2],
                    'bbox_max_col': prop['bbox'][3],
                    'eccentricity': prop['eccentricity'],
                    'solidity': prop['solidity'],
                    'major_axis': prop['major_axis_length'],
                    'minor_axis': prop['minor_axis_length']
                }
            else:
                # Fallback if properties not found
                centroid = np.mean(np.argwhere(seed_mask), axis=0)
                info = {
                    'seed_id': int(label),
                    'index': idx,
                    'centroid_y': centroid[0],
                    'centroid_x': centroid[1],
                    'area': np.sum(seed_mask),
                    'bbox_min_row': np.min(np.where(seed_mask)[0]),
                    'bbox_min_col': np.min(np.where(seed_mask)[1]),
                    'bbox_max_row': np.max(np.where(seed_mask)[0]),
                    'bbox_max_col': np.max(np.where(seed_mask)[1]),
                    'eccentricity': np.nan,
                    'solidity': np.nan,
                    'major_axis': np.nan,
                    'minor_axis': np.nan
                }

            # Add statistical measures if requested
            if compute_stats:
                info['mean_reflectance'] = float(np.mean(avg_spectrum))
                info['std_reflectance'] = float(np.std(avg_spectrum))
                info['min_reflectance'] = float(np.min(avg_spectrum))
                info['max_reflectance'] = float(np.max(avg_spectrum))

                # Spectral indices (example indices)
                if wavelengths is not None and len(wavelengths) > 0:
                    # Find bands close to specific wavelengths
                    # NDVI-like index using NIR and RED equivalents
                    nir_idx = np.argmin(np.abs(wavelengths - 1650))  # ~1650nm
                    red_idx = np.argmin(np.abs(wavelengths - 1450))  # ~1450nm

                    if nir_idx != red_idx:
                        nir_val = avg_spectrum[nir_idx]
                        red_val = avg_spectrum[red_idx]
                        if (nir_val + red_val) > 0:
                            info['spectral_index_1'] = (nir_val - red_val) / (nir_val + red_val)
                        else:
                            info['spectral_index_1'] = 0.0

            seed_info.append(info)

            logger.debug(f"Extracted spectrum for seed {label} ({idx + 1}/{n_seeds})")

        # Store results
        self.spectra = spectra
        self.wavelengths = wavelengths
        self.seed_info = seed_info

        results = {
            'n_seeds': n_seeds,
            'n_bands': n_bands,
            'spectra': spectra,
            'seed_info': seed_info,
            'wavelengths': wavelengths
        }

        logger.info(f"Extraction complete: {n_seeds} seeds, {n_bands} bands")

        return results

    def remove_outliers(
        self,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, List[Dict], List[int]]:
        """Remove outlier seeds based on area and shape properties.

        Args:
            config: Outlier detection configuration dictionary.
            verbose: Whether to log detailed information.

        Returns:
            Tuple of (filtered_spectra, filtered_seed_info, removed_indices).
        """
        if self.spectra is None or self.seed_info is None:
            raise ValueError("No spectra extracted. Run extract() first.")

        # Default configuration
        default_config = {
            'remove_outliers': True,
            'outlier_min_area': 50,
            'outlier_max_area': 2000,
            'outlier_iqr_lower': 1.5,
            'outlier_iqr_upper': 3.0,
            'outlier_eccentricity': 0.95,
            'outlier_solidity': 0.7,
            'use_shape_filtering': False
        }

        if config is None:
            config = default_config
        else:
            # Merge with defaults
            config = {**default_config, **config}

        if not config['remove_outliers']:
            return self.spectra, self.seed_info, []

        # Extract area values
        areas = np.array([s['area'] for s in self.seed_info])
        n_seeds = len(areas)

        # Initialize mask for seeds to keep
        keep_mask = np.ones(n_seeds, dtype=bool)

        # Step 1: Absolute area bounds
        area_min_mask = areas >= config['outlier_min_area']
        area_max_mask = areas <= config['outlier_max_area']
        keep_mask &= area_min_mask & area_max_mask

        if verbose:
            n_removed_min = np.sum(~area_min_mask)
            n_removed_max = np.sum(~area_max_mask)
            if n_removed_min > 0:
                logger.info(f"Removed {n_removed_min} seeds below {config['outlier_min_area']} pixels")
            if n_removed_max > 0:
                logger.info(f"Removed {n_removed_max} seeds above {config['outlier_max_area']} pixels")

        # Step 2: IQR-based filtering on remaining seeds
        if np.sum(keep_mask) > 3:  # Need at least 3 seeds for IQR
            remaining_areas = areas[keep_mask]
            Q1 = np.percentile(remaining_areas, 25)
            Q3 = np.percentile(remaining_areas, 75)
            IQR = Q3 - Q1

            # Calculate bounds (asymmetric multipliers)
            lower_bound = max(config['outlier_min_area'], Q1 - config['outlier_iqr_lower'] * IQR)
            upper_bound = min(config['outlier_max_area'], Q3 + config['outlier_iqr_upper'] * IQR)

            # Apply IQR filtering
            iqr_mask = (areas >= lower_bound) & (areas <= upper_bound)
            n_before_iqr = np.sum(keep_mask)
            keep_mask &= iqr_mask
            n_removed_iqr = n_before_iqr - np.sum(keep_mask)

            if verbose and n_removed_iqr > 0:
                logger.info(f"Removed {n_removed_iqr} seeds by IQR method (bounds: {lower_bound:.1f}-{upper_bound:.1f})")

        # Step 3: Shape-based filtering (optional)
        if config['use_shape_filtering']:
            shape_mask = np.ones(n_seeds, dtype=bool)

            for i, info in enumerate(self.seed_info):
                # Check eccentricity (elongation)
                if not np.isnan(info.get('eccentricity', np.nan)):
                    if info['eccentricity'] > config['outlier_eccentricity']:
                        shape_mask[i] = False

                # Check solidity (regularity)
                if not np.isnan(info.get('solidity', np.nan)):
                    if info['solidity'] < config['outlier_solidity']:
                        shape_mask[i] = False

            n_before_shape = np.sum(keep_mask)
            keep_mask &= shape_mask
            n_removed_shape = n_before_shape - np.sum(keep_mask)

            if verbose and n_removed_shape > 0:
                logger.info(f"Removed {n_removed_shape} seeds by shape criteria")

        # Get indices of removed seeds
        removed_indices = np.where(~keep_mask)[0].tolist()

        # Filter spectra and seed info
        filtered_spectra = self.spectra[keep_mask]
        filtered_seed_info = [info for i, info in enumerate(self.seed_info) if keep_mask[i]]

        # Update indices in filtered seed info
        for i, info in enumerate(filtered_seed_info):
            info['index'] = i

        # Report summary
        n_removed = len(removed_indices)
        n_kept = np.sum(keep_mask)

        if verbose:
            logger.info(f"Outlier removal complete: {n_kept} seeds kept, {n_removed} removed")
            if n_removed > 0:
                removed_areas = areas[~keep_mask]
                logger.info(f"Removed seed areas: min={np.min(removed_areas):.0f}, "
                          f"max={np.max(removed_areas):.0f}, mean={np.mean(removed_areas):.0f}")

        # Update instance variables with filtered data
        self.spectra = filtered_spectra
        self.seed_info = filtered_seed_info

        return filtered_spectra, filtered_seed_info, removed_indices

    def to_dataframe(self, include_wavelengths: bool = True) -> pd.DataFrame:
        """Convert extracted spectra to pandas DataFrame.

        Args:
            include_wavelengths: Whether to use wavelengths as column names.

        Returns:
            DataFrame with seed information and spectra.

        Raises:
            ValueError: If no spectra have been extracted.
        """
        if self.spectra is None or self.seed_info is None:
            raise ValueError("No spectra extracted. Run extract() first.")

        # Create DataFrame from seed info
        df = pd.DataFrame(self.seed_info)

        # Add spectral data
        if include_wavelengths and self.wavelengths is not None:
            # Use wavelengths as column names
            band_columns = [f'band_{wl:.2f}nm' for wl in self.wavelengths]
        else:
            # Use band indices
            band_columns = [f'band_{i}' for i in range(self.spectra.shape[1])]

        # Add spectral data to DataFrame
        spectra_df = pd.DataFrame(self.spectra, columns=band_columns)

        # Combine info and spectra
        df = pd.concat([df, spectra_df], axis=1)

        return df

    def save_csv(
        self,
        path: Union[str, Path],
        include_wavelengths: bool = True,
        separator: str = ","
    ) -> None:
        """Save extracted spectra to CSV file.

        Args:
            path: Path to save CSV file.
            include_wavelengths: Whether to use wavelengths as column names.
            separator: CSV separator character.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe(include_wavelengths=include_wavelengths)
        df.to_csv(path, sep=separator, index=False)

        logger.info(f"Saved {len(df)} seed spectra to {path}")

    def save_hdf5(
        self,
        path: Union[str, Path],
        compression: str = "gzip"
    ) -> None:
        """Save extracted spectra to HDF5 file.

        Args:
            path: Path to save HDF5 file.
            compression: Compression algorithm to use.
        """
        if self.spectra is None or self.seed_info is None:
            raise ValueError("No spectra extracted. Run extract() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, 'w') as f:
            # Save spectra
            f.create_dataset(
                'spectra', data=self.spectra,
                compression=compression
            )

            # Save wavelengths if available
            if self.wavelengths is not None:
                f.create_dataset(
                    'wavelengths', data=self.wavelengths,
                    compression=compression
                )

            # Save seed information
            info_group = f.create_group('seed_info')
            for key in self.seed_info[0].keys():
                values = [seed[key] for seed in self.seed_info]
                info_group.create_dataset(key, data=values)

            # Save metadata
            for key, value in self.metadata.items():
                f.attrs[key] = value

        logger.info(f"Saved spectra to HDF5 file: {path}")

    def plot_spectra(
        self,
        seed_indices: Optional[List[int]] = None,
        show_mean: bool = True,
        show_std: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        max_display: int = 10
    ) -> plt.Figure:
        """Plot extracted spectra with enhanced visualization.

        Args:
            seed_indices: Indices of seeds to plot (None for first N).
            show_mean: Whether to show mean spectrum.
            show_std: Whether to show standard deviation.
            save_path: Path to save the figure.
            max_display: Maximum number of individual seeds to display.

        Returns:
            Matplotlib figure object.
        """
        if self.spectra is None:
            raise ValueError("No spectra extracted. Run extract() first.")

        fig, ax = plt.subplots(figsize=(14, 8))

        # Determine x-axis values
        if self.wavelengths is not None:
            x_values = self.wavelengths
            x_label = 'Wavelength (nm)'
        else:
            x_values = np.arange(self.spectra.shape[1])
            x_label = 'Band Index'

        # Select seeds to plot
        if seed_indices is None:
            # Plot first N seeds for consistency
            n_to_plot = min(max_display, len(self.spectra))
            indices = list(range(n_to_plot))
        else:
            indices = seed_indices[:max_display]

        # Use colormap for individual spectra
        colors = plt.cm.tab20(np.linspace(0, 1, len(indices)))

        # Plot individual spectra
        for i, idx in enumerate(indices):
            label = f"Seed {self.seed_info[idx]['seed_id']}"
            ax.plot(x_values, self.spectra[idx], alpha=0.7, linewidth=1.2,
                   color=colors[i], label=label)

        # Plot mean spectrum
        if show_mean:
            mean_spectrum = np.mean(self.spectra, axis=0)
            ax.plot(x_values, mean_spectrum, 'k-', linewidth=2.5,
                   label='Mean', zorder=10)

            if show_std:
                std_spectrum = np.std(self.spectra, axis=0)
                ax.fill_between(
                    x_values,
                    mean_spectrum - std_spectrum,
                    mean_spectrum + std_spectrum,
                    alpha=0.3, color='gray', label='±1 STD'
                )

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Reflectance', fontsize=12)
        ax.set_title(f'Extracted Seed Spectra (n={len(self.spectra)})', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add legend
        ax.legend(loc='upper right', fontsize=9, ncol=2 if len(indices) > 5 else 1)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved spectra plot to {save_path}")

            # Generate additional statistics plot
            self._plot_statistics(save_path.parent / f"{save_path.stem}_statistics.png")

        return fig

    def plot_distribution(
        self,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Generate seed spatial and area distribution plots.

        Args:
            save_path: Path to save the figure.

        Returns:
            Matplotlib figure object.
        """
        if self.seed_info is None or len(self.seed_info) == 0:
            raise ValueError("No seed information available. Run extract() first.")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: Spatial distribution
        ax = axes[0]
        x_coords = [info['centroid_x'] for info in self.seed_info]
        y_coords = [info['centroid_y'] for info in self.seed_info]
        areas = [info['area'] for info in self.seed_info]

        # Create scatter plot with area as color
        scatter = ax.scatter(x_coords, y_coords, c=areas, s=100,
                           cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)

        # Add seed numbers
        for info in self.seed_info:
            ax.text(info['centroid_x'], info['centroid_y'], str(info['seed_id']),
                   fontsize=8, ha='center', va='center', color='white',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))

        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('Seed Spatial Distribution', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Invert Y axis to match image coordinates

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Area (pixels)', fontsize=10)

        # Right plot: Area distribution histogram
        ax = axes[1]
        areas = [info['area'] for info in self.seed_info]
        mean_area = np.mean(areas)

        n, bins, patches = ax.hist(areas, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(mean_area, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_area:.0f}')
        ax.set_xlabel('Seed Area (pixels)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Seed Area Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Seed Distribution Analysis (n={len(self.seed_info)})', fontsize=16)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved distribution plot to {save_path}")

        return fig

    def _plot_statistics(self, save_path: Path) -> None:
        """Generate additional statistics plot."""
        if self.spectra is None:
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Determine x-axis values
        if self.wavelengths is not None:
            x_values = self.wavelengths
            x_label = 'Wavelength (nm)'
        else:
            x_values = np.arange(self.spectra.shape[1])
            x_label = 'Band Index'

        # Top plot: All spectra with transparency
        ax = axes[0]
        for i in range(len(self.spectra)):
            ax.plot(x_values, self.spectra[i], alpha=0.1, color='blue', linewidth=0.5)

        mean_spectrum = np.mean(self.spectra, axis=0)
        ax.plot(x_values, mean_spectrum, 'r-', linewidth=2, label='Mean')
        ax.set_ylabel('Reflectance', fontsize=12)
        ax.set_title(f'All Seed Spectra (n={len(self.spectra)})', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Bottom plot: Percentile ranges
        ax = axes[1]
        ax.plot(x_values, mean_spectrum, 'b-', linewidth=2, label='Mean')
        ax.fill_between(x_values,
                        np.percentile(self.spectra, 25, axis=0),
                        np.percentile(self.spectra, 75, axis=0),
                        alpha=0.3, color='blue', label='25-75 percentile')
        ax.fill_between(x_values,
                        np.percentile(self.spectra, 10, axis=0),
                        np.percentile(self.spectra, 90, axis=0),
                        alpha=0.2, color='blue', label='10-90 percentile')
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Reflectance', fontsize=12)
        ax.set_title('Spectral Statistics', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved statistics plot to {save_path}")
        plt.close(fig)

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for extracted spectra.

        Returns:
            Dictionary containing statistical measures.
        """
        if self.spectra is None:
            raise ValueError("No spectra extracted. Run extract() first.")

        stats = {
            'n_seeds': len(self.spectra),
            'n_bands': self.spectra.shape[1],
            'mean_spectrum': np.mean(self.spectra, axis=0),
            'std_spectrum': np.std(self.spectra, axis=0),
            'min_spectrum': np.min(self.spectra, axis=0),
            'max_spectrum': np.max(self.spectra, axis=0),
            'global_mean': float(np.mean(self.spectra)),
            'global_std': float(np.std(self.spectra)),
            'global_min': float(np.min(self.spectra)),
            'global_max': float(np.max(self.spectra))
        }

        # Add seed area statistics
        areas = [info['area'] for info in self.seed_info]
        stats['area_mean'] = float(np.mean(areas))
        stats['area_std'] = float(np.std(areas))
        stats['area_min'] = float(np.min(areas))
        stats['area_max'] = float(np.max(areas))

        return stats

    def filter_seeds(
        self,
        min_area: Optional[int] = None,
        max_area: Optional[int] = None,
        min_mean: Optional[float] = None,
        max_mean: Optional[float] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Filter extracted seeds based on criteria.

        Args:
            min_area: Minimum seed area in pixels.
            max_area: Maximum seed area in pixels.
            min_mean: Minimum mean reflectance.
            max_mean: Maximum mean reflectance.

        Returns:
            Tuple of (filtered spectra, filtered seed info).
        """
        if self.spectra is None or self.seed_info is None:
            raise ValueError("No spectra extracted. Run extract() first.")

        valid_indices = []

        for idx, info in enumerate(self.seed_info):
            valid = True

            if min_area is not None and info['area'] < min_area:
                valid = False
            if max_area is not None and info['area'] > max_area:
                valid = False

            if 'mean_reflectance' in info:
                if min_mean is not None and info['mean_reflectance'] < min_mean:
                    valid = False
                if max_mean is not None and info['mean_reflectance'] > max_mean:
                    valid = False

            if valid:
                valid_indices.append(idx)

        filtered_spectra = self.spectra[valid_indices]
        filtered_info = [self.seed_info[i] for i in valid_indices]

        logger.info(
            f"Filtered seeds: {len(self.seed_info)} → {len(filtered_info)}"
        )

        return filtered_spectra, filtered_info

    def __repr__(self) -> str:
        """String representation of the extractor."""
        if self.spectra is not None:
            return (
                f"SpectralExtractor(seeds={len(self.spectra)}, "
                f"bands={self.spectra.shape[1]})"
            )
        else:
            return "SpectralExtractor(no data extracted)"