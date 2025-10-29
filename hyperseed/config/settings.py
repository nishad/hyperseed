"""Configuration settings for hyperseed analysis pipeline.

This module defines configuration schemas and default settings for the
hyperspectral seed analysis pipeline using Pydantic for validation.
"""

from pathlib import Path
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class PreprocessingConfig(BaseModel):
    """Configuration for spectral preprocessing.

    Attributes:
        method: Preprocessing method preset or 'custom'.
        snv: Apply Standard Normal Variate normalization.
        smoothing: Apply Savitzky-Golay smoothing.
        smoothing_window: Window length for smoothing (must be odd).
        smoothing_polyorder: Polynomial order for smoothing.
        baseline_correction: Apply baseline correction.
        baseline_order: Polynomial order for baseline fitting.
        derivative: Apply derivative (0, 1, or 2).
        msc: Apply Multiplicative Scatter Correction.
        detrend: Apply detrending.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["standard", "advanced", "minimal", "custom", "none"] = "minimal"
    snv: bool = False
    smoothing: bool = True
    smoothing_window: int = Field(default=11, ge=3, le=51)
    smoothing_polyorder: int = Field(default=3, ge=1, le=5)
    baseline_correction: bool = False
    baseline_order: int = Field(default=2, ge=1, le=5)
    derivative: Literal[0, 1, 2] = 0
    msc: bool = False
    detrend: bool = False


class SegmentationConfig(BaseModel):
    """Configuration for seed segmentation.

    Attributes:
        algorithm: Segmentation algorithm to use.
        min_pixels: Minimum number of pixels for valid seed.
        max_pixels: Maximum number of pixels for valid seed.
        reject_overlapping: Reject seeds that touch/overlap.
        threshold_method: Method for automatic thresholding.
        morphology_operations: Apply morphological operations.
        morphology_kernel_size: Size of morphology kernel.
        filter_border_seeds: Remove seeds touching image borders.
        border_width: Width of border region to check (pixels).
        remove_outliers: Enable outlier detection and removal.
        outlier_min_area: Minimum area threshold for outlier removal.
        outlier_max_area: Maximum area threshold for outlier removal.
        outlier_iqr_lower: IQR multiplier for lower bound calculation.
        outlier_iqr_upper: IQR multiplier for upper bound calculation.
        outlier_eccentricity: Maximum eccentricity for shape-based filtering.
        outlier_solidity: Minimum solidity for shape-based filtering.
        use_shape_filtering: Enable shape-based outlier filtering.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["threshold", "watershed", "connected", "combined"] = "watershed"
    min_pixels: int = Field(default=200, ge=10, le=10000)
    max_pixels: Optional[int] = Field(default=None, ge=10, le=100000)
    reject_overlapping: bool = True
    threshold_method: Literal["otsu", "adaptive", "manual"] = "otsu"
    morphology_operations: bool = True
    morphology_kernel_size: int = Field(default=3, ge=1, le=21)
    filter_border_seeds: bool = False
    border_width: int = Field(default=2, ge=0, le=50)

    # Outlier detection settings
    remove_outliers: bool = True  # Default: enabled
    outlier_min_area: int = Field(default=50, ge=0, description="Minimum seed area in pixels")
    outlier_max_area: int = Field(default=2000, ge=100, description="Maximum seed area in pixels")
    outlier_iqr_lower: float = Field(default=1.5, ge=0, description="IQR multiplier for lower bound")
    outlier_iqr_upper: float = Field(default=3.0, ge=0, description="IQR multiplier for upper bound")
    outlier_eccentricity: float = Field(default=0.95, ge=0, le=1, description="Max eccentricity (elongation)")
    outlier_solidity: float = Field(default=0.7, ge=0, le=1, description="Min solidity (shape regularity)")
    use_shape_filtering: bool = False  # Shape-based filtering disabled by default


class WavelengthSelectionConfig(BaseModel):
    """Configuration for wavelength selection/reduction.

    Attributes:
        method: Selection method.
        ranges: List of wavelength ranges to include (nm).
        bands: Specific band indices to include.
        n_components: Number of components for PCA/variance selection.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["all", "ranges", "indices", "pca", "variance"] = "all"
    ranges: Optional[List[tuple[float, float]]] = None
    bands: Optional[List[int]] = None
    n_components: int = Field(default=20, ge=1, le=200)


class CalibrationConfig(BaseModel):
    """Configuration for reflectance calibration.

    Attributes:
        apply_calibration: Apply white/dark reference calibration.
        apply_bad_pixels: Apply bad pixel replacement.
        clip_negative: Clip negative reflectance values to 0.
        clip_max: Maximum reflectance value to clip to.
    """

    model_config = ConfigDict(extra="forbid")

    apply_calibration: bool = True
    apply_bad_pixels: bool = True
    clip_negative: bool = True
    clip_max: Optional[float] = Field(default=1.0, ge=0.5, le=10.0)


class OutputConfig(BaseModel):
    """Configuration for output generation.

    Attributes:
        format: Primary output format.
        include_plots: Generate visualization plots.
        include_coordinates: Include seed coordinates in output.
        include_metadata: Include metadata in output.
        plot_format: Format for plot files.
        plot_dpi: DPI for plot output.
        csv_separator: Separator for CSV output.
    """

    model_config = ConfigDict(extra="forbid")

    format: Literal["csv", "hdf5", "both"] = "csv"
    include_plots: bool = True
    include_coordinates: bool = True
    include_metadata: bool = True
    plot_format: Literal["png", "pdf", "svg"] = "png"
    plot_dpi: int = Field(default=150, ge=72, le=600)
    csv_separator: str = ","


class ProcessingConfig(BaseModel):
    """Configuration for processing performance.

    Attributes:
        device: Computing device to use.
        parallel_workers: Number of parallel workers.
        chunk_size: Chunk size for processing large files.
        memory_limit_gb: Memory limit in GB.
        use_mmap: Use memory mapping for large files.
    """

    model_config = ConfigDict(extra="forbid")

    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    parallel_workers: int = Field(default=4, ge=1, le=32)
    chunk_size: Optional[int] = Field(default=None, ge=10, le=1000)
    memory_limit_gb: Optional[float] = Field(default=None, ge=0.5, le=256)
    use_mmap: bool = True


class Settings(BaseModel):
    """Main settings container for hyperseed pipeline.

    This class combines all configuration sections into a single
    settings object that can be loaded from YAML or created programmatically.

    Example:
        >>> from hyperseed.config import Settings
        >>> settings = Settings()  # Use defaults
        >>> settings.segmentation.min_pixels = 150
        >>> settings.save("config.yaml")
    """

    model_config = ConfigDict(extra="forbid")

    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    wavelength_selection: WavelengthSelectionConfig = Field(
        default_factory=WavelengthSelectionConfig
    )
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Load settings from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Settings object loaded from file.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            ValueError: If configuration is invalid.
        """
        import yaml

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def save(self, path: Path) -> None:
        """Save settings to YAML file.

        Args:
            path: Path to save configuration file.
        """
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(
                self.model_dump(exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False
            )

    def apply_preset(self, preset: str) -> None:
        """Apply a preprocessing preset.

        Args:
            preset: Name of preset ('standard', 'advanced', 'minimal').

        Raises:
            ValueError: If preset name is invalid.
        """
        if preset == "standard":
            self.preprocessing = PreprocessingConfig(
                method="standard",
                snv=True,
                smoothing=True,
                baseline_correction=True,
                derivative=0
            )
        elif preset == "advanced":
            self.preprocessing = PreprocessingConfig(
                method="advanced",
                snv=True,
                smoothing=True,
                baseline_correction=True,
                derivative=2,
                msc=True,
                detrend=True
            )
        elif preset == "minimal":
            self.preprocessing = PreprocessingConfig(
                method="minimal",
                smoothing=True,
                snv=False,
                baseline_correction=False
            )
        else:
            raise ValueError(f"Unknown preset: {preset}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary.

        Returns:
            Dictionary representation of settings.
        """
        return self.model_dump(exclude_none=True)