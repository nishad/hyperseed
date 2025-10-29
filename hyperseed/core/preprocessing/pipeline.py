"""Preprocessing pipeline for hyperspectral data.

This module provides a configurable pipeline for applying multiple
preprocessing steps to hyperspectral data in sequence.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import numpy as np

from hyperseed.config.settings import PreprocessingConfig
from hyperseed.core.preprocessing.methods import (
    apply_snv,
    apply_smoothing,
    apply_derivative,
    apply_baseline_correction,
    apply_msc,
    apply_detrend,
    apply_normalization,
)


logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Configurable pipeline for spectral preprocessing.

    Applies a sequence of preprocessing methods to hyperspectral data
    based on configuration settings.

    Example:
        >>> from hyperseed.config import Settings
        >>> settings = Settings()
        >>> settings.preprocessing.method = "standard"
        >>> pipeline = PreprocessingPipeline(settings.preprocessing)
        >>> processed_data = pipeline.fit_transform(raw_data)
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize preprocessing pipeline.

        Args:
            config: Preprocessing configuration object.
        """
        self.config = config or PreprocessingConfig()
        self.reference_spectrum = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build the preprocessing pipeline based on configuration."""
        self.steps = []

        # Apply preset if specified
        if self.config.method != "custom":
            self._apply_preset(self.config.method)

        # Build pipeline steps based on configuration
        if self.config.smoothing:
            self.steps.append(("smoothing", self._apply_smoothing))

        if self.config.baseline_correction:
            self.steps.append(("baseline", self._apply_baseline))

        if self.config.derivative > 0:
            self.steps.append(("derivative", self._apply_derivative))

        if self.config.detrend:
            self.steps.append(("detrend", self._apply_detrend))

        if self.config.msc:
            self.steps.append(("msc", self._apply_msc))

        if self.config.snv:
            self.steps.append(("snv", self._apply_snv))

        logger.info(f"Built preprocessing pipeline with {len(self.steps)} steps")

    def _apply_preset(self, preset: str) -> None:
        """Apply a preprocessing preset configuration.

        Args:
            preset: Name of the preset to apply.
        """
        if preset == "standard":
            self.config.snv = True
            self.config.smoothing = True
            self.config.baseline_correction = True
            self.config.derivative = 0
            self.config.msc = False
            self.config.detrend = False

        elif preset == "advanced":
            self.config.snv = True
            self.config.smoothing = True
            self.config.baseline_correction = True
            self.config.derivative = 2
            self.config.msc = True
            self.config.detrend = True

        elif preset == "minimal":
            self.config.smoothing = True
            self.config.snv = False
            self.config.baseline_correction = False
            self.config.derivative = 0
            self.config.msc = False
            self.config.detrend = False

    def fit(self, data: np.ndarray) -> "PreprocessingPipeline":
        """Fit the pipeline to reference data.

        Some preprocessing methods (like MSC) require fitting to reference data.

        Args:
            data: Reference data to fit to.

        Returns:
            Self for method chaining.
        """
        # Calculate reference spectrum for MSC if needed
        if self.config.msc:
            # Reshape to 2D if needed
            if data.ndim == 3:
                # Assume shape is (lines, samples, bands)
                data_2d = data.reshape(-1, data.shape[-1])
            else:
                data_2d = data

            self.reference_spectrum = np.mean(data_2d, axis=0)
            logger.info(f"Fitted reference spectrum with {len(self.reference_spectrum)} bands")

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline to data.

        Args:
            data: Input hyperspectral data.

        Returns:
            Preprocessed data.
        """
        processed = data.copy()
        original_shape = processed.shape

        # Track if we need to reshape data
        needs_reshape = False
        if processed.ndim == 3:
            # Reshape to 2D for processing (pixels, bands)
            needs_reshape = True
            processed = processed.reshape(-1, processed.shape[-1])

        # Apply each preprocessing step
        for step_name, step_func in self.steps:
            logger.debug(f"Applying {step_name}")
            processed = step_func(processed)

            # Check for invalid values
            if np.any(np.isnan(processed)):
                logger.warning(f"NaN values detected after {step_name}")
                processed = np.nan_to_num(processed, nan=0.0)

            if np.any(np.isinf(processed)):
                logger.warning(f"Inf values detected after {step_name}")
                processed = np.nan_to_num(processed, posinf=1.0, neginf=0.0)

        # Reshape back if needed
        if needs_reshape:
            processed = processed.reshape(original_shape)

        logger.info(f"Applied {len(self.steps)} preprocessing steps")
        return processed

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit the pipeline and transform data in one step.

        Args:
            data: Input hyperspectral data.

        Returns:
            Preprocessed data.
        """
        return self.fit(data).transform(data)

    def _apply_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Apply smoothing based on configuration."""
        return apply_smoothing(
            data,
            window_length=self.config.smoothing_window,
            polyorder=self.config.smoothing_polyorder,
            axis=-1,
            method="savgol"
        )

    def _apply_baseline(self, data: np.ndarray) -> np.ndarray:
        """Apply baseline correction based on configuration."""
        return apply_baseline_correction(
            data,
            order=self.config.baseline_order,
            axis=-1,
            method="polynomial"
        )

    def _apply_derivative(self, data: np.ndarray) -> np.ndarray:
        """Apply derivative based on configuration."""
        if self.config.derivative == 0:
            return data

        return apply_derivative(
            data,
            order=self.config.derivative,
            window_length=self.config.smoothing_window,
            polyorder=self.config.smoothing_polyorder,
            axis=-1
        )

    def _apply_snv(self, data: np.ndarray) -> np.ndarray:
        """Apply SNV transformation."""
        return apply_snv(data, axis=-1)

    def _apply_msc(self, data: np.ndarray) -> np.ndarray:
        """Apply MSC transformation."""
        return apply_msc(data, reference=self.reference_spectrum, axis=-1)

    def _apply_detrend(self, data: np.ndarray) -> np.ndarray:
        """Apply detrending."""
        return apply_detrend(data, axis=-1, type="linear")

    def get_step_names(self) -> List[str]:
        """Get list of preprocessing step names.

        Returns:
            List of step names in order of application.
        """
        return [name for name, _ in self.steps]

    def describe(self) -> Dict[str, Any]:
        """Get description of the preprocessing pipeline.

        Returns:
            Dictionary describing the pipeline configuration.
        """
        return {
            "method": self.config.method,
            "steps": self.get_step_names(),
            "config": self.config.model_dump(),
        }

    def save_config(self, path: Union[str, Path]) -> None:
        """Save pipeline configuration to file.

        Args:
            path: Path to save configuration.
        """
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.describe(), f, default_flow_style=False)

        logger.info(f"Saved preprocessing config to {path}")

    @classmethod
    def from_config_file(cls, path: Union[str, Path]) -> "PreprocessingPipeline":
        """Load pipeline from configuration file.

        Args:
            path: Path to configuration file.

        Returns:
            PreprocessingPipeline instance.
        """
        import yaml

        path = Path(path)
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if "config" in config_dict:
            config = PreprocessingConfig(**config_dict["config"])
        else:
            config = PreprocessingConfig(**config_dict)

        return cls(config)

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        steps_str = ", ".join(self.get_step_names())
        return f"PreprocessingPipeline(method='{self.config.method}', steps=[{steps_str}])"