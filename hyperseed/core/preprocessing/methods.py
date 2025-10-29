"""Spectral preprocessing methods for hyperspectral data.

This module provides various preprocessing techniques commonly used in
hyperspectral data analysis including SNV, derivatives, smoothing, etc.
"""

import logging
from typing import Optional, Union, Literal

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


def apply_snv(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """Apply Standard Normal Variate (SNV) transformation.

    SNV removes scatter effects by centering and scaling each spectrum
    individually to have zero mean and unit variance.

    Args:
        data: Input spectral data array.
        axis: Axis along which to apply SNV (default: -1, last axis).

    Returns:
        SNV transformed data.

    Example:
        >>> spectra = np.random.rand(100, 272)  # 100 spectra, 272 bands
        >>> snv_spectra = apply_snv(spectra)
    """
    # Calculate mean and std along spectral axis
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)

    # Avoid division by zero
    std = np.where(std == 0, 1, std)

    # Apply SNV: (x - mean) / std
    snv_data = (data - mean) / std

    logger.debug(f"Applied SNV transformation to shape {data.shape}")
    return snv_data


def apply_smoothing(
    data: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    axis: int = -1,
    method: Literal["savgol", "moving_average", "gaussian"] = "savgol"
) -> np.ndarray:
    """Apply smoothing to spectral data.

    Args:
        data: Input spectral data array.
        window_length: Length of the smoothing window (must be odd).
        polyorder: Polynomial order for Savitzky-Golay filter.
        axis: Axis along which to apply smoothing.
        method: Smoothing method to use.

    Returns:
        Smoothed data.

    Raises:
        ValueError: If window_length is even or invalid parameters.

    Example:
        >>> spectra = np.random.rand(100, 272)
        >>> smooth_spectra = apply_smoothing(spectra, window_length=11)
    """
    if window_length % 2 == 0:
        raise ValueError("Window length must be odd")

    if window_length < 3:
        raise ValueError("Window length must be at least 3")

    if method == "savgol":
        if polyorder >= window_length:
            raise ValueError("Polyorder must be less than window length")

        # Apply Savitzky-Golay filter
        smoothed = signal.savgol_filter(
            data, window_length, polyorder, axis=axis
        )

    elif method == "moving_average":
        # Simple moving average
        smoothed = uniform_filter1d(
            data, size=window_length, axis=axis, mode='nearest'
        )

    elif method == "gaussian":
        # Gaussian filter
        from scipy.ndimage import gaussian_filter1d
        sigma = window_length / 4.0  # Approximate conversion
        smoothed = gaussian_filter1d(data, sigma=sigma, axis=axis)

    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    logger.debug(f"Applied {method} smoothing with window {window_length}")
    return smoothed


def apply_derivative(
    data: np.ndarray,
    order: Literal[1, 2] = 1,
    window_length: int = 11,
    polyorder: int = 3,
    axis: int = -1
) -> np.ndarray:
    """Apply derivative transformation to spectral data.

    Uses Savitzky-Golay filter to compute derivatives while smoothing.

    Args:
        data: Input spectral data array.
        order: Derivative order (1 or 2).
        window_length: Length of the filter window (must be odd).
        polyorder: Polynomial order for the filter.
        axis: Axis along which to apply derivative.

    Returns:
        Derivative of the input data.

    Raises:
        ValueError: If parameters are invalid.

    Example:
        >>> spectra = np.random.rand(100, 272)
        >>> first_deriv = apply_derivative(spectra, order=1)
    """
    if order not in [1, 2]:
        raise ValueError("Derivative order must be 1 or 2")

    if window_length % 2 == 0:
        raise ValueError("Window length must be odd")

    if polyorder >= window_length:
        raise ValueError("Polyorder must be less than window length")

    # Apply Savitzky-Golay derivative
    derivative = signal.savgol_filter(
        data, window_length, polyorder, deriv=order, axis=axis
    )

    logger.debug(f"Applied {order}st/nd derivative with window {window_length}")
    return derivative


def apply_baseline_correction(
    data: np.ndarray,
    order: int = 2,
    axis: int = -1,
    method: Literal["polynomial", "rubberband", "asls"] = "polynomial"
) -> np.ndarray:
    """Apply baseline correction to spectral data.

    Args:
        data: Input spectral data array.
        order: Polynomial order for baseline fitting.
        axis: Axis along which to apply correction.
        method: Baseline correction method.

    Returns:
        Baseline corrected data.

    Example:
        >>> spectra = np.random.rand(100, 272)
        >>> corrected = apply_baseline_correction(spectra)
    """
    if method == "polynomial":
        # Polynomial baseline fitting
        corrected = _polynomial_baseline(data, order, axis)

    elif method == "rubberband":
        # Rubberband baseline correction
        corrected = _rubberband_baseline(data, axis)

    elif method == "asls":
        # Asymmetric Least Squares
        corrected = _asls_baseline(data, axis)

    else:
        raise ValueError(f"Unknown baseline method: {method}")

    logger.debug(f"Applied {method} baseline correction with order {order}")
    return corrected


def _polynomial_baseline(
    data: np.ndarray, order: int, axis: int
) -> np.ndarray:
    """Fit and remove polynomial baseline."""
    # Get shape info
    shape = data.shape
    n_features = shape[axis]

    # Create x values
    x = np.arange(n_features)

    # Move axis to last position for easier processing
    data_moved = np.moveaxis(data, axis, -1)
    original_shape = data_moved.shape
    data_flat = data_moved.reshape(-1, n_features)

    # Fit polynomial to each spectrum
    corrected_flat = np.zeros_like(data_flat)

    for i in range(data_flat.shape[0]):
        spectrum = data_flat[i]

        # Fit polynomial
        coeffs = np.polyfit(x, spectrum, order)
        baseline = np.polyval(coeffs, x)

        # Subtract baseline
        corrected_flat[i] = spectrum - baseline

    # Reshape and move axis back
    corrected = corrected_flat.reshape(original_shape)
    corrected = np.moveaxis(corrected, -1, axis)

    return corrected


def _rubberband_baseline(data: np.ndarray, axis: int) -> np.ndarray:
    """Rubberband baseline correction using convex hull."""
    from scipy.spatial import ConvexHull

    # Move axis to last position
    data_moved = np.moveaxis(data, axis, -1)
    original_shape = data_moved.shape
    data_flat = data_moved.reshape(-1, original_shape[-1])
    n_features = original_shape[-1]

    corrected_flat = np.zeros_like(data_flat)
    x = np.arange(n_features)

    for i in range(data_flat.shape[0]):
        spectrum = data_flat[i]

        # Create points for convex hull
        points = np.column_stack([x, spectrum])

        try:
            # Compute convex hull
            hull = ConvexHull(points)

            # Find lower hull points
            lower_hull_indices = []
            for simplex in hull.simplices:
                if np.all(points[simplex, 1] <= spectrum[simplex[0]]):
                    lower_hull_indices.extend(simplex.tolist())

            lower_hull_indices = sorted(set(lower_hull_indices))

            if len(lower_hull_indices) < 2:
                # Fallback to simple linear baseline
                baseline = np.linspace(spectrum[0], spectrum[-1], n_features)
            else:
                # Interpolate baseline from lower hull
                baseline = np.interp(x, x[lower_hull_indices],
                                    spectrum[lower_hull_indices])

            corrected_flat[i] = spectrum - baseline

        except Exception:
            # Fallback to no correction if convex hull fails
            corrected_flat[i] = spectrum

    # Reshape and move axis back
    corrected = corrected_flat.reshape(original_shape)
    corrected = np.moveaxis(corrected, -1, axis)

    return corrected


def _asls_baseline(
    data: np.ndarray,
    axis: int,
    lam: float = 1e6,
    p: float = 0.01,
    niter: int = 10
) -> np.ndarray:
    """Asymmetric Least Squares baseline correction."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    # Move axis to last position
    data_moved = np.moveaxis(data, axis, -1)
    original_shape = data_moved.shape
    data_flat = data_moved.reshape(-1, original_shape[-1])
    n_features = original_shape[-1]

    # Prepare difference matrix
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_features, n_features - 2))
    D = lam * D.dot(D.T)

    corrected_flat = np.zeros_like(data_flat)

    for i in range(data_flat.shape[0]):
        spectrum = data_flat[i]
        w = np.ones(n_features)
        baseline = np.zeros(n_features)

        for _ in range(niter):
            W = sparse.diags(w, 0, shape=(n_features, n_features))
            Z = W + D
            baseline = spsolve(Z, w * spectrum)
            w = p * (spectrum > baseline) + (1 - p) * (spectrum <= baseline)

        corrected_flat[i] = spectrum - baseline

    # Reshape and move axis back
    corrected = corrected_flat.reshape(original_shape)
    corrected = np.moveaxis(corrected, -1, axis)

    return corrected


def apply_msc(
    data: np.ndarray,
    reference: Optional[np.ndarray] = None,
    axis: int = -1
) -> np.ndarray:
    """Apply Multiplicative Scatter Correction (MSC).

    MSC corrects for multiplicative and additive effects in spectra.

    Args:
        data: Input spectral data array.
        reference: Reference spectrum (if None, use mean spectrum).
        axis: Axis along which spectra are defined.

    Returns:
        MSC corrected data.

    Example:
        >>> spectra = np.random.rand(100, 272)
        >>> msc_spectra = apply_msc(spectra)
    """
    # Move axis to last position
    data_moved = np.moveaxis(data, axis, -1)
    original_shape = data_moved.shape
    data_flat = data_moved.reshape(-1, original_shape[-1])

    # Calculate reference spectrum if not provided
    if reference is None:
        reference = np.mean(data_flat, axis=0)
    else:
        if reference.ndim > 1:
            reference = reference.flatten()

    corrected_flat = np.zeros_like(data_flat)

    # Apply MSC to each spectrum
    for i in range(data_flat.shape[0]):
        spectrum = data_flat[i]

        # Fit linear regression: spectrum = a + b * reference
        # Using least squares
        A = np.vstack([reference, np.ones(len(reference))]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, spectrum, rcond=None)
        b, a = coeffs

        # Correct spectrum
        corrected_flat[i] = (spectrum - a) / b

    # Reshape and move axis back
    corrected = corrected_flat.reshape(original_shape)
    corrected = np.moveaxis(corrected, -1, axis)

    logger.debug("Applied MSC correction")
    return corrected


def apply_detrend(
    data: np.ndarray,
    axis: int = -1,
    type: Literal["linear", "constant"] = "linear"
) -> np.ndarray:
    """Apply detrending to remove linear or constant trends.

    Args:
        data: Input spectral data array.
        axis: Axis along which to detrend.
        type: Type of detrending ('linear' or 'constant').

    Returns:
        Detrended data.

    Example:
        >>> spectra = np.random.rand(100, 272)
        >>> detrended = apply_detrend(spectra)
    """
    detrended = signal.detrend(data, axis=axis, type=type)

    logger.debug(f"Applied {type} detrending")
    return detrended


def apply_normalization(
    data: np.ndarray,
    method: Literal["minmax", "max", "area", "vector"] = "minmax",
    axis: int = -1
) -> np.ndarray:
    """Apply normalization to spectral data.

    Args:
        data: Input spectral data array.
        method: Normalization method.
        axis: Axis along which to normalize.

    Returns:
        Normalized data.

    Example:
        >>> spectra = np.random.rand(100, 272)
        >>> normalized = apply_normalization(spectra, method="minmax")
    """
    if method == "minmax":
        # Min-max normalization to [0, 1]
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        normalized = (data - min_val) / range_val

    elif method == "max":
        # Normalize by maximum value
        max_val = np.max(np.abs(data), axis=axis, keepdims=True)
        max_val = np.where(max_val == 0, 1, max_val)
        normalized = data / max_val

    elif method == "area":
        # Area/sum normalization
        area = np.sum(np.abs(data), axis=axis, keepdims=True)
        area = np.where(area == 0, 1, area)
        normalized = data / area

    elif method == "vector":
        # Vector normalization (L2 norm)
        norm = np.linalg.norm(data, axis=axis, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        normalized = data / norm

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    logger.debug(f"Applied {method} normalization")
    return normalized