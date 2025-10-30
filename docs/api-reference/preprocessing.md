# Preprocessing API

Programmatic interface for spectral preprocessing.

## PreprocessingPipeline

The `PreprocessingPipeline` class provides a complete preprocessing pipeline with configurable methods.

### Basic Usage

```python
from hyperseed.core.preprocessing.pipeline import PreprocessingPipeline
from hyperseed.config.settings import Settings

# Create pipeline with default settings
settings = Settings()
pipeline = PreprocessingPipeline(settings.preprocessing)

# Apply preprocessing
preprocessed_data = pipeline.fit_transform(calibrated_data)
```

### With Custom Configuration

```python
from hyperseed.config.settings import PreprocessingConfig

# Create custom configuration
config = PreprocessingConfig(
    method="custom",
    snv=True,
    smoothing=True,
    smoothing_window=15,
    smoothing_polyorder=3,
    baseline_correction=True,
    baseline_order=2,
    derivative=1,
    msc=False,
    detrend=False
)

# Create pipeline
pipeline = PreprocessingPipeline(config)

# Fit and transform
preprocessed = pipeline.fit_transform(data)
```

### Methods

#### `__init__(config: PreprocessingConfig)`

Initialize the preprocessing pipeline with configuration.

**Parameters:**
- `config`: PreprocessingConfig object with preprocessing settings

**Example:**
```python
from hyperseed.config.settings import PreprocessingConfig

config = PreprocessingConfig(method="standard")
pipeline = PreprocessingPipeline(config)
```

#### `fit(data: np.ndarray) -> PreprocessingPipeline`

Fit the preprocessing pipeline on data (e.g., compute MSC reference spectrum).

**Parameters:**
- `data`: Hyperspectral data (Y, X, Bands) or (Samples, Bands)

**Returns:**
- Self (for method chaining)

**Example:**
```python
pipeline.fit(training_data)
```

#### `transform(data: np.ndarray) -> np.ndarray`

Apply preprocessing transformations to data.

**Parameters:**
- `data`: Hyperspectral data to transform

**Returns:**
- Preprocessed data with same shape as input

**Example:**
```python
preprocessed = pipeline.transform(test_data)
```

**Note:** Must call `fit()` before `transform()` if using MSC.

#### `fit_transform(data: np.ndarray) -> np.ndarray`

Fit the pipeline and transform data in one step.

**Parameters:**
- `data`: Hyperspectral data to fit and transform

**Returns:**
- Preprocessed data

**Example:**
```python
preprocessed = pipeline.fit_transform(data)
```

**This is equivalent to:**
```python
pipeline.fit(data)
preprocessed = pipeline.transform(data)
```

#### `get_step_names() -> list[str]`

Get list of enabled preprocessing steps.

**Returns:**
- List of step names that will be applied

**Example:**
```python
steps = pipeline.get_step_names()
print(f"Preprocessing steps: {', '.join(steps)}")
# Output: Preprocessing steps: SNV, Smoothing, Baseline Correction
```

#### `describe() -> str`

Get human-readable description of the pipeline.

**Returns:**
- String describing the preprocessing configuration

**Example:**
```python
print(pipeline.describe())
# Output: Preprocessing Pipeline:
#   - SNV: enabled
#   - Smoothing: Savitzky-Golay (window=11, polyorder=3)
#   - Baseline Correction: Polynomial (order=2)
#   ...
```

## Individual Preprocessing Functions

For more control, use individual preprocessing functions from `hyperseed.core.preprocessing.methods`.

### apply_snv

Standard Normal Variate transformation.

```python
from hyperseed.core.preprocessing.methods import apply_snv

# Apply SNV to spectra
snv_data = apply_snv(data, axis=-1)
```

**Parameters:**
- `data`: Input array
- `axis`: Axis along which to apply SNV (default: -1)

**Returns:**
- SNV-transformed data

### apply_smoothing

Smooth spectra using various methods.

```python
from hyperseed.core.preprocessing.methods import apply_smoothing

# Savitzky-Golay smoothing (default)
smoothed = apply_smoothing(
    data,
    window_length=11,
    polyorder=3,
    method="savgol"
)

# Moving average
smoothed = apply_smoothing(
    data,
    window_length=11,
    method="moving_average"
)

# Gaussian filter
smoothed = apply_smoothing(
    data,
    window_length=11,
    method="gaussian"
)
```

**Parameters:**
- `data`: Input array
- `window_length`: Window size (must be odd)
- `polyorder`: Polynomial order for Savitzky-Golay (default: 3)
- `method`: "savgol", "moving_average", or "gaussian" (default: "savgol")
- `axis`: Axis along which to smooth (default: -1)

**Returns:**
- Smoothed data

### apply_derivative

Compute spectral derivatives.

```python
from hyperseed.core.preprocessing.methods import apply_derivative

# First derivative
first_deriv = apply_derivative(
    data,
    order=1,
    window_length=11,
    polyorder=3
)

# Second derivative
second_deriv = apply_derivative(
    data,
    order=2,
    window_length=11,
    polyorder=3
)
```

**Parameters:**
- `data`: Input array
- `order`: Derivative order (1 or 2)
- `window_length`: Window size for Savitzky-Golay (default: 11)
- `polyorder`: Polynomial order (default: 3)
- `axis`: Axis along which to compute (default: -1)

**Returns:**
- Derivative spectra

### apply_baseline_correction

Remove baseline from spectra.

```python
from hyperseed.core.preprocessing.methods import apply_baseline_correction

# Polynomial baseline
corrected = apply_baseline_correction(
    data,
    order=2,
    method="polynomial"
)

# Rubberband baseline
corrected = apply_baseline_correction(
    data,
    method="rubberband"
)

# ASLS baseline
corrected = apply_baseline_correction(
    data,
    method="asls"
)
```

**Parameters:**
- `data`: Input array
- `order`: Polynomial order (for polynomial method, default: 2)
- `method`: "polynomial", "rubberband", or "asls" (default: "polynomial")
- `axis`: Axis along which to correct (default: -1)

**Returns:**
- Baseline-corrected data

**Methods:**
- **polynomial**: Fits polynomial and subtracts
- **rubberband**: Convex hull method
- **asls**: Asymmetric Least Squares (lam=1e6, p=0.01, niter=10)

### apply_msc

Multiplicative Scatter Correction.

```python
from hyperseed.core.preprocessing.methods import apply_msc

# Use mean spectrum as reference
corrected = apply_msc(data)

# Use custom reference
reference = data[0, :]  # Use first spectrum
corrected = apply_msc(data, reference=reference)
```

**Parameters:**
- `data`: Input array
- `reference`: Reference spectrum (if None, uses mean)
- `axis`: Axis along which to apply (default: -1)

**Returns:**
- MSC-corrected data

### apply_detrend

Remove linear trends.

```python
from hyperseed.core.preprocessing.methods import apply_detrend

# Linear detrending
detrended = apply_detrend(data, type="linear")

# Constant detrending (remove mean)
detrended = apply_detrend(data, type="constant")
```

**Parameters:**
- `data`: Input array
- `type`: "linear" or "constant" (default: "linear")
- `axis`: Axis along which to detrend (default: -1)

**Returns:**
- Detrended data

### apply_normalization

Normalize spectra (not exposed in config, but available via API).

```python
from hyperseed.core.preprocessing.methods import apply_normalization

# Min-max normalization [0, 1]
normalized = apply_normalization(data, method="minmax")

# Max normalization
normalized = apply_normalization(data, method="max")

# Area normalization
normalized = apply_normalization(data, method="area")

# Vector (L2) normalization
normalized = apply_normalization(data, method="vector")
```

**Parameters:**
- `data`: Input array
- `method`: "minmax", "max", "area", or "vector" (default: "minmax")
- `axis`: Axis along which to normalize (default: -1)

**Returns:**
- Normalized data

## Complete Example

```python
import numpy as np
from hyperseed.core.io.envi_reader import ENVIReader
from hyperseed.core.calibration.reflectance import ReflectanceCalibrator
from hyperseed.core.preprocessing.pipeline import PreprocessingPipeline
from hyperseed.config.settings import PreprocessingConfig

# Load data
reader = ENVIReader("path/to/data.hdr")
data = reader.read_data()
wavelengths = reader.get_wavelengths()

# Calibrate
calibrator = ReflectanceCalibrator(clip_negative=True, clip_max=1.0)
calibrated, _ = calibrator.calibrate_from_directory("path/to/dataset")

# Configure preprocessing
config = PreprocessingConfig(
    method="custom",
    snv=True,
    smoothing=True,
    smoothing_window=15,
    baseline_correction=True,
    derivative=1
)

# Preprocess
pipeline = PreprocessingPipeline(config)
preprocessed = pipeline.fit_transform(calibrated)

print(f"Original shape: {calibrated.shape}")
print(f"Preprocessed shape: {preprocessed.shape}")
print(f"Preprocessing steps: {', '.join(pipeline.get_step_names())}")

# Use preprocessed data for segmentation or analysis
```

## Using Individual Functions

```python
from hyperseed.core.preprocessing.methods import (
    apply_snv,
    apply_smoothing,
    apply_baseline_correction,
    apply_derivative
)

# Manual preprocessing pipeline
data_snv = apply_snv(calibrated, axis=-1)
data_smooth = apply_smoothing(data_snv, window_length=15, polyorder=3)
data_baseline = apply_baseline_correction(data_smooth, order=2)
data_deriv = apply_derivative(data_baseline, order=1)

# Now use data_deriv for analysis
```

## Notes

- All preprocessing functions preserve data shape
- Operations are applied along the last axis by default (spectral axis)
- For hyperspectral cubes (Y, X, Bands), reshaping may be needed
- The PreprocessingPipeline handles reshaping automatically
- MSC requires fitting before transformation (computes reference spectrum)
- Normalization is available via API but not exposed in configuration

## See Also

- **[Preprocessing Guide](../user-guide/preprocessing.md)**: Detailed method descriptions
- **[Configuration](../getting-started/configuration.md#preprocessing-settings)**: Configuration reference
- **[CLI Reference](../cli-reference/extract.md)**: Command-line preprocessing
