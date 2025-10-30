# Configuration

Learn how to customize Hyperseed's analysis pipeline using configuration files.

## Overview

Hyperseed can be configured in three ways:

1. **Command-line arguments** (quick, one-off adjustments)
2. **YAML configuration files** (reproducible, shareable settings)
3. **Python API** (programmatic control)

This guide focuses on YAML configuration files, which provide the most flexibility and reproducibility.

## Generating a Configuration File

Generate a configuration template:

```bash
hyperseed config --output my_config.yaml --preset minimal
```

Available presets:

| Preset | Description | Use Case |
|--------|-------------|----------|
| `minimal` | Minimal preprocessing | Best for segmentation (recommended) |
| `standard` | Moderate preprocessing | Balanced for most applications |
| `advanced` | Full preprocessing pipeline | Maximum spectral transformation |
| `none` | No preprocessing | Raw data analysis |

## Configuration Structure

A complete configuration file has four main sections:

```yaml
calibration:    # White/dark reference correction
preprocessing:  # Spectral preprocessing
segmentation:   # Seed detection and isolation
output:         # Results and visualization
```

## Calibration Settings

Configure white and dark reference calibration:

```yaml
calibration:
  apply_calibration: true    # Enable/disable calibration
  clip_negative: true        # Clip negative reflectance values to 0
  clip_max: 1.0             # Clip maximum reflectance to 1.0
  interpolate_bad_pixels: true  # Automatically fix bad pixels
```

### Parameters Explained

**`apply_calibration`** *(boolean, default: true)*
: Enable or disable reflectance calibration using white/dark references.

**`clip_negative`** *(boolean, default: true)*
: Clip negative reflectance values to zero (physically meaningful).

**`clip_max`** *(float, default: 1.0)*
: Maximum reflectance value to clip at (1.0 = 100% reflectance).

**`interpolate_bad_pixels`** *(boolean, default: true)*
: Automatically detect and interpolate bad pixels in references.

!!! tip "Recommended Settings"
    Keep `apply_calibration: true` and `interpolate_bad_pixels: true` for best results.

## Preprocessing Settings

Configure spectral preprocessing methods:

```yaml
preprocessing:
  method: minimal              # Preset: minimal, standard, advanced, none

  # Individual preprocessing options
  snv: false                   # Standard Normal Variate
  smoothing: true              # Savitzky-Golay smoothing
  smoothing_window: 11         # Smoothing window size (must be odd, range: 3-51)
  smoothing_polyorder: 3       # Polynomial order for smoothing (range: 1-5)
  baseline_correction: false   # Baseline correction
  baseline_order: 2            # Polynomial order for baseline (range: 1-5)
  derivative: 0                # Derivative order (0, 1, or 2)
  msc: false                   # Multiplicative Scatter Correction
  detrend: false               # Remove linear trends
```

### Preprocessing Methods

The `method` parameter provides preset combinations:

=== "minimal (Recommended)"
    ```yaml
    preprocessing:
      method: minimal
      # Enables: smoothing only
      # Best for: Segmentation quality
    ```

=== "standard"
    ```yaml
    preprocessing:
      method: standard
      # Enables: SNV, smoothing, baseline correction
      # Best for: Balanced preprocessing
    ```

=== "advanced"
    ```yaml
    preprocessing:
      method: advanced
      # Enables: SNV, smoothing, baseline correction, 2nd derivative, MSC, detrend
      # Best for: Chemometrics and advanced spectral analysis
    ```

=== "none"
    ```yaml
    preprocessing:
      method: none
      # Enables: nothing
      # Best for: Raw data analysis
    ```

### Individual Parameters

**`snv`** *(boolean, default: false)*
: Standard Normal Variate - removes multiplicative scatter effects.

**`smoothing`** *(boolean, default: true)*
: Savitzky-Golay smoothing to reduce noise.

**`smoothing_window`** *(int, default: 11)*
: Window size for smoothing (must be odd number, range: 3-51).

**`smoothing_polyorder`** *(int, default: 3)*
: Polynomial order for Savitzky-Golay filter (range: 1-5).

**`baseline_correction`** *(boolean, default: false)*
: Remove baseline drift in spectra using polynomial fitting.

**`baseline_order`** *(int, default: 2)*
: Polynomial order for baseline correction (range: 1-5).

**`derivative`** *(int, default: 0)*
: Derivative order: 0 (none), 1 (first), 2 (second). Computed using Savitzky-Golay filter.

**`msc`** *(boolean, default: false)*
: Multiplicative Scatter Correction - corrects for scatter effects by normalizing to a reference spectrum.

**`detrend`** *(boolean, default: false)*
: Remove linear trends from spectra.

!!! warning "Preprocessing and Segmentation"
    Heavy preprocessing can reduce segmentation quality. Use `minimal` for best segmentation results.

## Segmentation Settings

Configure seed detection and isolation:

```yaml
segmentation:
  algorithm: watershed         # Algorithm choice (threshold/watershed/connected/combined)
  min_pixels: 200             # Minimum seed size in pixels
  max_pixels: null            # Maximum seed size (null = no limit)
  reject_overlapping: true    # Reject overlapping seeds

  # Thresholding options
  threshold_method: otsu      # Method for threshold algorithm (otsu/adaptive/manual)

  # Morphological operations
  morphology_operations: true # Apply morphological cleanup
  morphology_kernel_size: 3   # Kernel size for morphology (1-21)

  # Border filtering
  filter_border_seeds: false  # Remove seeds touching image borders
  border_width: 2            # Border region width in pixels (0-50)

  # Outlier removal (automatic)
  remove_outliers: true       # Enable outlier detection
  outlier_min_area: 50        # Minimum seed area threshold
  outlier_max_area: 2000      # Maximum seed area threshold
  outlier_iqr_lower: 1.5      # IQR multiplier for lower bound
  outlier_iqr_upper: 3.0      # IQR multiplier for upper bound

  # Shape-based filtering (optional)
  use_shape_filtering: false  # Enable shape-based outlier removal
  outlier_eccentricity: 0.95  # Maximum eccentricity (elongation, 0-1)
  outlier_solidity: 0.7       # Minimum solidity (regularity, 0-1)
```

### Segmentation Algorithms

**`threshold`**
: Simple intensity-based thresholding using Otsu or adaptive methods
: **Pros**: Fast, simple | **Cons**: May not separate touching seeds well

**`watershed`** *(recommended)*
: Watershed segmentation with distance transform
: **Pros**: Effectively separates touching seeds | **Cons**: Slightly slower

**`connected`**
: Connected components labeling with shape filtering
: **Pros**: Simple, fast for well-separated seeds | **Cons**: Cannot separate touching seeds

**`combined`**
: Combines multiple algorithms using consensus voting
: **Pros**: Most robust, best accuracy | **Cons**: Slowest

### Algorithm Parameters

**`algorithm`** *(choice, default: "watershed")*
: Segmentation algorithm to use. Choose based on your seed arrangement.

**`threshold_method`** *(choice, default: "otsu")*
: Thresholding method for threshold algorithm:
: - `otsu`: Automatic global threshold (recommended)
: - `adaptive`: Local adaptive thresholding for uneven illumination
: - `manual`: Requires manual threshold value

**`morphology_operations`** *(boolean, default: true)*
: Apply morphological operations (closing, opening) to clean up segmentation.

**`morphology_kernel_size`** *(int, default: 3, range: 1-21)*
: Size of kernel for morphological operations. Larger values = more aggressive cleanup.

**`filter_border_seeds`** *(boolean, default: false)*
: Remove seeds touching image borders (useful for incomplete seeds at edges).

**`border_width`** *(int, default: 2, range: 0-50)*
: Width of border region to check when filtering border seeds.

### Size Filtering

**`min_pixels`** *(int, default: 200, range: 10-10000)*
: Minimum seed size in pixels. Objects smaller than this are filtered out.

**`max_pixels`** *(int, default: None, range: 10-100000)*
: Maximum seed size in pixels. Objects larger than this are filtered out. Set to `null` for no upper limit.

**`reject_overlapping`** *(boolean, default: true)*
: Reject seeds that overlap or touch each other.

!!! tip "Choosing min_pixels"
    The default of 200 pixels works for most seeds. Adjust based on your imaging resolution:
    - Small seeds or high resolution: `min_pixels: 100`
    - Medium seeds (default): `min_pixels: 200`
    - Large seeds or low resolution: `min_pixels: 300`

### Outlier Removal

Automatic outlier removal uses a three-step process to eliminate:
- Reference objects (calibration targets)
- Anomalous large/small objects
- Debris and artifacts
- Irregularly shaped objects (optional)

**`remove_outliers`** *(boolean, default: true)*
: Enable automatic outlier detection and removal. Highly recommended.

**Step 1: Absolute Area Bounds**

**`outlier_min_area`** *(int, default: 50)*
: Minimum area threshold. Seeds below this are always removed.

**`outlier_max_area`** *(int, default: 2000)*
: Maximum area threshold. Seeds above this are always removed.

**Step 2: IQR-Based Filtering**

Uses Interquartile Range (IQR) to detect statistical outliers in seed size distribution.

**`outlier_iqr_lower`** *(float, default: 1.5)*
: IQR multiplier for lower bound calculation. Lower bound = Q1 - (iqr_lower × IQR)

**`outlier_iqr_upper`** *(float, default: 3.0)*
: IQR multiplier for upper bound calculation. Upper bound = Q3 + (iqr_upper × IQR)

!!! info "Asymmetric IQR Multipliers"
    The default uses asymmetric multipliers (1.5 lower, 3.0 upper) because large outliers (reference objects) are more common than small outliers.

**Step 3: Shape-Based Filtering** (Optional)

**`use_shape_filtering`** *(boolean, default: false)*
: Enable shape-based outlier removal. Filters seeds based on eccentricity and solidity.

**`outlier_eccentricity`** *(float, default: 0.95, range: 0-1)*
: Maximum eccentricity (elongation). 0 = circle, 1 = line. Seeds above threshold are removed.

**`outlier_solidity`** *(float, default: 0.7, range: 0-1)*
: Minimum solidity (area/convex hull area). Measures shape regularity. Seeds below threshold are removed.

## Output Settings

Configure results and visualizations:

```yaml
output:
  format: csv                  # Output format (currently only csv)
  include_plots: true          # Generate visualization plots
  include_coordinates: true    # Include spatial coordinates
  include_morphology: true     # Include shape properties
  save_mask: false            # Save segmentation mask
```

**`include_plots`** *(boolean, default: true)*
: Generate distribution, segmentation, and spectral plots.

**`include_coordinates`** *(boolean, default: true)*
: Include centroid coordinates in output CSV.

**`include_morphology`** *(boolean, default: true)*
: Include morphological properties (area, eccentricity, solidity).

## Using Configuration Files

### From Command Line

```bash
# Generate config
hyperseed config --output my_config.yaml --preset minimal

# Use config for analysis
hyperseed analyze dataset/sample_001 --config my_config.yaml

# Use config for batch processing
hyperseed batch dataset/ --config my_config.yaml --output-dir results/
```

### Overriding Config with CLI Arguments

Command-line arguments override configuration file settings:

```bash
# Config file has min_pixels: 50, but override to 100
hyperseed analyze dataset/sample_001 \
    --config my_config.yaml \
    --min-pixels 100
```

## Example Configurations

### Example 1: High-Quality Segmentation

```yaml
calibration:
  apply_calibration: true
  clip_negative: true
  clip_max: 1.0

preprocessing:
  method: minimal
  smoothing: true
  smoothing_window: 11

segmentation:
  algorithm: watershed
  min_pixels: 50
  remove_outliers: true

output:
  include_plots: true
  include_coordinates: true
```

### Example 2: Advanced Spectral Analysis

```yaml
calibration:
  apply_calibration: true
  clip_negative: true
  clip_max: 1.0

preprocessing:
  method: advanced
  snv: true
  smoothing: true
  smoothing_window: 11
  baseline_correction: true
  derivative: 1

segmentation:
  algorithm: combined
  min_pixels: 50
  remove_outliers: true

output:
  include_plots: true
  include_coordinates: true
  save_mask: true
```

### Example 3: Fast Batch Processing

```yaml
calibration:
  apply_calibration: true
  clip_negative: true

preprocessing:
  method: none  # Skip preprocessing for speed

segmentation:
  algorithm: threshold  # Fastest algorithm
  min_pixels: 50
  remove_outliers: false  # Skip outlier detection

output:
  include_plots: false  # Skip plots for speed
  include_coordinates: true
```

## Validation

Hyperseed validates your configuration and provides helpful error messages:

```bash
$ hyperseed analyze dataset/sample_001 --config invalid_config.yaml

Error: Invalid configuration
- smoothing_window must be an odd number (got 10)
- min_pixels must be greater than 0 (got -5)
```

## Next Steps

Now that you understand configuration:

- **[User Guide](../user-guide/index.md)**: Learn about specific features
- **[Preprocessing Guide](../user-guide/preprocessing.md)**: Deep dive into preprocessing methods
- **[Segmentation Guide](../user-guide/segmentation.md)**: Deep dive into segmentation algorithms
- **[CLI Reference](../cli-reference/index.md)**: Complete command documentation
