# Hyperseed CLI Complete Documentation

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Commands Overview](#commands-overview)
4. [Analyze Command](#analyze-command)
5. [Batch Command](#batch-command)
6. [Segment Command](#segment-command)
7. [Config Command](#config-command)
8. [Info Command](#info-command)
9. [Preprocessing Options](#preprocessing-options)
10. [Segmentation Options](#segmentation-options)
11. [Output Formats](#output-formats)
12. [Advanced Usage](#advanced-usage)
13. [Troubleshooting](#troubleshooting)

## Installation

```bash
# Install from source
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```bash
# Basic analysis with visualization
hyperseed analyze dataset/sample_data -o results.csv --export-plots

# Batch processing
hyperseed batch dataset/ --output-dir results/

# Generate configuration
hyperseed config --output my_config.yaml --preset minimal
```

## Commands Overview

The hyperseed CLI provides 5 main commands:

| Command | Purpose | Example |
|---------|---------|---------|
| `analyze` | Process single dataset | `hyperseed analyze dataset/sample -o output.csv` |
| `batch` | Process multiple datasets | `hyperseed batch datasets/ -o results/` |
| `segment` | Segmentation only | `hyperseed segment dataset/sample --visualize` |
| `config` | Generate configuration | `hyperseed config --preset minimal` |
| `info` | Show version/system info | `hyperseed info` |

## Analyze Command

The main command for processing hyperspectral seed imagery.

### Basic Syntax
```bash
hyperseed analyze INPUT_PATH [OPTIONS]
```

### All Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --output` | PATH | None | Output file path (CSV or HDF5) |
| `-c, --config` | PATH | None | Configuration file path (YAML) |
| `--preprocess` | CHOICE | standard | Preprocessing preset (minimal/standard/advanced/none) |
| `--segmentation` | CHOICE | watershed | Algorithm (threshold/watershed/connected/combined) |
| `--min-pixels` | INT | 200 | Minimum seed size in pixels |
| `--export-plots` | FLAG | False | Export visualization plots |
| `--export-mask` | FLAG | False | Export segmentation mask |
| `--no-outlier-removal` | FLAG | False | Disable automatic outlier removal |
| `-v, --verbose` | FLAG | False | Verbose output |
| `-d, --debug` | FLAG | False | Debug mode |

### Examples

#### Basic Analysis
```bash
hyperseed analyze dataset/sample_001 -o results.csv
```

#### With Visualizations
```bash
hyperseed analyze dataset/sample_001 \
    --output results.csv \
    --export-plots
```

#### Optimal Settings for Seeds
```bash
hyperseed analyze dataset/sample_data \
    --output seed_analysis.csv \
    --min-pixels 50 \
    --preprocess minimal \
    --segmentation watershed \
    --export-plots \
    --export-mask
```

#### Without Outlier Removal
```bash
hyperseed analyze dataset/sample_data \
    --output results.csv \
    --min-pixels 50 \
    --no-outlier-removal
```

#### Using Custom Configuration
```bash
hyperseed analyze dataset/sample \
    --output results.csv \
    --config custom_settings.yaml
```

#### HDF5 Output Format
```bash
hyperseed analyze dataset/sample \
    --output results.h5 \
    --export-plots
```

### Generated Files

For input `dataset/sample` with output `results.csv` and `--export-plots`:

1. **results.csv** - Extracted seed spectra with metadata
2. **results_distribution.png** - Spatial and size distribution
3. **results_segmentation.png** - Numbered seed visualization
4. **results_spectra.png** - Individual and mean spectra
5. **results_spectra_statistics.png** - Statistical analysis
6. **results_mask.npy** - Segmentation mask (if --export-mask)

## Batch Command

Process multiple datasets in parallel.

### Basic Syntax
```bash
hyperseed batch INPUT_DIR [OPTIONS]
```

### All Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --output-dir` | PATH | input_dir/results | Output directory |
| `-c, --config` | PATH | None | Configuration file (YAML) |
| `--parallel` | INT | 1 | Number of parallel workers |
| `--pattern` | TEXT | * | Pattern to match datasets |
| `--min-pixels` | INT | 200 | Minimum seed size |
| `--no-outlier-removal` | FLAG | False | Disable outlier removal |

### Examples

#### Process All Datasets
```bash
hyperseed batch dataset/
```

#### Custom Output Directory
```bash
hyperseed batch dataset/ --output-dir analysis_results/
```

#### Parallel Processing
```bash
hyperseed batch dataset/ \
    --output-dir results/ \
    --parallel 4
```

#### Filter by Pattern
```bash
# Process only SWIR_L* datasets
hyperseed batch dataset/ --pattern "SXX_X*"

# Process only specific datasets
hyperseed batch dataset/ --pattern "SYYY_YY*"
```

#### With Custom Settings
```bash
hyperseed batch dataset/ \
    --output-dir results/ \
    --min-pixels 50 \
    --config batch_config.yaml
```

### Batch Output Structure
```
results/
├── sample_001_spectra.csv
├── sample_001_distribution.png
├── sample_001_segmentation.png
├── sample_001_spectra.png
├── sample_001_spectra_statistics.png
├── sample_002_spectra.csv
├── sample_002_distribution.png
└── ...
```

## Segment Command

Perform segmentation only without spectral extraction.

### Basic Syntax
```bash
hyperseed segment INPUT_PATH [OPTIONS]
```

### All Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --output` | PATH | None | Output mask file path |
| `--algorithm` | CHOICE | watershed | Segmentation algorithm |
| `--min-pixels` | INT | 200 | Minimum seed size |
| `--visualize` | FLAG | False | Show segmentation visualization |

### Examples

#### Visualize Segmentation
```bash
hyperseed segment dataset/sample_data --visualize
```

#### Save Segmentation Mask
```bash
hyperseed segment dataset/sample_data \
    --output segmentation_mask.npy \
    --min-pixels 50
```

#### Test Different Algorithms
```bash
# Watershed (default)
hyperseed segment dataset/sample --algorithm watershed --visualize

# Threshold-based
hyperseed segment dataset/sample --algorithm threshold --visualize

# Connected components
hyperseed segment dataset/sample --algorithm connected --visualize

# Combined approach
hyperseed segment dataset/sample --algorithm combined --visualize
```

## Config Command

Generate configuration files with presets.

### Basic Syntax
```bash
hyperseed config [OPTIONS]
```

### All Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --output` | PATH | config.yaml | Output configuration file |
| `--preset` | CHOICE | standard | Preset (minimal/standard/advanced) |

### Examples

#### Generate Default Configuration
```bash
hyperseed config
```

#### Generate with Specific Preset
```bash
# For segmentation-focused work
hyperseed config --output seg_config.yaml --preset minimal

# For spectral analysis
hyperseed config --output spectral_config.yaml --preset standard

# For chemometric analysis
hyperseed config --output chemometric_config.yaml --preset advanced
```

### Configuration File Structure

```yaml
calibration:
  apply_calibration: true
  apply_bad_pixels: true
  clip_negative: true
  clip_max: 1.0

preprocessing:
  method: minimal
  snv: false
  smoothing: true
  smoothing_window: 11
  smoothing_polyorder: 3
  baseline_correction: false
  baseline_order: 2
  derivative: 0
  msc: false
  detrend: false

segmentation:
  algorithm: watershed
  min_pixels: 200
  max_pixels: null
  reject_overlapping: true
  threshold_method: otsu
  morphology_operations: true
  morphology_kernel_size: 3
  filter_border_seeds: false
  border_width: 2
  remove_outliers: true
  outlier_min_area: 50
  outlier_max_area: 2000
  outlier_iqr_lower: 1.5
  outlier_iqr_upper: 3.0
  outlier_eccentricity: 0.95
  outlier_solidity: 0.7
  use_shape_filtering: false

wavelength_selection:
  method: all
  ranges: null
  bands: null
  n_components: 20

output:
  format: csv
  include_plots: true
  include_coordinates: true
  include_metadata: true
  plot_format: png
  plot_dpi: 150
  csv_separator: ','

processing:
  device: auto
  parallel_workers: 4
  chunk_size: null
  memory_limit_gb: null
  use_mmap: true
```

### Editing Configuration

After generating, edit the YAML file to customize:

```bash
# Generate base config
hyperseed config --output custom.yaml --preset minimal

# Edit with your preferred editor
nano custom.yaml

# Use the custom config
hyperseed analyze dataset/sample --config custom.yaml
```

## Info Command

Display system and dependency information.

### Basic Syntax
```bash
hyperseed info
```

### Example Output
```
Hyperseed Information

┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Component       ┃ Version/Info ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Hyperseed       │        0.1.0 │
│ Python version  │       3.10.0 │
│ NumPy          │      1.24.3  │
│ SciPy          │      1.11.4  │
│ scikit-image   │      0.22.0  │
│ Spectral       │       0.23.1 │
└─────────────────┴──────────────┘
```

## Preprocessing Options

### Comparison Table

| Preset | SNV | Smoothing | Baseline | Derivative | MSC | Detrend | Best For |
|--------|-----|-----------|----------|------------|-----|---------|----------|
| **minimal** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | Segmentation |
| **standard** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | General analysis |
| **advanced** | ✅ | ✅ | ✅ | 2nd | ✅ | ✅ | Chemometrics |
| **none** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Raw data |

### Usage Examples

```bash
# Best for segmentation
hyperseed analyze dataset/sample --preprocess minimal

# Default balanced approach
hyperseed analyze dataset/sample --preprocess standard

# Maximum preprocessing
hyperseed analyze dataset/sample --preprocess advanced

# No preprocessing
hyperseed analyze dataset/sample --preprocess none
```

## Segmentation Options

### Available Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **watershed** | Marker-based watershed | Touching/overlapping seeds |
| **threshold** | Otsu thresholding | Well-separated seeds |
| **connected** | Connected components | Simple scenes |
| **combined** | Multi-method approach | Complex scenes |

### Usage Examples

```bash
# Default watershed
hyperseed analyze dataset/sample --segmentation watershed

# Simple thresholding
hyperseed analyze dataset/sample --segmentation threshold

# Connected components
hyperseed analyze dataset/sample --segmentation connected

# Combined approach
hyperseed analyze dataset/sample --segmentation combined
```

## Output Formats

### CSV Format

Default output format with seed metadata and spectra:

```csv
seed_id,index,centroid_y,centroid_x,area,eccentricity,solidity,band_995.00nm,band_999.14nm,...
1,0,234.5,156.2,435,0.34,0.92,0.234,0.237,...
2,1,345.6,234.1,421,0.28,0.94,0.229,0.232,...
```

### HDF5 Format

Hierarchical format for large datasets:

```bash
hyperseed analyze dataset/sample --output results.h5
```

Structure:
```
results.h5
├── spectra (n_seeds × n_bands array)
├── wavelengths (n_bands array)
├── seed_info (structured array)
└── metadata (attributes)
```

### Visualization Outputs

1. **Distribution Plot** (`*_distribution.png`)
   - Left: Spatial distribution with numbered seeds
   - Right: Area distribution histogram

2. **Segmentation Plot** (`*_segmentation.png`)
   - Left: Original image
   - Middle: Numbered seeds
   - Right: Boundary overlay

3. **Spectra Plot** (`*_spectra.png`)
   - Individual seed spectra
   - Mean spectrum (bold)
   - Standard deviation shading

4. **Statistics Plot** (`*_spectra_statistics.png`)
   - Top: All spectra overlaid
   - Bottom: Percentile ranges

## Advanced Usage

### Pipeline Integration

```bash
# Step 1: Generate configuration
hyperseed config --output pipeline.yaml --preset minimal

# Step 2: Process with custom config
hyperseed batch dataset/ --config pipeline.yaml --output-dir results/

# Step 3: Post-process results
python post_process.py results/
```

### Parallel Batch Processing

```bash
# Use all available cores
hyperseed batch dataset/ --parallel $(nproc)

# Limit to 4 cores
hyperseed batch dataset/ --parallel 4
```

### Custom Outlier Settings

Create custom configuration:
```yaml
segmentation:
  remove_outliers: true
  outlier_min_area: 100     # Larger minimum
  outlier_max_area: 1500    # Smaller maximum
  outlier_iqr_lower: 2.0    # Stricter lower bound
  outlier_iqr_upper: 2.5    # Stricter upper bound
  use_shape_filtering: true # Enable shape filtering
```

### Debugging Mode

```bash
# Enable verbose output
hyperseed analyze dataset/sample -v

# Enable debug mode with full traceback
hyperseed analyze dataset/sample -d

# Combine for maximum information
hyperseed analyze dataset/sample -v -d
```

## Troubleshooting

### Common Issues and Solutions

#### 1. No Seeds Detected
```bash
# Solution: Lower minimum pixel threshold
hyperseed analyze dataset/sample --min-pixels 50

# Try minimal preprocessing
hyperseed analyze dataset/sample --preprocess minimal
```

#### 2. Too Many False Detections
```bash
# Solution: Increase minimum pixels
hyperseed analyze dataset/sample --min-pixels 300

# Enable outlier removal (default)
hyperseed analyze dataset/sample  # Outlier removal is on by default
```

#### 3. Poor Segmentation Quality
```bash
# Try different algorithm
hyperseed analyze dataset/sample --segmentation watershed
hyperseed analyze dataset/sample --segmentation threshold

# Adjust preprocessing
hyperseed analyze dataset/sample --preprocess minimal
```

#### 4. Memory Issues with Large Datasets
```bash
# Process sequentially
hyperseed batch dataset/ --parallel 1

# Or process individually
for dir in dataset/*/; do
    name=$(basename "$dir")
    hyperseed analyze "$dir" -o "results/${name}.csv"
done
```

### Performance Tips

1. **For Fast Processing**:
   ```bash
   hyperseed analyze dataset/sample --preprocess none
   ```

2. **For Best Segmentation**:
   ```bash
   hyperseed analyze dataset/sample --preprocess minimal --min-pixels 50
   ```

3. **For Best Spectral Quality**:
   ```bash
   hyperseed analyze dataset/sample --preprocess standard
   ```

4. **For Batch Processing**:
   ```bash
   hyperseed batch dataset/ --parallel 4 --min-pixels 50
   ```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Processing error |

## Getting Help

```bash
# General help
hyperseed --help

# Command-specific help
hyperseed analyze --help
hyperseed batch --help
hyperseed segment --help
hyperseed config --help
hyperseed info --help
```

## Best Practices

### Key Features

- **Automatic Bad Pixel Correction**: The tool automatically detects and corrects bad pixels using interpolation when bad pixel maps are present in the dataset
- **Intelligent Outlier Removal**: Automatically removes reference objects and anomalies (enabled by default, can be disabled with `--no-outlier-removal`)
- **Multiple Preprocessing Options**: Choose from minimal, standard, advanced, or none based on your analysis needs
- **Comprehensive Visualizations**: Automatically generates distribution, segmentation, and spectral plots

### Recommended Workflow

1. **Test on Single Dataset**:
   ```bash
   hyperseed analyze dataset/sample_001 \
       --min-pixels 50 \
       --preprocess minimal \
       --export-plots
   ```

2. **Review Visualizations**:
   - Check segmentation quality
   - Verify outlier removal
   - Inspect spectral quality

3. **Adjust Settings if Needed**:
   ```bash
   hyperseed config --output optimized.yaml --preset minimal
   # Edit optimized.yaml as needed
   ```

4. **Run Batch Processing**:
   ```bash
   hyperseed batch dataset/ \
       --config optimized.yaml \
       --output-dir final_results/ \
       --parallel 4
   ```

### Quality Control Checklist

- ✅ Check distribution plots for outliers
- ✅ Verify segmentation accuracy in visualization
- ✅ Inspect spectral curves for noise/artifacts
- ✅ Confirm seed counts are reasonable
- ✅ Review mean reflectance values
- ✅ Check for processing warnings/errors
