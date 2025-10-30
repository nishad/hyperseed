# analyze

The main command for processing hyperspectral seed imagery.

## Synopsis

```bash
hyperseed analyze INPUT_PATH [OPTIONS]
```

## Description

The `analyze` command performs complete hyperspectral seed analysis:

1. Loads ENVI format hyperspectral data
2. Applies reflectance calibration with white/dark references
3. Performs spectral preprocessing
4. Segments individual seeds
5. Extracts spectral signatures
6. Exports results to CSV or HDF5

## Arguments

**`INPUT_PATH`** *(required)*
: Path to the hyperspectral dataset directory

## Options

### Output Options

**`-o, --output PATH`** *(required)*
: Output file path (`.csv` or `.h5`)

**`--export-plots`**
: Generate visualization plots (distribution, segmentation, spectra)

**`--export-mask`**
: Export segmentation mask as `.npy` file

### Processing Options

**`-c, --config PATH`**
: Path to YAML configuration file

**`--preprocess CHOICE`**
: Preprocessing preset: `minimal`, `standard`, `advanced`, `none`
: Default: `standard`

**`--segmentation CHOICE`**
: Segmentation algorithm: `threshold`, `watershed`, `connected`, `combined`
: Default: `watershed`

**`--min-pixels INT`**
: Minimum seed size in pixels
: Default: `200`

**`--no-outlier-removal`**
: Disable automatic outlier detection and removal

### Logging Options

**`-v, --verbose`**
: Enable verbose logging

**`-d, --debug`**
: Enable debug mode with detailed logging

## Examples

### Basic Analysis

```bash
hyperseed analyze dataset/sample_001 --output results.csv
```

### With Visualizations

```bash
hyperseed analyze dataset/sample_001 \
    --output results.csv \
    --export-plots
```

### Recommended Settings for Seeds

```bash
hyperseed analyze dataset/sample_data \
    --output seed_analysis.csv \
    --min-pixels 50 \
    --preprocess minimal \
    --export-plots
```

### Using Custom Configuration

```bash
hyperseed analyze dataset/sample \
    --output results.csv \
    --config custom_settings.yaml
```

### Disable Outlier Removal

```bash
hyperseed analyze dataset/sample_data \
    --output results.csv \
    --min-pixels 50 \
    --no-outlier-removal
```

## Output Files

For input `dataset/sample` with output `results.csv` and `--export-plots`:

1. **results.csv** - Extracted seed spectra with metadata
2. **results_distribution.png** - Spatial and size distribution
3. **results_segmentation.png** - Numbered seed visualization
4. **results_spectra.png** - Individual and mean spectra
5. **results_spectra_statistics.png** - Statistical analysis
6. **results_mask.npy** - Segmentation mask (if `--export-mask`)

## CSV Output Format

```csv
seed_id,index,centroid_y,centroid_x,area,eccentricity,solidity,band_1000nm,band_1005nm,...
1,0,234.5,156.2,435,0.34,0.92,0.234,0.237,...
2,1,345.6,234.1,421,0.28,0.94,0.229,0.232,...
```

Columns:
- **seed_id**: Sequential seed identifier
- **index**: Original label from segmentation
- **centroid_y, centroid_x**: Seed center coordinates
- **area**: Seed area in pixels
- **eccentricity**: Shape eccentricity (0=circle, 1=line)
- **solidity**: Shape solidity (convex hull ratio)
- **band_***: Reflectance values at each wavelength

## See Also

- **[batch](batch.md)**: Process multiple datasets
- **[Configuration Guide](../getting-started/configuration.md)**: Detailed configuration options
- **[Quick Start](../getting-started/quick-start.md)**: Tutorial walkthrough
