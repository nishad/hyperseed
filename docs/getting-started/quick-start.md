# Quick Start

Get started with Hyperseed in 5 minutes! This guide walks you through your first analysis.

## Prerequisites

Before starting, ensure you have:

- [x] Installed Hyperseed (`pip install hyperseed`)
- [x] Hyperspectral data in ENVI format with white/dark references

## Your First Analysis

### Step 1: Prepare Your Data

Ensure your dataset follows this structure:

```
dataset/sample_001/
├── capture/
│   ├── data.raw              # Main hyperspectral data
│   ├── data.hdr              # ENVI header file
│   ├── WHITEREF_data.raw     # White reference
│   ├── WHITEREF_data.hdr
│   ├── DARKREF_data.raw      # Dark reference
│   └── DARKREF_data.hdr
```

!!! info "Data Structure"
    See [Data Preparation](../user-guide/data-preparation.md) for detailed information about expected data structure.

### Step 2: Run Basic Analysis

Run your first analysis with default settings:

```bash
hyperseed analyze dataset/sample_001 --output results.csv
```

This command:
- ✅ Loads the hyperspectral data
- ✅ Applies automatic calibration
- ✅ Segments individual seeds
- ✅ Extracts spectral signatures
- ✅ Saves results to `results.csv`

### Step 3: View Results

The analysis generates a CSV file with spectral data:

```csv
seed_id,index,centroid_y,centroid_x,area,eccentricity,solidity,band_1000nm,band_1005nm,...
1,0,234.5,156.2,435,0.34,0.92,0.234,0.237,...
2,1,345.6,234.1,421,0.28,0.94,0.229,0.232,...
```

Each row represents one seed with:
- **Spatial information**: centroid coordinates
- **Morphological properties**: area, eccentricity, solidity
- **Spectral data**: reflectance values for each wavelength band

## Adding Visualizations

To generate plots along with the CSV data:

```bash
hyperseed analyze dataset/sample_001 \
    --output results.csv \
    --export-plots
```

This creates four visualization files:

1. **`sample_001_distribution.png`**: Spatial and area distribution
2. **`sample_001_segmentation.png`**: Numbered seeds with boundaries
3. **`sample_001_spectra.png`**: Individual spectral curves
4. **`sample_001_spectra_statistics.png`**: Statistical analysis

!!! tip "Recommended First Run"
    Always use `--export-plots` on your first analysis to visually verify the segmentation quality.

## Recommended Settings

For optimal results with seed analysis:

```bash
hyperseed analyze dataset/sample_001 \
    --output results.csv \
    --min-pixels 50 \
    --preprocess minimal \
    --export-plots
```

Parameters explained:

- `--min-pixels 50`: Filter out objects smaller than 50 pixels (reduces noise)
- `--preprocess minimal`: Light preprocessing optimized for segmentation
- `--export-plots`: Generate visualization plots

## Batch Processing

To process multiple datasets:

```bash
# Process all datasets in the directory
hyperseed batch dataset/ \
    --output-dir results/ \
    --min-pixels 50
```

This will:
- Process all subdirectories in `dataset/`
- Save results to `results/` directory
- Apply consistent settings to all datasets

## Common Options

Here are the most commonly used options:

| Option | Description | Example |
|--------|-------------|---------|
| `--output` | Output CSV file path | `results.csv` |
| `--export-plots` | Generate visualization plots | *(flag)* |
| `--min-pixels` | Minimum seed size in pixels | `50` |
| `--preprocess` | Preprocessing method | `minimal`, `standard`, `advanced`, `none` |
| `--no-outlier-removal` | Disable automatic outlier removal | *(flag)* |
| `--config` | Use custom configuration file | `config.yaml` |

## What's Next?

### Customize Your Analysis

Learn how to create custom configurations:

```bash
# Generate a configuration template
hyperseed config --output my_config.yaml --preset minimal

# Use your custom configuration
hyperseed analyze dataset/sample_001 --config my_config.yaml
```

See [Configuration Guide →](configuration.md)

### Advanced Workflows

Explore advanced features:

- **[Preprocessing Options](../user-guide/preprocessing.md)**: SNV, derivatives, baseline correction
- **[Segmentation Algorithms](../user-guide/segmentation.md)**: Threshold, watershed, combined methods
- **[Batch Processing](../user-guide/batch-processing.md)**: Processing multiple datasets efficiently

### Command-Line Reference

For complete command documentation:

- **[CLI Reference](../cli-reference/index.md)**: All commands and options
- **[Analyze Command](../cli-reference/extract.md)**: Detailed analysis options
- **[Batch Command](../cli-reference/batch.md)**: Batch processing guide

## Example: Complete Workflow

Here's a complete example workflow:

```bash
# 1. Analyze a single dataset with visualizations
hyperseed analyze dataset/sample_001 \
    --output results/sample_001.csv \
    --min-pixels 50 \
    --preprocess minimal \
    --export-plots

# 2. Review the plots and adjust if needed

# 3. Process all datasets with the same settings
hyperseed batch dataset/ \
    --output-dir results/ \
    --min-pixels 50 \
    --preprocess minimal

# 4. Results are now in results/ directory
ls results/
# sample_001.csv  sample_002.csv  sample_003.csv  ...
```

## Getting Help

Need assistance?

```bash
# Get general help
hyperseed --help

# Get help for specific command
hyperseed analyze --help
hyperseed batch --help
```

Or check the [Troubleshooting Guide](../user-guide/troubleshooting.md) for common issues.

---

**Congratulations!** You've completed your first Hyperseed analysis. Continue to the [Configuration Guide](configuration.md) to learn about customization options.
