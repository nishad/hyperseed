# batch

Process multiple hyperspectral datasets sequentially.

## Synopsis

```bash
hyperseed batch INPUT_DIR [OPTIONS]
```

## Description

The `batch` command processes multiple datasets sequentially (one after another) with consistent settings. It applies the same analysis pipeline to each dataset and saves results to a structured output directory.

## Arguments

### INPUT_DIR

Directory containing multiple dataset subdirectories.

**Required:** Yes

**Format:** Each subdirectory should contain:
- `capture/data.raw` and `capture/data.hdr` (main data)
- `capture/WHITEREF_data.raw` and `.hdr` (white reference)
- `capture/DARKREF_data.raw` and `.hdr` (dark reference)

**Example:**
```
datasets/
├── sample_001/
│   └── capture/
│       ├── data.raw, data.hdr
│       ├── WHITEREF_data.raw, WHITEREF_data.hdr
│       └── DARKREF_data.raw, DARKREF_data.hdr
├── sample_002/
│   └── capture/
│       └── ...
└── sample_003/
    └── capture/
        └── ...
```

## Options

### -o, --output-dir PATH

Output directory for results.

**Type:** Path
**Default:** `INPUT_DIR/results`

All output files are saved to this directory with dataset names as prefixes.

**Example:**
```bash
hyperseed batch datasets/ --output-dir analysis_results/
```

### -c, --config PATH

Path to YAML configuration file.

**Type:** Path
**Default:** None (uses default settings)

Applies consistent preprocessing, segmentation, and output settings across all datasets.

**Example:**
```bash
hyperseed batch datasets/ --config batch_config.yaml
```

### --pattern TEXT

Pattern to match dataset directories (glob-style).

**Type:** Text
**Default:** `*` (matches all subdirectories)

Use glob patterns to filter which datasets to process.

**Example:**
```bash
# Process only datasets starting with "sample_"
hyperseed batch datasets/ --pattern "sample_*"

# Process only SWIR datasets
hyperseed batch datasets/ --pattern "SWIR_*"

# Process specific range
hyperseed batch datasets/ --pattern "sample_00[1-5]"
```

### --min-pixels INTEGER

Minimum seed size in pixels.

**Type:** Integer
**Default:** 200
**Range:** 10-10000

Overrides the min_pixels setting from configuration.

**Example:**
```bash
hyperseed batch datasets/ --min-pixels 150
```

### --no-outlier-removal

Disable automatic outlier removal.

**Type:** Flag (boolean)
**Default:** False (outlier removal enabled)

Disables outlier detection and removal for all datasets.

**Example:**
```bash
hyperseed batch datasets/ --no-outlier-removal
```

## Complete Examples

### Basic Batch Processing

```bash
hyperseed batch datasets/
```

**What it does:**
1. Finds all subdirectories in `datasets/`
2. Processes each sequentially
3. Saves results to `datasets/results/`

### Custom Output Directory

```bash
hyperseed batch datasets/ --output-dir analysis_results/
```

**Output location:** `analysis_results/`

### Filter by Pattern

```bash
# Process only datasets starting with "sample_"
hyperseed batch datasets/ --pattern "sample_*"

# Process only specific samples
hyperseed batch datasets/ --pattern "sample_00[1-5]"
```

### With Configuration File

```bash
hyperseed batch datasets/ \
    --config batch_config.yaml \
    --output-dir results/
```

**batch_config.yaml:**
```yaml
preprocessing:
  method: minimal  # Fast processing for batch

segmentation:
  algorithm: watershed
  min_pixels: 200
  remove_outliers: true

output:
  format: csv
  include_plots: true
```

### Override Settings

```bash
# Use config but override min_pixels
hyperseed batch datasets/ \
    --config batch_config.yaml \
    --min-pixels 150 \
    --output-dir results/
```

## Output Structure

For input directory `datasets/` containing `sample_001/`, `sample_002/`, etc., the batch command generates:

```
results/
├── sample_001_spectra.csv
├── sample_001_distribution.png
├── sample_001_segmentation.png
├── sample_001_spectra.png
├── sample_002_spectra.csv
├── sample_002_distribution.png
├── sample_002_segmentation.png
├── sample_002_spectra.png
├── sample_003_spectra.csv
└── ...
```

### Generated Files Per Dataset

**For each dataset that contains seeds:**

1. **{name}_spectra.csv** - Extracted spectral data with metadata
   - Seed IDs, coordinates, areas, morphology
   - Complete spectral signatures (all wavelengths)

2. **{name}_distribution.png** - Spatial and size distribution
   - Left panel: Spatial distribution of seeds
   - Right panel: Area distribution histogram

3. **{name}_segmentation.png** - Seed visualization
   - Left panel: Original image
   - Middle panel: Numbered seeds with colors
   - Right panel: Seed boundaries overlay

4. **{name}_spectra.png** - Spectral curves
   - Individual seed spectra (light lines)
   - Mean spectrum (bold line)
   - Standard deviation band (shaded)

**For datasets with no seeds:**
- No files are generated
- Warning is displayed in console

## Processing Pipeline

For each dataset, the batch command performs:

1. **Dataset Discovery**
   - Searches INPUT_DIR for subdirectories matching pattern
   - Filters to directories with `capture/` folder

2. **Sequential Processing** (for each dataset)
   - Load and calibrate hyperspectral data
   - Apply preprocessing (from config or defaults)
   - Segment seeds
   - Extract spectra
   - Apply outlier removal (if enabled)
   - Save CSV and generate plots

3. **Error Handling**
   - If a dataset fails, error is logged
   - Processing continues with next dataset
   - Summary shows success/failure counts

4. **Summary Display**
   - Total datasets processed
   - Successful count
   - Failed datasets (if any)

## Dataset Discovery

The batch command automatically finds datasets with this structure:

```bash
# Searches for directories matching pattern
INPUT_DIR/{pattern}/capture/data.hdr

# Examples found:
datasets/sample_001/capture/data.hdr  ✓
datasets/sample_002/capture/data.hdr  ✓
datasets/other_file.txt               ✗ (not a directory)
datasets/no_capture/data.hdr          ✗ (missing capture/ folder)
```

## Error Handling

Batch processing continues even if individual datasets fail:

```bash
$ hyperseed batch datasets/ --output-dir results/

[1/5] Processing sample_001...
  ✓ Processed: 47 seeds → sample_001_spectra.csv

[2/5] Processing sample_002...
  ✗ Failed: ENVI header not found

[3/5] Processing sample_003...
  ✓ Processed: 52 seeds → sample_003_spectra.csv

Batch Processing Summary:
  Successful: 3/5
  Failed: sample_002, sample_004
```

**Failed datasets:**
- Error message is displayed
- Processing continues with next dataset
- Failed datasets listed in summary

**Common failure reasons:**
- Missing data files
- Corrupted ENVI headers
- No seeds detected (if validation too strict)
- Insufficient disk space

## Performance Notes

### Processing Time

Batch processing is **sequential** (one dataset at a time):

- **Time per dataset:** ~30-60 seconds (typical)
- **Total time:** num_datasets × time_per_dataset

**Example:**
```
10 datasets × 45 seconds = ~7.5 minutes total
```

### Reducing Processing Time

```yaml
# fast_config.yaml - Optimized for speed
preprocessing:
  method: minimal  # Minimal preprocessing (fastest)

segmentation:
  algorithm: threshold  # Faster than watershed
  min_pixels: 200
  morphology_operations: false  # Skip cleanup
  remove_outliers: false  # Skip outlier detection
```

```bash
hyperseed batch datasets/ --config fast_config.yaml
```

### Memory Usage

- **Per dataset:** ~1-2GB RAM
- **Total:** Same as single dataset (sequential processing)

## Troubleshooting

### Issue: No datasets found

**Error:** `No datasets found matching '*'`

**Solutions:**
```bash
# Check directory structure
ls -la datasets/

# Verify capture folders exist
find datasets/ -name "capture" -type d

# Check pattern
hyperseed batch datasets/ --pattern "*" -v
```

### Issue: All datasets failing

**Possible causes:**
- Incorrect directory structure
- Missing reference files
- Corrupted data

**Solutions:**
```bash
# Test single dataset first
hyperseed analyze datasets/sample_001 --output test.csv

# Enable debug mode
hyperseed batch datasets/ --debug
```

### Issue: Some seeds missing

**Possible causes:**
- min_pixels threshold too high
- Outlier removal too aggressive

**Solutions:**
```bash
# Lower min_pixels
hyperseed batch datasets/ --min-pixels 100

# Disable outlier removal
hyperseed batch datasets/ --no-outlier-removal

# Use custom config with looser thresholds
```

### Issue: Processing too slow

**Solutions:**
```bash
# Use minimal preprocessing
hyperseed batch datasets/ --config fast_config.yaml

# Reduce plot generation (custom config)
```

```yaml
# fast_config.yaml
output:
  include_plots: false  # Skip plots for speed
```

## Comparison with analyze Command

| Feature | batch | analyze |
|---------|-------|---------|
| **Datasets** | Multiple | Single |
| **Processing** | Sequential | Single run |
| **Output** | Multiple files | Single file set |
| **Error handling** | Continues on failure | Stops on error |
| **Progress** | Shows count (1/N) | Shows progress bar |
| **Interactive** | No | Optional (--export-plots) |
| **Use case** | Process many datasets | Detailed single analysis |

**When to use batch:**
- Processing multiple datasets with same settings
- Automated workflows
- Consistent analysis across samples

**When to use analyze:**
- Single dataset analysis
- Testing different parameters
- Need detailed progress information

## Advanced Usage

### Batch with Different Settings Per Type

Process different dataset types with different configs:

```bash
# Process SWIR datasets with one config
hyperseed batch datasets/ \
    --pattern "SWIR_*" \
    --config swir_config.yaml \
    --output-dir results_swir/

# Process VIS datasets with another config
hyperseed batch datasets/ \
    --pattern "VIS_*" \
    --config vis_config.yaml \
    --output-dir results_vis/
```

### Progress Monitoring

```bash
# Run with verbose output
hyperseed batch datasets/ -v --output-dir results/

# Monitor output directory
watch -n 5 'ls -lh results/*.csv | wc -l'
```

### Resume Failed Processing

```bash
# First run - some fail
hyperseed batch datasets/ --output-dir results/

# Find what succeeded
ls results/*.csv

# Process only missing datasets
hyperseed batch datasets/ \
    --pattern "sample_00[6-9]" \
    --output-dir results/
```

## See Also

- **[analyze command](extract.md)**: Process single dataset
- **[Configuration Guide](../getting-started/configuration.md)**: Create batch configurations
- **[Batch Processing Guide](../user-guide/batch-processing.md)**: Detailed workflows and examples
