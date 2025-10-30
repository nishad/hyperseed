# Batch Processing

Process multiple hyperspectral datasets efficiently with consistent settings.

## Overview

Batch processing allows you to analyze multiple hyperspectral datasets with the same analysis pipeline. The `batch` command:

- **Processes datasets sequentially** (one after another)
- **Applies consistent settings** across all datasets
- **Handles errors gracefully** - continues if one dataset fails
- **Generates organized output** with clear naming
- **Provides progress feedback** and summary statistics

## When to Use Batch Processing

### Good Use Cases

✅ **Processing experimental datasets**
```bash
# Process all treatment groups
hyperseed batch experiments/treatment_A/ --config settings.yaml
hyperseed batch experiments/treatment_B/ --config settings.yaml
```

✅ **Consistent analysis across time points**
```bash
# Process all time points with same settings
hyperseed batch timeseries/ --pattern "day_*" --output-dir results/
```

✅ **Quality control across samples**
```bash
# Quick batch analysis of all samples
hyperseed batch samples/ --config minimal_config.yaml
```

✅ **Re-analyzing with different parameters**
```bash
# First pass
hyperseed batch datasets/ --config pass1.yaml --output-dir results_v1/

# Second pass with adjusted settings
hyperseed batch datasets/ --config pass2.yaml --output-dir results_v2/
```

### Not Recommended

❌ **Single dataset** - Use `analyze` command instead
```bash
# Don't use batch for one dataset
hyperseed analyze dataset/sample_001 --output results.csv
```

❌ **Different settings per dataset** - Process individually
```bash
# Process each with different config
hyperseed analyze dataset/sample_001 --config config_A.yaml
hyperseed analyze dataset/sample_002 --config config_B.yaml
```

❌ **Interactive parameter tuning** - Use `segment` or `analyze`
```bash
# Use analyze for testing parameters
hyperseed analyze dataset/sample --min-pixels 100 --export-plots
hyperseed analyze dataset/sample --min-pixels 200 --export-plots
```

## Quick Start

### Basic Batch Processing

```bash
# Process all datasets in directory
hyperseed batch datasets/
```

**What happens:**
1. Finds all subdirectories in `datasets/`
2. Processes each sequentially
3. Saves results to `datasets/results/`

### With Custom Output

```bash
hyperseed batch datasets/ --output-dir analysis_results/
```

### With Configuration

```bash
hyperseed batch datasets/ \
    --config batch_config.yaml \
    --output-dir results/
```

## Directory Structure

### Input Structure Required

```
datasets/
├── sample_001/
│   └── capture/
│       ├── data.raw
│       ├── data.hdr
│       ├── WHITEREF_data.raw
│       ├── WHITEREF_data.hdr
│       ├── DARKREF_data.raw
│       └── DARKREF_data.hdr
├── sample_002/
│   └── capture/
│       └── ...
└── sample_003/
    └── capture/
        └── ...
```

**Requirements:**
- Each dataset must be in its own subdirectory
- Each must have a `capture/` folder
- Must contain `data.hdr`, white reference, dark reference

### Output Structure Generated

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
└── ...
```

## Configuration for Batch Processing

### Creating a Batch Configuration

```bash
# Generate template
hyperseed config --output batch_config.yaml --preset minimal
```

### Recommended Batch Settings

```yaml
# batch_config.yaml - Optimized for batch processing

calibration:
  apply_calibration: true
  clip_negative: true
  clip_max: 1.0
  interpolate_bad_pixels: true

preprocessing:
  method: minimal  # Fast, good for segmentation

segmentation:
  algorithm: watershed  # Best balance
  min_pixels: 200
  reject_overlapping: true
  remove_outliers: true  # Automatic quality control
  outlier_min_area: 50
  outlier_max_area: 2000
```

### Fast Batch Configuration

For maximum speed when processing many datasets:

```yaml
# fast_batch.yaml - Speed-optimized

preprocessing:
  method: none  # Skip preprocessing

segmentation:
  algorithm: threshold  # Fastest algorithm
  min_pixels: 200
  morphology_operations: false
  remove_outliers: false
```

```bash
hyperseed batch large_dataset/ --config fast_batch.yaml
```

## Pattern Matching

Use glob patterns to selectively process datasets.

### Match All (Default)

```bash
hyperseed batch datasets/
# Processes: sample_001, sample_002, sample_003, ...
```

### Match Prefix

```bash
# Process only datasets starting with "SWIR_"
hyperseed batch datasets/ --pattern "SWIR_*"
# Processes: SWIR_001, SWIR_002, SWIR_003
# Skips: VIS_001, other_data
```

### Match Specific Range

```bash
# Process samples 1-5
hyperseed batch datasets/ --pattern "sample_00[1-5]"
# Processes: sample_001, sample_002, ..., sample_005
# Skips: sample_006, sample_007, ...
```

### Match Multiple Patterns

```bash
# Process treatment A samples
hyperseed batch experiments/ --pattern "A_*"

# Then process treatment B samples
hyperseed batch experiments/ --pattern "B_*" --output-dir results_B/
```

### Complex Patterns

```bash
# Process all SWIR samples from experiment 1
hyperseed batch data/ --pattern "exp1_SWIR_*"

# Process time point zero across all experiments
hyperseed batch data/ --pattern "*_t00_*"
```

## Error Handling

Batch processing continues even when individual datasets fail.

### Example Output with Failures

```bash
$ hyperseed batch datasets/ --output-dir results/

[1/5] Processing sample_001...
  ✓ Processed: 47 seeds → sample_001_spectra.csv
     Generated visualizations:
       - sample_001_distribution.png (spatial & size)
       - sample_001_segmentation.png (numbered seeds)
       - sample_001_spectra.png (spectral data)

[2/5] Processing sample_002...
  ✗ Failed: ENVI header file not found

[3/5] Processing sample_003...
  ⚠ No seeds found in sample_003 (check min-pixels threshold)

[4/5] Processing sample_004...
  ✓ Processed: 52 seeds → sample_004_spectra.csv

[5/5] Processing sample_005...
  ✓ Processed: 39 seeds → sample_005_spectra.csv

Batch Processing Summary:
  Successful: 3/5
  Failed: sample_002, sample_003
```

### Common Failure Reasons

**Missing files:**
```
Error: ENVI header file not found
```
**Solution:** Check dataset structure, ensure `capture/data.hdr` exists

**Corrupted data:**
```
Error: Unable to read ENVI data
```
**Solution:** Verify data files are not corrupted, check file permissions

**No seeds detected:**
```
⚠ No seeds found (check min-pixels threshold)
```
**Solution:** Lower `--min-pixels` threshold or check image quality

### Debugging Failed Datasets

```bash
# Test failed dataset individually
hyperseed analyze datasets/sample_002 --output test.csv -v

# Run batch with debug mode
hyperseed batch datasets/ --debug --output-dir results/
```

## Performance and Timing

### Typical Processing Times

**Per dataset** (typical seed image):
- Calibration: ~5-10 seconds
- Preprocessing: ~2-5 seconds
- Segmentation: ~5-10 seconds
- Extraction: ~3-5 seconds
- Plotting: ~5-10 seconds
- **Total:** ~30-60 seconds per dataset

**Batch processing time:**
```
Total time ≈ Number of datasets × Time per dataset

Examples:
- 10 datasets × 45 sec = ~7.5 minutes
- 50 datasets × 45 sec = ~38 minutes
- 100 datasets × 45 sec = ~75 minutes
```

### Speed Optimization

**1. Use minimal preprocessing**
```yaml
preprocessing:
  method: minimal  # 2-3x faster than advanced
```

**2. Use faster segmentation**
```yaml
segmentation:
  algorithm: threshold  # Faster than watershed
```

**3. Disable outlier removal**
```bash
hyperseed batch datasets/ --no-outlier-removal
```

**Combined fast configuration:**
```yaml
preprocessing:
  method: none

segmentation:
  algorithm: threshold
  morphology_operations: false
  remove_outliers: false
```

**Speed improvement:** ~2-3x faster (20-30 seconds per dataset)

## Workflow Examples

### Example 1: Research Experiment

Process multiple treatment groups with consistent settings.

```bash
# Create configuration
cat > experiment_config.yaml << EOF
preprocessing:
  method: minimal

segmentation:
  algorithm: watershed
  min_pixels: 200
  remove_outliers: true
EOF

# Process each treatment group
for group in control treatment_A treatment_B; do
    hyperseed batch experiments/$group/ \
        --config experiment_config.yaml \
        --output-dir results/$group/
done

# Compare results
ls results/*/sample_*.csv
```

### Example 2: Time Series Analysis

Process all time points with same settings.

```bash
# Process all time points
hyperseed batch timeseries/ \
    --pattern "day_*" \
    --config timeseries_config.yaml \
    --output-dir timeseries_results/

# Results organized by day
ls timeseries_results/day_*_spectra.csv
```

### Example 3: Quality Control

Quick batch processing to identify problem datasets.

```bash
# Fast processing with minimal settings
hyperseed batch samples/ \
    --config minimal_config.yaml \
    --output-dir qc_results/ \
    -v

# Review which samples failed
grep "Failed" qc_results/*.log

# Check seed counts
for f in qc_results/*_spectra.csv; do
    echo "$f: $(wc -l < $f) seeds"
done
```

### Example 4: Re-analysis with Different Parameters

Compare results with different min_pixels thresholds.

```bash
# First pass - default
hyperseed batch datasets/ \
    --min-pixels 200 \
    --output-dir results_p200/

# Second pass - lower threshold
hyperseed batch datasets/ \
    --min-pixels 100 \
    --output-dir results_p100/

# Compare seed counts
diff <(ls results_p200/*.csv | wc -l) \
     <(ls results_p100/*.csv | wc -l)
```

### Example 5: Selective Re-processing

Re-process only failed datasets.

```bash
# Initial batch run
hyperseed batch datasets/ --output-dir results/

# Identify successful datasets
successful=$(ls results/*_spectra.csv | xargs -n1 basename | sed 's/_spectra.csv//')

# Find failed datasets
cd datasets/
for dataset in *; do
    if ! echo "$successful" | grep -q "$dataset"; then
        echo "Failed: $dataset"
    fi
done

# Re-process failed datasets manually
hyperseed analyze datasets/failed_sample_002 --output results/failed_sample_002_spectra.csv
```

## Monitoring Progress

### Real-time Monitoring

```bash
# Terminal 1: Run batch processing
hyperseed batch datasets/ --output-dir results/ -v

# Terminal 2: Monitor output files
watch -n 5 'ls -lh results/*.csv | wc -l'

# Terminal 3: Monitor disk usage
watch -n 10 'du -sh results/'
```

### Progress Estimation

```bash
# Count total datasets
total=$(ls -d datasets/*/ | wc -l)

# Monitor completion
while true; do
    completed=$(ls results/*_spectra.csv 2>/dev/null | wc -l)
    echo "Progress: $completed / $total"
    sleep 10
done
```

## Batch vs. Individual Analysis

| Aspect | batch command | analyze command |
|--------|---------------|-----------------|
| **Number of datasets** | Multiple | Single |
| **Processing** | Sequential | One-time |
| **Settings** | Consistent across all | Per-run |
| **Error handling** | Continues on failure | Stops on error |
| **Progress display** | Dataset count (1/N) | Detailed progress bar |
| **Output organization** | All in one directory | Specified per run |
| **Interactive tuning** | Not suitable | Good for testing |
| **Automation** | Excellent | Requires scripting |

**Use batch when:**
- Processing 3+ datasets with same settings
- Need consistent analysis pipeline
- Running automated workflows
- Don't need to test parameters

**Use analyze when:**
- Processing single dataset
- Testing different parameters
- Need detailed progress feedback
- Want to inspect results interactively

## Troubleshooting

### Issue: No datasets found

**Error:** `No datasets found matching '*'`

**Causes:**
- Wrong directory structure
- No `capture/` folders
- Pattern doesn't match

**Solutions:**
```bash
# Check structure
ls -la datasets/

# Verify capture folders
find datasets/ -type d -name "capture"

# Try explicit pattern
hyperseed batch datasets/ --pattern "*" -v
```

### Issue: All datasets failing

**Solutions:**
```bash
# Test one dataset individually
hyperseed analyze datasets/sample_001 --output test.csv -v

# Check data files
ls datasets/sample_001/capture/

# Run with debug
hyperseed batch datasets/ --debug
```

### Issue: Inconsistent seed counts

**Possible causes:**
- Variable image quality
- min_pixels threshold not appropriate
- Outlier removal too aggressive

**Solutions:**
```bash
# Disable outlier removal to see raw counts
hyperseed batch datasets/ --no-outlier-removal

# Lower min_pixels
hyperseed batch datasets/ --min-pixels 100

# Use custom config with looser thresholds
```

### Issue: Memory errors

**Solutions:**
```bash
# Process fewer datasets at once
hyperseed batch datasets/ --pattern "sample_00[1-3]"
hyperseed batch datasets/ --pattern "sample_00[4-6]"

# Close other applications
# Check available memory: free -h (Linux) or top (macOS)
```

### Issue: Slow processing

**Solutions:**
```bash
# Use fast configuration
hyperseed batch datasets/ --config fast_config.yaml

# Skip plots
hyperseed batch datasets/ --config no_plots.yaml

# Process in chunks
```

## See Also

- **[CLI Reference: batch](../cli-reference/batch.md)**: Complete command documentation
- **[Configuration Guide](../getting-started/configuration.md)**: Create batch configurations
- **[analyze command](../cli-reference/extract.md)**: Single dataset analysis
- **[Segmentation Guide](segmentation.md)**: Optimize segmentation settings
