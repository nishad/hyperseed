# segment

Perform seed segmentation only, without spectral extraction.

The `segment` command is useful for:

- **Testing segmentation parameters** before full analysis
- **Visualizing segmentation results** interactively
- **Exporting segmentation masks** for external processing
- **Quick quality checks** on imaging setup

## Syntax

```bash
hyperseed segment INPUT_PATH [OPTIONS]
```

## Arguments

### INPUT_PATH

Path to the hyperspectral dataset directory.

**Required:** Yes

**Format:** Directory containing:
- `capture/data.raw` and `capture/data.hdr` (main data)
- `capture/WHITEREF_data.raw` and `.hdr` (white reference)
- `capture/DARKREF_data.raw` and `.hdr` (dark reference)

**Example:**
```bash
hyperseed segment dataset/sample_001
```

## Options

### -o, --output PATH

Output file path for segmentation mask.

**Type:** Path
**Default:** None (no mask saved)
**Format:** `.npy` (NumPy array format)

**Example:**
```bash
hyperseed segment dataset/sample \
    --output segmentation_mask.npy
```

**Note:** The mask is a 2D labeled array where each seed has a unique integer ID (0 = background).

### --algorithm CHOICE

Segmentation algorithm to use.

**Type:** Choice
**Choices:** `threshold`, `watershed`, `connected`, `combined`
**Default:** `watershed`

**Example:**
```bash
# Fast thresholding
hyperseed segment dataset/sample --algorithm threshold

# Watershed (default, recommended)
hyperseed segment dataset/sample --algorithm watershed

# Connected components
hyperseed segment dataset/sample --algorithm connected

# Combined (most robust)
hyperseed segment dataset/sample --algorithm combined
```

**Algorithm descriptions:**

| Algorithm | Speed | Separates touching seeds | Best for |
|-----------|-------|-------------------------|----------|
| `threshold` | ⚡⚡⚡ | ❌ | Well-separated seeds, quick tests |
| `watershed` | ⚡⚡ | ✅ | General use (recommended) |
| `connected` | ⚡⚡⚡ | ❌ | Simple, well-separated seeds |
| `combined` | ⚡ | ✅ | Maximum accuracy, robust |

### --min-pixels INTEGER

Minimum seed size in pixels.

**Type:** Integer
**Default:** 200
**Range:** 10-10000

Seeds smaller than this threshold are filtered out as noise or debris.

**Example:**
```bash
# Small seeds or high resolution
hyperseed segment dataset/sample --min-pixels 100

# Default
hyperseed segment dataset/sample --min-pixels 200

# Large seeds or low resolution
hyperseed segment dataset/sample --min-pixels 300
```

**Guidelines:**
- Small seeds (< 5mm) or high-resolution images: 100-150
- Medium seeds (default): 200
- Large seeds (> 10mm) or low-resolution: 300-500

### --visualize

Show segmentation visualization.

**Type:** Flag (boolean)
**Default:** False

Displays an interactive visualization showing:
- Original image
- Segmentation with numbered seeds
- Seed boundaries overlay

**Example:**
```bash
hyperseed segment dataset/sample --visualize
```

**Note:** Requires display (won't work in headless environments). Press any key to close the visualization window.

### --help

Show help message and exit.

**Example:**
```bash
hyperseed segment --help
```

## Complete Examples

### Basic Segmentation with Visualization

```bash
hyperseed segment dataset/sample_001 --visualize
```

**What it does:**
1. Loads and calibrates hyperspectral data
2. Applies minimal preprocessing (smoothing only)
3. Segments seeds using watershed algorithm
4. Displays interactive visualization

### Save Segmentation Mask

```bash
hyperseed segment dataset/sample_001 \
    --output results/sample_001_mask.npy \
    --algorithm watershed \
    --min-pixels 200
```

**Output:** `results/sample_001_mask.npy` - NumPy array with labeled seeds

**Load mask in Python:**
```python
import numpy as np
mask = np.load('results/sample_001_mask.npy')
print(f"Found {mask.max()} seeds")
```

### Test Different Algorithms

```bash
# Test threshold
hyperseed segment dataset/sample --algorithm threshold --visualize

# Test watershed
hyperseed segment dataset/sample --algorithm watershed --visualize

# Test combined
hyperseed segment dataset/sample --algorithm combined --visualize
```

**Use case:** Compare algorithms visually to choose the best for your data.

### Adjust Size Threshold

```bash
# Lower threshold for small seeds
hyperseed segment dataset/sample --min-pixels 100 --visualize

# Higher threshold to remove noise
hyperseed segment dataset/sample --min-pixels 300 --visualize
```

## Processing Pipeline

The `segment` command performs these steps:

1. **Load data** from ENVI format files
2. **Calibrate** using white/dark references
3. **Preprocess** with minimal preset (smoothing only)
4. **Segment** using selected algorithm
5. **Filter** by size (`min_pixels`)
6. **Visualize** (if `--visualize` flag set)
7. **Export mask** (if `--output` specified)

**Note:** The segment command does NOT perform:
- Outlier removal (use `analyze` command for this)
- Spectral extraction (use `analyze` command)
- Advanced preprocessing (always uses minimal)

## Output Format

### Segmentation Mask (.npy)

A 2D NumPy array with the same spatial dimensions as the input image.

**Format:**
- **dtype:** `int32` or `int64`
- **Values:**
  - `0` = background (no seed)
  - `1, 2, 3, ...` = seed IDs

**Example structure:**
```python
import numpy as np
mask = np.load('mask.npy')

# mask.shape: (lines, samples)
# Example: (384, 512)

# Get seed IDs
seed_ids = np.unique(mask[mask > 0])
print(f"Seeds detected: {seed_ids}")
# Output: Seeds detected: [1 2 3 4 5 ... 47]

# Get pixels for seed #5
seed_5_pixels = mask == 5
seed_5_area = np.sum(seed_5_pixels)
print(f"Seed 5 area: {seed_5_area} pixels")
```

## Comparison with analyze Command

| Feature | segment | analyze |
|---------|---------|---------|
| **Segmentation** | ✅ Yes | ✅ Yes |
| **Spectral extraction** | ❌ No | ✅ Yes |
| **Outlier removal** | ❌ No | ✅ Yes (default) |
| **Preprocessing** | Minimal only | All presets |
| **Visualization** | Interactive | Exported plots |
| **Output** | Mask only | CSV + plots + mask |
| **Speed** | Faster | Slower |
| **Use case** | Testing, quick checks | Full analysis |

**When to use segment:**
- Testing segmentation parameters
- Visualizing results quickly
- You only need the mask
- Performing custom spectral analysis later

**When to use analyze:**
- Complete analysis workflow
- Need spectral data extracted
- Want outlier removal
- Need CSV output with spectra

## Troubleshooting

### Issue: No display available

**Error:** `Cannot connect to display`

**Solution:** Run without `--visualize` in headless environments:
```bash
hyperseed segment dataset/sample --output mask.npy
```

### Issue: Too many/too few seeds detected

**Solution:** Adjust `min_pixels`:
```bash
# Increase to reduce detections
hyperseed segment dataset/sample --min-pixels 300 --visualize

# Decrease to increase detections
hyperseed segment dataset/sample --min-pixels 150 --visualize
```

### Issue: Touching seeds not separated

**Solution:** Use watershed or combined algorithm:
```bash
hyperseed segment dataset/sample \
    --algorithm watershed \
    --visualize
```

### Issue: Segmentation quality poor

**Possible causes:**
- Algorithm not suitable for seed arrangement
- Preprocessing too aggressive (segment always uses minimal)
- Poor image quality/contrast

**Solutions:**
```bash
# Try different algorithm
hyperseed segment dataset/sample --algorithm combined --visualize

# For full control, use analyze command with custom config
hyperseed analyze dataset/sample --config custom_config.yaml
```

## Advanced Usage

### Batch Testing Algorithms

Test all algorithms on the same dataset:

```bash
#!/bin/bash
for algo in threshold watershed connected combined; do
    echo "Testing $algo..."
    hyperseed segment dataset/sample \
        --algorithm $algo \
        --output masks/${algo}_mask.npy
done
```

### Compare Segmentation with Different min_pixels

```bash
#!/bin/bash
for pixels in 100 150 200 250 300; do
    echo "Testing min_pixels=$pixels..."
    hyperseed segment dataset/sample \
        --min-pixels $pixels \
        --output masks/mask_${pixels}px.npy
done
```

### Extract Statistics from Mask

```python
import numpy as np
from skimage import measure

# Load mask
mask = np.load('segmentation_mask.npy')

# Get region properties
props = measure.regionprops(mask)

# Print statistics
print(f"Total seeds: {len(props)}")
print(f"Mean area: {np.mean([p.area for p in props]):.1f} pixels")
print(f"Area range: {min(p.area for p in props)}-{max(p.area for p in props)} pixels")

# Export to CSV
import pandas as pd
data = [{
    'seed_id': p.label,
    'area': p.area,
    'centroid_y': p.centroid[0],
    'centroid_x': p.centroid[1],
    'eccentricity': p.eccentricity,
    'solidity': p.solidity
} for p in props]
df = pd.DataFrame(data)
df.to_csv('seed_properties.csv', index=False)
```

## See Also

- **[analyze command](extract.md)**: Complete analysis with spectral extraction
- **[Segmentation Guide](../user-guide/segmentation.md)**: Detailed algorithm descriptions
- **[Configuration](../getting-started/configuration.md#segmentation-settings)**: All segmentation parameters
- **[API Reference](../api-reference/segmentation.md)**: Programmatic segmentation
