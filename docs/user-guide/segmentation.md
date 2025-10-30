# Segmentation

Seed segmentation is the process of detecting and isolating individual seeds in hyperspectral images. Hyperseed provides multiple algorithms optimized for different scenarios.

## Overview

Segmentation converts hyperspectral imagery into a labeled mask where each seed is assigned a unique ID. This enables:

- **Spectral extraction** from individual seeds
- **Morphological analysis** (size, shape, eccentricity)
- **Spatial tracking** of seed positions
- **Quality control** through outlier removal

## When to Use Each Algorithm

### threshold (Fast & Simple)

**Best for:**
- Well-separated seeds with uniform illumination
- Quick initial testing
- Processing large batches where speed is critical

**How it works:**
- Applies global (Otsu) or adaptive thresholding
- Labels connected components
- Applies morphological cleanup (erosion/dilation)

**Limitations:**
- May not separate touching seeds effectively
- Sensitive to illumination variations
- Less robust than watershed

**Usage:**
```bash
hyperseed analyze dataset/sample \
    --segmentation threshold \
    --min-pixels 200
```

### watershed (Recommended)

**Best for:**
- Seeds that are touching or close together
- Most general use cases
- High-quality segmentation

**How it works:**
- Computes distance transform from binary mask
- Finds local maxima as seed centers
- Applies watershed algorithm to separate regions
- Effectively "floods" from seed centers

**Advantages:**
- Separates touching seeds reliably
- Robust to minor variations
- Good balance of speed and accuracy

**Usage:**
```bash
hyperseed analyze dataset/sample \
    --segmentation watershed \
    --min-pixels 200
```

### connected (Simple & Fast)

**Best for:**
- Well-separated seeds with clear gaps
- Seeds with consistent size and shape
- When watershed is too aggressive

**How it works:**
- Binary thresholding with Otsu
- Morphological cleanup (closing/opening)
- Connected component labeling
- Filters by shape (eccentricity < 0.95, solidity > 0.7)

**Limitations:**
- Cannot separate touching seeds
- May fail with irregular spacing

**Usage:**
```bash
hyperseed analyze dataset/sample \
    --segmentation connected \
    --min-pixels 200
```

### combined (Most Robust)

**Best for:**
- Critical applications requiring maximum accuracy
- Mixed seed arrangements (some touching, some separated)
- When you want consensus from multiple methods

**How it works:**
- Runs threshold and watershed algorithms
- Uses majority voting to create consensus mask
- Applies final size filtering
- Slower but most robust

**Trade-offs:**
- Slowest algorithm (runs multiple methods)
- Best accuracy and robustness
- Good for challenging images

**Usage:**
```bash
hyperseed analyze dataset/sample \
    --segmentation combined \
    --min-pixels 200
```

## Algorithm Comparison

| Feature | threshold | watershed | connected | combined |
|---------|-----------|-----------|-----------|----------|
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡ Medium | ⚡⚡⚡ Fast | ⚡ Slow |
| **Separates touching seeds** | ❌ No | ✅ Yes | ❌ No | ✅ Yes |
| **Robustness** | ⭐⭐ Good | ⭐⭐⭐ Excellent | ⭐⭐ Good | ⭐⭐⭐⭐ Best |
| **Illumination tolerance** | ⚠️ Low | ✅ High | ⚠️ Medium | ✅ High |
| **Best use case** | Quick tests | General use | Well-separated | Critical apps |

## Size Filtering

All algorithms apply size filtering to remove noise and artifacts.

### min_pixels

Minimum seed size in pixels. Objects smaller than this are removed.

**Default:** 200 pixels

**How to choose:**
```bash
# Small seeds or high-resolution images
hyperseed analyze dataset/sample --min-pixels 100

# Medium seeds (default, recommended)
hyperseed analyze dataset/sample --min-pixels 200

# Large seeds or low-resolution images
hyperseed analyze dataset/sample --min-pixels 300
```

### max_pixels

Maximum seed size in pixels. Objects larger than this are removed.

**Default:** None (no upper limit)

**Usage:**
```yaml
# configuration.yaml
segmentation:
  min_pixels: 200
  max_pixels: 5000  # Remove objects larger than 5000 pixels
```

## Morphological Operations

Morphological operations clean up segmentation results by filling holes and smoothing boundaries.

### morphology_operations

**Default:** `true` (enabled)

**Operations applied:**
- **Closing**: Fills small holes within seeds
- **Opening**: Removes small protrusions

**When to disable:**
```yaml
segmentation:
  morphology_operations: false  # Disable if seeds have irregular shapes
```

### morphology_kernel_size

Size of the structuring element for morphological operations.

**Default:** 3 pixels
**Range:** 1-21 pixels

**Effect:**
- Small (1-3): Minimal cleanup, preserves detail
- Medium (3-7): Balanced (recommended)
- Large (7-21): Aggressive cleanup, may merge close seeds

```yaml
segmentation:
  morphology_kernel_size: 5  # More aggressive cleanup
```

## Border Seed Filtering

Remove seeds that touch the image borders (often incomplete or partially visible).

### filter_border_seeds

**Default:** `false` (disabled)

**When to enable:**
- Seeds at edges are incomplete
- Want to exclude partial seeds
- Analyzing seed distributions (avoid edge bias)

```yaml
segmentation:
  filter_border_seeds: true
  border_width: 2  # Pixels from edge
```

## Threshold Method Options

The `threshold` algorithm supports three thresholding methods:

### otsu (Default)

Automatic global threshold using Otsu's method.

**Best for:**
- Uniform illumination
- Clear seed/background separation
- Most common use case

```yaml
segmentation:
  algorithm: threshold
  threshold_method: otsu
```

### adaptive

Local adaptive thresholding for uneven illumination.

**Best for:**
- Non-uniform lighting
- Gradients across image
- Shadows present

```yaml
segmentation:
  algorithm: threshold
  threshold_method: adaptive
```

### manual

Requires explicit threshold value (not exposed in CLI, API only).

## Outlier Removal

Automatic outlier removal eliminates reference objects, debris, and anomalies.

### Why Remove Outliers?

Hyperspectral seed datasets often contain:

- **Calibration targets** (white/dark references in view)
- **Reference objects** (rulers, labels, markers)
- **Debris** (dust, chaff, broken seeds)
- **Anomalies** (clumped seeds, imaging artifacts)

Outlier removal automatically filters these without manual intervention.

### Three-Step Process

#### Step 1: Absolute Area Bounds

Hard thresholds that always apply.

```yaml
segmentation:
  outlier_min_area: 50    # Remove anything < 50 pixels
  outlier_max_area: 2000  # Remove anything > 2000 pixels
```

**Use case:** Remove obviously too-small (debris) or too-large (reference targets) objects.

#### Step 2: IQR-Based Filtering

Statistical outlier detection using Interquartile Range.

**How it works:**
1. Calculate Q1 (25th percentile) and Q3 (75th percentile) of seed areas
2. Compute IQR = Q3 - Q1
3. Define bounds:
   - Lower bound = Q1 - (iqr_lower × IQR)
   - Upper bound = Q3 + (iqr_upper × IQR)
4. Remove seeds outside bounds

**Configuration:**
```yaml
segmentation:
  outlier_iqr_lower: 1.5  # Stricter lower bound
  outlier_iqr_upper: 3.0  # Looser upper bound (large outliers more common)
```

**Why asymmetric?**
Large outliers (calibration targets, rulers) are more common than small outliers, so the upper multiplier is larger.

#### Step 3: Shape-Based Filtering (Optional)

Filter by shape properties (disabled by default).

```yaml
segmentation:
  use_shape_filtering: true
  outlier_eccentricity: 0.95  # Max elongation (0=circle, 1=line)
  outlier_solidity: 0.7       # Min regularity (area/convex_hull)
```

**When to enable:**
- Seeds should be round or oval
- Want to exclude elongated debris
- Reject irregularly shaped clumps

**Examples:**
- Eccentricity 0.5 = moderately oval seed ✅
- Eccentricity 0.98 = elongated debris ❌
- Solidity 0.85 = solid seed ✅
- Solidity 0.55 = irregular clump ❌

### Disabling Outlier Removal

```bash
hyperseed analyze dataset/sample --no-outlier-removal
```

```yaml
segmentation:
  remove_outliers: false
```

**When to disable:**
- All objects in image are seeds
- Want to manually inspect all detections
- Performing custom filtering later

## Complete Configuration Examples

### Example 1: Default (Recommended)

```yaml
segmentation:
  algorithm: watershed
  min_pixels: 200
  max_pixels: null
  reject_overlapping: true
  threshold_method: otsu
  morphology_operations: true
  morphology_kernel_size: 3
  filter_border_seeds: false
  remove_outliers: true
  outlier_min_area: 50
  outlier_max_area: 2000
  outlier_iqr_lower: 1.5
  outlier_iqr_upper: 3.0
  use_shape_filtering: false
```

### Example 2: Strict Outlier Removal

```yaml
segmentation:
  algorithm: watershed
  min_pixels: 200
  remove_outliers: true
  outlier_min_area: 100        # Larger minimum
  outlier_max_area: 1500       # Smaller maximum
  outlier_iqr_lower: 1.0       # Stricter IQR
  outlier_iqr_upper: 2.0
  use_shape_filtering: true    # Enable shape filtering
  outlier_eccentricity: 0.90   # Reject more elongated seeds
  outlier_solidity: 0.75       # Require more regular shapes
```

### Example 3: Fast Processing

```yaml
segmentation:
  algorithm: threshold         # Fastest algorithm
  threshold_method: otsu
  min_pixels: 200
  morphology_operations: false # Skip cleanup for speed
  remove_outliers: false       # Skip outlier detection
```

### Example 4: Maximum Accuracy

```yaml
segmentation:
  algorithm: combined          # Most robust
  min_pixels: 150
  max_pixels: 5000
  morphology_operations: true
  morphology_kernel_size: 5    # More cleanup
  filter_border_seeds: true    # Remove edge seeds
  remove_outliers: true
  use_shape_filtering: true    # All filtering enabled
```

## Troubleshooting

### Issue: Seeds not detected

**Possible causes:**
- `min_pixels` too high
- Preprocessing removed too much signal
- Poor illumination/contrast

**Solutions:**
```bash
# Lower min_pixels threshold
hyperseed analyze dataset/sample --min-pixels 100

# Try different algorithm
hyperseed analyze dataset/sample --segmentation combined

# Use minimal preprocessing
hyperseed analyze dataset/sample --preprocess minimal
```

### Issue: Touching seeds not separated

**Solutions:**
```bash
# Use watershed algorithm (best for separation)
hyperseed analyze dataset/sample --segmentation watershed

# Try combined method
hyperseed analyze dataset/sample --segmentation combined
```

### Issue: Too many small detections (noise)

**Solutions:**
```bash
# Increase min_pixels
hyperseed analyze dataset/sample --min-pixels 300

# Enable outlier removal (should be default)
hyperseed analyze dataset/sample  # outlier removal is on by default

# Increase morphology cleanup
```

```yaml
segmentation:
  morphology_kernel_size: 7
```

### Issue: Reference objects detected as seeds

**Solution:**
Enable outlier removal with appropriate thresholds:

```yaml
segmentation:
  remove_outliers: true
  outlier_max_area: 1500  # Adjust based on reference object size
```

### Issue: Over-segmentation (seeds split into multiple parts)

**Possible causes:**
- Morphology operations too aggressive
- Algorithm too sensitive

**Solutions:**
```yaml
segmentation:
  morphology_operations: false  # Disable cleanup
  # Or reduce kernel size
  morphology_kernel_size: 1
```

Try less aggressive algorithm:
```bash
hyperseed analyze dataset/sample --segmentation connected
```

## Workflow Recommendations

### 1. Start with defaults

```bash
hyperseed analyze dataset/sample \
    --output results.csv \
    --export-plots
```

Review the segmentation visualization to assess quality.

### 2. Adjust min_pixels if needed

```bash
# If too many small detections
hyperseed analyze dataset/sample --min-pixels 300

# If seeds not detected
hyperseed analyze dataset/sample --min-pixels 100
```

### 3. Try different algorithms

```bash
# If seeds are touching
hyperseed analyze dataset/sample --segmentation watershed

# If watershed over-separates
hyperseed analyze dataset/sample --segmentation connected

# If unsure
hyperseed analyze dataset/sample --segmentation combined
```

### 4. Fine-tune outlier removal

Create custom configuration file:

```yaml
# config.yaml
segmentation:
  algorithm: watershed
  min_pixels: 200
  remove_outliers: true
  outlier_min_area: 75      # Adjust based on your seeds
  outlier_max_area: 1800
  outlier_iqr_lower: 1.5
  outlier_iqr_upper: 2.5    # Stricter upper bound
```

```bash
hyperseed analyze dataset/sample --config config.yaml
```

## See Also

- **[Configuration Guide](../getting-started/configuration.md#segmentation-settings)**: Complete parameter reference
- **[API Reference](../api-reference/segmentation.md)**: Programmatic segmentation
- **[CLI Reference](../cli-reference/segment.md)**: segment command documentation
