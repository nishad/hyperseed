# Segmentation API

Programmatic interface for seed detection and segmentation.

## SeedSegmenter

The `SeedSegmenter` class provides a unified interface for applying various segmentation algorithms with validation and visualization.

### Basic Usage

```python
from hyperseed.core.segmentation.segmenter import SeedSegmenter
from hyperseed.config.settings import Settings

# Create segmenter with default settings
settings = Settings()
segmenter = SeedSegmenter(settings.segmentation)

# Segment seeds
mask, n_seeds = segmenter.segment(preprocessed_data)

# Get seed properties
properties = segmenter.get_seed_properties()
```

### With Custom Configuration

```python
from hyperseed.config.settings import SegmentationConfig

# Create custom configuration
config = SegmentationConfig(
    algorithm="watershed",
    min_pixels=200,
    max_pixels=5000,
    reject_overlapping=True,
    morphology_operations=True,
    morphology_kernel_size=5,
    remove_outliers=True,
    outlier_min_area=75,
    outlier_max_area=1800
)

# Create segmenter
segmenter = SeedSegmenter(config)

# Segment
mask, n_seeds = segmenter.segment(data)
```

## SeedSegmenter Methods

### \_\_init\_\_(config: Optional[SegmentationConfig] = None)

Initialize the seed segmenter.

**Parameters:**
- `config` *(SegmentationConfig, optional)*: Segmentation configuration object. If None, uses defaults.

**Example:**
```python
from hyperseed.config.settings import SegmentationConfig

config = SegmentationConfig(algorithm="watershed", min_pixels=200)
segmenter = SeedSegmenter(config)
```

### segment(data: np.ndarray, band_index: Optional[int] = None, validate: bool = True) -> Tuple[np.ndarray, int]

Perform seed segmentation on hyperspectral data.

**Parameters:**
- `data` *(np.ndarray)*: Hyperspectral data array with shape (lines, samples, bands) or (lines, samples)
- `band_index` *(int, optional)*: Specific band to use for segmentation. If None, uses automatic selection based on variance
- `validate` *(bool, default: True)*: Whether to apply validation after segmentation (size filtering, overlap rejection)

**Returns:**
- `mask` *(np.ndarray)*: Labeled mask array where each seed has a unique ID (0 = background)
- `n_seeds` *(int)*: Number of seeds detected

**Example:**
```python
# Segment using all bands (automatic selection)
mask, n_seeds = segmenter.segment(data)
print(f"Found {n_seeds} seeds")

# Segment using specific band
mask, n_seeds = segmenter.segment(data, band_index=50)

# Segment without validation
mask, n_seeds = segmenter.segment(data, validate=False)
```

**Note:** The segmentation process includes:
1. Apply selected algorithm (threshold/watershed/connected/combined)
2. Apply morphological operations (if enabled)
3. Validate seeds (if validate=True): size filtering, overlap rejection
4. Filter border seeds (if enabled)

### visualize(data: np.ndarray, band_index: Optional[int] = None, save_path: Optional[Union[str, Path]] = None, show_labels: bool = True, show_boundaries: bool = True) -> plt.Figure

Visualize segmentation results.

**Parameters:**
- `data` *(np.ndarray)*: Original hyperspectral data
- `band_index` *(int, optional)*: Band to use for background image. If None, uses mean of middle bands
- `save_path` *(str or Path, optional)*: Path to save the figure. If None, displays interactive plot
- `show_labels` *(bool, default: True)*: Whether to show seed ID numbers
- `show_boundaries` *(bool, default: True)*: Whether to show seed boundaries

**Returns:**
- `fig` *(matplotlib.figure.Figure)*: Matplotlib figure object

**Example:**
```python
# Visualize with all features
fig = segmenter.visualize(data, show_labels=True, show_boundaries=True)

# Save to file
segmenter.visualize(
    data,
    save_path="segmentation_result.png",
    band_index=50
)

# Simple visualization without labels
fig = segmenter.visualize(data, show_labels=False)
plt.show()
```

**Visualization layout:**
- **Left panel**: Original image
- **Middle panel**: Colored segmentation with numbered seeds
- **Right panel**: Seed boundaries overlay

### get_seed_properties() -> list

Get morphological properties of all segmented seeds.

**Returns:**
- `properties` *(list of dict)*: List of property dictionaries, one per seed

**Property dictionary keys:**
- `label` *(int)*: Seed ID
- `area` *(int)*: Seed area in pixels
- `centroid` *(tuple)*: (y, x) centroid coordinates
- `bbox` *(tuple)*: (min_row, min_col, max_row, max_col) bounding box
- `eccentricity` *(float)*: Elongation (0=circle, 1=line)
- `solidity` *(float)*: Area/convex_hull_area (shape regularity)
- `coords` *(np.ndarray)*: All pixel coordinates belonging to seed

**Example:**
```python
properties = segmenter.get_seed_properties()

for prop in properties:
    print(f"Seed {prop['label']}:")
    print(f"  Area: {prop['area']} pixels")
    print(f"  Centroid: {prop['centroid']}")
    print(f"  Eccentricity: {prop['eccentricity']:.3f}")
    print(f"  Solidity: {prop['solidity']:.3f}")
```

### get_validation_stats() -> Optional[Dict[str, Any]]

Get validation statistics from the last segmentation.

**Returns:**
- `stats` *(dict or None)*: Validation statistics dictionary, or None if validation not performed

**Statistics dictionary keys:**
- `initial_count` *(int)*: Number of seeds before validation
- `final_count` *(int)*: Number of seeds after validation
- `removed_by_size` *(int)*: Seeds removed by size filtering
- `removed_by_overlap` *(int)*: Seeds removed by overlap rejection
- `removed_by_shape` *(int)*: Seeds removed by shape criteria

**Example:**
```python
mask, n_seeds = segmenter.segment(data, validate=True)
stats = segmenter.get_validation_stats()

if stats:
    print(f"Initial seeds: {stats['initial_count']}")
    print(f"Final seeds: {stats['final_count']}")
    print(f"Removed by size: {stats['removed_by_size']}")
    print(f"Removed by overlap: {stats['removed_by_overlap']}")
```

### export_mask(path: Union[str, Path], format: str = "npy") -> None

Export segmentation mask to file.

**Parameters:**
- `path` *(str or Path)*: Output file path
- `format` *(str, default: "npy")*: Export format. Options: "npy", "png", "tiff"

**Example:**
```python
# Export as NumPy array (recommended)
segmenter.export_mask("results/mask.npy", format="npy")

# Export as PNG image (scaled to 0-255)
segmenter.export_mask("results/mask.png", format="png")

# Export as TIFF
segmenter.export_mask("results/mask.tiff", format="tiff")
```

**Format details:**
- **npy**: Native NumPy format, preserves exact labels, recommended for further processing
- **png**: 8-bit PNG, labels scaled to 0-255, useful for visualization
- **tiff**: TIFF format, preserves labels

### export_properties(path: Union[str, Path], format: str = "csv") -> None

Export seed properties to file.

**Parameters:**
- `path` *(str or Path)*: Output file path
- `format` *(str, default: "csv")*: Export format. Options: "csv", "json"

**Example:**
```python
# Export as CSV
segmenter.export_properties("results/seed_props.csv", format="csv")

# Export as JSON
segmenter.export_properties("results/seed_props.json", format="json")
```

**CSV format:** Tabular format with columns for label, area, centroid_y, centroid_x, bbox, eccentricity, solidity

**JSON format:** List of property dictionaries (excludes pixel coordinates)

### describe() -> Dict[str, Any]

Get description of segmentation results.

**Returns:**
- `description` *(dict)*: Dictionary describing the segmentation

**Description dictionary keys:**
- `algorithm` *(str)*: Algorithm used
- `n_seeds` *(int)*: Number of seeds detected
- `mask_shape` *(tuple)*: Shape of segmentation mask
- `config` *(dict)*: Configuration parameters
- `validation` *(dict, optional)*: Validation statistics
- `seed_statistics` *(dict, optional)*: Area statistics (min, max, mean, std)

**Example:**
```python
desc = segmenter.describe()
print(f"Algorithm: {desc['algorithm']}")
print(f"Seeds detected: {desc['n_seeds']}")
print(f"Mask shape: {desc['mask_shape']}")

if 'seed_statistics' in desc:
    stats = desc['seed_statistics']
    print(f"Area range: {stats['min_area']}-{stats['max_area']} pixels")
    print(f"Mean area: {stats['mean_area']:.1f} pixels")
```

## Individual Segmentation Functions

For advanced use, individual segmentation algorithms can be called directly.

### threshold_segmentation

```python
from hyperseed.core.segmentation.algorithms import threshold_segmentation

mask, n_seeds = threshold_segmentation(
    data,
    method="otsu",  # or "adaptive", "manual"
    threshold_value=None,  # for manual method only
    min_seed_size=200,
    max_seed_size=None,
    band_index=None
)
```

**Parameters:**
- `data` *(np.ndarray)*: Hyperspectral data
- `method` *(str)*: "otsu", "adaptive", or "manual"
- `threshold_value` *(float, optional)*: Manual threshold (for method="manual")
- `min_seed_size` *(int, default: 200)*: Minimum seed size in pixels
- `max_seed_size` *(int, optional)*: Maximum seed size in pixels
- `band_index` *(int, optional)*: Specific band to use

**Returns:**
- `mask` *(np.ndarray)*: Labeled mask
- `n_seeds` *(int)*: Number of seeds

**Methods:**
- **otsu**: Automatic global threshold using Otsu's method
- **adaptive**: Local adaptive thresholding (block size=35)
- **manual**: Requires explicit threshold_value parameter

### watershed_segmentation

```python
from hyperseed.core.segmentation.algorithms import watershed_segmentation

mask, n_seeds = watershed_segmentation(
    data,
    min_seed_size=200,
    max_seed_size=None,
    band_index=None,
    min_distance=20
)
```

**Parameters:**
- `data` *(np.ndarray)*: Hyperspectral data
- `min_seed_size` *(int, default: 200)*: Minimum seed size in pixels
- `max_seed_size` *(int, optional)*: Maximum seed size in pixels
- `band_index` *(int, optional)*: Specific band to use
- `min_distance` *(int, default: 20)*: Minimum distance between seed centers

**Returns:**
- `mask` *(np.ndarray)*: Labeled mask
- `n_seeds` *(int)*: Number of seeds

**How it works:**
1. Initial Otsu thresholding
2. Distance transform computation
3. Local maxima detection (seed centers)
4. Watershed algorithm to separate regions
5. Size filtering

### connected_components_segmentation

```python
from hyperseed.core.segmentation.algorithms import connected_components_segmentation

mask, n_seeds = connected_components_segmentation(
    data,
    min_seed_size=200,
    max_seed_size=None,
    band_index=None,
    connectivity=2
)
```

**Parameters:**
- `data` *(np.ndarray)*: Hyperspectral data
- `min_seed_size` *(int, default: 200)*: Minimum seed size in pixels
- `max_seed_size` *(int, optional)*: Maximum seed size in pixels
- `band_index` *(int, optional)*: Specific band to use
- `connectivity` *(int, default: 2)*: Connectivity for labeling (1 or 2)

**Returns:**
- `mask` *(np.ndarray)*: Labeled mask
- `n_seeds` *(int)*: Number of seeds

**How it works:**
1. Otsu thresholding
2. Morphological cleanup (closing + opening)
3. Connected component labeling
4. Size filtering
5. Shape filtering (eccentricity < 0.95, solidity > 0.7)

### combined_segmentation

```python
from hyperseed.core.segmentation.algorithms import combined_segmentation

mask, n_seeds = combined_segmentation(
    data,
    min_seed_size=200,
    max_seed_size=None,
    band_index=None,
    methods=["threshold", "watershed"]
)
```

**Parameters:**
- `data` *(np.ndarray)*: Hyperspectral data
- `min_seed_size` *(int, default: 200)*: Minimum seed size in pixels
- `max_seed_size` *(int, optional)*: Maximum seed size in pixels
- `band_index` *(int, optional)*: Specific band to use
- `methods` *(list, default: ["threshold", "watershed"])*: List of algorithms to combine

**Returns:**
- `mask` *(np.ndarray)*: Labeled mask
- `n_seeds` *(int)*: Number of seeds

**How it works:**
1. Runs each specified algorithm
2. Converts results to binary masks
3. Combines using majority voting (consensus threshold = ceil(n_methods/2))
4. Labels final consensus mask
5. Size filtering

### apply_morphological_operations

```python
from hyperseed.core.segmentation.algorithms import apply_morphological_operations

cleaned_mask = apply_morphological_operations(
    mask,
    operations=["closing", "opening"],
    kernel_size=3
)
```

**Parameters:**
- `mask` *(np.ndarray)*: Binary or labeled mask
- `operations` *(list, default: ["closing", "opening"])*: Operations to apply in order
- `kernel_size` *(int, default: 3)*: Size of morphological kernel (disk shape)

**Returns:**
- `cleaned_mask` *(np.ndarray)*: Processed mask

**Available operations:**
- **closing**: Fills small holes (dilation → erosion)
- **opening**: Removes small protrusions (erosion → dilation)
- **erosion**: Shrinks objects
- **dilation**: Expands objects

## Complete Example

```python
import numpy as np
from hyperseed.core.io.envi_reader import ENVIReader
from hyperseed.core.calibration.reflectance import ReflectanceCalibrator
from hyperseed.core.preprocessing.pipeline import PreprocessingPipeline
from hyperseed.core.segmentation.segmenter import SeedSegmenter
from hyperseed.config.settings import Settings, SegmentationConfig

# Load data
reader = ENVIReader("path/to/data.hdr")
data = reader.read_data()
wavelengths = reader.get_wavelengths()

# Calibrate
calibrator = ReflectanceCalibrator(clip_negative=True, clip_max=1.0)
calibrated, reader = calibrator.calibrate_from_directory("path/to/dataset")

# Preprocess (minimal for segmentation)
settings = Settings()
settings.preprocessing.method = "minimal"
preprocessor = PreprocessingPipeline(settings.preprocessing)
processed = preprocessor.fit_transform(calibrated)

# Configure segmentation
seg_config = SegmentationConfig(
    algorithm="watershed",
    min_pixels=200,
    morphology_operations=True,
    morphology_kernel_size=5,
    filter_border_seeds=True,
    remove_outliers=True,
    outlier_min_area=75,
    outlier_max_area=1800,
    outlier_iqr_lower=1.5,
    outlier_iqr_upper=3.0,
    use_shape_filtering=True,
    outlier_eccentricity=0.90,
    outlier_solidity=0.75
)

# Segment
segmenter = SeedSegmenter(seg_config)
mask, n_seeds = segmenter.segment(processed, validate=True)

print(f"Segmented {n_seeds} seeds")

# Get properties
properties = segmenter.get_seed_properties()
for prop in properties[:5]:  # Show first 5
    print(f"Seed {prop['label']}: {prop['area']} pixels at {prop['centroid']}")

# Get validation stats
stats = segmenter.get_validation_stats()
if stats:
    print(f"\nValidation:")
    print(f"  Initial: {stats['initial_count']} seeds")
    print(f"  Final: {stats['final_count']} seeds")
    print(f"  Removed by size: {stats['removed_by_size']}")

# Visualize
segmenter.visualize(
    calibrated,
    save_path="segmentation_result.png",
    show_labels=True,
    show_boundaries=True
)

# Export
segmenter.export_mask("segmentation_mask.npy")
segmenter.export_properties("seed_properties.csv")

# Get description
desc = segmenter.describe()
print(f"\nDescription: {desc}")
```

## See Also

- **[Segmentation Guide](../user-guide/segmentation.md)**: Detailed algorithm descriptions and usage
- **[Configuration](../getting-started/configuration.md#segmentation-settings)**: Configuration reference
- **[CLI Reference](../cli-reference/segment.md)**: Command-line segmentation
