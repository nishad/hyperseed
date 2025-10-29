# Hyperseed 🌱

An experimental Python tool for hyperspectral seed image analysis. Inspired by (Reddy, et.al 2023, Sensors)[https://pmc.ncbi.nlm.nih.gov/articles/PMC9961513/]

## 🌟 Features

- **ENVI Format Support**: Read and process ENVI format hyperspectral data from Specim SWIR cameras
- **Automatic Calibration**: White/dark reference correction with automatic bad pixel interpolation
- **Intelligent Outlier Removal**: Automatically detect and remove reference objects, calibration targets, and anomalies
- **Advanced Preprocessing**: Multiple spectral preprocessing methods (SNV, derivatives, baseline correction, etc.)
- **Smart Segmentation**: Multiple algorithms for accurate seed detection and isolation
- **Spectral Extraction**: Extract average spectral signatures from individual seeds
- **Spatial Preservation**: Maintain seed coordinates and morphological properties
- **Comprehensive Visualizations**: Auto-generate distribution, segmentation, and spectral plots
- **Batch Processing**: Process multiple datasets efficiently with parallel support
- **Flexible Configuration**: YAML-based configuration system
- **User-Friendly CLI**: Intuitive command-line interface with rich output

## 📋 Requirements

- Python 3.10 or higher
- 8GB+ RAM recommended
- Optional: GPU with Metal (macOS) or CUDA support for acceleration

## 🚀 Installation

### From Source

```bash
# Clone the repository
git clone [repository-url]
cd hyperseed

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## 📖 Quick Start

### Basic Usage

Analyze a single hyperspectral dataset:

```bash
hyperseed analyze dataset/sample_001 --output results.csv --export-plots
```

### Recommended Usage

```bash
# Optimal settings for seed analysis with visualizations
hyperseed analyze dataset/sample_data \
    --output results.csv \
    --min-pixels 50 \
    --preprocess minimal \
    --export-plots
```

### Advanced Usage

```bash
# Batch process multiple datasets in parallel
hyperseed batch dataset/ \
    --output-dir results/ \
    --min-pixels 50 \
    --parallel 4

# Disable outlier removal if needed
hyperseed analyze dataset/sample \
    --output results.csv \
    --no-outlier-removal

# Generate and use custom configuration
hyperseed config --output my_config.yaml --preset minimal
hyperseed analyze dataset/sample --config my_config.yaml
```

## 📁 Expected Data Structure

Hyperseed expects hyperspectral data in the following directory structure:

```
dataset/
├── capture/
│   ├── data.raw              # Main hyperspectral data
│   ├── data.hdr              # ENVI header file
│   ├── WHITEREF_data.raw     # White reference
│   ├── WHITEREF_data.hdr
│   ├── DARKREF_data.raw      # Dark reference
│   └── DARKREF_data.hdr
├── calibrations/bpr/         # Optional bad pixel maps
│   ├── bprmap.bpr
│   └── bprmap.hdr
└── metadata/                 # Optional metadata
    └── data.xml
```

## ⚙️ Configuration

Create a configuration file to customize the analysis pipeline:

```yaml
# config.yaml
calibration:
  apply_calibration: true
  clip_negative: true
  clip_max: 1.0

preprocessing:
  method: minimal  # Options: minimal, standard, advanced, none
  snv: false
  smoothing: true
  smoothing_window: 11
  baseline_correction: false

segmentation:
  algorithm: watershed  # Options: threshold, watershed, connected, combined
  min_pixels: 50
  reject_overlapping: true
  remove_outliers: true  # Automatic outlier removal (enabled by default)
  outlier_min_area: 50
  outlier_max_area: 2000

output:
  format: csv
  include_plots: true
  include_coordinates: true
```

## 📊 Output Format

The tool generates multiple outputs:

### CSV Spectra File
```csv
seed_id,index,centroid_y,centroid_x,area,eccentricity,solidity,band_1000nm,band_1005nm,...
1,0,234.5,156.2,435,0.34,0.92,0.234,0.237,...
2,1,345.6,234.1,421,0.28,0.94,0.229,0.232,...
```

### Visualization Plots (when using --export-plots)
- `*_distribution.png`: Spatial and area distribution of seeds
- `*_segmentation.png`: Numbered seed visualization with boundaries
- `*_spectra.png`: Individual and mean spectral curves
- `*_spectra_statistics.png`: Statistical analysis of spectral variability

## 🔬 Processing Pipeline

1. **Data Loading**: Read ENVI format hyperspectral data
2. **Calibration**: Apply white/dark reference correction with bad pixel interpolation
3. **Preprocessing**: Apply spectral preprocessing methods (minimal recommended for segmentation)
4. **Segmentation**: Detect and isolate individual seeds using smart algorithms
5. **Validation**: Filter seeds based on size and shape criteria
6. **Outlier Removal**: Automatically remove reference objects and anomalies
7. **Extraction**: Extract average spectrum for each valid seed
8. **Export**: Save results with comprehensive spatial and spectral information

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hyperseed

# Run specific test module
pytest tests/test_preprocessing.py -v
```

### Code Quality

```bash
# Format code
black hyperseed/

# Check code style
ruff check hyperseed/

# Type checking
mypy hyperseed/
```

### Building Documentation

```bash
cd docs
make html
```

## 📚 API Usage

```python
from hyperseed import ENVIReader, Settings
from hyperseed.core.calibration import ReflectanceCalibrator
from hyperseed.core.preprocessing import PreprocessingPipeline
from hyperseed.core.segmentation import SeedSegmenter
from hyperseed.core.extraction import SpectralExtractor

# Load data
reader = ENVIReader("path/to/data.hdr")
data = reader.read_data()
wavelengths = reader.get_wavelengths()

# Calibrate (automatically handles bad pixel correction)
calibrator = ReflectanceCalibrator(clip_negative=True, clip_max=1.0)
calibrated, reader = calibrator.calibrate_from_directory("path/to/dataset")

# Preprocess
settings = Settings()
preprocessor = PreprocessingPipeline(settings.preprocessing)
processed = preprocessor.fit_transform(calibrated)

# Segment
segmenter = SeedSegmenter(settings.segmentation)
mask, n_seeds = segmenter.segment(processed)

# Extract spectra
extractor = SpectralExtractor()
results = extractor.extract(calibrated, mask, wavelengths)

# Save results
extractor.save_csv("results.csv")
```
