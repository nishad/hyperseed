# API Reference

Python API documentation for Hyperseed.

## Overview

Hyperseed provides a Python API for programmatic control of the hyperspectral analysis pipeline. This allows you to:

- Integrate Hyperseed into your own workflows
- Customize processing beyond CLI capabilities
- Build custom applications using Hyperseed components

## Modules

<div class="grid cards" markdown>

-   **[Core Modules](core.md)**

    Core functionality and base classes

-   **[I/O Operations](io.md)**

    ENVI file reading and data loading

-   **[Calibration](calibration.md)**

    Reflectance calibration with white/dark references

-   **[Preprocessing](preprocessing.md)**

    Spectral preprocessing methods

-   **[Segmentation](segmentation.md)**

    Seed detection and isolation algorithms

-   **[Extraction](extraction.md)**

    Spectral signature extraction

</div>

## Quick Start Example

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

## Installation for API Use

```bash
# Install from PyPI
pip install hyperseed

# Or from source for development
git clone https://github.com/nishad/hyperseed
cd hyperseed
pip install -e ".[dev]"
```

## Jupyter Notebooks

Hyperseed works great in Jupyter notebooks for interactive analysis:

```python
# In Jupyter
import hyperseed
from hyperseed import ENVIReader

# Load and visualize
reader = ENVIReader("data.hdr")
data = reader.read_data()

# Plot RGB composite
import matplotlib.pyplot as plt
rgb = data[:, :, [100, 50, 10]]  # Select R, G, B bands
plt.imshow(rgb)
```

## Next Steps

Explore detailed API documentation for each module:

- **[Core](core.md)**: Base classes and utilities
- **[I/O](io.md)**: Data loading and ENVI format handling
- **[Calibration](calibration.md)**: Reflectance calibration
- **[Preprocessing](preprocessing.md)**: Spectral preprocessing
- **[Segmentation](segmentation.md)**: Seed segmentation
- **[Extraction](extraction.md)**: Spectral extraction
