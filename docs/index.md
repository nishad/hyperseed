# Welcome to Hyperseed

[![PyPI version](https://badge.fury.io/py/hyperseed.svg)](https://badge.fury.io/py/hyperseed)
[![Downloads](https://pepy.tech/badge/hyperseed)](https://pepy.tech/project/hyperseed)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/hyperseed.svg)](https://pypi.org/project/hyperseed/)

**Hyperseed** is an experimental Python tool for hyperspectral seed image analysis. Analyze hyperspectral imagery of plant seeds to extract spectral signatures with automatic calibration, intelligent segmentation, and comprehensive visualizations.

!!! info "Research Foundation"
    Inspired by [Reddy, et al. 2023, Sensors](https://pmc.ncbi.nlm.nih.gov/articles/PMC9961513/)

## Features

<div class="grid cards" markdown>

-   :material-file-eye:{ .lg .middle } **ENVI Format Support**

    ---

    Read and process ENVI format hyperspectral data from Specim SWIR cameras

-   :material-auto-fix:{ .lg .middle } **Automatic Calibration**

    ---

    White/dark reference correction with automatic bad pixel interpolation

-   :material-chart-scatter-plot:{ .lg .middle } **Intelligent Outlier Removal**

    ---

    Automatically detect and remove reference objects, calibration targets, and anomalies

-   :material-waveform:{ .lg .middle } **Advanced Preprocessing**

    ---

    Multiple spectral preprocessing methods: SNV, derivatives, baseline correction, and more

-   :material-grid:{ .lg .middle } **Smart Segmentation**

    ---

    Multiple algorithms for accurate seed detection and isolation

-   :material-eye:{ .lg .middle } **Spectral Extraction**

    ---

    Extract average spectral signatures from individual seeds with spatial preservation

-   :material-chart-line:{ .lg .middle } **Comprehensive Visualizations**

    ---

    Auto-generate distribution, segmentation, and spectral plots

-   :material-rocket-launch:{ .lg .middle } **Batch Processing**

    ---

    Process multiple datasets efficiently

</div>

## Quick Start

Get started with Hyperseed in minutes:

=== "Installation"

    ```bash
    pip install hyperseed
    ```

=== "Basic Usage"

    ```bash
    # Analyze a single dataset
    hyperseed analyze dataset/sample_001 \
        --output results.csv \
        --export-plots
    ```

=== "Batch Processing"

    ```bash
    # Process multiple datasets
    hyperseed batch dataset/ \
        --output-dir results/ \
        --min-pixels 50
    ```

## Requirements

- **Python**: 3.10 or higher
- **RAM**: 8GB+ recommended

## Processing Pipeline

```mermaid
graph LR
    A[ENVI Data] --> B[Calibration]
    B --> C[Preprocessing]
    C --> D[Segmentation]
    D --> E[Validation]
    E --> F[Outlier Removal]
    F --> G[Spectral Extraction]
    G --> H[Export Results]
```

1. **Data Loading**: Read ENVI format hyperspectral data
2. **Calibration**: Apply white/dark reference correction with bad pixel interpolation
3. **Preprocessing**: Apply spectral preprocessing methods
4. **Segmentation**: Detect and isolate individual seeds
5. **Validation**: Filter seeds based on size and shape criteria
6. **Outlier Removal**: Automatically remove reference objects and anomalies
7. **Extraction**: Extract average spectrum for each valid seed
8. **Export**: Save results with comprehensive information

## Output Examples

### CSV Spectra File
```csv
seed_id,index,centroid_y,centroid_x,area,eccentricity,solidity,band_1000nm,band_1005nm,...
1,0,234.5,156.2,435,0.34,0.92,0.234,0.237,...
2,1,345.6,234.1,421,0.28,0.94,0.229,0.232,...
```

### Visualization Plots

When using `--export-plots`, Hyperseed generates:

- **Distribution Plot**: Spatial and area distribution of seeds
- **Segmentation Plot**: Numbered seed visualization with boundaries
- **Spectra Plot**: Individual and mean spectral curves
- **Statistics Plot**: Statistical analysis of spectral variability

## Next Steps

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **[Quick Start Guide](getting-started/quick-start.md)**

    ---

    Learn the basics in 5 minutes with a step-by-step tutorial

-   :material-book-open-variant:{ .lg .middle } **[User Guide](user-guide/index.md)**

    ---

    Comprehensive documentation for all features and workflows

-   :material-console:{ .lg .middle } **[CLI Reference](cli-reference/index.md)**

    ---

    Detailed command-line interface documentation

-   :material-code-braces:{ .lg .middle } **[API Reference](api-reference/index.md)**

    ---

    Python API documentation for advanced users

</div>

## Development Status

!!! warning "Alpha Release"
    Hyperseed is currently in **alpha** (v0.1.0-alpha.3). The API may change between releases. See the [changelog](about/changelog.md) for version history.

## License

Hyperseed is released under the [MIT License](about/license.md).

## Credits

Logo icon ["Sprouting Seed"](https://thenounproject.com/icon/sprouting-seed-5803652/) by [4urbrand](https://thenounproject.com/4urbrand/) from [The Noun Project](https://thenounproject.com/), used under [Creative Commons license](https://creativecommons.org/licenses/by/3.0/).
