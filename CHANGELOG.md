# Changelog

All notable changes to the Hyperseed project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.3] - 2025-01-29

### Added
- Initial release of Hyperseed - Hyperspectral Seed Image Analysis Tool
- Core functionality for hyperspectral image processing and analysis
- Support for ENVI format hyperspectral data files
- Comprehensive preprocessing pipeline with multiple methods:
  - Standard Normal Variate (SNV)
  - Smoothing (Savitzky-Golay, Gaussian, Moving Average, Median)
  - Baseline correction (Polynomial, Rubberband, ASLS)
  - Derivative computation (1st, 2nd order)
  - Normalization (Min-Max, Max, Area, Vector, Peak)
  - Multiplicative Scatter Correction (MSC)
  - Detrending
- Advanced segmentation algorithms for seed detection:
  - Threshold-based segmentation (Otsu, adaptive, manual)
  - Watershed segmentation
  - Connected components analysis
  - Combined approach for robust detection
- Spectral extraction and analysis capabilities:
  - Extract mean spectra from segmented regions
  - Statistical analysis of spectral signatures
  - Export to CSV and HDF5 formats
- Reflectance calibration with white and dark references
- Command-line interface (CLI) with multiple commands:
  - `analyze`: Full pipeline analysis
  - `segment`: Segmentation only
  - `batch`: Batch processing
  - `config`: Configuration management
  - `info`: System information
- Comprehensive test suite with ~80% code coverage
- Visualization tools for segmentation results
- Support for batch processing of multiple datasets
- Configuration system with presets (minimal, standard, advanced)
- Validation metrics for segmentation quality (IoU, Dice, F1-score)

### Features
- **Performance**: Optimized for large hyperspectral datasets
- **Flexibility**: Modular architecture allowing custom pipelines
- **Extensibility**: Plugin-ready architecture for custom algorithms
- **Documentation**: Comprehensive documentation and examples
- **Testing**: Extensive test coverage ensuring reliability

### Dependencies
- Python >=3.10
- NumPy, SciPy, scikit-learn, scikit-image
- OpenCV for image processing
- Pandas for data manipulation
- Matplotlib for visualization
- Click for CLI interface
- Rich for enhanced terminal output

### Known Issues
- Some empty module directories reserved for future features
- Visualization requires display environment (not suitable for headless servers without proper configuration)

### Contributors
- Nishad Thalhath - Lead Developer
- Deepa Kasaragod - Contributor

### Note
This is an **alpha release** (alpha.3). The API and features are still under active development and may change in future releases.

[0.1.0-alpha.3]: https://github.com/nishad/hyperseed/releases/tag/v0.1.0a3