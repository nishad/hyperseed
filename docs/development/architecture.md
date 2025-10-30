# Architecture

Project structure and design overview.

## Project Structure

```
hyperseed/
├── hyperseed/
│   ├── cli/                 # Command-line interface
│   ├── config/              # Configuration management
│   ├── core/                # Core functionality
│   │   ├── io/              # ENVI file reading
│   │   ├── calibration/     # Reflectance calibration
│   │   ├── preprocessing/   # Spectral preprocessing
│   │   ├── segmentation/    # Seed detection
│   │   └── extraction/      # Spectral extraction
│   └── tests/               # Test suite
├── docs/                    # Documentation
├── pyproject.toml          # Project configuration
└── README.md
```

## Design Principles

- **Modular**: Each component is independent
- **Extensible**: Easy to add new algorithms
- **Testable**: Comprehensive test coverage
- **Documented**: Clear documentation for all APIs

## Core Components

### I/O Module
Handles reading ENVI format hyperspectral data.

### Calibration Module
Applies white/dark reference correction.

### Preprocessing Module
Spectral preprocessing transformations.

### Segmentation Module
Seed detection and isolation.

### Extraction Module
Spectral signature extraction.
