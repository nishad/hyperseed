# Installation

This guide covers different methods to install Hyperseed.

## Using pip (Recommended)

The simplest way to install Hyperseed is via pip from PyPI:

```bash
pip install hyperseed
```

### Verify Installation

After installation, verify that Hyperseed is installed correctly:

```bash
hyperseed --version
```

You should see output like:
```
Hyperseed version 0.1.0-alpha.3
```

## Using pip from GitHub

To install the latest development version directly from GitHub:

```bash
pip install --no-cache-dir --force-reinstall https://github.com/nishad/hyperseed/archive/main.zip
```

This method is useful for:
- Getting the absolute latest features
- Testing bug fixes before they're released
- Contributing to development

## From Source

For development or customization, install from source:

```bash
# 1. Clone the repository
git clone https://github.com/nishad/hyperseed
cd hyperseed

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Development Installation Benefits

Installing with `-e ".[dev]"` provides:

- **Editable mode**: Changes to code are immediately available
- **Development tools**: Testing, linting, formatting tools included
- **Documentation tools**: MkDocs for building these docs

### Development Dependencies

The development installation includes:

- **Testing**: pytest, pytest-cov, pytest-mock
- **Code Quality**: black, ruff, mypy
- **Documentation**: mkdocs-material, mkdocstrings
- **Notebooks**: jupyter, ipykernel

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|------------|
| **Python** | 3.10 or higher |
| **RAM** | 4GB (minimum) |
| **Disk Space** | 500MB for installation |
| **OS** | macOS, Linux, or Windows |

### Recommended Requirements

| Component | Recommendation |
|-----------|---------------|
| **Python** | 3.11 or 3.12 |
| **RAM** | 8GB+ |
| **Disk Space** | 2GB+ for data and results |
| **CPU** | Multi-core for batch processing |

## Virtual Environments

We strongly recommend using a virtual environment to avoid dependency conflicts.

### Using venv (built-in)

```bash
# Create environment
python -m venv hyperseed-env

# Activate (macOS/Linux)
source hyperseed-env/bin/activate

# Activate (Windows)
hyperseed-env\Scripts\activate

# Install hyperseed
pip install hyperseed
```

### Using conda

```bash
# Create environment
conda create -n hyperseed python=3.11

# Activate
conda activate hyperseed

# Install hyperseed
pip install hyperseed
```

## Updating Hyperseed

To update to the latest version:

```bash
pip install --upgrade hyperseed
```

## Uninstalling

To remove Hyperseed:

```bash
pip uninstall hyperseed
```

## Troubleshooting Installation

### Common Issues

??? question "Error: Python version not supported"

    Hyperseed requires Python 3.10+. Check your Python version:
    ```bash
    python --version
    ```

    If needed, install a newer Python version from [python.org](https://www.python.org/downloads/).

??? question "Error: Permission denied"

    On macOS/Linux, you may need to use:
    ```bash
    pip install --user hyperseed
    ```

    Or use a virtual environment (recommended).

??? question "Error: Could not find a version that satisfies the requirement"

    Update pip to the latest version:
    ```bash
    pip install --upgrade pip
    ```

??? question "Installation is very slow"

    Some dependencies (like NumPy, SciPy) can be large. Consider:

    - Using a wired internet connection
    - Installing with conda if you have it available
    - Using a mirror closer to your location

### Getting Help

If you encounter issues not covered here:

1. Check [GitHub Issues](https://github.com/nishad/hyperseed/issues) for similar problems
2. Create a new issue with:
    - Your OS and Python version
    - Complete error message
    - Installation method you tried

## Next Steps

Once installed, proceed to the [Quick Start Guide →](quick-start.md)
