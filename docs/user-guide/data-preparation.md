# Data Preparation

Learn how to prepare hyperspectral data for analysis with Hyperseed.

## Expected Data Structure

Hyperseed expects hyperspectral data in ENVI format with specific file organization:

```
dataset/
└── sample_name/
    ├── capture/
    │   ├── data.raw              # Main hyperspectral datacube
    │   ├── data.hdr              # ENVI header file
    │   ├── WHITEREF_data.raw     # White reference
    │   ├── WHITEREF_data.hdr
    │   ├── DARKREF_data.raw      # Dark reference
    │   └── DARKREF_data.hdr
    ├── calibrations/bpr/         # Optional: bad pixel maps
    │   ├── bprmap.bpr
    │   └── bprmap.hdr
    └── metadata/                 # Optional: metadata
        └── data.xml
```

## Required Files

### Main Data Files

**`data.raw`** *(required)*
: Binary file containing the hyperspectral datacube (Y × X × Wavelengths)

**`data.hdr`** *(required)*
: ENVI header file with metadata (dimensions, wavelengths, data type)

### Reference Files

**`WHITEREF_data.raw`** *(required)*
: White reference image for reflectance calibration

**`WHITEREF_data.hdr`** *(required)*
: Header file for white reference

**`DARKREF_data.raw`** *(required)*
: Dark reference image for reflectance calibration

**`DARKREF_data.hdr`** *(required)*
: Header file for dark reference

!!! warning "Reference Requirements"
    White and dark references are **required** for reflectance calibration. Without them, analysis cannot proceed.

## ENVI Format

### What is ENVI Format?

ENVI (Environment for Visualizing Images) is a standard format for hyperspectral data consisting of:

1. **Binary data file** (`.raw`, `.dat`, or no extension)
2. **ASCII header file** (`.hdr`)

### Header File Structure

A minimal ENVI header contains:

```text
ENVI
samples = 640
lines = 480
bands = 224
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bil
byte order = 0
wavelength = {
 1000.0, 1005.0, 1010.0, ...
}
```

Important parameters:

**`samples`**
: Number of pixels per line (X dimension)

**`lines`**
: Number of lines (Y dimension)

**`bands`**
: Number of spectral bands (wavelengths)

**`data type`**
: Data format (1=byte, 2=int16, 4=float32, etc.)

**`interleave`**
: Data organization (BIL, BIP, or BSQ)

**`wavelength`**
: List of wavelengths in nanometers

## Supported Configurations

### Data Types

Hyperseed supports these ENVI data types:

| Data Type | Description | Typical Use |
|-----------|-------------|-------------|
| 1 | 8-bit unsigned integer | Raw sensor data |
| 2 | 16-bit signed integer | Raw sensor data |
| 4 | 32-bit float | Reflectance data |
| 5 | 64-bit float | High-precision reflectance |
| 12 | 16-bit unsigned integer | Raw sensor data |

### Interleave Formats

All three interleave formats are supported:

**BIL (Band Interleaved by Line)** *(recommended)*
: `[band1_line1, band2_line1, ..., band1_line2, band2_line2, ...]`
: Most common for pushbroom sensors

**BIP (Band Interleaved by Pixel)**
: `[band1_pixel1, band2_pixel1, ..., band1_pixel2, band2_pixel2, ...]`
: Good for per-pixel processing

**BSQ (Band Sequential)**
: `[all_band1, all_band2, all_band3, ...]`
: Good for band-wise processing

## Data from Specim Cameras

Hyperseed is optimized for Specim SWIR cameras but works with any ENVI-format data.

### Specim Data Export

When exporting from Specim software:

1. ✅ Export in **ENVI format**
2. ✅ Include **white reference** image
3. ✅ Include **dark reference** image
4. ✅ Export **wavelength calibration** in header
5. ✅ Export **bad pixel map** (optional but recommended)

### Typical Specim Wavelength Ranges

| Camera | Wavelength Range | Spectral Resolution |
|--------|------------------|---------------------|
| FX10 | 400-1000 nm | ~5 nm |
| FX17 | 900-1700 nm | ~8 nm |
| FX50 | 1000-2500 nm | ~10 nm |

## Verifying Your Data

### Check File Structure

```bash
# Verify directory structure
tree dataset/sample_001

# Should show:
# sample_001/
# └── capture/
#     ├── data.raw
#     ├── data.hdr
#     ├── WHITEREF_data.raw
#     ├── WHITEREF_data.hdr
#     ├── DARKREF_data.raw
#     └── DARKREF_data.hdr
```

### Check Header Files

```bash
# View header file
cat dataset/sample_001/capture/data.hdr

# Verify:
# - samples, lines, bands are positive integers
# - wavelength list has correct number of entries
# - data type is valid (1, 2, 4, 5, or 12)
```

### Test Load

```bash
# Test if Hyperseed can load your data
hyperseed analyze dataset/sample_001 --output test.csv

# If successful, data is correctly formatted
```

## Common Data Issues

### Issue 1: Missing Reference Files

??? question "Error: Could not find white/dark reference"

    **Problem**: Reference files are missing or incorrectly named.

    **Solution**:
    ```bash
    # References must be named:
    WHITEREF_data.raw / WHITEREF_data.hdr
    DARKREF_data.raw / DARKREF_data.hdr

    # Check your capture/ directory
    ls dataset/sample_001/capture/
    ```

### Issue 2: Dimension Mismatch

??? question "Error: White reference dimensions don't match data"

    **Problem**: References have different dimensions than main data.

    **Solution**: All files must have the same `samples` and `bands` values in their headers.

    ```bash
    # Check dimensions in each header
    grep -E "samples|bands" dataset/sample_001/capture/*.hdr
    ```

### Issue 3: Missing Wavelengths

??? question "Error: No wavelength information in header"

    **Problem**: Header file doesn't contain wavelength list.

    **Solution**: Add wavelength information to `.hdr` file:
    ```text
    wavelength = {
     1000.0, 1005.0, 1010.0, ...
    }
    ```

### Issue 4: Bad Pixel Issues

??? question "Warning: Detected bad pixels in reference"

    **Problem**: Bad/dead pixels in reference images.

    **Solution**: Hyperseed automatically interpolates bad pixels. If you have a bad pixel map (`.bpr`), place it in `calibrations/bpr/` directory:
    ```
    dataset/sample_001/
    └── calibrations/bpr/
        ├── bprmap.bpr
        └── bprmap.hdr
    ```

## Organizing Multiple Datasets

For batch processing, organize datasets consistently:

```
dataset/
├── sample_001/
│   └── capture/
│       ├── data.raw, data.hdr
│       ├── WHITEREF_*.raw, WHITEREF_*.hdr
│       └── DARKREF_*.raw, DARKREF_*.hdr
├── sample_002/
│   └── capture/
│       └── ...
├── sample_003/
│   └── capture/
│       └── ...
└── ...
```

Then batch process:

```bash
hyperseed batch dataset/ --output-dir results/
```

## Next Steps

- **[Quick Start](../getting-started/quick-start.md)**: Run your first analysis
- **[Preprocessing](preprocessing.md)**: Learn about spectral preprocessing
- **[Batch Processing](batch-processing.md)**: Process multiple datasets
- **[Troubleshooting](troubleshooting.md)**: Solve common problems
