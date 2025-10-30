# Extraction

Spectral signature extraction from segmented seeds.

## SpectralExtractor

The SpectralExtractor class extracts spectral signatures from segmented regions.

```python
from hyperseed.core.extraction.extractor import SpectralExtractor

# Create extractor
extractor = SpectralExtractor()

# Extract spectra
results = extractor.extract(
    calibrated_data,
    segmentation_mask,
    wavelengths
)

# Save results
extractor.save_csv("results.csv")
extractor.save_hdf5("results.h5")
```

## Methods

**`__init__()`**
: Initialize extractor

**`extract(data, mask, wavelengths)`**
: Extract spectral signatures from masked regions

**`save_csv(filepath)`**
: Save results to CSV file

**`save_hdf5(filepath)`**
: Save results to HDF5 file

Full API documentation coming soon. For now, use Python's built-in help:

```python
from hyperseed.core.extraction.extractor import SpectralExtractor
help(SpectralExtractor)
```
