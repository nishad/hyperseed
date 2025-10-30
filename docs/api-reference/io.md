# I/O Operations

ENVI file reading and data loading.

## ENVIReader

The ENVIReader class handles reading ENVI format hyperspectral data.

```python
from hyperseed.core.io.envi_reader import ENVIReader

# Load ENVI data
reader = ENVIReader("path/to/data.hdr")
data = reader.read_data()
wavelengths = reader.get_wavelengths()
metadata = reader.get_metadata()
```

## Methods

**`__init__(header_path)`**
: Initialize reader with ENVI header file path

**`read_data()`**
: Read the hyperspectral datacube

**`get_wavelengths()`**
: Get wavelength list from header

**`get_metadata()`**
: Get all metadata from header

Full API documentation coming soon. For now, use Python's built-in help:

```python
from hyperseed.core.io.envi_reader import ENVIReader
help(ENVIReader)
```
