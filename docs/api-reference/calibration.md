# Calibration

Reflectance calibration with white and dark references.

## ReflectanceCalibrator

The ReflectanceCalibrator class applies reflectance calibration.

```python
from hyperseed.core.calibration.reflectance import ReflectanceCalibrator

# Create calibrator
calibrator = ReflectanceCalibrator(
    clip_negative=True,
    clip_max=1.0
)

# Calibrate data
calibrated_data, reader = calibrator.calibrate_from_directory("path/to/dataset")
```

## Methods

**`__init__(clip_negative, clip_max)`**
: Initialize calibrator with settings

**`calibrate(data, white_ref, dark_ref)`**
: Calibrate using references

**`calibrate_from_directory(dataset_path)`**
: Load and calibrate from directory

Full API documentation coming soon. For now, use Python's built-in help:

```python
from hyperseed.core.calibration.reflectance import ReflectanceCalibrator
help(ReflectanceCalibrator)
```
