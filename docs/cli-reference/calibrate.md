# Calibration (via analyze command)

!!! info "No Standalone Calibrate Command"
    Hyperseed does not have a standalone `calibrate` command. Calibration is performed automatically as part of the `analyze` workflow.

## How Calibration Works

Calibration is automatically performed when you run the `analyze` command:

```bash
hyperseed analyze dataset/sample_001 --output results.csv
```

The calibration process:

1. Automatically finds white and dark reference files in the dataset
2. Applies reflectance calibration with bad pixel interpolation
3. Clips negative values and maximum reflectance (configurable)
4. Passes calibrated data to preprocessing and segmentation

## Calibration via Python API

For programmatic control over calibration, use the Python API:

```python
from hyperseed.core.calibration.reflectance import ReflectanceCalibrator

# Create calibrator with custom settings
calibrator = ReflectanceCalibrator(
    clip_negative=True,
    clip_max=1.0
)

# Calibrate from directory (auto-finds references)
calibrated_data, reader = calibrator.calibrate_from_directory(
    "path/to/dataset"
)

# Or calibrate with explicit references
calibrated = calibrator.calibrate(
    raw_data=raw_data,
    white_ref=white_reference,
    dark_ref=dark_reference
)
```

## Configuration Options

Configure calibration in your YAML config file:

```yaml
calibration:
  apply_calibration: true
  clip_negative: true
  clip_max: 1.0
  apply_bad_pixels: true
```

Then use with analyze:

```bash
hyperseed analyze dataset/sample_001 --config config.yaml
```

## See Also

- **[analyze command](extract.md)**: Full analysis pipeline including calibration
- **[Configuration Guide](../getting-started/configuration.md#calibration-settings)**: Calibration configuration options
- **[API Reference](../api-reference/calibration.md)**: ReflectanceCalibrator class documentation
