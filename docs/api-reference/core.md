# Core Modules

Core functionality and base classes for Hyperseed.

## Settings

The Settings class provides configuration management for Hyperseed's analysis pipeline.

```python
from hyperseed.config.settings import Settings

# Create settings with defaults
settings = Settings()

# Access configuration sections
calibration_config = settings.calibration
preprocessing_config = settings.preprocessing
segmentation_config = settings.segmentation
```

See the [Configuration Guide](../getting-started/configuration.md) for detailed configuration options.

## Module Documentation

Full API documentation coming soon. For now, use Python's built-in help:

```python
import hyperseed
help(hyperseed.config.settings.Settings)
```
