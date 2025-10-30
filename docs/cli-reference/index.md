# CLI Reference

Complete command-line interface reference for Hyperseed.

## Commands Overview

Hyperseed provides the following commands:

| Command | Purpose | Documentation |
|---------|---------|---------------|
| **analyze** | Process single dataset | [Details →](extract.md) |
| **batch** | Process multiple datasets | [Details →](batch.md) |
| **segment** | Segmentation only | [Details →](segment.md) |
| **config** | Generate configuration | Via `--help` |
| **info** | Show version/system info | Via `--help` |

## Global Options

Options available for all commands:

```bash
--help              Show help message and exit
--version           Show version and exit
-v, --verbose       Enable verbose output
-d, --debug         Enable debug mode
```

## Getting Help

Get help for any command:

```bash
# General help
hyperseed --help

# Command-specific help
hyperseed analyze --help
hyperseed batch --help
hyperseed segment --help
```

## Quick Examples

### Analyze a Single Dataset

```bash
hyperseed analyze dataset/sample_001 \
    --output results.csv \
    --export-plots
```

### Batch Process Multiple Datasets

```bash
hyperseed batch dataset/ \
    --output-dir results/
```

### Generate Configuration

```bash
hyperseed config --output my_config.yaml --preset minimal
```

## Next Steps

<div class="grid cards" markdown>

-   **[analyze command](extract.md)**

    Process single hyperspectral dataset

-   **[batch command](batch.md)**

    Process multiple datasets sequentially

-   **[segment command](segment.md)**

    Segmentation-only mode

-   **[config command]()**

    Generate configuration files

</div>
