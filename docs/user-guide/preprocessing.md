# Preprocessing

Spectral preprocessing transforms raw hyperspectral data to improve analysis quality, reduce noise, and enhance relevant features.

## Overview

Hyperseed provides comprehensive preprocessing methods for hyperspectral data, including:

- **SNV** (Standard Normal Variate) - Remove multiplicative scatter effects
- **Smoothing** - Reduce noise with Savitzky-Golay filtering
- **Baseline Correction** - Remove baseline drift
- **Derivatives** - Emphasize spectral features
- **MSC** (Multiplicative Scatter Correction) - Normalize scatter
- **Detrending** - Remove linear trends

## When to Use Preprocessing

### For Segmentation

!!! tip "Minimal Preprocessing Recommended"
    Heavy preprocessing can reduce segmentation quality. Use the `minimal` preset (smoothing only) for best results.

```bash
hyperseed analyze dataset/sample_001 \
    --output results.csv \
    --preprocess minimal
```

### For Spectral Analysis

For analyzing extracted spectra, more preprocessing may improve results:

```bash
hyperseed analyze dataset/sample_001 \
    --output results.csv \
    --preprocess standard  # or advanced
```

## Preprocessing Presets

Hyperseed provides four preprocessing presets via the `--preprocess` CLI option or configuration files.

### minimal (Recommended for Segmentation)

**Enables**: Smoothing only

**Use when**:
- Performing seed segmentation
- Maximizing segmentation accuracy
- Keeping spectral features intact

**Configuration**:
```yaml
preprocessing:
  method: minimal
```

**What it does**:
- Applies Savitzky-Golay smoothing (window=11, polyorder=3)
- Reduces noise while preserving spectral features
- No other transformations

### standard (Balanced)

**Enables**: SNV + Smoothing + Baseline Correction

**Use when**:
- Performing general spectral analysis
- Balancing noise reduction and feature preservation
- Working with spectra that have scatter and baseline issues

**Configuration**:
```yaml
preprocessing:
  method: standard
```

**What it does**:
1. Applies SNV to remove multiplicative scatter
2. Smooths with Savitzky-Golay filter
3. Removes baseline drift with polynomial fitting

### advanced (Chemometrics)

**Enables**: SNV + Smoothing + Baseline + 2nd Derivative + MSC + Detrend

**Use when**:
- Performing chemometric analysis
- Building classification/regression models
- Maximizing spectral transformation for pattern recognition

**Configuration**:
```yaml
preprocessing:
  method: advanced
```

**What it does**:
1. Applies SNV for scatter correction
2. Smooths with Savitzky-Golay filter
3. Removes baseline drift
4. Computes 2nd derivative (emphasizes subtle features)
5. Applies MSC for additional scatter normalization
6. Removes linear trends

!!! warning "Advanced Preprocessing"
    The `advanced` preset heavily transforms spectra. Use with caution - may not be suitable for all applications.

### none (Raw Data)

**Enables**: Nothing

**Use when**:
- Analyzing raw, uncalibrated data
- Comparing with/without preprocessing
- Data is already preprocessed externally

**Configuration**:
```yaml
preprocessing:
  method: none
```

## Individual Preprocessing Methods

### Standard Normal Variate (SNV)

**Purpose**: Remove multiplicative scatter effects caused by particle size differences and light scattering.

**How it works**: Each spectrum is centered (zero mean) and scaled (unit variance):

```
SNV(x) = (x - mean(x)) / std(x)
```

**When to use**:
- Spectra show scatter-related variations
- Working with powder or particulate samples
- Preparing data for chemometrics

**Configuration**:
```yaml
preprocessing:
  method: custom
  snv: true
```

### Smoothing

**Purpose**: Reduce random noise while preserving spectral features.

**Methods available** (hardcoded to Savitzky-Golay in presets):
- Savitzky-Golay filter (default)
- Moving average
- Gaussian filter

**How it works**: Fits local polynomials to smooth data points.

**Parameters**:
- `smoothing_window`: Window size (must be odd, range 3-51, default: 11)
- `smoothing_polyorder`: Polynomial order (range 1-5, default: 3)

**When to use**:
- Data has high-frequency noise
- Before derivative computation
- Always (included in all presets except `none`)

**Configuration**:
```yaml
preprocessing:
  method: custom
  smoothing: true
  smoothing_window: 11
  smoothing_polyorder: 3
```

!!! tip "Window Size Selection"
    - Smaller windows (5-9): Less smoothing, preserves detail
    - Medium windows (11-15): Balanced (recommended)
    - Larger windows (17-25): More smoothing, may lose features

### Baseline Correction

**Purpose**: Remove baseline drift or offset in spectra caused by instrument or sample effects.

**Method**: Polynomial fitting (hardcoded in presets).

**How it works**: Fits a polynomial to the baseline and subtracts it from the spectrum.

**Parameters**:
- `baseline_order`: Polynomial order (range 1-5, default: 2)
  - Order 1: Linear baseline
  - Order 2: Quadratic baseline (default)
  - Order 3+: Higher-order curves

**When to use**:
- Spectra have visible baseline drift
- Background fluorescence is present
- Offset between spectra needs normalization

**Configuration**:
```yaml
preprocessing:
  method: custom
  baseline_correction: true
  baseline_order: 2
```

### Derivatives

**Purpose**: Emphasize subtle spectral features and remove baseline effects by computing the rate of change.

**Orders available**:
- 0: No derivative (original spectrum)
- 1: First derivative (rate of change)
- 2: Second derivative (curvature)

**How it works**: Computes derivatives using Savitzky-Golay filter.

**When to use**:
- Overlapping peaks need separation
- Baseline variations must be eliminated
- Building classification models (especially 2nd derivative)

**Trade-offs**:
- ✅ Enhances subtle features
- ✅ Removes baseline completely
- ⚠️ Amplifies noise (smooth first!)
- ⚠️ Changes interpretation (not absorbance anymore)

**Configuration**:
```yaml
preprocessing:
  method: custom
  derivative: 2  # 0, 1, or 2
```

!!! warning "Derivative + Noise"
    Always smooth before computing derivatives, or noise will be amplified!

### Multiplicative Scatter Correction (MSC)

**Purpose**: Correct for scatter effects by normalizing all spectra to a reference spectrum.

**How it works**:
1. Computes a reference spectrum (mean of all spectra)
2. Fits each spectrum to the reference via linear regression
3. Corrects using the regression coefficients

**When to use**:
- Scatter effects vary between samples
- After SNV, if scatter remains
- Preparing data for multivariate analysis

**Configuration**:
```yaml
preprocessing:
  method: custom
  msc: true
```

**Comparison with SNV**:
- SNV: Operates on each spectrum independently
- MSC: Uses a reference spectrum for correction
- Often used together in chemometrics

### Detrending

**Purpose**: Remove linear trends from spectra.

**Method**: Linear detrending (hardcoded).

**How it works**: Fits and subtracts a straight line from each spectrum.

**When to use**:
- Spectra have linear slope
- Simple baseline correction needed
- Alternative to polynomial baseline correction

**Configuration**:
```yaml
preprocessing:
  method: custom
  detrend: true
```

## Custom Preprocessing

For complete control over preprocessing methods, use `method: custom` in your configuration:

```yaml
preprocessing:
  method: custom

  # Enable only what you need
  snv: true
  smoothing: true
  smoothing_window: 15
  smoothing_polyorder: 3
  baseline_correction: false
  derivative: 1
  msc: false
  detrend: false
```

Then use with analyze:

```bash
hyperseed analyze dataset/sample_001 \
    --config custom_preprocess.yaml
```

## Method Comparison

| Method | Purpose | When to Use | Avoid When |
|--------|---------|-------------|------------|
| **SNV** | Remove scatter | Powder samples, varied particles | Need absolute values |
| **Smoothing** | Reduce noise | Always (except raw analysis) | Over-smoothing loses features |
| **Baseline** | Remove drift | Visible baseline issues | Baseline is meaningful |
| **Derivative** | Enhance features | Overlapping peaks | High noise |
| **MSC** | Normalize scatter | Multivariate analysis | Single spectrum analysis |
| **Detrend** | Remove slope | Linear baseline | Non-linear baseline |

## Preprocessing Order

When using `method: custom`, preprocessing is applied in this order:

1. **SNV** (if enabled)
2. **Smoothing** (if enabled)
3. **Baseline Correction** (if enabled)
4. **Derivative** (if > 0)
5. **MSC** (if enabled)
6. **Detrending** (if enabled)

!!! info "Order Matters"
    The order of operations is fixed and optimized. Smoothing before derivatives is crucial to avoid noise amplification.

## Examples

### Example 1: Segmentation Focus

```bash
# Use minimal preprocessing for best segmentation
hyperseed analyze dataset/seeds \
    --preprocess minimal \
    --export-plots
```

### Example 2: Spectral Analysis

```yaml
# config.yaml
preprocessing:
  method: standard  # SNV + smooth + baseline
  smoothing_window: 15  # More smoothing
```

```bash
hyperseed analyze dataset/seeds --config config.yaml
```

### Example 3: Custom Pipeline

```yaml
# custom.yaml
preprocessing:
  method: custom
  snv: true           # Remove scatter
  smoothing: true
  smoothing_window: 13
  smoothing_polyorder: 3
  baseline_correction: true
  baseline_order: 2
  derivative: 1       # 1st derivative
  msc: false
  detrend: false
```

```bash
hyperseed analyze dataset/seeds --config custom.yaml
```

## Troubleshooting

### Issue: Segmentation quality decreased after preprocessing

**Solution**: Use `minimal` preset for segmentation:
```bash
hyperseed analyze dataset/sample --preprocess minimal
```

### Issue: Spectra still noisy after smoothing

**Solution**: Increase smoothing window:
```yaml
preprocessing:
  method: custom
  smoothing: true
  smoothing_window: 17  # Larger window
```

### Issue: Baseline correction over-corrected

**Solution**: Reduce polynomial order:
```yaml
preprocessing:
  baseline_order: 1  # Linear instead of quadratic
```

### Issue: 2nd derivative too noisy

**Solution**: Increase smoothing before derivative:
```yaml
preprocessing:
  method: custom
  smoothing: true
  smoothing_window: 17  # More smoothing
  derivative: 2
```

## See Also

- **[Configuration Guide](../getting-started/configuration.md#preprocessing-settings)**: Complete configuration reference
- **[API Reference](../api-reference/preprocessing.md)**: Programmatic preprocessing
- **[CLI Reference](../cli-reference/extract.md)**: Command-line options
