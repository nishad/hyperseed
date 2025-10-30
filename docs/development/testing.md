# Testing

Guide to running tests and ensuring code quality.

## Running Tests

### All Tests

```bash
pytest
```

### With Coverage

```bash
pytest --cov=hyperseed --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Specific Module

```bash
pytest tests/test_preprocessing.py -v
```

## Test Structure

```
tests/
├── fixtures/          # Test data and fixtures
├── test_calibration.py
├── test_preprocessing.py
├── test_segmentation.py
└── test_extraction.py
```

## Writing Tests

Use pytest fixtures for setup:

```python
import pytest

@pytest.fixture
def sample_data():
    return np.random.rand(100, 100, 224)

def test_preprocessing(sample_data):
    from hyperseed.core.preprocessing import PreprocessingPipeline
    pipeline = PreprocessingPipeline()
    result = pipeline.fit_transform(sample_data)
    assert result.shape == sample_data.shape
```

## Code Quality

Ensure code quality before submitting:

```bash
# Format
black hyperseed/

# Lint
ruff check hyperseed/

# Type check
mypy hyperseed/

# Test
pytest --cov=hyperseed
```
