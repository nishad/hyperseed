# Contributing to Hyperseed

Thank you for considering contributing to Hyperseed! This document provides guidelines for contributing.

## Ways to Contribute

- **Report bugs**: Open an issue on GitHub
- **Suggest features**: Propose new features via issues
- **Improve documentation**: Fix typos, add examples, clarify
- **Submit code**: Fix bugs or implement features

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your changes
4. Make your changes
5. Run tests and ensure they pass
6. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/hyperseed
cd hyperseed

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Code Style

We use **black** for code formatting:

```bash
black hyperseed/
```

Lint with **ruff**:

```bash
ruff check hyperseed/
```

Type check with **mypy**:

```bash
mypy hyperseed/
```

## Testing

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=hyperseed --cov-report=html
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit pull request with clear description

## Code of Conduct

Be respectful and constructive in all interactions.

## Recognition

All contributors are automatically recognized on the [GitHub Contributors page](https://github.com/nishad/hyperseed/graphs/contributors). Your contributions, whether code, documentation, bug reports, or feature suggestions, are valued and appreciated!

## Questions?

Open a GitHub issue or discussion if you have questions.
