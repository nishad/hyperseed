# Development

Information for developers and contributors.

## Overview

Hyperseed is an open-source project and welcomes contributions! This section provides resources for developers who want to:

- Contribute code, documentation, or bug fixes
- Understand the project architecture
- Run tests and ensure code quality
- Build and deploy the project

## For Contributors

<div class="grid cards" markdown>

-   **[Contributing Guide](contributing.md)**

    How to contribute to Hyperseed

-   **[Architecture](architecture.md)**

    Project structure and design

-   **[Testing](testing.md)**

    Running tests and ensuring quality

</div>

## Quick Start for Developers

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/hyperseed
cd hyperseed
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### 3. Make Changes

```bash
# Create a new branch
git checkout -b feature/my-new-feature

# Make your changes
# ...

# Format code
black hyperseed/
ruff check hyperseed/

# Run tests
pytest
```

### 4. Submit Pull Request

```bash
# Commit and push
git add .
git commit -m "Add my new feature"
git push origin feature/my-new-feature

# Create pull request on GitHub
```

## Development Tools

Hyperseed uses modern Python development tools:

| Tool | Purpose |
|------|---------|
| **pytest** | Testing framework |
| **black** | Code formatting |
| **ruff** | Fast linting |
| **mypy** | Type checking |
| **mkdocs-material** | Documentation |

## Code Quality Standards

- **Test coverage**: Aim for >80% coverage
- **Code style**: Follow PEP 8 (enforced by black)
- **Type hints**: Add type hints to public functions
- **Documentation**: Document all public APIs

## Resources

- **GitHub Repository**: [nishad/hyperseed](https://github.com/nishad/hyperseed)
- **Issue Tracker**: [GitHub Issues](https://github.com/nishad/hyperseed/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nishad/hyperseed/discussions)
