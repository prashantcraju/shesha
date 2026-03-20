# Contributing to Shesha

First, thank you for your interest in contributing to `Shesha`! We welcome contributions from the community, whether they are bug reports, feature requests, documentation improvements, or code contributions.

By participating in this project, you agree to abide by our Code of Conduct.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please open an issue on GitHub. Include as much detail as possible:

* A clear and descriptive title.
* The exact version of `shesha-geometry`, Python, and relevant dependencies (`numpy`, `scipy`, `anndata`, etc.) you are using.
* A minimal reproducible example (code snippet) that triggers the bug.
* The full traceback of any errors.

### Suggesting Enhancements

If you have an idea for a new feature, a new stability metric, or an improvement to the `shesha.bio` or `shesha.core` modules, please open an issue first to discuss it before writing any code. This ensures your effort aligns with the project's roadmap.

## Setting Up Your Development Environment

To contribute code, you will need to set up a local development environment.

1. **Fork the repository** on GitHub (click the "Fork" button at the top right)

2. **Clone your fork** locally (replace `YOUR_USERNAME` with your GitHub username):
```bash
git clone https://github.com/YOUR_USERNAME/shesha.git
cd shesha
```

3. **Add the upstream repository** to sync with the original:
```bash
git remote add upstream https://github.com/prashantcraju/shesha.git
git fetch upstream
```

4. **Create a virtual environment** (using `venv`, `conda`, etc.) to isolate your dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

5. **Install the package in editable mode** along with all testing and development dependencies:
```bash
pip install -e .[dev]
```

### Keeping Your Fork Up to Date

Before starting new work, sync your fork with the upstream repository:

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

## Development Guidelines

### Code Style

We follow standard Python PEP 8 conventions. To maintain a consistent codebase, please format your code before committing.

* Use `black` for code formatting.
* Use `flake8` for linting.
* Ensure all new functions and classes have descriptive docstrings detailing expected input shapes and types (especially critical for high-dimensional arrays and `AnnData` objects).

### Running Tests

`Shesha` uses `pytest` for unit testing. JOSS places a high value on test coverage to ensure scientific reliability. Before submitting a pull request, verify that all tests pass:

```bash
pytest tests/
```

If you are adding a new feature or metric, please include corresponding unit tests in the `tests/` directory to verify its mathematical correctness and edge-case handling.

## Pull Request Process

1. **Sync your fork** with the upstream repository (see "Keeping Your Fork Up to Date" above)

2. **Create a new branch** for your feature or bugfix:
```bash
git checkout -b feature/your-feature-name
```

3. **Make your changes** and commit them with clear, descriptive commit messages.

4. **Run tests locally** to ensure everything works:
```bash
pytest tests/
```

5. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

6. **Submit a Pull Request (PR)** against the `main` branch of the upstream `Shesha` repository.

7. **Review:** A maintainer will review your code. We may request changes to ensure performance scaling and API consistency before merging.
