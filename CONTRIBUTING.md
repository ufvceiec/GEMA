# Contributing to GEMA

Thank you for your interest in contributing to GEMA! This document describes how to set up your environment, run the tests, and submit changes.

---

## Table of Contents

- [Getting started](#getting-started)
- [Running the tests](#running-the-tests)
- [Code style](#code-style)
- [Submitting a pull request](#submitting-a-pull-request)
- [Reporting bugs](#reporting-bugs)

---

## Getting started

1. **Fork** the repository on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/GEMA.git
   cd GEMA
   ```

2. Create a virtual environment and install the package in editable mode with test dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -e ".[test]"
   ```

3. Create a feature branch:

   ```bash
   git checkout -b feature/my-improvement
   ```

---

## Running the tests

```bash
pytest tests/ -v
```

All 51 tests should pass before you submit a pull request. To also check coverage:

```bash
pytest tests/ --cov=GEMA --cov-report=term-missing
```

---

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Keep public methods documented with NumPy-style docstrings (consistent with the existing codebase).
- Do not break backward compatibility without a clear reason and a CHANGELOG entry.

---

## Submitting a pull request

1. Make sure all tests pass and no new warnings are introduced.
2. Update `CHANGELOG.md` under an `[Unreleased]` section describing your changes.
3. Push your branch and open a pull request against `master`.
4. A maintainer will review and merge your PR.

---

## Reporting bugs

Please open a GitHub issue at https://github.com/ufvceiec/GEMA/issues and include:

- GEMA version (`pip show GEMA`)
- Python version
- A minimal reproducible example
- The full traceback
