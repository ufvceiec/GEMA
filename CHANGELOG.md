# Changelog

All notable changes to GEMA are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.5.1] — 2026-06-21

### Added

- **Hexagonal topology** (`topology='hexagonal'` in `Map`) — neurons are placed on an offset hex lattice where every neuron has 6 equidistant neighbours, giving smoother and more uniform cluster boundaries than a rectangular grid.  The `topology` parameter is persisted in `save_classifier` / `load_classifier` (backwards-compatible: old JSON files default to `'rectangular'`).
- **`IterativeSOM` fully implemented** — `calculate_range()` now derives a sensible `[min_size, max_size]` range automatically from the dataset size using the heuristic `centre = sqrt(5 * sqrt(N))`.  The constructor trains one `Map` per size, optionally evaluates each with `Classification` (quantization error), stores the scores in `self.scores`, and exposes the best map via `get_best_map()`.  Added `get_scores()` helper and forwarding of arbitrary `**map_kwargs` to each `Map`.

### Changed

- **Euclidean normalisation vectorised** — `__normalize(method='euclidean')` replaced nested Python loops with `np.linalg.norm(axis=1, keepdims=True)` division; zero-norm rows are left unchanged. Delivers the same result with a fraction of the runtime.
- **`'sample'` weight init vectorised** — replaced the scalar loop with a single `np.random.randint` call that produces all indices at once; weights are drawn as whole sample vectors rather than individual scalar components, which is both faster and statistically sounder.
- **`reinforce()` learning rate bug fixed** — `origin_initial_lr` was reset to `self.initial_lr` (the original value) instead of the compressed value each round, meaning the compression had no cumulative effect. Fixed to accumulate correctly and update `self.initial_lr`.
- **`__adjust_weights` hex-aware** — BMU position is now looked up in `__ids_matrix` before computing neighbourhood distances, so hex and rectangular grids both use their correct 2-D positions.

---

## [0.5.0] — 2026-06-19

### Added

- **GitHub Actions CI** — automated test suite runs on Python 3.10 and 3.11 on every push and pull request.
- **Sphinx documentation** — full API docs built with `sphinx_rtd_theme`; automatically deployed to GitHub Pages via `peaceiris/actions-gh-pages`.
- **Test suite** (`tests/test_gema.py`) — 51 unit tests covering `Map`, `Classification`, all normalization methods, weight initializations, save/load, and distance metrics.
- **`CITATION.cff`** — machine-readable citation metadata for GitHub's "Cite this repository" button (García-Tejedor & Nogales, *Software Impacts* 2022, DOI `10.1016/j.simpa.2022.100280`).
- **Benchmark table** in README — wall-clock timing comparison of GEMA, MiniSom, and sklearn-som on small/medium/large datasets.

### Fixed

- **`classification.py`** — replaced deprecated chained assignment (`df['col'][row] = val`) with `.loc[row, 'col'] = val` to prevent `ChainedAssignmentError` under pandas 2.0+.
- **`visualization.py`** — made `IPython.display` import optional with a no-op fallback so GEMA works outside Jupyter notebooks without raising `ModuleNotFoundError`.

---

# Previous changelog

All notable changes to GEMA are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.4.3] — 2026-06-12

### Fixed

- **`map.py`** — `np.random.randint(0, self.num_data - 1)` in `train()` and `reinforce()` excluded the last data sample from random presentation. Fixed to `np.random.randint(0, self.num_data)`.
- **`map.py`** — Same off-by-one in `__init_weights()` (`'sample'` method): last data sample and last feature dimension were never selected.
- **`map.py`** — Euclidean normalization loops in `__normalize()` used `range(shape - 1)`, skipping the last row and last feature of every vector. Fixed to `range(shape)`.
- **`map.py`** — `initial_neighbourhood is not 0` used identity comparison instead of equality. Changed to `!= 0`.
- **`map.py`** — `method is not 'none'` used identity comparison for string. Changed to `!= 'none'`.
- **`visualization.py`** — `cmax is 0` used identity comparison for integer. Changed to `== 0`.
- **`visualization.py`** — `header is not 'none'` used identity comparison for string. Changed to `!= 'none'`.
- **`visualization.py`** — `ax[i, j].xticks = (...)` was a no-op attribute assignment. Changed to `ax[i, j].set_xticks(...)`.
- **`iterativesom.py`** — `__init__` was incorrectly decorated with `@staticmethod`, preventing instantiation. Removed decorator and added `self` parameter.
- **`iterativesom.py`** — `if range_from == np.array([0, 0])` produced a boolean array, causing an ambiguous truth-value error. Changed to `np.array_equal(...)`.
- **`iterativesom.py`** — `map[x] = Map.train(...)` referenced the Python built-in `map` as a dict and called `Map.train` as a static constructor. Fixed to `self.maps[x] = Map(...)`.
- **`iterativesom.py`** — `calculate_range` was missing `@staticmethod` decorator.

### Changed

- `requirements.txt` updated to include `scikit-learn` and `scipy` (already used by `map.py` and referenced in the paper).
- `setup.py` version bumped to `0.4.3`; added `install_requires`, license classifier, and Python version constraint.
- `README.md` fully rewritten with Quick Start, full API reference, and parameter tables.

---

## [0.4.2] — prior release

- PCA weight initialization added.
- Reinforcement learning (`reinforce()`) added.
- U-matrix computation added to `Classification`.
