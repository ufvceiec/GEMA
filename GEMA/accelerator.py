"""
GEMA accelerator — optional numba (CPU JIT) and CuPy (GPU) back-ends.

Importing this module never raises an error: if numba or cupy are not
installed the corresponding capabilities are simply unavailable and GEMA
falls back to plain NumPy automatically.

Public API
----------
CUPY_AVAILABLE   : bool  — True when CuPy is installed and a CUDA GPU is present.
NUMBA_AVAILABLE  : bool  — True when numba is installed.
get_array_module : returns the array library (numpy or cupy) that owns *arr*.
to_device        : move a numpy array to the requested device ('cpu' / 'gpu').
to_numpy         : always return a plain numpy array (copies from GPU if needed).
"""

import numpy as np

# ── CuPy (GPU) ────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    # Verify a device is actually reachable
    cp.cuda.runtime.getDeviceCount()
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

# ── Numba (CPU JIT) ───────────────────────────────────────────────────────────
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ── Device helpers ────────────────────────────────────────────────────────────

def get_array_module(arr):
    """Return the array library (numpy or cupy) that owns *arr*.

    Safe to call even when CuPy is not installed — always returns numpy
    in that case.
    """
    if CUPY_AVAILABLE:
        return cp.get_array_module(arr)
    return np


def to_device(arr, device: str):
    """Move *arr* to *device* and cast to float64.

    :param arr: A numpy (or cupy) array.
    :param device: ``'cpu'`` → numpy float64 array.
                   ``'gpu'`` → cupy float64 array on the default CUDA device.
    :raises RuntimeError: when ``device='gpu'`` but CuPy is unavailable.
    """
    if device == 'gpu':
        if not CUPY_AVAILABLE:
            raise RuntimeError(
                "GPU acceleration requires CuPy, but CuPy is not installed or "
                "no CUDA GPU was detected.\n"
                "Install CuPy for your CUDA version, e.g.:\n"
                "    pip install cupy-cuda12x      # CUDA 12.x\n"
                "    pip install cupy-cuda11x      # CUDA 11.x\n"
                "See https://docs.cupy.dev/en/stable/install.html for details."
            )
        return cp.asarray(arr, dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def to_numpy(arr) -> np.ndarray:
    """Always return a plain numpy ndarray, copying from GPU memory if needed."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def device_info() -> str:
    """Return a short string describing available acceleration.

    Useful for quick diagnostics::

        from GEMA.accelerator import device_info
        print(device_info())
    """
    parts = ["NumPy (always available)"]
    if NUMBA_AVAILABLE:
        import numba
        parts.append(f"numba {numba.__version__} (CPU JIT, parallel)")
    else:
        parts.append("numba NOT installed (pip install numba)")
    if CUPY_AVAILABLE:
        import cupy as cp_local
        n = cp_local.cuda.runtime.getDeviceCount()
        names = [cp_local.cuda.runtime.getDeviceProperties(i)['name'].decode()
                 for i in range(n)]
        parts.append(f"CuPy {cp_local.__version__} — {n} GPU(s): {', '.join(names)}")
    else:
        parts.append("CuPy NOT installed (pip install cupy-cuda12x)")
    return "\n".join(f"  • {p}" for p in parts)


# ── Numba CPU kernels ─────────────────────────────────────────────────────────
# Only defined when numba is installed; otherwise these names are absent from
# the module namespace and GEMA falls back to the numpy/cupy vector paths.

if NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True)
    def _bmu_distances_numba(weights, pattern):
        """Squared distance from every neuron weight to *pattern* (parallelised).

        :param weights: float64 array of shape (H, W, D).
        :param pattern: float64 1-D array of length D.
        :return: float64 array of shape (H, W).
        """
        H = weights.shape[0]
        W = weights.shape[1]
        D = weights.shape[2]
        dists = np.empty((H, W), dtype=np.float64)
        for y in prange(H):
            for x in range(W):
                acc = np.float64(0.0)
                for k in range(D):
                    diff = weights[y, x, k] - pattern[k]
                    acc += diff * diff
                dists[y, x] = acc
        return dists

    @njit(cache=True, parallel=True)
    def _euclidean_grid_distances_numba(ids_matrix, bmu_y, bmu_x):
        """Euclidean distance on the neuron grid from every neuron to the BMU.

        :param ids_matrix: float64 array of shape (H, W, 2) — neuron positions.
        :param bmu_y: int — row index of the BMU.
        :param bmu_x: int — column index of the BMU.
        :return: float64 array of shape (H, W).
        """
        H = ids_matrix.shape[0]
        W = ids_matrix.shape[1]
        dists = np.empty((H, W), dtype=np.float64)
        py = ids_matrix[bmu_y, bmu_x, 0]
        px = ids_matrix[bmu_y, bmu_x, 1]
        for y in prange(H):
            for x in range(W):
                dy = ids_matrix[y, x, 0] - py
                dx = ids_matrix[y, x, 1] - px
                dists[y, x] = (dy * dy + dx * dx) ** 0.5
        return dists

    @njit(cache=True, parallel=True)
    def _chebyshev_grid_distances_numba(ids_matrix, bmu_y, bmu_x):
        """Chebyshev (L∞) distance on the neuron grid from every neuron to the BMU.

        :param ids_matrix: float64 array of shape (H, W, 2) — neuron positions.
        :param bmu_y: int — row index of the BMU.
        :param bmu_x: int — column index of the BMU.
        :return: float64 array of shape (H, W).
        """
        H = ids_matrix.shape[0]
        W = ids_matrix.shape[1]
        dists = np.empty((H, W), dtype=np.float64)
        py = ids_matrix[bmu_y, bmu_x, 0]
        px = ids_matrix[bmu_y, bmu_x, 1]
        for y in prange(H):
            for x in range(W):
                dy = abs(ids_matrix[y, x, 0] - py)
                dx = abs(ids_matrix[y, x, 1] - px)
                dists[y, x] = dy if dy > dx else dx
        return dists

    @njit(cache=True, parallel=True)
    def _weight_update_numba(weights, grid_distances, v, eta, pattern, use_decay):
        """Update *weights* in-place for all neurons within neighbourhood *v*.

        :param weights: float64 array (H, W, D) — modified in-place.
        :param grid_distances: float64 array (H, W) — pre-computed distances to BMU.
        :param v: float — current neighbourhood radius.
        :param eta: float — current learning rate.
        :param pattern: float64 1-D array (D,) — current training sample.
        :param use_decay: bool — apply Gaussian decay when True.
        :return: updated *weights* array.
        """
        H = weights.shape[0]
        W = weights.shape[1]
        D = weights.shape[2]
        for y in prange(H):
            for x in range(W):
                dist = grid_distances[y, x]
                if dist <= v:
                    if use_decay:
                        h = np.exp(-(dist * dist) / (2.0 * v * v))
                    else:
                        h = np.float64(1.0)
                    for k in range(D):
                        weights[y, x, k] += eta * h * (pattern[k] - weights[y, x, k])
        return weights
