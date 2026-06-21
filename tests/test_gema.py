"""
Comprehensive pytest test suite for the GEMA SOM library.

Coverage
--------
- TestMapInit               basic construction and attribute defaults
- TestMapStaticMethods      distance / schedule / decay helpers
- TestMapTrain              training, BMU, sequential presentation
- TestMapNormalization      all four normalisation modes
- TestMapWeightsInit        all four weight-init strategies
- TestMapDistance           euclidean and chebyshev dispatch
- TestMapSaveLoad           JSON round-trip (rectangular + hexagonal)
- TestMapTopology           hexagonal grid construction and training
- TestMapDevice             device parameter validation and CPU training
- TestMapReinforce          reinforce() modifies weights correctly
- TestMapDecay              use_decay=True path
- TestAccelerator           get_array_module / to_device / to_numpy / device_info
- TestNumbaKernels          numba JIT kernels (skipped when numba absent)
- TestGPUDevice             device='gpu' (skipped when CuPy absent)
- TestClassification        metrics, maps, U-matrix
- TestClassificationTagged  label-column extraction
- TestClassificationOther   extra DataFrame columns
- TestIterativeSOM          calculate_range, multi-size training, scoring
"""

import os
import numpy as np
import pandas as pd
import pytest

from GEMA.map import Map
from GEMA.classification import Classification
from GEMA.iterativesom import IterativeSOM
from GEMA.accelerator import (
    CUPY_AVAILABLE,
    NUMBA_AVAILABLE,
    get_array_module,
    to_device,
    to_numpy,
    device_info,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)

N_SAMPLES  = 50
N_FEATURES = 4
MAP_SIZE   = 5
PERIOD     = 20

DATA = RNG.random((N_SAMPLES, N_FEATURES)).astype(np.float64)


def make_map(**kwargs):
    """Return a freshly trained Map with sensible defaults."""
    defaults = dict(
        data=DATA,
        size=MAP_SIZE,
        period=PERIOD,
        initial_lr=0.1,
        distance="euclidean",
        use_decay=False,
        normalization="none",
        presentation="random",
        weights="random",
        topology="rectangular",
        device="cpu",
    )
    defaults.update(kwargs)
    np.random.seed(0)
    return Map(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
# TestMapInit
# ══════════════════════════════════════════════════════════════════════════════

class TestMapInit:
    def test_weights_shape_after_train(self):
        assert make_map().weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    def test_default_neighbourhood_equals_size(self):
        m = Map(size=MAP_SIZE, period=PERIOD)
        assert m.neighbourhood == MAP_SIZE

    def test_explicit_neighbourhood(self):
        m = Map(size=MAP_SIZE, period=PERIOD, initial_neighbourhood=3)
        assert m.neighbourhood == 3

    def test_map_size_stored(self):
        assert Map(size=MAP_SIZE, period=PERIOD).map_size == MAP_SIZE

    def test_period_stored(self):
        assert Map(size=MAP_SIZE, period=PERIOD).period == PERIOD

    def test_default_topology_is_rectangular(self):
        assert Map(size=MAP_SIZE, period=PERIOD).topology == "rectangular"

    def test_default_device_is_cpu(self):
        assert Map(size=MAP_SIZE, period=PERIOD).device == "cpu"

    def test_invalid_topology_raises(self):
        with pytest.raises(AssertionError):
            Map(size=MAP_SIZE, period=PERIOD, topology="triangular")

    def test_invalid_device_raises(self):
        with pytest.raises(AssertionError):
            Map(size=MAP_SIZE, period=PERIOD, device="tpu")

    def test_size_below_2_raises(self):
        with pytest.raises(AssertionError):
            Map(size=1, period=PERIOD)

    def test_period_below_2_raises(self):
        with pytest.raises(AssertionError):
            Map(size=MAP_SIZE, period=1)

    def test_weights_are_numpy_after_init(self):
        m = make_map()
        assert isinstance(m.weights, np.ndarray)

    def test_ids_matrix_rectangular_shape(self):
        m = Map(size=MAP_SIZE, period=PERIOD, topology="rectangular")
        ids = m._Map__ids_matrix
        assert ids.shape == (MAP_SIZE, MAP_SIZE, 2)

    def test_ids_matrix_rectangular_values(self):
        m = Map(size=3, period=PERIOD, topology="rectangular")
        ids = m._Map__ids_matrix
        assert ids[0, 0, 0] == pytest.approx(0.0)   # row 0, col 0 → y=0
        assert ids[0, 0, 1] == pytest.approx(0.0)   # x=0
        assert ids[2, 2, 0] == pytest.approx(2.0)   # y=2
        assert ids[2, 2, 1] == pytest.approx(2.0)   # x=2


# ══════════════════════════════════════════════════════════════════════════════
# TestMapStaticMethods
# ══════════════════════════════════════════════════════════════════════════════

class TestMapStaticMethods:
    def test_vector_distance_zero(self):
        v = np.array([1.0, 2.0, 3.0])
        assert Map.vector_distance(v, v) == pytest.approx(0.0)

    def test_vector_distance_known(self):
        assert Map.vector_distance(np.array([0., 0.]),
                                   np.array([3., 4.])) == pytest.approx(5.0)

    def test_chebyshev_distance_zero(self):
        v = np.array([1.0, 2.0, 3.0])
        assert Map.chebyshev_distance(v, v) == pytest.approx(0.0)

    def test_chebyshev_distance_known(self):
        assert Map.chebyshev_distance(np.array([0., 0., 0.]),
                                      np.array([1., 5., 3.])) == pytest.approx(5.0)

    def test_chebyshev_le_euclidean(self):
        v1, v2 = RNG.random(10), RNG.random(10)
        assert Map.chebyshev_distance(v1, v2) <= Map.vector_distance(v1, v2)

    def test_variation_lr_decreases(self):
        lr = [Map.variation_learning_rate(0.5, i, 100) for i in (1, 50, 99)]
        assert lr[0] > lr[1] > lr[2]

    def test_variation_lr_at_zero(self):
        assert Map.variation_learning_rate(0.3, 0, 100) == pytest.approx(0.3)

    def test_variation_lr_at_end_near_zero(self):
        assert Map.variation_learning_rate(0.5, 100, 100) == pytest.approx(0.0)

    def test_variation_neighbourhood_decreases(self):
        n = [Map.variation_neighbourhood(5, i, 100, 0) for i in (1, 50, 99)]
        assert n[0] > n[1] > n[2]

    def test_variation_neighbourhood_final(self):
        assert Map.variation_neighbourhood(5, 100, 100, final=1) == pytest.approx(1.0)

    def test_decay_shape(self):
        d = np.array([0., 1., 2., 3.])
        assert Map.decay(d, 2.0).shape == d.shape

    def test_decay_max_at_bmu(self):
        d = np.array([0., 1., 2.])
        r = Map.decay(d, 2.0)
        assert r[0] == pytest.approx(1.0)
        assert r[0] > r[1] > r[2]

    def test_vector_distance_batch(self):
        """vector_distance should work on arrays of vectors too."""
        a = np.zeros((3, 2))
        b = np.array([[3., 4.], [0., 0.], [1., 0.]])
        result = Map.vector_distance(a, b)
        np.testing.assert_allclose(result, [5.0, 0.0, 1.0])


# ══════════════════════════════════════════════════════════════════════════════
# TestMapTrain
# ══════════════════════════════════════════════════════════════════════════════

class TestMapTrain:
    def test_weights_shape(self):
        assert make_map().weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    def test_trained_flag(self):
        assert make_map()._Map__trainned is True

    def test_bmu_returns_four_items(self):
        assert len(make_map().calculate_bmu(DATA[0])) == 4

    def test_bmu_pos_within_bounds(self):
        m = make_map()
        _, bmu, _, sbmu = m.calculate_bmu(DATA[0])
        for pos in (bmu, sbmu):
            assert 0 <= pos[0] < MAP_SIZE
            assert 0 <= pos[1] < MAP_SIZE

    def test_bmu_distance_non_negative(self):
        assert make_map().calculate_bmu(DATA[0])[0] >= 0.0

    def test_second_bmu_dist_gte_bmu(self):
        m = make_map()
        d1, _, d2, _ = m.calculate_bmu(DATA[0])
        assert d2 >= d1

    def test_sequential_presentation(self):
        m = make_map(presentation="sequential")
        assert m._Map__trainned is True

    def test_num_data_stored(self):
        assert make_map().num_data == N_SAMPLES

    def test_input_data_dimension_stored(self):
        assert make_map().input_data_dimension == N_FEATURES

    def test_weights_are_numpy(self):
        assert isinstance(make_map().weights, np.ndarray)

    def test_weights_finite(self):
        assert np.all(np.isfinite(make_map().weights))

    def test_train_without_data_in_constructor(self):
        m = Map(size=MAP_SIZE, period=PERIOD)
        m.train(DATA)
        assert m.weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)


# ══════════════════════════════════════════════════════════════════════════════
# TestMapNormalization
# ══════════════════════════════════════════════════════════════════════════════

class TestMapNormalization:
    @pytest.mark.parametrize("method", ["none", "fwn", "01scale", "euclidean"])
    def test_does_not_crash(self, method):
        m = make_map(normalization=method)
        assert m.weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    @pytest.mark.parametrize("method", ["none", "fwn", "01scale", "euclidean"])
    def test_weights_finite(self, method):
        m = make_map(normalization=method)
        assert np.all(np.isfinite(m.weights))

    def test_euclidean_unit_norms(self):
        """After euclidean normalisation each row should have unit L2 norm."""
        normalise = Map._Map__normalize
        normed = normalise(DATA.copy(), "euclidean")
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_01scale_range(self):
        normalise = Map._Map__normalize
        scaled = normalise(DATA.copy(), "01scale")
        assert scaled.min() == pytest.approx(0.0, abs=1e-10)
        assert scaled.max() == pytest.approx(1.0, abs=1e-10)

    def test_fwn_zero_mean(self):
        normalise = Map._Map__normalize
        normed = normalise(DATA.copy(), "fwn")
        np.testing.assert_allclose(normed.mean(axis=0), 0.0, atol=1e-10)

    def test_fwn_unit_std(self):
        normalise = Map._Map__normalize
        normed = normalise(DATA.copy(), "fwn")
        np.testing.assert_allclose(normed.std(axis=0), 1.0, atol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# TestMapWeightsInit
# ══════════════════════════════════════════════════════════════════════════════

class TestMapWeightsInit:
    @pytest.mark.parametrize("method", ["random", "random_negative", "sample", "PCA"])
    def test_shape(self, method):
        assert make_map(weights=method).weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    def test_random_in_01(self):
        w = make_map(weights="random").weights
        assert w.min() >= 0.0
        assert w.max() <= 1.0

    def test_random_negative_in_m1_1(self):
        w = make_map(weights="random_negative").weights
        assert w.min() >= -1.0
        assert w.max() <= 1.0

    def test_sample_initial_weights_come_from_data(self):
        """The 'sample' initialiser must draw whole rows from training data.

        We test the initialiser directly rather than the post-training weights,
        because the update loop moves weights away from their initial values.
        """
        m = Map(size=MAP_SIZE, period=PERIOD, weights="sample")
        m.num_data = N_SAMPLES
        m.input_data_dimension = N_FEATURES
        init_w = m._Map__init_weights(DATA.astype(np.float64), "sample")
        flat = init_w.reshape(-1, N_FEATURES)
        for vec in flat:
            assert any(np.allclose(vec, row) for row in DATA)

    def test_pca_weights_finite(self):
        assert np.all(np.isfinite(make_map(weights="PCA").weights))


# ══════════════════════════════════════════════════════════════════════════════
# TestMapDistance
# ══════════════════════════════════════════════════════════════════════════════

class TestMapDistance:
    def test_euclidean_dispatch(self):
        m = Map(size=MAP_SIZE, period=PERIOD, distance="euclidean")
        assert m.calculate_distance(np.array([0., 0.]),
                                    np.array([3., 4.])) == pytest.approx(5.0)

    def test_chebyshev_dispatch(self):
        m = Map(size=MAP_SIZE, period=PERIOD, distance="chebyshev")
        assert m.calculate_distance(np.array([0., 0., 0.]),
                                    np.array([1., 5., 3.])) == pytest.approx(5.0)

    def test_chebyshev_training_works(self):
        m = make_map(distance="chebyshev")
        assert m.weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    def test_chebyshev_le_euclidean_batch(self):
        v1, v2 = RNG.random(10), RNG.random(10)
        assert Map.chebyshev_distance(v1, v2) <= Map.vector_distance(v1, v2)


# ══════════════════════════════════════════════════════════════════════════════
# TestMapSaveLoad
# ══════════════════════════════════════════════════════════════════════════════

class TestMapSaveLoad:
    def test_weights_round_trip(self, tmp_path):
        m = make_map()
        path = str(tmp_path / "model")
        m.save_classifier(path)
        loaded = Map.load_classifier(path)
        np.testing.assert_array_almost_equal(m.weights, loaded.weights)

    def test_attributes_round_trip(self, tmp_path):
        m = make_map()
        path = str(tmp_path / "model")
        m.save_classifier(path)
        loaded = Map.load_classifier(path)
        assert loaded.map_size == m.map_size
        assert loaded.period == m.period
        assert loaded.distance == m.distance
        assert loaded.topology == m.topology

    def test_json_file_created(self, tmp_path):
        m = make_map()
        path = str(tmp_path / "model")
        m.save_classifier(path)
        assert os.path.exists(path + ".json")

    def test_hexagonal_topology_persisted(self, tmp_path):
        m = make_map(topology="hexagonal")
        path = str(tmp_path / "hex_model")
        m.save_classifier(path)
        loaded = Map.load_classifier(path)
        assert loaded.topology == "hexagonal"

    def test_loaded_map_can_classify(self, tmp_path):
        m = make_map()
        path = str(tmp_path / "model")
        m.save_classifier(path)
        loaded = Map.load_classifier(path)
        _, pos, _, _ = loaded.calculate_bmu(DATA[0])
        assert 0 <= pos[0] < MAP_SIZE
        assert 0 <= pos[1] < MAP_SIZE


# ══════════════════════════════════════════════════════════════════════════════
# TestMapTopology
# ══════════════════════════════════════════════════════════════════════════════

class TestMapTopology:
    def test_hex_ids_matrix_shape(self):
        m = Map(size=MAP_SIZE, period=PERIOD, topology="hexagonal")
        assert m._Map__ids_matrix.shape == (MAP_SIZE, MAP_SIZE, 2)

    def test_hex_even_rows_no_offset(self):
        m = Map(size=4, period=PERIOD, topology="hexagonal")
        ids = m._Map__ids_matrix
        # Even rows (0, 2, …) — x coords should be 0, 1, 2, 3 (no offset)
        for row in (0, 2):
            for col in range(4):
                assert ids[row, col, 1] == pytest.approx(col)

    def test_hex_odd_rows_half_offset(self):
        m = Map(size=4, period=PERIOD, topology="hexagonal")
        ids = m._Map__ids_matrix
        # Odd rows (1, 3, …) — x coords should be 0.5, 1.5, 2.5, 3.5
        for row in (1, 3):
            for col in range(4):
                assert ids[row, col, 1] == pytest.approx(col + 0.5)

    def test_hex_y_spacing_is_sqrt3_over_2(self):
        m = Map(size=4, period=PERIOD, topology="hexagonal")
        ids = m._Map__ids_matrix
        expected_dy = np.sqrt(3) / 2
        for row in range(1, 4):
            dy = ids[row, 0, 0] - ids[row - 1, 0, 0]
            assert dy == pytest.approx(expected_dy)

    def test_hex_six_neighbours_equidistant(self):
        """Interior neuron on a hex grid should have 6 equidistant neighbours."""
        m = Map(size=5, period=PERIOD, topology="hexagonal")
        ids = m._Map__ids_matrix
        # Pick an interior neuron (row=2, col=2, even row → no offset)
        centre = ids[2, 2]
        neighbours = [
            ids[1, 1], ids[1, 2],   # row above (odd → offset)
            ids[2, 1], ids[2, 3],   # same row
            ids[3, 1], ids[3, 2],   # row below (odd → offset)
        ]
        dists = [np.linalg.norm(n - centre) for n in neighbours]
        assert all(pytest.approx(dists[0], rel=1e-6) == d for d in dists)

    def test_hex_training_weights_shape(self):
        assert make_map(topology="hexagonal").weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    def test_hex_training_bmu_within_bounds(self):
        m = make_map(topology="hexagonal")
        _, pos, _, _ = m.calculate_bmu(DATA[0])
        assert 0 <= pos[0] < MAP_SIZE
        assert 0 <= pos[1] < MAP_SIZE

    def test_hex_weights_finite(self):
        assert np.all(np.isfinite(make_map(topology="hexagonal").weights))

    def test_hex_classification_works(self):
        m = make_map(topology="hexagonal")
        clf = Classification(m, DATA, verbose=0)
        assert clf.activations_map.sum() == N_SAMPLES

    def test_hex_chebyshev_training(self):
        m = make_map(topology="hexagonal", distance="chebyshev")
        assert m.weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)


# ══════════════════════════════════════════════════════════════════════════════
# TestMapDevice
# ══════════════════════════════════════════════════════════════════════════════

class TestMapDevice:
    def test_cpu_device_stored(self):
        assert make_map(device="cpu").device == "cpu"

    def test_weights_numpy_after_cpu_training(self):
        assert isinstance(make_map(device="cpu").weights, np.ndarray)

    def test_cpu_weights_shape(self):
        assert make_map(device="cpu").weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    def test_cpu_weights_finite(self):
        assert np.all(np.isfinite(make_map(device="cpu").weights))

    def test_gpu_raises_without_cupy(self):
        if CUPY_AVAILABLE:
            pytest.skip("CuPy is available — GPU test skipped in this branch")
        with pytest.raises(RuntimeError, match="CuPy"):
            make_map(device="gpu")

    def test_invalid_device_raises(self):
        with pytest.raises(AssertionError):
            Map(size=MAP_SIZE, period=PERIOD, device="mps")

    def test_cpu_result_deterministic_with_seed(self):
        np.random.seed(7)
        w1 = make_map(device="cpu").weights.copy()
        np.random.seed(7)
        w2 = make_map(device="cpu").weights.copy()
        np.testing.assert_array_equal(w1, w2)


# ══════════════════════════════════════════════════════════════════════════════
# TestMapReinforce
# ══════════════════════════════════════════════════════════════════════════════

class TestMapReinforce:
    def test_reinforce_changes_weights(self):
        m = make_map()
        w_before = m.weights.copy()
        m.reinforce(DATA, reinforcement=1, extension=1, compression=0.5)
        assert not np.array_equal(w_before, m.weights)

    def test_reinforce_shape_unchanged(self):
        m = make_map()
        m.reinforce(DATA, reinforcement=1, extension=1, compression=0.5)
        assert m.weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    def test_reinforce_weights_finite(self):
        m = make_map()
        m.reinforce(DATA, reinforcement=2, extension=1, compression=0.5)
        assert np.all(np.isfinite(m.weights))

    def test_reinforce_lr_compressed(self):
        """After one round with compression=0.5, initial_lr should be halved."""
        m = make_map(initial_lr=0.1)
        original_lr = m.initial_lr
        m.reinforce(DATA, reinforcement=1, extension=1, compression=0.5)
        assert m.initial_lr == pytest.approx(original_lr * 0.5)

    def test_reinforce_zero_rounds_no_change(self):
        m = make_map()
        w_before = m.weights.copy()
        m.reinforce(DATA, reinforcement=0)
        np.testing.assert_array_equal(w_before, m.weights)


# ══════════════════════════════════════════════════════════════════════════════
# TestMapDecay
# ══════════════════════════════════════════════════════════════════════════════

class TestMapDecay:
    def test_decay_training_works(self):
        m = make_map(use_decay=True)
        assert m.weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    def test_decay_weights_finite(self):
        assert np.all(np.isfinite(make_map(use_decay=True).weights))

    def test_decay_bmu_within_bounds(self):
        m = make_map(use_decay=True)
        _, pos, _, _ = m.calculate_bmu(DATA[0])
        assert 0 <= pos[0] < MAP_SIZE
        assert 0 <= pos[1] < MAP_SIZE

    def test_decay_hex_works(self):
        m = make_map(use_decay=True, topology="hexagonal")
        assert m.weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)


# ══════════════════════════════════════════════════════════════════════════════
# TestAccelerator
# ══════════════════════════════════════════════════════════════════════════════

class TestAccelerator:
    def test_cupy_available_is_bool(self):
        assert isinstance(CUPY_AVAILABLE, bool)

    def test_numba_available_is_bool(self):
        assert isinstance(NUMBA_AVAILABLE, bool)

    def test_get_array_module_numpy(self):
        arr = np.array([1.0, 2.0])
        assert get_array_module(arr) is np

    def test_to_device_cpu_returns_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = to_device(arr, "cpu")
        assert isinstance(result, np.ndarray)

    def test_to_device_cpu_dtype(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = to_device(arr, "cpu")
        assert result.dtype == np.float64

    def test_to_device_gpu_raises_without_cupy(self):
        if CUPY_AVAILABLE:
            pytest.skip("CuPy available — skip no-CuPy branch")
        with pytest.raises(RuntimeError, match="CuPy"):
            to_device(np.array([1.0]), "gpu")

    def test_to_numpy_from_numpy(self):
        arr = np.array([1.0, 2.0])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_to_numpy_preserves_values(self):
        arr = np.arange(10, dtype=np.float64)
        np.testing.assert_array_equal(to_numpy(arr), arr)

    def test_device_info_returns_string(self):
        assert isinstance(device_info(), str)

    def test_device_info_contains_numpy(self):
        assert "NumPy" in device_info()

    def test_device_info_mentions_numba(self):
        info = device_info()
        assert "numba" in info.lower()

    def test_device_info_mentions_cupy(self):
        info = device_info()
        assert "cupy" in info.lower()


# ══════════════════════════════════════════════════════════════════════════════
# TestNumbaKernels
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
class TestNumbaKernels:
    """Tests for the numba JIT-compiled kernels in GEMA.accelerator."""

    def setup_method(self):
        from GEMA.accelerator import (
            _bmu_distances_numba,
            _euclidean_grid_distances_numba,
            _chebyshev_grid_distances_numba,
            _weight_update_numba,
        )
        self.bmu_dists     = _bmu_distances_numba
        self.euc_grid      = _euclidean_grid_distances_numba
        self.cheb_grid     = _chebyshev_grid_distances_numba
        self.weight_update = _weight_update_numba

        # Small weight tensor: 3×3 map, 4 features
        self.W = RNG.random((3, 3, N_FEATURES)).astype(np.float64)
        self.pattern = RNG.random(N_FEATURES).astype(np.float64)
        # Rectangular grid positions
        ids = []
        for y in range(3):
            row = []
            for x in range(3):
                row.append([float(y), float(x)])
            ids.append(row)
        self.ids = np.array(ids, dtype=np.float64)

    def test_bmu_distances_shape(self):
        d = self.bmu_dists(self.W, self.pattern)
        assert d.shape == (3, 3)

    def test_bmu_distances_non_negative(self):
        d = self.bmu_dists(self.W, self.pattern)
        assert np.all(d >= 0.0)

    def test_bmu_distances_matches_numpy(self):
        """Numba kernel must match the numpy reference implementation."""
        d_numba = self.bmu_dists(self.W, self.pattern)
        d_numpy = np.sum((self.W - self.pattern) ** 2, axis=-1)
        np.testing.assert_allclose(d_numba, d_numpy, rtol=1e-10)

    def test_euclidean_grid_distances_shape(self):
        d = self.euc_grid(self.ids, 1, 1)   # BMU at centre
        assert d.shape == (3, 3)

    def test_euclidean_grid_distances_bmu_is_zero(self):
        d = self.euc_grid(self.ids, 1, 1)
        assert d[1, 1] == pytest.approx(0.0)

    def test_euclidean_grid_distances_non_negative(self):
        d = self.euc_grid(self.ids, 0, 0)
        assert np.all(d >= 0.0)

    def test_euclidean_grid_distances_corner_known(self):
        # Corner (0,0) to opposite corner (2,2) → sqrt(8) ≈ 2.828
        d = self.euc_grid(self.ids, 0, 0)
        assert d[2, 2] == pytest.approx(np.sqrt(8), rel=1e-6)

    def test_chebyshev_grid_distances_shape(self):
        d = self.cheb_grid(self.ids, 1, 1)
        assert d.shape == (3, 3)

    def test_chebyshev_grid_distances_bmu_is_zero(self):
        assert self.cheb_grid(self.ids, 1, 1)[1, 1] == pytest.approx(0.0)

    def test_chebyshev_le_euclidean_grid(self):
        e = self.euc_grid(self.ids, 0, 0)
        c = self.cheb_grid(self.ids, 0, 0)
        assert np.all(c <= e + 1e-10)

    def test_weight_update_changes_weights(self):
        W = self.W.copy()
        grid_d = self.euc_grid(self.ids, 1, 1)
        W_new = self.weight_update(W, grid_d, 2.0, 0.1, self.pattern, False)
        assert not np.array_equal(W_new, self.W)

    def test_weight_update_shape_preserved(self):
        W = self.W.copy()
        grid_d = self.euc_grid(self.ids, 1, 1)
        W_new = self.weight_update(W, grid_d, 2.0, 0.1, self.pattern, False)
        assert W_new.shape == (3, 3, N_FEATURES)

    def test_weight_update_outside_neighbourhood_unchanged(self):
        """Neurons beyond neighbourhood radius v should not be updated."""
        W = self.W.copy()
        # BMU at corner (0,0), v=0.5 → only (0,0) updated
        grid_d = self.euc_grid(self.ids, 0, 0)
        W_new = self.weight_update(W.copy(), grid_d, 0.5, 0.1, self.pattern, False)
        # Neuron (2,2) is sqrt(8) away — must be unchanged
        np.testing.assert_array_equal(W_new[2, 2], W[2, 2])

    def test_weight_update_with_decay(self):
        W = self.W.copy()
        grid_d = self.euc_grid(self.ids, 1, 1)
        W_new = self.weight_update(W, grid_d, 2.0, 0.1, self.pattern, True)
        assert W_new.shape == (3, 3, N_FEATURES)
        assert np.all(np.isfinite(W_new))

    def test_numba_cpu_training_matches_numpy(self):
        """Map trained with numba should produce valid BMU positions."""
        m = make_map(device="cpu")   # numba auto-used if installed
        _, pos, _, _ = m.calculate_bmu(DATA[0])
        assert 0 <= pos[0] < MAP_SIZE
        assert 0 <= pos[1] < MAP_SIZE


# ══════════════════════════════════════════════════════════════════════════════
# TestGPUDevice
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy / CUDA GPU not available")
class TestGPUDevice:
    """Tests that run only when CuPy and a CUDA GPU are present."""

    def test_gpu_device_stored(self):
        m = make_map(device="gpu")
        assert m.device == "gpu"

    def test_gpu_weights_are_numpy(self):
        """Weights must be copied back to CPU after training."""
        m = make_map(device="gpu")
        assert isinstance(m.weights, np.ndarray)

    def test_gpu_weights_shape(self):
        assert make_map(device="gpu").weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    def test_gpu_weights_finite(self):
        assert np.all(np.isfinite(make_map(device="gpu").weights))

    def test_gpu_bmu_within_bounds(self):
        m = make_map(device="gpu")
        _, pos, _, _ = m.calculate_bmu(DATA[0])
        assert 0 <= pos[0] < MAP_SIZE
        assert 0 <= pos[1] < MAP_SIZE

    def test_gpu_classification_works(self):
        m = make_map(device="gpu")
        clf = Classification(m, DATA, verbose=0)
        assert clf.activations_map.sum() == N_SAMPLES

    def test_gpu_save_load(self, tmp_path):
        m = make_map(device="gpu")
        path = str(tmp_path / "gpu_model")
        m.save_classifier(path)
        loaded = Map.load_classifier(path)
        assert loaded.device == "cpu"
        np.testing.assert_array_almost_equal(m.weights, loaded.weights)

    def test_gpu_hex_topology(self):
        m = make_map(device="gpu", topology="hexagonal")
        assert m.weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    def test_gpu_decay(self):
        m = make_map(device="gpu", use_decay=True)
        assert np.all(np.isfinite(m.weights))

    def test_gpu_chebyshev(self):
        m = make_map(device="gpu", distance="chebyshev")
        assert m.weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    def test_get_array_module_cupy(self):
        import cupy as cp
        arr = cp.array([1.0, 2.0])
        assert get_array_module(arr) is cp

    def test_to_device_gpu_returns_cupy(self):
        import cupy as cp
        result = to_device(np.array([1.0, 2.0]), "gpu")
        assert isinstance(result, cp.ndarray)

    def test_to_numpy_from_cupy(self):
        import cupy as cp
        arr = cp.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


# ══════════════════════════════════════════════════════════════════════════════
# TestClassification
# ══════════════════════════════════════════════════════════════════════════════

class TestClassification:
    @pytest.fixture(scope="class")
    def clf(self):
        return Classification(make_map(), DATA, verbose=0)

    def test_activations_sum_to_n(self, clf):
        assert clf.activations_map.sum() == N_SAMPLES

    def test_activations_shape(self, clf):
        assert clf.activations_map.shape == (MAP_SIZE, MAP_SIZE)

    def test_umatriz_shape(self, clf):
        expected = 2 * MAP_SIZE - 1
        assert clf.umatriz.shape == (expected, expected)

    def test_classification_map_columns(self, clf):
        for col in ("x", "y", "dist"):
            assert col in clf.classification_map.columns

    def test_classification_map_rows(self, clf):
        assert len(clf.classification_map) == N_SAMPLES

    def test_topological_error_in_01(self, clf):
        assert 0.0 <= clf.topological_error <= 1.0

    def test_quantization_error_non_negative(self, clf):
        assert clf.quantization_error >= 0.0

    def test_distances_map_non_negative(self, clf):
        assert (clf.distances_map >= 0).all()

    def test_num_activations_positive(self, clf):
        assert clf.num_activations > 0

    def test_mean_distance_non_negative(self, clf):
        assert clf.mean_distance_map >= 0.0

    def test_bmu_coords_in_bounds(self, clf):
        xs = clf.classification_map["x"].to_numpy()
        ys = clf.classification_map["y"].to_numpy()
        assert (xs >= 0).all() and (xs < MAP_SIZE).all()
        assert (ys >= 0).all() and (ys < MAP_SIZE).all()

    def test_dist_column_non_negative(self, clf):
        assert (clf.classification_map["dist"] >= 0).all()

    def test_topological_error_map_shape(self, clf):
        assert clf.topological_error_map.shape == (MAP_SIZE, MAP_SIZE)

    def test_quantization_error_map_shape(self, clf):
        assert clf.quantization_error_map.shape == (MAP_SIZE, MAP_SIZE)

    def test_classification_hex_map(self):
        m = make_map(topology="hexagonal")
        clf = Classification(m, DATA, verbose=0)
        assert clf.activations_map.sum() == N_SAMPLES
        assert 0.0 <= clf.topological_error <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# TestClassificationTagged
# ══════════════════════════════════════════════════════════════════════════════

class TestClassificationTagged:
    def test_data_shape(self):
        m = make_map()
        tagged = np.hstack([np.arange(N_SAMPLES, dtype=float).reshape(-1, 1), DATA])
        clf = Classification(m, tagged, tagged=True, verbose=0)
        assert clf.classification_data.shape == (N_SAMPLES, N_FEATURES)

    def test_labels_extracted(self):
        m = make_map()
        labels = np.arange(N_SAMPLES, dtype=float)
        tagged = np.hstack([labels.reshape(-1, 1), DATA])
        clf = Classification(m, tagged, tagged=True, verbose=0)
        np.testing.assert_array_equal(clf.classification_labels, labels)

    def test_activations_sum(self):
        m = make_map()
        tagged = np.hstack([np.zeros((N_SAMPLES, 1)), DATA])
        clf = Classification(m, tagged, tagged=True, verbose=0)
        assert clf.activations_map.sum() == N_SAMPLES


# ══════════════════════════════════════════════════════════════════════════════
# TestClassificationOther
# ══════════════════════════════════════════════════════════════════════════════

class TestClassificationOther:
    def test_extra_column_present(self):
        m = make_map()
        other = pd.DataFrame({"extra": RNG.integers(0, 10, N_SAMPLES)})
        clf = Classification(m, DATA, other=other, verbose=0)
        assert "extra" in clf.classification_map.columns

    def test_other_does_not_affect_activations(self):
        m = make_map()
        other = pd.DataFrame({"extra": np.zeros(N_SAMPLES)})
        clf1 = Classification(m, DATA, other=other, verbose=0)
        clf2 = Classification(m, DATA, verbose=0)
        np.testing.assert_array_equal(clf1.activations_map, clf2.activations_map)


# ══════════════════════════════════════════════════════════════════════════════
# TestIterativeSOM
# ══════════════════════════════════════════════════════════════════════════════

class TestIterativeSOM:
    def test_calculate_range_returns_array(self):
        r = IterativeSOM.calculate_range(DATA)
        assert isinstance(r, np.ndarray)
        assert r.shape == (2,)

    def test_calculate_range_min_le_max(self):
        r = IterativeSOM.calculate_range(DATA)
        assert r[0] <= r[1]

    def test_calculate_range_min_at_least_2(self):
        assert IterativeSOM.calculate_range(DATA)[0] >= 2

    def test_calculate_range_small_data(self):
        tiny = RNG.random((10, 2))
        r = IterativeSOM.calculate_range(tiny)
        assert r[0] >= 2
        assert r[0] <= r[1]

    def test_maps_dict_has_correct_keys(self):
        isom = IterativeSOM(DATA, period=PERIOD, initial_lr=0.1,
                            range_from=np.array([2, 4]))
        assert set(isom.maps.keys()) == {2, 3, 4}

    def test_maps_are_map_instances(self):
        isom = IterativeSOM(DATA, period=PERIOD, initial_lr=0.1,
                            range_from=np.array([2, 3]))
        for m in isom.maps.values():
            assert isinstance(m, Map)

    def test_weights_shape_per_size(self):
        isom = IterativeSOM(DATA, period=PERIOD, initial_lr=0.1,
                            range_from=np.array([2, 4]))
        for size, m in isom.maps.items():
            assert m.weights.shape == (size, size, N_FEATURES)

    def test_try_best_sets_best_map(self):
        isom = IterativeSOM(DATA, period=PERIOD, initial_lr=0.1,
                            range_from=np.array([2, 4]), try_best=True)
        assert isinstance(isom.get_best_map(), Map)

    def test_try_best_scores_populated(self):
        isom = IterativeSOM(DATA, period=PERIOD, initial_lr=0.1,
                            range_from=np.array([2, 4]), try_best=True)
        scores = isom.get_scores()
        assert set(scores.keys()) == {2, 3, 4}
        assert all(v >= 0 for v in scores.values())

    def test_best_map_has_lowest_score(self):
        isom = IterativeSOM(DATA, period=PERIOD, initial_lr=0.1,
                            range_from=np.array([2, 4]), try_best=True)
        scores = isom.get_scores()
        best = isom.get_best_map()
        best_size = best.map_size
        assert scores[best_size] == min(scores.values())

    def test_no_try_best_returns_none(self):
        isom = IterativeSOM(DATA, period=PERIOD, initial_lr=0.1,
                            range_from=np.array([2, 3]))
        assert isom.get_best_map() is None

    def test_no_try_best_scores_empty(self):
        isom = IterativeSOM(DATA, period=PERIOD, initial_lr=0.1,
                            range_from=np.array([2, 3]))
        assert isom.get_scores() == {}

    def test_give_best_alias(self):
        isom = IterativeSOM(DATA, period=PERIOD, initial_lr=0.1,
                            range_from=np.array([2, 3]), give_best=True)
        assert isinstance(isom.get_best_map(), Map)

    def test_default_range_from_auto_computed(self):
        """When range_from=[0,0] a valid range must be derived automatically."""
        isom = IterativeSOM(DATA, period=PERIOD, initial_lr=0.1)
        assert len(isom.maps) >= 1
        for m in isom.maps.values():
            assert m.map_size >= 2

    def test_map_kwargs_forwarded(self):
        """topology kwarg must be forwarded to every Map."""
        isom = IterativeSOM(DATA, period=PERIOD, initial_lr=0.1,
                            range_from=np.array([2, 3]),
                            topology="hexagonal")
        for m in isom.maps.values():
            assert m.topology == "hexagonal"

    def test_best_map_can_classify(self):
        isom = IterativeSOM(DATA, period=PERIOD, initial_lr=0.1,
                            range_from=np.array([2, 4]), try_best=True)
        best = isom.get_best_map()
        clf = Classification(best, DATA, verbose=0)
        assert clf.activations_map.sum() == N_SAMPLES
