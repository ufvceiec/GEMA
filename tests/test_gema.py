"""
Comprehensive pytest test suite for the GEMA SOM library.
"""

import os
import numpy as np
import pandas as pd
import pytest

from GEMA.map import Map
from GEMA.classification import Classification

# ── Shared fixture ────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)

N_SAMPLES = 50
N_FEATURES = 4
MAP_SIZE = 5
PERIOD = 20

DATA = RNG.random((N_SAMPLES, N_FEATURES)).astype(float)


def make_trained_map(distance="euclidean", weights="random", normalization="none"):
    """Return a freshly trained Map."""
    np.random.seed(0)  # keep numpy global RNG deterministic
    m = Map(
        data=DATA,
        size=MAP_SIZE,
        period=PERIOD,
        initial_lr=0.1,
        distance=distance,
        use_decay=False,
        normalization=normalization,
        presentation="random",
        weights=weights,
    )
    return m


# ── TestMapInit ───────────────────────────────────────────────────────────────


class TestMapInit:
    def test_weights_shape_after_train(self):
        m = make_trained_map()
        assert m.weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    def test_default_neighbourhood_equals_size(self):
        # When initial_neighbourhood=0 (default), neighbourhood is set to size
        m = Map(size=MAP_SIZE, period=PERIOD)
        assert m.neighbourhood == MAP_SIZE

    def test_explicit_neighbourhood(self):
        m = Map(size=MAP_SIZE, period=PERIOD, initial_neighbourhood=3)
        assert m.neighbourhood == 3

    def test_map_size_stored(self):
        m = Map(size=MAP_SIZE, period=PERIOD)
        assert m.map_size == MAP_SIZE

    def test_period_stored(self):
        m = Map(size=MAP_SIZE, period=PERIOD)
        assert m.period == PERIOD


# ── TestMapStaticMethods ──────────────────────────────────────────────────────


class TestMapStaticMethods:
    def test_vector_distance_zero(self):
        v = np.array([1.0, 2.0, 3.0])
        assert Map.vector_distance(v, v) == pytest.approx(0.0)

    def test_vector_distance_known(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        assert Map.vector_distance(v1, v2) == pytest.approx(5.0)

    def test_chebyshev_distance_zero(self):
        v = np.array([1.0, 2.0, 3.0])
        assert Map.chebyshev_distance(v, v) == pytest.approx(0.0)

    def test_chebyshev_distance_known(self):
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 5.0, 3.0])
        # max absolute difference = 5
        assert Map.chebyshev_distance(v1, v2) == pytest.approx(5.0)

    def test_chebyshev_less_than_euclidean(self):
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 1.0, 1.0])
        assert Map.chebyshev_distance(v1, v2) <= Map.vector_distance(v1, v2)

    def test_variation_learning_rate_decreases(self):
        lr0 = Map.variation_learning_rate(0.5, 1, 100)
        lr1 = Map.variation_learning_rate(0.5, 50, 100)
        lr2 = Map.variation_learning_rate(0.5, 99, 100)
        assert lr0 > lr1 > lr2

    def test_variation_learning_rate_start(self):
        # At i=0 the value equals initial_lr
        assert Map.variation_learning_rate(0.3, 0, 100) == pytest.approx(0.3)

    def test_variation_neighbourhood_decreases(self):
        n0 = Map.variation_neighbourhood(5, 1, 100, 0)
        n1 = Map.variation_neighbourhood(5, 50, 100, 0)
        n2 = Map.variation_neighbourhood(5, 99, 100, 0)
        assert n0 > n1 > n2

    def test_variation_neighbourhood_with_final(self):
        # final should be approached at i == iterations_number
        val = Map.variation_neighbourhood(5, 100, 100, final=1)
        assert val == pytest.approx(1.0)

    def test_decay_shape(self):
        distances = np.array([0.0, 1.0, 2.0, 3.0])
        result = Map.decay(distances, 2.0)
        assert result.shape == distances.shape

    def test_decay_max_at_bmu(self):
        distances = np.array([0.0, 1.0, 2.0])
        result = Map.decay(distances, 2.0)
        assert result[0] == pytest.approx(1.0)
        assert result[0] > result[1] > result[2]


# ── TestMapTrain ──────────────────────────────────────────────────────────────


class TestMapTrain:
    def test_weights_shape(self):
        m = make_trained_map()
        assert m.weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    def test_trainned_flag(self):
        m = make_trained_map()
        # The attribute is name-mangled due to the double underscore prefix
        assert m._Map__trainned is True

    def test_calculate_bmu_returns_four_items(self):
        m = make_trained_map()
        result = m.calculate_bmu(DATA[0])
        assert len(result) == 4

    def test_bmu_pos_within_bounds(self):
        m = make_trained_map()
        _, bmu_pos, _, second_bmu_pos = m.calculate_bmu(DATA[0])
        assert 0 <= bmu_pos[0] < MAP_SIZE
        assert 0 <= bmu_pos[1] < MAP_SIZE
        assert 0 <= second_bmu_pos[0] < MAP_SIZE
        assert 0 <= second_bmu_pos[1] < MAP_SIZE

    def test_bmu_distance_non_negative(self):
        m = make_trained_map()
        bmu_dist, _, _, _ = m.calculate_bmu(DATA[0])
        assert bmu_dist >= 0

    def test_second_bmu_distance_gte_bmu(self):
        m = make_trained_map()
        bmu_dist, _, second_bmu_dist, _ = m.calculate_bmu(DATA[0])
        assert second_bmu_dist >= bmu_dist

    def test_sequential_presentation(self):
        np.random.seed(0)
        m = Map(
            data=DATA,
            size=MAP_SIZE,
            period=PERIOD,
            initial_lr=0.1,
            presentation="sequential",
        )
        assert m._Map__trainned is True


# ── TestMapNormalization ──────────────────────────────────────────────────────


class TestMapNormalization:
    @pytest.mark.parametrize("method", ["none", "fwn", "01scale", "euclidean"])
    def test_normalization_does_not_crash(self, method):
        np.random.seed(0)
        m = Map(
            data=DATA,
            size=MAP_SIZE,
            period=PERIOD,
            normalization=method,
        )
        assert m.weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)


# ── TestMapWeightsInit ────────────────────────────────────────────────────────


class TestMapWeightsInit:
    @pytest.mark.parametrize("method", ["random", "random_negative", "sample", "PCA"])
    def test_weights_init_does_not_crash(self, method):
        np.random.seed(0)
        m = Map(
            data=DATA,
            size=MAP_SIZE,
            period=PERIOD,
            weights=method,
        )
        assert m.weights.shape == (MAP_SIZE, MAP_SIZE, N_FEATURES)

    def test_random_weights_in_01(self):
        np.random.seed(0)
        m = Map(data=DATA, size=MAP_SIZE, period=PERIOD, weights="random")
        assert m.weights.min() >= 0.0
        assert m.weights.max() <= 1.0

    def test_random_negative_weights_in_minus1_1(self):
        np.random.seed(0)
        m = Map(data=DATA, size=MAP_SIZE, period=PERIOD, weights="random_negative")
        assert m.weights.min() >= -1.0
        assert m.weights.max() <= 1.0


# ── TestMapSaveLoad ───────────────────────────────────────────────────────────


class TestMapSaveLoad:
    def test_save_and_reload_weights_match(self, tmp_path):
        m = make_trained_map()
        filepath = str(tmp_path / "test_model")
        m.save_classifier(filepath)
        assert os.path.exists(filepath + ".json")

        loaded = Map.load_classifier(filepath)
        np.testing.assert_array_almost_equal(m.weights, loaded.weights)

    def test_loaded_map_attributes(self, tmp_path):
        m = make_trained_map()
        filepath = str(tmp_path / "model2")
        m.save_classifier(filepath)
        loaded = Map.load_classifier(filepath)

        assert loaded.map_size == m.map_size
        assert loaded.period == m.period
        assert loaded.distance == m.distance


# ── TestMapDistance ───────────────────────────────────────────────────────────


class TestMapDistance:
    def test_calculate_distance_euclidean(self):
        m = Map(size=MAP_SIZE, period=PERIOD, distance="euclidean")
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        assert m.calculate_distance(v1, v2) == pytest.approx(5.0)

    def test_calculate_distance_chebyshev(self):
        m = Map(size=MAP_SIZE, period=PERIOD, distance="chebyshev")
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 5.0, 3.0])
        assert m.calculate_distance(v1, v2) == pytest.approx(5.0)

    def test_chebyshev_vs_euclidean_ordering(self):
        """Chebyshev <= Euclidean for same pair of vectors."""
        v1 = RNG.random(10)
        v2 = RNG.random(10)
        assert Map.chebyshev_distance(v1, v2) <= Map.vector_distance(v1, v2)


# ── TestClassification ────────────────────────────────────────────────────────


class TestClassification:
    @pytest.fixture(scope="class")
    def trained_map_and_clf(self):
        m = make_trained_map()
        clf = Classification(m, DATA, verbose=0)
        return m, clf

    def test_activations_map_sums_to_n_samples(self, trained_map_and_clf):
        _, clf = trained_map_and_clf
        assert clf.activations_map.sum() == N_SAMPLES

    def test_umatriz_shape(self, trained_map_and_clf):
        _, clf = trained_map_and_clf
        expected = 2 * MAP_SIZE - 1
        assert clf.umatriz.shape == (expected, expected)

    def test_classification_map_columns(self, trained_map_and_clf):
        _, clf = trained_map_and_clf
        for col in ("x", "y", "dist"):
            assert col in clf.classification_map.columns

    def test_classification_map_rows(self, trained_map_and_clf):
        _, clf = trained_map_and_clf
        assert len(clf.classification_map) == N_SAMPLES

    def test_topological_error_in_0_1(self, trained_map_and_clf):
        _, clf = trained_map_and_clf
        assert 0.0 <= clf.topological_error <= 1.0

    def test_quantization_error_non_negative(self, trained_map_and_clf):
        _, clf = trained_map_and_clf
        assert clf.quantization_error >= 0.0

    def test_distances_map_non_negative(self, trained_map_and_clf):
        _, clf = trained_map_and_clf
        assert (clf.distances_map >= 0).all()

    def test_num_activations_positive(self, trained_map_and_clf):
        _, clf = trained_map_and_clf
        assert clf.num_activations > 0

    def test_mean_distance_map_non_negative(self, trained_map_and_clf):
        _, clf = trained_map_and_clf
        assert clf.mean_distance_map >= 0.0


# ── TestClassificationTagged ──────────────────────────────────────────────────


class TestClassificationTagged:
    def test_tagged_classification(self):
        m = make_trained_map()
        # First column = integer labels 0..N_SAMPLES-1
        labels = np.arange(N_SAMPLES).reshape(-1, 1).astype(float)
        tagged_data = np.hstack([labels, DATA])

        clf = Classification(m, tagged_data, tagged=True, verbose=0)

        assert clf.classification_data.shape == (N_SAMPLES, N_FEATURES)
        assert len(clf.classification_labels) == N_SAMPLES
        assert clf.activations_map.sum() == N_SAMPLES

    def test_tagged_labels_extracted_correctly(self):
        m = make_trained_map()
        label_values = np.arange(N_SAMPLES, dtype=float)
        tagged_data = np.hstack([label_values.reshape(-1, 1), DATA])

        clf = Classification(m, tagged_data, tagged=True, verbose=0)
        np.testing.assert_array_equal(clf.classification_labels, label_values)


# ── TestClassificationOther ───────────────────────────────────────────────────


class TestClassificationOther:
    def test_other_dataframe_appended_as_columns(self):
        m = make_trained_map()
        other_df = pd.DataFrame(
            {"extra_col": RNG.integers(0, 10, size=N_SAMPLES)}
        )

        clf = Classification(m, DATA, other=other_df, verbose=0)
        assert "extra_col" in clf.classification_map.columns

    def test_other_does_not_affect_activations(self):
        m = make_trained_map()
        other_df = pd.DataFrame({"extra_col": np.zeros(N_SAMPLES)})

        clf_with = Classification(m, DATA, other=other_df, verbose=0)
        clf_without = Classification(m, DATA, verbose=0)

        np.testing.assert_array_equal(
            clf_with.activations_map, clf_without.activations_map
        )
