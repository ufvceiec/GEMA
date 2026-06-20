import json
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler

from .accelerator import (
    CUPY_AVAILABLE, NUMBA_AVAILABLE,
    get_array_module, to_device, to_numpy,
)

if NUMBA_AVAILABLE:
    from .accelerator import (
        _bmu_distances_numba,
        _euclidean_grid_distances_numba,
        _chebyshev_grid_distances_numba,
        _weight_update_numba,
    )


class Map:
    """
    Map class is the main component of GEMA. It contains the classifying map
    that allows for classification and is subject of analysis in search of data
    information.

    Training can be accelerated in two independent ways:

    * **CPU / numba** (``device='cpu'``, numba installed): the BMU search and
      weight-update kernels are compiled to native machine code and run in
      parallel across all CPU cores via ``numba.prange``.

    * **GPU / CuPy** (``device='gpu'``, CuPy installed and a CUDA GPU present):
      the weight tensor and training data are transferred to the GPU once at the
      start of training.  All inner-loop operations (BMU search, neighbourhood
      distances, weight update) run as native CUDA kernels via CuPy.  The
      finished weight tensor is copied back to CPU automatically so that
      :class:`Classification`, :meth:`save_classifier`, and all
      :class:`Visualization` methods work without any changes.
    """

    def __init__(self,
                 data=None,
                 size=-1,
                 period=10,
                 initial_lr=0.1,
                 initial_neighbourhood=0,
                 distance='euclidean',
                 use_decay=False,
                 normalization='none',
                 presentation='random',
                 weights='random',
                 topology='rectangular',
                 device='cpu'):
        """Initialise a Self-Organizing Map.

        :param data: numpy array (N × D). If provided, training starts
            immediately after construction.
        :param size: Side length of the square map (≥ 2).
        :param period: Number of training iterations.
        :param initial_lr: Initial learning rate (0 < lr < 1).
        :param initial_neighbourhood: Initial neighbourhood radius.
            Defaults to ``size`` when 0.
        :param distance: Distance metric for neighbourhood calculations:
            ``'euclidean'`` (default) or ``'chebyshev'``.
        :param use_decay: Apply Gaussian decay so neurons closer to the BMU
            learn more than distant ones.
        :param normalization: Pre-processing applied to training data:
            ``'none'`` | ``'fwn'`` | ``'01scale'`` | ``'euclidean'``.
        :param presentation: Pattern selection order:
            ``'random'`` (default) or ``'sequential'``.
        :param weights: Weight initialisation strategy:
            ``'random'`` | ``'random_negative'`` | ``'sample'`` | ``'PCA'``.
        :param topology: Neuron grid layout:
            ``'rectangular'`` (default) or ``'hexagonal'``.
            Hexagonal grids give each neuron 6 equidistant neighbours and
            produce more uniform cluster boundaries.
        :param device: Compute device for the training loop:

            * ``'cpu'`` (default) — plain NumPy; if **numba** is installed
              the inner kernels are JIT-compiled and run in parallel across
              all CPU cores automatically.
            * ``'gpu'`` — requires **CuPy** and a CUDA-capable GPU.
              All weight/data tensors live on the GPU during training and are
              automatically transferred back to CPU when training finishes.
        """

        # ── Input validation ──────────────────────────────────────────────────
        self.__trainned = False
        assert period > 1, 'Period must be a positive value'
        assert initial_lr < 1 or initial_lr > 0, \
            'Learning rate initial must be between 0 and 1'
        assert size >= 2, 'Map size must be a value higher than 1'
        assert topology in ('rectangular', 'hexagonal'), \
            "topology must be 'rectangular' or 'hexagonal'"
        assert device in ('cpu', 'gpu'), \
            "device must be 'cpu' or 'gpu'"

        # Validate GPU availability early so the user gets a clear error message
        # before spending time on weight initialisation.
        if device == 'gpu' and not CUPY_AVAILABLE:
            raise RuntimeError(
                "device='gpu' requested but CuPy is not installed or no CUDA GPU "
                "was detected.\nInstall CuPy: pip install cupy-cuda12x"
            )

        # ── Attributes ────────────────────────────────────────────────────────
        self.map_size = size
        self.presentation = presentation
        self.initial_lr = initial_lr
        self.distance = distance
        self.use_decay = use_decay
        self.num_data = 0
        self.input_data_dimension = 0
        self.period = period
        self.neighbourhood = initial_neighbourhood if initial_neighbourhood != 0 \
            else size
        self.normalization = normalization
        self.weights_init = weights
        self.topology = topology
        self.device = device

        # Placeholder; real weights are set in train()
        self.weights = np.random.random(1)

        # Neuron position matrix — (size, size, 2) float64.
        # Rectangular: integer [y, x] coords.
        # Hexagonal  : offset-coordinate float positions so that all 6
        #              nearest neighbours are equidistant.
        self.__ids_matrix = self.__build_ids_matrix()

        if data is not None:
            self.train(data)

    # ──────────────────────────────────────────────────────────────────────────
    # GRID HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def __build_ids_matrix(self) -> np.ndarray:
        """Build the (map_size, map_size, 2) float64 array of neuron positions."""
        if self.topology == 'rectangular':
            ys, xs = np.meshgrid(np.arange(self.map_size, dtype=np.float64),
                                 np.arange(self.map_size, dtype=np.float64),
                                 indexing='ij')
            return np.stack([ys, xs], axis=-1)   # (S, S, 2)

        else:  # hexagonal
            # Odd rows are shifted +0.5 in x; rows are spaced √3/2 apart in y.
            sqrt3_2 = np.sqrt(3.0) / 2.0
            ids = np.empty((self.map_size, self.map_size, 2), dtype=np.float64)
            for y in range(self.map_size):
                x_offset = 0.5 if (y % 2 == 1) else 0.0
                ids[y, :, 0] = y * sqrt3_2
                ids[y, :, 1] = np.arange(self.map_size) + x_offset
            return ids   # (S, S, 2)

    # ──────────────────────────────────────────────────────────────────────────
    # TRAINING
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, data):
        """Train the SOM on *data*.

        :param data: numpy array (N × D).
        """
        data = np.asarray(data, dtype=np.float64)
        self.num_data = data.shape[0]
        self.input_data_dimension = data.shape[1]

        # Normalise on CPU (one-time, cheap)
        training_data_np = self.__normalize(data, method=self.normalization)
        weights_np = self.__init_weights(data=data, method=self.weights_init)

        # ── Move tensors to compute device ────────────────────────────────────
        _W = to_device(weights_np, self.device)            # (S, S, D)
        _T = to_device(training_data_np, self.device)      # (N, D)
        _I = to_device(self.__ids_matrix, self.device)     # (S, S, 2)

        accel = []
        if self.device == 'gpu':
            accel.append('CuPy GPU')
        if self.device == 'cpu' and NUMBA_AVAILABLE:
            accel.append('numba CPU parallel')
        tag = ', '.join(accel) if accel else 'NumPy CPU'
        print(f"TRAINING... [{tag}]")

        for numPresentation in tqdm(range(1, self.period + 1)):
            # ── Pattern selection ──────────────────────────────────────────────
            if self.presentation == 'sequential':
                idx = numPresentation % self.num_data
            else:
                idx = int(np.random.randint(0, self.num_data))
            new_pattern = _T[idx]

            # ── BMU search ─────────────────────────────────────────────────────
            bmu_pos = self.__find_bmu(_W, new_pattern)

            # ── Schedule ───────────────────────────────────────────────────────
            eta = self.variation_learning_rate(self.initial_lr, numPresentation,
                                               self.period)
            v_final = 1.0 if self.use_decay else 0.0
            v = self.variation_neighbourhood(self.neighbourhood, numPresentation,
                                             self.period, v_final)

            # ── Weight update ──────────────────────────────────────────────────
            _W = self.__update_weights(_W, _I, bmu_pos, v, eta, new_pattern)

        # Always keep self.weights as a plain numpy array so that
        # Classification, save_classifier, and Visualization work unchanged.
        self.weights = to_numpy(_W)
        self.__trainned = True
        print("FINISHED.")

    def reinforce(self,
                  training_data,
                  reinforcement=0,
                  extension=1,
                  compression=0.5):
        """Fine-tune the map with additional training at a compressed learning rate.

        :param training_data: numpy array (N × D).
        :param reinforcement: Number of reinforcement rounds.
        :param extension: Multiplier applied to ``period`` each round (> 1 extends).
        :param compression: Multiplier applied to the learning rate each round
            (< 1 shrinks it, e.g. 0.5 halves it every round).
        """
        training_data = np.asarray(training_data, dtype=np.float64)

        for _ in range(reinforcement):
            self.period = int(self.period * extension)
            self.initial_lr *= compression

            _W = to_device(self.weights, self.device)
            _T = to_device(training_data, self.device)
            _I = to_device(self.__ids_matrix, self.device)

            for numPresentation in tqdm(range(1, self.period + 1)):
                if self.presentation == 'sequential':
                    idx = numPresentation % self.num_data
                else:
                    idx = int(np.random.randint(0, self.num_data))
                new_pattern = _T[idx]

                bmu_pos = self.__find_bmu(_W, new_pattern)
                eta = self.variation_learning_rate(self.initial_lr, numPresentation,
                                                   self.period)
                _W = self.__update_weights(_W, _I, bmu_pos, 1.0, eta, new_pattern)

            self.weights = to_numpy(_W)

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNAL ACCELERATED KERNELS
    # ──────────────────────────────────────────────────────────────────────────

    def __find_bmu(self, W, pattern):
        """Return the (row, col) index of the Best Matching Unit.

        Uses the numba kernel when on CPU and numba is available;
        otherwise uses the vectorised numpy/cupy path.
        """
        if self.device == 'cpu' and NUMBA_AVAILABLE:
            sq_dists = _bmu_distances_numba(W, pattern)
            return np.unravel_index(np.argmin(sq_dists), sq_dists.shape)
        else:
            xp = get_array_module(W)
            sq_dists = xp.sum((W - pattern) ** 2, axis=-1)
            return tuple(int(i) for i in
                         xp.unravel_index(xp.argmin(sq_dists), sq_dists.shape))

    def __update_weights(self, W, I, bmu_pos, v, eta, pattern):
        """Update weights for all neurons within neighbourhood *v* of *bmu_pos*.

        Uses numba when on CPU; uses cupy vector ops when on GPU.
        Returns the (possibly mutated) weight array.
        """
        if self.device == 'cpu' and NUMBA_AVAILABLE:
            if self.distance == 'chebyshev':
                grid_dists = _chebyshev_grid_distances_numba(
                    I, bmu_pos[0], bmu_pos[1])
            else:
                grid_dists = _euclidean_grid_distances_numba(
                    I, bmu_pos[0], bmu_pos[1])
            W = _weight_update_numba(W, grid_dists, float(v), float(eta),
                                     pattern, self.use_decay)
        else:
            xp = get_array_module(W)
            bmu_pos_2d = I[bmu_pos[0], bmu_pos[1]]           # (2,)
            diff = I - bmu_pos_2d                             # (S, S, 2)

            if self.distance == 'chebyshev':
                distances = xp.max(xp.abs(diff), axis=-1)
            else:
                distances = xp.sqrt(xp.sum(diff ** 2, axis=-1))

            to_update = distances <= v

            if self.use_decay:
                decay_vals = xp.exp(-(distances ** 2) / (2.0 * float(v) ** 2))
            else:
                decay_vals = xp.ones_like(distances)

            W[to_update] = (W[to_update]
                            + eta
                            * xp.expand_dims(decay_vals[to_update], axis=1)
                            * (pattern - W[to_update]))
        return W

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC INFERENCE
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_bmu(self, pattern):
        """Return the BMU and second-BMU for *pattern*.

        Always operates on CPU (weights are stored as numpy after training),
        so this method works identically regardless of the ``device`` used for
        training.

        :param pattern: 1-D array of length D.
        :return: ``(bmu_dist, bmu_pos, second_bmu_dist, second_bmu_pos)``
        """
        pattern = np.asarray(pattern, dtype=np.float64)
        distancias = np.sum((self.weights - pattern) ** 2, axis=-1)

        bmu_dist = float(np.min(distancias))
        bmu_pos = np.unravel_index(np.argmin(distancias), distancias.shape)

        distancias[bmu_pos] = np.inf

        second_bmu_dist = float(np.min(distancias))
        second_bmu_pos = np.unravel_index(np.argmin(distancias), distancias.shape)

        return bmu_dist, bmu_pos, second_bmu_dist, second_bmu_pos

    def calculate_distance(self, vector1, vector2):
        """Dispatch to the configured distance metric.

        Works with both numpy and cupy arrays (device-agnostic).
        """
        if self.distance == 'chebyshev':
            return Map.chebyshev_distance(vector1, vector2)
        return Map.vector_distance(vector1, vector2)

    # ──────────────────────────────────────────────────────────────────────────
    # STATIC MATH HELPERS  (device-agnostic via get_array_module)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def variation_learning_rate(initial_lr, i, iterations_number):
        """Linear decay of the learning rate.

        :param initial_lr: Initial learning rate.
        :param i: Current iteration (1-based).
        :param iterations_number: Total number of iterations.
        :return: Learning rate for iteration *i*.
        """
        return initial_lr + ((-initial_lr * i) / iterations_number)

    @staticmethod
    def variation_neighbourhood(initial_neighbourhood, i,
                                iterations_number, final=0):
        """Linear decay of the neighbourhood radius.

        :param initial_neighbourhood: Initial neighbourhood radius.
        :param i: Current iteration (1-based).
        :param iterations_number: Total number of iterations.
        :param final: Neighbourhood radius at the last iteration.
        :return: Neighbourhood radius for iteration *i*.
        """
        return final + initial_neighbourhood * (1 - (i / iterations_number))

    @staticmethod
    def decay(distance_BMU, current_neighbourhood):
        """Gaussian decay kernel.

        Works with both numpy and cupy arrays.

        :param distance_BMU: Array of distances to the BMU.
        :param current_neighbourhood: Current neighbourhood radius.
        :return: Decay multiplier array (same shape as *distance_BMU*).
        """
        xp = get_array_module(distance_BMU)
        return xp.exp(-(distance_BMU ** 2) / (2 * (current_neighbourhood ** 2)))

    @staticmethod
    def vector_distance(vector1, vector2):
        """Euclidean (L2) distance. Works with numpy and cupy arrays.

        :param vector1: First vector or array of vectors.
        :param vector2: Second vector or array of vectors (same shape).
        :return: Scalar or array of distances.
        """
        xp = get_array_module(vector1)
        return xp.sqrt(xp.sum(xp.square(vector1 - vector2), axis=-1))

    @staticmethod
    def chebyshev_distance(vector1, vector2):
        """Chebyshev (L∞) distance. Works with numpy and cupy arrays.

        :param vector1: First vector or array of vectors.
        :param vector2: Second vector or array of vectors (same shape).
        :return: Scalar or array of distances.
        """
        xp = get_array_module(vector1)
        return xp.max(xp.abs(vector1 - vector2), axis=-1)

    # ──────────────────────────────────────────────────────────────────────────
    # NORMALISATION
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def __normalize(data, method):
        """Normalise *data* (numpy float64 array) according to *method*.

        :param data: numpy array (N × D).
        :param method: ``'none'`` | ``'fwn'`` | ``'01scale'`` | ``'euclidean'``.
        :return: Normalised numpy array (same shape).
        """
        if method == 'none':
            return data

        if method == 'fwn':
            # Feature-wise normalisation: zero mean, unit variance per feature.
            data = data - data.mean(axis=0)
            std = data.std(axis=0)
            std[std == 0] = 1.0          # avoid div-by-zero for constant features
            data = data / std

        elif method == 'euclidean':
            # L2 normalisation: project each sample onto the unit hypersphere.
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            data = data / norms

        elif method == '01scale':
            lo, hi = data.min(), data.max()
            data = (data - lo) / (hi - lo) if hi != lo else data

        return data

    # ──────────────────────────────────────────────────────────────────────────
    # WEIGHT INITIALISATION
    # ──────────────────────────────────────────────────────────────────────────

    def __init_weights(self, data, method):
        """Initialise the weight tensor (always on CPU as numpy float64).

        :param data: numpy array (N × D) — training data.
        :param method: ``'random'`` | ``'random_negative'`` | ``'sample'`` | ``'PCA'``.
        :return: numpy array (map_size, map_size, D).
        """
        S, D = self.map_size, self.input_data_dimension

        if method == 'random':
            return np.random.random((S, S, D))

        elif method == 'random_negative':
            return np.random.uniform(-1.0, 1.0, (S, S, D))

        elif method == 'sample':
            # Pick S×S random training rows as initial neuron weights.
            idx = np.random.randint(0, self.num_data, size=(S, S))
            return data[idx].astype(np.float64)   # (S, S, D)

        elif method == 'PCA':
            scaler = StandardScaler()
            std_data = scaler.fit_transform(data)
            pca_weights = np.zeros((S, S, D))
            cov = np.cov(std_data.T)
            pc_length, pc = np.linalg.eig(cov)
            pc_order = np.argsort(-pc_length)
            for i, c1 in enumerate(np.linspace(-1, 1, S)):
                for j, c2 in enumerate(np.linspace(-1, 1, S)):
                    pca_weights[i, j] = (c1 * pc[:, pc_order[0]]
                                         + c2 * pc[:, pc_order[1]])
            return pca_weights.real.astype(np.float64)

        raise ValueError(f"Unknown weight initialisation method: '{method}'")

    # ──────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def load_classifier(cls, filename='Model'):
        """Load a previously saved SOM from a JSON file.

        :param filename: Path without ``.json`` extension.
        :return: A fully initialised :class:`Map` instance (on CPU).
        """
        with open(filename + '.json') as f:
            data = json.load(f)

        model = data['model'][0]
        new_map = cls(
            data=None,
            size=model['map_size'],
            period=model['period'],
            initial_lr=model['initial_lr'],
            initial_neighbourhood=model['neighbourhood'],
            distance=model['distance'],
            use_decay=model['use_decay'],
            topology=model.get('topology', 'rectangular'),
            device='cpu',   # always load to CPU
        )
        new_map.weights = np.array(model['weights'])
        new_map.input_data_dimension = model['input_data_dimension']
        new_map.presentation = model['presentation']
        new_map.num_data = model['num_data']
        new_map.__trainned = True

        print('Imported successfully')
        return new_map

    def save_classifier(self, filename='Model'):
        """Save the trained SOM to a JSON file.

        :param filename: Path without ``.json`` extension.
        """
        # to_numpy handles the case where weights might still be on GPU.
        weights_cpu = to_numpy(self.weights)

        payload = {'model': [{
            'map_size': self.map_size,
            'input_data_dimension': self.input_data_dimension,
            'presentation': self.presentation,
            'initial_lr': self.initial_lr,
            'distance': self.distance,
            'use_decay': self.use_decay,
            'num_data': self.num_data,
            'period': self.period,
            'neighbourhood': self.neighbourhood,
            'topology': self.topology,
            'weights': weights_cpu.tolist(),
        }]}

        with open(filename + '.json', 'w') as f:
            json.dump(payload, f)

        print('Saved successfully')
