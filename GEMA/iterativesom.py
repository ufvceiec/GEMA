import numpy as np

from .map import Map
from .classification import Classification


class IterativeSOM:
    """Trains multiple SOMs of different sizes and optionally selects the best one.

    This is useful when the optimal map size is not known in advance.
    A range of sizes is explored and each resulting map is evaluated using the
    quantization error so the user can pick — or automatically obtain — the
    best-performing map.

    Usage example::

        isom = IterativeSOM(
            data=my_data,
            period=1000,
            initial_lr=0.1,
            range_from=np.array([5, 15]),   # try sizes 5 → 15
            try_best=True,
        )
        best = isom.get_best_map()
        print(isom.scores)   # {size: quantization_error, ...}
    """

    def __init__(self,
                 data,
                 period,
                 initial_lr,
                 range_from=np.array([0, 0]),
                 try_best=False,
                 give_best=False,
                 **map_kwargs):
        """
        :param data: numpy array (N × D) — training data.
        :param period: Number of training iterations for each map.
        :param initial_lr: Initial learning rate for each map.
        :param range_from: 2-element array [min_size, max_size].
            When left as the default [0, 0] a sensible range is derived
            automatically from the size of the dataset.
        :param try_best: If True, classify the training data with every trained
            map and record the quantization error.  The map with the lowest
            error is stored as ``best_map``.
        :param give_best: Alias for ``try_best`` (kept for backwards
            compatibility).
        :param map_kwargs: Any extra keyword arguments are forwarded to every
            :class:`Map` constructor call (e.g. ``topology``, ``distance``,
            ``normalization``).
        """
        self.maps = {}
        self.scores = {}
        self.best_map = None
        evaluate = try_best or give_best

        if np.array_equal(range_from, np.array([0, 0])):
            range_from = IterativeSOM.calculate_range(data)

        min_size = int(range_from[0])
        max_size = int(range_from[1])

        for size in range(min_size, max_size + 1):
            print(f"\n=== Training SOM size {size}×{size} ===")
            m = Map(data=data, size=size, period=period,
                    initial_lr=initial_lr, **map_kwargs)
            self.maps[size] = m

            if evaluate:
                c = Classification(m, data)
                self.scores[size] = c.quantization_error
                print(f"    Quantization error: {c.quantization_error:.6f}")

        if evaluate and self.scores:
            best_size = min(self.scores, key=self.scores.get)
            self.best_map = self.maps[best_size]
            print(f"\nBest map: size {best_size}×{best_size} "
                  f"(quantization error {self.scores[best_size]:.6f})")

    # ------------------------------------------------------------------
    # PUBLIC HELPERS
    # ------------------------------------------------------------------

    def get_best_map(self):
        """Return the :class:`Map` with the lowest quantization error.

        :return: Best :class:`Map` instance, or ``None`` if ``try_best``
                 was not set.
        """
        return self.best_map

    def get_scores(self):
        """Return a dict mapping each map size to its quantization error.

        :return: ``{size: quantization_error}`` — empty if ``try_best`` was
                 not set.
        """
        return dict(self.scores)

    # ------------------------------------------------------------------
    # STATIC HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_range(data):
        """Derive a sensible [min_size, max_size] range from the dataset.

        The heuristic is based on a common rule of thumb for SOM sizing:
        the map should have roughly ``5 * sqrt(N)`` neurons in total,
        meaning a side length of about ``sqrt(5 * sqrt(N))``.  We explore
        ±2 sizes around that centre point, with a minimum side length of 2.

        :param data: numpy array (N × D).
        :return: numpy array ``[min_size, max_size]``.
        """
        n = data.shape[0]
        centre = max(2, int(np.sqrt(5 * np.sqrt(n))))
        min_size = max(2, centre - 2)
        max_size = centre + 2
        return np.array([min_size, max_size])
