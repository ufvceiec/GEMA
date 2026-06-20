from .map import Map
from .classification import Classification
from .visualization import Visualization
from .iterativesom import IterativeSOM
from .accelerator import CUPY_AVAILABLE, NUMBA_AVAILABLE, device_info

name = "GEMA"
__version__ = "0.5.2"
