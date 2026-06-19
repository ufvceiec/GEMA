Usage Guide
===========

This page walks through the full GEMA workflow from data preparation to quality evaluation.

.. contents:: Contents
   :local:
   :depth: 2

----

Preparing data
--------------

GEMA expects a 2-D :class:`numpy.ndarray` where rows are samples and columns are features.
The library includes built-in normalisation options, but we recommend normalising your data
beforehand for maximum control.

.. code-block:: python

    import numpy as np

    # Example: three 2-D clusters
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=[0.2, 0.2], scale=0.05, size=(40, 2))
    cluster_b = rng.normal(loc=[0.5, 0.8], scale=0.05, size=(40, 2))
    cluster_c = rng.normal(loc=[0.8, 0.2], scale=0.05, size=(40, 2))
    data = np.vstack([cluster_a, cluster_b, cluster_c])   # shape (120, 2)

----

Training the SOM
----------------

Pass data directly to :class:`~GEMA.map.Map` to train in one step:

.. code-block:: python

    from GEMA.map import Map

    som = Map(
        data=data,
        size=6,                   # 6×6 neuron grid
        period=200,               # number of training iterations
        initial_lr=0.5,           # initial learning rate (0–1)
        normalization='none',     # 'none' | 'fwn' | '01scale' | 'euclidean'
        weights='sample',         # 'random' | 'random_negative' | 'sample' | 'PCA'
        distance='euclidean',     # 'euclidean' | 'chebyshev'
    )

Key parameters
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Parameter
     - Description
     - Default
   * - ``size``
     - Side length of the square map (``size × size`` neurons)
     - required
   * - ``period``
     - Number of training iterations
     - ``10``
   * - ``initial_lr``
     - Starting learning rate
     - ``0.1``
   * - ``initial_neighbourhood``
     - Starting neighbourhood radius (defaults to ``size``)
     - ``0``
   * - ``normalization``
     - Built-in data normalisation method
     - ``'none'``
   * - ``weights``
     - Weight initialisation strategy
     - ``'random'``
   * - ``distance``
     - Distance metric for BMU search
     - ``'euclidean'``
   * - ``use_decay``
     - Apply distance-based weight decay
     - ``False``

----

Classifying data
----------------

:class:`~GEMA.classification.Classification` maps every sample to its Best Matching Unit (BMU)
and computes quality metrics:

.. code-block:: python

    from GEMA.classification import Classification

    clf = Classification(som, data, verbose=0)

    print(f"Active neurons    : {clf.num_activations} / {som.map_size ** 2}")
    print(f"Quantization error: {clf.quantization_error:.6f}")
    print(f"Topological error : {clf.topological_error:.6f}")
    print(clf.activations_map)       # count of patterns per neuron
    print(clf.classification_map)    # DataFrame: label, x, y, dist per sample

Using labelled data
~~~~~~~~~~~~~~~~~~~

Pass ``tagged=True`` when the first column of your array holds sample labels:

.. code-block:: python

    labels = np.arange(data.shape[0]).reshape(-1, 1)
    tagged_data = np.hstack([labels, data])
    clf = Classification(som, tagged_data, tagged=True)

----

Quality metrics
---------------

After classification the following attributes are available on the
:class:`~GEMA.classification.Classification` object:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Description
   * - ``quantization_error``
     - Mean distance between each sample and its BMU
   * - ``topological_error``
     - Fraction of samples whose 2nd BMU is non-adjacent to the 1st
   * - ``activations_map``
     - 2-D array counting how many samples mapped to each neuron
   * - ``distances_map``
     - 2-D array of cumulative BMU distances per neuron
   * - ``umatriz``
     - Unified distance matrix (U-matrix) for cluster boundary visualisation

----

Saving and loading a model
--------------------------

.. code-block:: python

    # Save
    som.save_classifier(filename='my_model')   # writes my_model.json

    # Load
    loaded = Map.load_classifier(filename='my_model')

----

Visualisation
-------------

.. note::

   Most visualisation methods use `Plotly <https://plotly.com/python/>`_ and are
   designed to run inside Jupyter notebooks.

.. code-block:: python

    from GEMA.visualization import Visualization

    # Heat map of BMU activations
    Visualization.heat_map(clf, colorscale='Reds')

    # 3-D elevation map
    Visualization.elevation_map(clf)

    # U-matrix (matplotlib)
    Visualization.umatrix(clf, colorscale='binary')

    # Weight profile for neuron at (row=2, col=3)
    Visualization.characteristics_graph(som, row=2, column=3)

    # Bar chart of activations distribution
    Visualization.neurons_per_num_activations_map(clf)
