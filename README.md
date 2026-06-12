# GEMA — Self-Organizing Maps in Python

**GEMA** (*GEnerador de Mapas Autoasociativos*) is an open-source Python library for building, training, and analysing Self-Organizing Maps (SOMs / Kohonen maps). It covers the full workflow: data normalisation → training → classification → quality metrics → interactive visualisation.

> **Cite as:**  
> García-Tejedor, Á. J., & Nogales, A. (2022). An Open-Source Python Library for Self-Organizing-Maps. *Software Impacts*, 12. https://doi.org/10.1016/j.simpa.2022.100280

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [Map](#map)
  - [Classification](#classification)
  - [Visualization](#visualization)
- [Normalization Options](#normalization-options)
- [Weight Initialization Options](#weight-initialization-options)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

---

## Features

- Train a SOM from scratch with a single call
- Sequential or random data presentation
- Euclidean and Chebyshev distance metrics
- Neighbourhood decay
- Four normalization strategies (none, FWN, 0-1 scale, Euclidean)
- Four weight-initialization strategies (random, random\_negative, sample, PCA)
- Save / load trained models as JSON
- Classification with topological and quantization error metrics
- U-matrix computation
- Interactive Plotly visualisations (heat map, elevation map, codebook vectors)
- Static Matplotlib visualisations (characteristics graph, bar graph, full weight map)

---

## Installation

```bash
pip install GEMA
```

Conda (Windows x64):

```bash
conda install -c ceiecadmin gema
```

---

## Quick Start

```python
import numpy as np
from GEMA import Map, Classification, Visualization

# 1. Load your data (samples × features)
data = np.loadtxt('my_data.csv', delimiter=',')

# 2. Train a 10×10 SOM for 5000 iterations
som = Map(
    data=data,
    size=10,
    period=5000,
    initial_lr=0.1,
    distance='euclidean',
    normalization='none',
    weights='random'
)

# 3. Save the trained model
som.save_classifier('my_model')

# 4. Classify data
classification = Classification(som, data)
print('Topological error :', classification.topological_error)
print('Quantization error:', classification.quantization_error)

# 5. Visualise
Visualization.heat_map(classification)
Visualization.elevation_map(classification)
```

### Load a pre-trained model

```python
som = Map.load_classifier('my_model')
```

---

## API Reference

### Map

```python
Map(data=None, size, period, initial_lr, initial_neighbourhood=0,
    distance='euclidean', use_decay=False, normalization='none',
    presentation='random', weights='random')
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `np.ndarray` (N×D) | `None` | Training data. If provided, training starts immediately. |
| `size` | `int` | — | Side length of the square map (minimum 2). |
| `period` | `int` | `10` | Number of training iterations. |
| `initial_lr` | `float` | `0.1` | Initial learning rate (0 < lr < 1). |
| `initial_neighbourhood` | `int` | `0` | Initial neighbourhood radius. Defaults to `size` when 0. |
| `distance` | `str` | `'euclidean'` | Distance metric: `'euclidean'` or `'chebyshev'`. |
| `use_decay` | `bool` | `False` | Apply Gaussian decay to neighbour weight updates. |
| `normalization` | `str` | `'none'` | Input normalization strategy (see [Normalization Options](#normalization-options)). |
| `presentation` | `str` | `'random'` | Data presentation order: `'random'` or `'sequential'`. |
| `weights` | `str` | `'random'` | Weight initialization method (see [Weight Initialization Options](#weight-initialization-options)). |

**Key methods:**

| Method | Description |
|---|---|
| `train(data)` | Train the map on `data`. |
| `reinforce(data, reinforcement, extension, compression)` | Continue training with compressed learning rate. |
| `calculate_bmu(pattern)` | Return BMU distance, position, second-BMU distance and position. |
| `save_classifier(filename)` | Save the trained model to `<filename>.json`. |
| `Map.load_classifier(filename)` | Class method — load a model from `<filename>.json`. |

---

### Classification

```python
Classification(som, classification_data, other=None, tagged=False, verbose=1)
```

| Parameter | Type | Description |
|---|---|---|
| `som` | `Map` | Trained SOM. |
| `classification_data` | `np.ndarray` | Data to classify. |
| `other` | `pd.DataFrame` | Optional extra columns to attach to the result table. |
| `tagged` | `bool` | If `True`, first column of `classification_data` is treated as labels. |
| `verbose` | `int` | `0` = silent, `1` = progress bar, `2` = debug output. |

**Key attributes after classification:**

| Attribute | Description |
|---|---|
| `activations_map` | 2-D array — how many samples each neuron won. |
| `distances_map` | 2-D array — cumulative quantization distances. |
| `topological_error` | Fraction of samples whose 2nd BMU is not adjacent to the 1st. |
| `quantization_error` | Mean distance between each sample and its BMU. |
| `classification_map` | Pandas DataFrame with label, coordinates and distance for each sample. |
| `umatriz` | U-matrix (unified distance matrix). |

---

### Visualization

All methods are `@staticmethod`.

| Method | Description |
|---|---|
| `heat_map(classification, ...)` | Interactive Plotly heat map of activations. |
| `elevation_map(classification, ...)` | Interactive Plotly 3-D elevation map. |
| `characteristics_graph(map, row, col, labels, ...)` | Line plot of a single neuron's weight vector. |
| `characteristics_bargraph(map, row, col, labels, ...)` | Bar chart of a single neuron's weight vector. |
| `codebook_vector(map, index, header, ...)` | Annotated heat map for one feature across the whole map. |
| `codebook_vectors(map, headers)` | Render all codebook vectors. |
| `umatrix(classification, colorscale)` | Matplotlib U-matrix plot. |
| `full_map_weights(map, labels, ...)` | Grid of weight plots for every neuron. |
| `neurons_per_num_activations_map(classification, ...)` | Bar chart of activation frequency distribution. |

---

## Normalization Options

| Value | Description |
|---|---|
| `'none'` | No normalization applied. |
| `'fwn'` | Feature-wise normalization (zero mean, unit variance per feature). |
| `'01scale'` | Scale all values to the [0, 1] interval. |
| `'euclidean'` | Euclidean (L2) normalization of each sample vector. |

> **Recommendation:** normalize your data before passing it to GEMA rather than relying on the built-in options.

---

## Weight Initialization Options

| Value | Description |
|---|---|
| `'random'` | Uniform random values in [0, 1]. |
| `'random_negative'` | Uniform random values in [−1, 1]. |
| `'sample'` | Random values sampled directly from the training data. Useful for unnormalized data. |
| `'PCA'` | Weights initialised along the hyperplane spanned by the two largest principal components. |

---

## Requirements

```
numpy
tqdm
pandas
matplotlib
plotly
scikit-learn
scipy
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes and add tests where applicable.
3. Open a pull request describing the change and its motivation.

Bug reports and feature requests are welcome via the [issue tracker](https://github.com/ufvceiec/GEMA/issues).  
Mailing list: gema-som@googlegroups.com

---

## Contact

| Role | Name | Email |
|---|---|---|
| Responsible | Alberto Nogales | alberto.nogales@ceiec.es |
| Supervisor | Álvaro José García-Tejedor | — |
| Developers | Adrián Prieto, Gonzalo de las Heras de Matías, Antonio Pérez Morales | — |
| Contributors | Santiago Donaher Naranjo, Afonso Reis (IST Lisbon) | — |

---

## License

Under license of CEIEC — http://www.ceiec.es
