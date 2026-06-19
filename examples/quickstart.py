"""
GEMA quickstart example
=======================

This script demonstrates the full GEMA workflow:
  1. Generate or load data
  2. Train a Self-Organizing Map
  3. Classify data on the trained map
  4. Inspect quality metrics
  5. Save and reload the model

Run from the repository root::

    python examples/quickstart.py
"""

import numpy as np
from GEMA.map import Map
from GEMA.classification import Classification

# ── 1. Create a small synthetic dataset ───────────────────────────────────────
# Three 2-D Gaussian clusters
rng = np.random.default_rng(42)
cluster_a = rng.normal(loc=[0.2, 0.2], scale=0.05, size=(40, 2))
cluster_b = rng.normal(loc=[0.5, 0.8], scale=0.05, size=(40, 2))
cluster_c = rng.normal(loc=[0.8, 0.2], scale=0.05, size=(40, 2))
data = np.vstack([cluster_a, cluster_b, cluster_c])

print(f"Dataset shape: {data.shape}")   # (120, 2)

# ── 2. Train the SOM ──────────────────────────────────────────────────────────
som = Map(
    data=data,
    size=6,                        # 6×6 grid
    period=200,                    # training iterations
    initial_lr=0.5,                # starting learning rate
    normalization='none',          # data is already in [0, 1]
    weights='sample',              # initialise weights from data samples
    distance='euclidean',
)

print(f"Training complete. Map size: {som.map_size}×{som.map_size}")

# ── 3. Classify the same data on the trained map ──────────────────────────────
clf = Classification(som, data, verbose=0)

print(f"\nActive neurons   : {clf.num_activations} / {som.map_size ** 2}")
print(f"Quantization error: {clf.quantization_error:.6f}")
print(f"Topological error : {clf.topological_error:.6f}")

# ── 4. Inspect the activation map ─────────────────────────────────────────────
print("\nActivation map (number of patterns per neuron):")
print(clf.activations_map)

# ── 5. Save and reload ────────────────────────────────────────────────────────
som.save_classifier(filename='gema_quickstart_model')
print("\nModel saved to gema_quickstart_model.json")

loaded_som = Map.load_classifier(filename='gema_quickstart_model')
print(f"Model reloaded — weights shape: {loaded_som.weights.shape}")

# Verify the reloaded model gives identical BMU positions
bmu_orig, pos_orig, _, _ = som.calculate_bmu(data[0])
bmu_load, pos_load, _, _ = loaded_som.calculate_bmu(data[0])
assert pos_orig == pos_load, "BMU mismatch after reload!"
print("Reload verification passed.")
