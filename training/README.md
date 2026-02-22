# HPO Graph Training Datasets

This directory now contains two dataset paths:

- **Current path (Phase 1):** JSON -> NPZ shards for Keras 3 + JAX
- **Legacy path:** PyG `.pt` -> LMDB for historical reproducibility

## Structure

```text
training/
  datasets/
    jax_npz_graph_dataset.py   # Keras/JAX NPZ loader (current)
    convert_to_lmdb.py         # Legacy converter from old PyG artifacts
    lmdb_hpo_dataset.py        # Legacy LMDB dataset for PyG training
    test.py                    # Legacy LMDB speed/integrity script
```

## Current: NPZ Dataset (Keras + JAX)

Dataset generation is handled by:

- `src/simulation/phenotype_simulation/create_hpo_dataset.py`

Artifacts:

```text
<dataset_root>/
  manifest.json
  shards/shard_XXXXX.npz
  splits/train_indices.npy   # optional
  splits/val_indices.npy     # optional
  splits/test_indices.npy    # optional
```

Loader usage:

```python
from training.datasets.jax_npz_graph_dataset import JAXNPZGraphDataset

train_ds = JAXNPZGraphDataset(
    dataset_root="data/simulation/output_jax_npz",
    split="train",
    batch_size=64,
    shuffle=True,
    seed=42,
)

for batch in train_ds:
    # x, node_mask, edge_index, edge_mask, y
    x = batch["x"]
    y = batch["y"]
    break
```

## Legacy: LMDB/PyG Dataset

`convert_to_lmdb.py` and `lmdb_hpo_dataset.py` are preserved for older experiments that depend on PyTorch Geometric + LMDB.

Important:

- The refactored JSON -> dataset pipeline no longer writes PyG `.pt` batches.
- New dataset generation should target NPZ shards.
- LMDB conversion remains available only for old artifacts.

## Runtime Validation for Keras + JAX

```bash
KERAS_BACKEND=jax python scripts/validate_jax_stack.py
```

## Requirements

- Keras 3 + JAX stack: see `requirements-jax-cpu.txt` or `requirements-jax-cuda12.txt`
- Legacy LMDB stack: PyTorch, PyTorch Geometric, `lmdb`, NumPy, tqdm
