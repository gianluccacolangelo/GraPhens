# Keras + JAX Dataset Format (JSON -> NPZ)

This document describes the dataset pipeline for
`JSON -> NPZ shards`.

## Workflow Diagram

```mermaid
flowchart TD
    A[Simulation JSON\n gene -> list[patient phenotypes]] --> B[create_hpo_dataset.py\nPass 1: compute global max nodes/edges]
    B --> C[GraPhens + ComponentFactory + StandardPipelineOrchestrator\nlookup embeddings, augmentation, adjacency, assembler]
    C --> D[Graph objects\nnode_features + edge_index]
    D --> E[Pass 2: JAXNPZShardWriter\npad + mask + shard]
    E --> F[shards/shard_XXXXX.npz]
    E --> G[manifest.json]
    G --> H[Optional split generation\nsplits/*.npy + gene_splits.json]
    F --> I[JAXNPZGraphDataset\ntraining/datasets/jax_npz_graph_dataset.py]
    G --> I
    H --> I
```

## Modules and Responsibilities

- `src/simulation/phenotype_simulation/create_hpo_dataset.py`
  - Entry point for JSON -> NPZ conversion.
  - Runs two-pass conversion to enforce fixed global shapes.
  - Optional split generation in `.npy` format.

- `src/simulation/phenotype_simulation/jax_npz_writer.py`
  - Writes padded NPZ shards and `manifest.json`.
  - Enforces canonical dtypes and sentinel values.

- `src/simulation/phenotype_simulation/jax_dataset_schema.py`
  - Schema constants and validation for manifest/shards.
  - Validates shape consistency, masks, padding, and label bounds.

- `training/datasets/jax_npz_graph_dataset.py`
  - Split-aware loader for Keras/JAX training.
  - Yields batched NumPy arrays with deterministic optional shuffle.


## Output Layout

```text
output_dir/
  manifest.json
  shards/
    shard_00000.npz
    shard_00001.npz
  splits/                  # optional
    train_indices.npy
    val_indices.npy
    test_indices.npy
    gene_splits.json
```

## NPZ Schema

Every shard stores:

- `x`: `[S, Nmax, D]`, `float32`
- `node_mask`: `[S, Nmax]`, `bool`
- `edge_index`: `[S, 2, Emax]`, `int32` (`-1` padded)
- `edge_mask`: `[S, Emax]`, `bool`
- `y`: `[S]`, `int32`
- `gene_idx`: `[S]`, `int32`
- `case_idx`: `[S]`, `int32`

## CLI Usage

```bash
python -m src.simulation.phenotype_simulation.create_hpo_dataset \
  --input data/simulation/output/simulated_patients_all_20251202_154807.json \
  --output-dir data/simulation/output_jax_npz \
  --shard-size 2048 \
  --embeddings data/embeddings/hpo_embeddings_gsarti_biobert-nli_20250317_162424.pkl \
  --create-splits \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --seed 42
```

Optional shape overrides:

```bash
--max-nodes 256 --max-edges 1024
```

## Loader Usage

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
    x = batch["x"]
    node_mask = batch["node_mask"]
    edge_index = batch["edge_index"]
    edge_mask = batch["edge_mask"]
    y = batch["y"]
    break
```

## Runtime Validation

Use the stack probe script:

```bash
KERAS_BACKEND=jax python scripts/validate_jax_stack.py
```

## Notes

- NPZ is the canonical artifact format.
- PyG `.pt` and LMDB conversion are not supported by the refactored JSON -> dataset pipeline.
