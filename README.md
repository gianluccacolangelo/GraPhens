# GraPhens

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## What is GraPhens?

GraPhens is a phenotype graph construction library and training pipeline. It builds graphs from HPO phenotype sets and supports dataset generation and model training in the current Keras+JAX stack.

## Project Status

GraPhens uses Keras+JAX for dataset generation and model training, with fixed-shape NPZ graph data consumed through Keras data sequences.

## Quick Start

```python
from graphens import GraPhens

# Create a graph from phenotypes
graphens = GraPhens()
graph = graphens.create_graph_from_phenotypes(["HP:0001250", "HP:0001251"])  # Seizure phenotypes

# Export to a format you need
graph_json = graphens.export_graph(graph, format="json")
```

## Core Capabilities

### Phenotype lookup

```python
# Search for phenotypes by keyword
seizure_phenotypes = graphens.phenotype_lookup("seizure")
for phenotype in seizure_phenotypes[:3]:
    print(f"{phenotype.id} - {phenotype.name}")
```

### Configurable graph pipeline

```python
# Chain methods for clear, readable configuration
graphens = (GraPhens()
           .with_embedding_model("openai", "text-embedding-3-small")
           .with_augmentation(include_ancestors=True)
           .with_visualization(enabled=True))

# Create and visualize your graph
graph = graphens.create_graph_from_phenotypes(phenotype_ids)
graphens.visualize(graph=graph, phenotypes=graph.metadata["phenotypes"])
```

### Pre-computed embeddings

```python
# Load biomedical embeddings directly from file
graphens = GraPhens().with_lookup_embeddings("data/embeddings/hpo_biobert.pkl")

# Create graph with domain-specific embeddings
graph = graphens.create_graph_from_phenotypes(phenotype_ids)
```

### Export formats

```python
# NetworkX
nx_graph = graphens.export_graph(graph, format="networkx")

# JSON
graphens.export_graph(graph, format="json", output_path="graph.json")
```

### Multi-patient graph creation

```python
patient_data = {
    "patient_1": ["HP:0001250", "HP:0002066"],
    "patient_2": ["HP:0000407", "HP:0001263"],
    # ... more patients
}

# Create graphs for multiple patients
patient_graphs = graphens.create_graphs_from_multiple_patients(patient_data)
```

## Installation

```bash
pip install graphens
```

## Dataset Generation (Keras + JAX)

Current training dataset path:

`Simulation JSON -> NPZ shards -> Keras/JAX loader`

- Dataset builder: `src/simulation/phenotype_simulation/create_hpo_dataset.py`
  - Two-pass process: collect `max_nodes/max_edges` statistics, then write padded NPZ shards.
- NPZ shard writer: `src/simulation/phenotype_simulation/jax_npz_writer.py`
  - Stores fixed-shape arrays and masks: `x`, `node_mask`, `edge_index`, `edge_mask`, `y`.
- Dataset loaders:
  - `training/datasets/jax_npz_graph_dataset.py`
  - `training/datasets/keras_npz_sequence.py`

Example command:

`python -m src.simulation.phenotype_simulation.create_hpo_dataset --input <simulated_json> --output-dir <dataset_dir> --shard-size 2048 --create-splits`

Validate runtime stack:

`KERAS_BACKEND=jax python scripts/validate_jax_stack.py`

See `docs/dataset_keras_jax.md` for schema, workflow, and dependencies.

## Keras+JAX Training Notes

- Graph samples are pre-sharded as fixed-shape NPZ tensors and consumed via Keras `Sequence` for `model.fit`.
- Batches use static padded tensors with explicit masks (`node_mask`, `edge_mask`) so JAX/XLA can compile stable programs.

## TPU Performance Tips

- Set backend to JAX: `export KERAS_BACKEND=jax`.
- Use larger `--shard-size` and training `batch_size` when memory allows to improve device utilization.
- Keep graph tensors fixed-shape (already done via NPZ + masks) so XLA can compile stable TPU programs.
- Avoid frequent shape changes between runs; keep `max_nodes`, `max_edges`, and model config stable for better compile reuse.

## Design Philosophy

GraPhens prioritizes clear APIs, reproducible preprocessing, and explicit training data contracts across the graph construction and model training pipeline.

## Learn More

```python
# Examples in the repo show you everything you need
from examples import quick_start, custom_embeddings, visualization_demo

# Methods have helpful docstrings
help(GraPhens.create_graph_from_phenotypes)
```

## Advanced Configuration

```python
# Custom embedding models
graphens.with_embedding_model("tfidf", max_features=512)

# Save configurations for reproducibility
graphens.save_config("my_config.json")
loaded_graphens = GraPhens().with_config_from_file("my_config.json")
```

## Examples

See the [examples directory](src/examples/) for end-to-end usage examples.
