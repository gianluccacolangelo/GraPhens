# GraPhens: The Intuitive Way to Build Phenotype Graphs

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## What is GraPhens?

GraPhens transforms phenotype data into powerful graph representations for machine learning—with almost no learning curve. Whether you're studying rare genetic disorders or exploring phenotype relationships, GraPhens makes building graph neural networks (GNNs) as simple as a few lines of code.

## Why You'll Love It

- **Just Works™:** Reasonable defaults handle the complex stuff behind the scenes
- **Speaks Your Language:** Methods named for what they do, not how they do it
- **Grows With You:** Simple by default, powerful when needed

## Quick Start (30 Seconds!)

```python
from graphens import GraPhens

# Create a graph from phenotypes
graphens = GraPhens()
graph = graphens.create_graph_from_phenotypes(["HP:0001250", "HP:0001251"])  # Seizure phenotypes

# Export for machine learning
pyg_graph = graphens.export_graph(graph, format="pytorch")

# That's it! Ready for your GNN model
```

## Key Features

### 1. Find Phenotypes Without Memorizing IDs

```python
# Search for phenotypes by keyword
seizure_phenotypes = graphens.phenotype_lookup("seizure")
for phenotype in seizure_phenotypes[:3]:
    print(f"{phenotype.id} - {phenotype.name}")
```

### 2. Customize With a Fluent Interface

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

### 3. Use Pre-computed Embeddings

```python
# Load biomedical embeddings directly from file
graphens = GraPhens().with_lookup_embeddings("data/embeddings/hpo_biobert.pkl")

# Create graph with domain-specific embeddings
graph = graphens.create_graph_from_phenotypes(phenotype_ids)
```

### 4. Export to Any ML Framework

```python
# PyTorch Geometric (most common for GNNs)
pyg_graph = graphens.export_graph(graph, format="pytorch")

# NetworkX (for general graph analysis)
nx_graph = graphens.export_graph(graph, format="networkx")

# JSON (for sharing/storage)
graphens.export_graph(graph, format="json", output_path="graph.json")
# Other formats like TensorFlow also available
```

### 5. Efficient Batch Processing for ML Training

```python
# Process multiple patients at once (40-70x faster)
patient_data = {
    "patient_1": ["HP:0001250", "HP:0002066"],
    "patient_2": ["HP:0000407", "HP:0001263"],
    # ... more patients
}

# Create all graphs efficiently
patient_graphs = graphens.create_graphs_from_multiple_patients(patient_data)

# Export as a single batched graph for efficient PyTorch training
batched_graph = graphens.export_graph(patient_graphs, format="pytorch", batch=True)

# Train your PyTorch Geometric model with the batched graph
# (See full example for model definition)
model = GNN() 
node_embeddings = model(batched_graph) # Processes entire batch

# Results maintain patient ID mapping (batched_graph.patient_ids, etc.)
```

## Installation

```bash
pip install graphens
```

## Dataset Generation (Keras + JAX)

The current dataset generation path for model training is:

`Simulation JSON -> NPZ shards -> Keras/JAX loader`

- Generate dataset:
  - `python -m src.simulation.phenotype_simulation.create_hpo_dataset --input <simulated_json> --output-dir <dataset_dir> --shard-size 2048 --create-splits`
- Consume dataset:
  - `training/datasets/jax_npz_graph_dataset.py`
- Validate runtime stack:
  - `KERAS_BACKEND=jax python scripts/validate_jax_stack.py`

See `docs/dataset_keras_jax.md` for schema, workflow, and module dependencies.

## Design Philosophy

GraPhens prioritizes usability and intuitive design, inspired by principles from Don Norman's "The Design of Everyday Things". It aims to be self-explanatory with sensible defaults, allowing complexity to be revealed progressively.

## Learn More

```python
# Examples in the repo show you everything you need
from examples import quick_start, custom_embeddings, visualization_demo

# Methods have helpful docstrings
help(GraPhens.create_graph_from_phenotypes)
```

## Need Advanced Features?

The simple facade hides a powerful, extensible system. When you need more:

```python
# Custom embedding models
graphens.with_embedding_model("tfidf", max_features=512)

# Save configurations for reproducibility
graphens.save_config("my_config.json")
loaded_graphens = GraPhens().with_config_from_file("my_config.json")
```

## Get Started Now

Check out the [examples directory](src/examples/) for complete working examples! 
