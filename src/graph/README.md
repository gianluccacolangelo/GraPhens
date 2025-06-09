# Graph Module

Responsible for constructing the final graph representation used in GNNs. This involves building the adjacency information (edges) and assembling the complete graph with node features and edge indices.

## Overview

This module implements two core interfaces from `src.core`:

- **`AdjacencyListBuilder`**: Defines how graph edges (connectivity) are created from a list of phenotypes.
    - **`HPOAdjacencyListBuilder`**: Implementation using HPO parent-child relationships (requires `HPOGraphProvider`).
- **`GraphAssembler`**: Defines how the final `Graph` object is created.
    - **`StandardGraphAssembler`**: Implementation that combines phenotypes, node features (from Embedding), and an edge index (from AdjacencyListBuilder) into a `Graph` object.

## Key Features

- **HPO-Based Edges**: `HPOAdjacencyListBuilder` creates edges based on ontology structure.
- **Flexible Assembly**: `StandardGraphAssembler` combines standard components into a GNN-ready graph.
- **Standard Output**: Produces `Graph` objects (defined in `src.core.types`) compatible with downstream processing and export.

## Usage Example

```python
from src.graph.adjacency import HPOAdjacencyListBuilder
from src.graph.assembler import StandardGraphAssembler
from src.ontology.hpo_graph import HPOGraphProvider 
# Assuming: 
# - `hpo_provider` is an initialized HPOGraphProvider
# - `augmented_phenotypes` is a List[Phenotype]
# - `node_features` is a np.ndarray from EmbeddingContext

# 1. Build adjacency list (edges) using HPO hierarchy
builder = HPOAdjacencyListBuilder(hpo_provider, include_reverse_edges=True)
edge_index = builder.build(augmented_phenotypes)

# 2. Assemble the final graph
assembler = StandardGraphAssembler()
graph = assembler.assemble(
    phenotypes=augmented_phenotypes, 
    node_features=node_features,
    edge_index=edge_index
)

# graph object is ready for use/export
print(f"Graph created with {graph.node_features.shape[0]} nodes and {graph.edge_index.shape[1]} edges.")
```

## Validation & Demos

- **Validation (`validation.py`)**: Includes checks for graph properties (e.g., node feature dimensions, edge index format).
- **Demos**: 
    - `demo_adjacency.py`: Shows `HPOAdjacencyListBuilder` usage.
    - `demo_assembler.py`: Demonstrates combining builder and assembler.
    - `demo_validation.py`: Shows validation usage.

Run demos like: `python -m src.graph.demo_assembler`

## Extending

- Create custom edge building logic (e.g., based on semantic similarity) by implementing `AdjacencyListBuilder`.
- Create custom graph assembly logic by implementing `GraphAssembler`.
