# Phenotype Visualization Module

Provides capabilities for visualizing phenotype hierarchies and graphs using the `PhenotypeVisualizer` interface.

## Overview

This module helps understand HPO relationships and the structure of generated graphs.

- **`GraphvizVisualizer`**: The primary implementation, using the Graphviz library to create PNG renderings of:
    - HPO hierarchies (showing parent-child relationships)
    - Full `Graph` objects (nodes and edges)
    - Augmentation results (highlighting added phenotypes)

## Key Features

- **Hierarchical Layouts**: Visualizes HPO structure.
- **Graph Rendering**: Displays nodes and edges from `Graph` objects.
- **Highlighting**: Differentiates initial vs. augmented phenotypes.
- **Customizable**: Output directory and titles can be set.

## Usage Example (`GraphvizVisualizer`)

```python
from src.visualization.graphviz import GraphvizVisualizer
from src.ontology.hpo_graph import HPOGraphProvider # Needed for hierarchy

# Initialize (requires graphviz library + system package)
visualizer = GraphvizVisualizer(output_dir="visualizations")
hpo_provider = HPOGraphProvider.get_instance(data_dir="data/ontology")

# Visualize hierarchy (using HPOGraphProvider)
visualizer.visualize_hierarchy(
    phenotypes=augmented_phenotypes, 
    initial_phenotypes=initial_phenotypes, 
    title="Augmented_Hierarchy",
    hpo_provider=hpo_provider 
)

# Visualize a Graph object 
# graph = assembler.assemble(...)
visualizer.visualize_graph(
    graph=graph, 
    phenotypes=phenotypes_in_graph, # List of Phenotype objects corresponding to graph nodes
    title="Phenotype_Graph"
)
```

## Requirements

- Python package: `pip install graphviz`
- System package: Graphviz (e.g., `apt-get install graphviz`, `brew install graphviz`)

## Extending

Create custom visualizers (e.g., using Matplotlib, Plotly) by implementing the `PhenotypeVisualizer` interface from `src.core.interfaces`. 