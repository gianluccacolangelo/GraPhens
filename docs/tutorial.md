# GraPhens Tutorial

This tutorial guides you through using the GraPhens system, from basic usage to advanced configurations. We follow the principle of "progressive disclosure" - starting with simple use cases and gradually introducing more sophisticated features.

## Installation

Before we begin, make sure you have installed GraPhens:

```bash
pip install graphens
```

## Basic Usage

Let's start with the simplest way to use GraPhens:

```python
from src.graphens import GraPhens

# Initialize with default settings
graphens = GraPhens()

# Create a graph from a list of phenotype IDs
phenotype_ids = ["HP:0001250", "HP:0002066", "HP:0100543"]
graph = graphens.create_graph_from_phenotypes(phenotype_ids)

# Print basic information about the graph
print(f"Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
```

This code initializes GraPhens with sensible defaults, creates a graph from a list of phenotype IDs, and prints basic information about the resulting graph.

## Working with Multiple Patients

GraPhens can efficiently process multiple patients at once:

```python
# Define patient data (patient ID -> list of phenotype IDs)
patient_data = {
    "patient_1": ["HP:0001250", "HP:0002066", "HP:0100543"],
    "patient_2": ["HP:0000407", "HP:0001263", "HP:0001290"],
    "patient_3": ["HP:0000256", "HP:0000486", "HP:0000556"]
}

# Process all patients efficiently
patient_graphs = graphens.create_graphs_from_multiple_patients(patient_data)

# Print information about each patient's graph
for patient_id, graph in patient_graphs.items():
    print(f"{patient_id}: Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
```

## Exporting Graphs for Machine Learning

Once you've created your graphs, you can export them to various formats for machine learning tasks:

```python
# Export a single graph to PyTorch Geometric
graph = graphens.create_graph_from_phenotypes(phenotype_ids)
pyg_graph = graphens.export_graph(graph, format="pytorch")

# Use the PyTorch Geometric graph in your GNN model
# pyg_graph can be used directly with PyG models
```

### Batch Processing for PyTorch Training

For efficient training, you can export multiple patient graphs as a single batched graph:

```python
# Process multiple patients
patient_data = {
    "patient_1": ["HP:0001250", "HP:0002066", "HP:0100543"],
    "patient_2": ["HP:0000407", "HP:0001263", "HP:0001290"],
    "patient_3": ["HP:0000256", "HP:0000486", "HP:0000556"]
}

# Create graphs for all patients
patient_graphs = graphens.create_graphs_from_multiple_patients(patient_data)

# Option 1: Export as individual graphs (dictionary of graphs)
individual_graphs = graphens.export_graph(patient_graphs, format="pytorch", batch=False)

# Option 2: Export as a single batched graph (more efficient for training)
batched_graph = graphens.export_graph(patient_graphs, format="pytorch", batch=True)

# Using the batched graph in a PyTorch Geometric model
import torch
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First Graph Conv layer
        x = self.conv1(x, edge_index)
        x = x.relu()
        
        # Second Graph Conv layer
        x = self.conv2(x, edge_index)
        
        return x

# Create and use the model
model = GNN(in_channels=768, hidden_channels=64, out_channels=16)
embeddings = model(batched_graph)

# Access node embeddings for each patient
patient_ids = batched_graph.patient_ids
node_mappings = batched_graph.node_mappings

# Process the results by patient
for i, patient_id in enumerate(patient_ids):
    # Get patient's node mapping to connect embeddings back to phenotypes
    mapping = node_mappings[patient_id]
    print(f"Processed embeddings for {patient_id}")
```

The batched graph contains all the necessary information to track which nodes belong to which patients, making it easy to use in PyTorch Geometric models while maintaining the ability to connect results back to individual patients.

## Configuration Management

### Using YAML Configuration

GraPhens can be configured using YAML files:

```python
# Initialize with a YAML configuration file
graphens = GraPhens(config_path="config/default.yaml")

# Alternatively, load configuration after initialization
graphens = GraPhens().load_config_from_yaml("config/custom_config.yaml")
```

### Saving Your Configuration

Once you've customized your GraPhens instance, you can save the configuration:

```python
# Save the current configuration to a YAML file
graphens.save_config_to_yaml("config/my_config.yaml")
```

## Customizing Components

### Embedding Strategy

GraPhens supports different embedding strategies:

```python
# Use memory-mapped embeddings for optimal performance
graphens.with_memmap_embeddings()

# Use pre-trained embeddings
graphens.with_pretrained_embeddings("path/to/embeddings.pkl")

# Use a custom embedding strategy with Sentence Transformers
graphens.with_sentence_transformer_embeddings("all-MiniLM-L6-v2")

# Use OpenAI embeddings
graphens.with_openai_embeddings(api_key="your-api-key")
```

### Augmentation Strategy

You can customize how phenotypes are augmented:

```python
# Include ancestors but not descendants
graphens.with_local_augmentation(include_ancestors=True, include_descendants=False)

# Use an external API for augmentation
graphens.with_api_augmentation(api_base_url="https://example.com/augment")
```

### Graph Structure

Control the structure of your generated graphs:

```python
# Include reverse edges in the graph
graphens.with_adjacency_settings(include_reverse_edges=True)

# Add global context to the graph (HPO root)
graphens.with_global_context("root")
```

## Visualization

Enable visualization for your graphs:

```python
# Enable visualization with GraphViz
graphens.with_visualization(output_dir="visualizations", format="png")

# Generate a visualization for a specific graph
phenotype_ids = ["HP:0001250", "HP:0002066", "HP:0100543"]
graph = graphens.create_graph_from_phenotypes(phenotype_ids)
graphens.visualize_graph(graph, filename="my_graph")
```

## Advanced Usage

### Custom Pipeline Components

For advanced users, GraPhens allows direct access to pipeline components:

```python
# Get the underlying orchestrator
orchestrator = graphens.get_orchestrator()

# Access the HPO graph provider
hpo_provider = graphens.get_hpo_provider()

# Get HPO terms for specific IDs
terms = hpo_provider.get_terms(["HP:0001250", "HP:0002066"])
```

### Performance Optimization

GraPhens includes several performance optimizations:

```python
# Enable caching when processing multiple patients
patient_data = {...}  # Dict mapping patient IDs to phenotype lists
patient_graphs = graphens.create_graphs_from_multiple_patients(
    patient_data,
    use_caching=True
)

# Measure performance with benchmarking
import time
start_time = time.time()
graph = graphens.create_graph_from_phenotypes(phenotype_ids)
elapsed = time.time() - start_time
print(f"Graph creation took {elapsed:.2f} seconds")
```

## Complete Example

Here's a complete example that combines several features:

```python
from src.graphens import GraPhens

# Initialize with configuration
graphens = GraPhens(config_path="config/default.yaml")

# Customize components as needed
graphens.with_memmap_embeddings()
graphens.with_local_augmentation(include_ancestors=True)
graphens.with_visualization(output_dir="output/visualizations")

# Process multiple patients
patient_data = {
    "patient_1": ["HP:0001250", "HP:0002066", "HP:0100543"],
    "patient_2": ["HP:0000407", "HP:0001263", "HP:0001290"],
    "patient_3": ["HP:0000256", "HP:0000486", "HP:0000556"]
}

# Create graphs efficiently
patient_graphs = graphens.create_graphs_from_multiple_patients(patient_data)

# Generate visualizations
for patient_id, graph in patient_graphs.items():
    graphens.visualize_graph(graph, filename=f"graph_{patient_id}")
    print(f"{patient_id}: Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

# Save the configuration for future use
graphens.save_config_to_yaml("config/my_project_config.yaml")
```

## Conclusion

This tutorial covered the basics of using GraPhens, from simple initialization to advanced customization. The system is designed to be intuitive while offering flexibility for complex use cases.

For more information, check out:
- The [API Reference](api_reference.md) for detailed documentation
- The [Configuration Guide](configuration.md) for all available settings
- The [Examples](../examples/) directory for complete working examples 