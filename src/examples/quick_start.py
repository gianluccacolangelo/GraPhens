#!/usr/bin/env python
"""
GraPhens Quick Start Example

This example demonstrates how to use the GraPhens facade to:
1. Create a phenotype graph with default settings
2. Customize the graph creation process
3. Visualize the results
4. Export to different formats

The example focuses on phenotypes related to "seizure" conditions.
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.graphens import GraPhens
from src.core.types import Phenotype

def main():
    print("=" * 80)
    print("GraPhens Quick Start Example")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    print("\n1. Basic Usage - Creating a graph with default settings")
    print("-" * 60)
    
    # Initialize GraPhens with default settings
    graphens = GraPhens()
    
    # Define some seizure-related phenotypes
    phenotype_ids = [
        "HP:0001250",  # Seizure
        "HP:0001251",  # Atonic seizure
        "HP:0002121"   # Absence seizure
    ]
    
    # Create a graph with progress indicators
    print("Creating graph from seizure phenotypes...")
    start_time = time.time()
    graph = graphens.create_graph_from_phenotypes(phenotype_ids, show_progress=True)
    elapsed = time.time() - start_time
    
    print(f"Graph created in {elapsed:.2f}s with {len(graph.node_mapping)} nodes and {graph.edge_index.shape[1]} edges")
    
    print("\n2. Customization - Creating a graph with custom settings")
    print("-" * 60)
    
    # Create a GraPhens instance with custom settings using the fluent interface
    custom_graphens = (GraPhens()
                      .with_embedding_model("tfidf", max_features=256)
                      .with_augmentation(include_ancestors=True, include_descendants=True)
                      .with_visualization(enabled=True, output_dir=str(output_dir)))
    
    # Save the configuration for future use
    config_path = output_dir / "custom_config.json"
    custom_graphens.save_config(str(config_path))
    print(f"Custom configuration saved to {config_path}")
    
    # Create a graph with the custom settings
    print("Creating graph with custom settings...")
    custom_graph = custom_graphens.create_graph_from_phenotypes(phenotype_ids, show_progress=True)
    
    print("\n3. Discovery - Finding relevant phenotypes")
    print("-" * 60)
    
    # Search for phenotypes related to "seizure"
    print("Searching for phenotypes related to 'seizure'...")
    seizure_phenotypes = graphens.phenotype_lookup("seizure")
    
    print(f"Found {len(seizure_phenotypes)} phenotypes related to 'seizure':")
    for i, phenotype in enumerate(seizure_phenotypes[:5]):  # Show only the first 5
        print(f"  {i+1}. {phenotype.id} - {phenotype.name}")
    
    if len(seizure_phenotypes) > 5:
        print(f"  ... and {len(seizure_phenotypes) - 5} more")
    
    print("\n4. Visualization - Creating visual representations")
    print("-" * 60)
    
    # Visualize the graph (if visualization is enabled)
    print("Generating visualization...")
    visualization_path = custom_graphens.visualize(
        graph=custom_graph,
        phenotypes=[Phenotype(id=pid, name=f"Phenotype {pid}") for pid in custom_graph.node_mapping.keys()],
        title="Seizure Phenotype Graph"
    )
    
    if visualization_path:
        print(f"Visualization created at {visualization_path}")
    else:
        print("Visualization is not enabled or could not be created")
    
    print("\n5. Export - Saving the graph for machine learning")
    print("-" * 60)
    
    # Export the graph to JSON format
    json_path = output_dir / "seizure_graph.json"
    print(f"Exporting graph to JSON format at {json_path}...")
    custom_graphens.export_graph(custom_graph, format="json", output_path=str(json_path))
    
    # Show how to export to PyTorch (without actually executing the import)
    print("\nTo use the graph with PyTorch Geometric:")
    print("```python")
    print("import torch_geometric as pyg")
    print("pyg_graph = graphens.export_graph(graph, format='pytorch')")
    print("# Use with GNN models")
    print("```")
    
    print("\n6. Advanced Usage - Loading configuration from file")
    print("-" * 60)
    
    # Create a new instance from the saved configuration
    print(f"Loading configuration from {config_path}...")
    loaded_graphens = GraPhens().with_config_from_file(str(config_path))
    
    # Create a graph with the loaded configuration
    print("Creating graph with loaded configuration...")
    loaded_graph = loaded_graphens.create_graph_from_phenotypes(phenotype_ids[0:1], show_progress=True)
    
    print("\nDone! 🎉")
    print("=" * 80)

if __name__ == "__main__":
    main() 