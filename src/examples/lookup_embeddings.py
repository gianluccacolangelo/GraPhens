#!/usr/bin/env python
"""
GraPhens Example: Using Pre-computed Embeddings

This example demonstrates how to use pre-computed embeddings with GraPhens.
Pre-computed embeddings are useful when:

1. You want to use specific embeddings trained on biomedical data
2. You want to ensure reproducibility across experiments
3. You want to avoid recomputing embeddings for the same phenotypes
4. You want faster graph creation by skipping the embedding computation

The example uses the HPO pre-computed embeddings stored in the data directory.
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
    print("GraPhens Example: Using Pre-computed Embeddings")
    print("=" * 80)
    
    # Define some seizure-related phenotypes
    phenotype_ids = [
        "HP:0001250",  # Seizure
        "HP:0001251",  # Atonic seizure
        "HP:0002121"   # Absence seizure
    ]
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Path to pre-computed embeddings
    embeddings_path = Path("/home/brainy/GraPhens/data/embeddings/dummy_embeddings.pkl")
    
    if not embeddings_path.exists():
        print(f"Error: Embeddings file not found at {embeddings_path}")
        print("Please update the path to point to a valid embeddings file.")
        return
    
    print("\n1. Using Pre-computed Embeddings with the Convenience Method")
    print("-" * 60)
    
    # Initialize GraPhens with pre-computed embeddings using the convenience method
    print(f"Loading embeddings from {embeddings_path}")
    start_time = time.time()
    graphens = GraPhens().with_lookup_embeddings(str(embeddings_path))
    
    # Create a graph using the pre-computed embeddings
    print("Creating graph with pre-computed embeddings...")
    graph = graphens.create_graph_from_phenotypes(phenotype_ids, show_progress=True)
    elapsed = time.time() - start_time
    
    print(f"Graph created in {elapsed:.2f}s with {len(graph.node_mapping)} nodes and {graph.edge_index.shape[1]} edges")
    
    print("\n2. Manual Approach (For Reference)")
    print("-" * 60)
    
    print("For reference, here's how to load embeddings manually:")
    print("```python")
    print("import pickle")
    print("with open(embeddings_path, 'rb') as f:")
    print("    embedding_dict = pickle.load(f)")
    print("graphens = GraPhens().with_embedding_model('lookup', embedding_dict=embedding_dict)")
    print("```")
    
    print("\n3. Visualizing the Graph")
    print("-" * 60)
    
    # Enable visualization
    graphens = graphens.with_visualization(enabled=True, output_dir=str(output_dir))
    
    # Visualize the graph
    print("Generating visualization...")
    visualization_path = graphens.visualize(
        graph=graph,
        phenotypes=[Phenotype(id=pid, name=f"Phenotype {pid}") for pid in graph.node_mapping.keys()],
        title="Seizure Phenotype Graph (With Pre-computed Embeddings)"
    )
    
    if visualization_path:
        print(f"Visualization created at {visualization_path}")
    
    print("\nDone! 🎉")
    print("=" * 80)

if __name__ == "__main__":
    main() 