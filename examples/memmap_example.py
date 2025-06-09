#!/usr/bin/env python3
"""
Example demonstrating memory-mapped embeddings for improved performance.

This script shows how to:
1. Convert pickle-based embeddings to memory-mapped format
2. Use memory-mapped embeddings with GraPhens
3. Compare performance between regular and memory-mapped embeddings
"""

import sys
import os
import time
import pickle
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import GraPhens
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graphens import GraPhens
from src.embedding.vector_db.memmap import (
    convert_embedding_to_memmap,
    list_memmap_files
)


def create_memmap_if_needed(embeddings_dir: str = "data/embeddings"):
    """
    Check if memory-mapped embedding files exist, and create them if not.
    
    Args:
        embeddings_dir: Directory containing embedding files
    """
    # Check if memory-mapped files already exist
    memmap_files = list_memmap_files(embeddings_dir)
    if memmap_files:
        print(f"Found {len(memmap_files)} existing memory-mapped embedding files:")
        for i, (data_path, index_path) in enumerate(memmap_files):
            print(f"  {i+1}. {os.path.basename(data_path)}")
        return
    
    # Find pickle files to convert
    pickle_files = []
    for file in os.listdir(embeddings_dir):
        if file.endswith(".pkl") and not file.endswith("_metadata.pkl"):
            pickle_files.append(os.path.join(embeddings_dir, file))
    
    if not pickle_files:
        print(f"No embedding files found in {embeddings_dir}")
        return
    
    # Ask user which file to convert
    print("\nNo memory-mapped embedding files found. Available pickle files:")
    for i, file in enumerate(pickle_files):
        print(f"  {i+1}. {os.path.basename(file)}")
    
    if len(pickle_files) == 1:
        # Only one file, convert it automatically
        file_to_convert = pickle_files[0]
    else:
        # Ask which file to convert
        selection = input("\nEnter the number of the file to convert (or 'all' to convert all): ")
        if selection.lower() == 'all':
            # Convert all files
            for file in pickle_files:
                try:
                    convert_embedding_to_memmap(file)
                except Exception as e:
                    print(f"Error converting {file}: {e}")
            return
        else:
            try:
                idx = int(selection) - 1
                if 0 <= idx < len(pickle_files):
                    file_to_convert = pickle_files[idx]
                else:
                    print("Invalid selection, converting the first file.")
                    file_to_convert = pickle_files[0]
            except ValueError:
                print("Invalid input, converting the first file.")
                file_to_convert = pickle_files[0]
    
    # Convert the selected file
    try:
        convert_embedding_to_memmap(file_to_convert)
    except Exception as e:
        print(f"Error converting {file_to_convert}: {e}")


def compare_performance():
    """Compare performance between regular and memory-mapped embeddings."""
    print("\nComparing performance: Regular vs. Memory-mapped Embeddings")
    print("=" * 60)
    
    # Function to measure initialization time
    def measure_init_time(use_memmap: bool):
        start_time = time.time()
        if use_memmap:
            graphens = GraPhens(use_default_embeddings=False)
            graphens.with_memmap_embeddings()
        else:
            graphens = GraPhens(use_default_embeddings=False)
            graphens.with_pretrained_embeddings()
        end_time = time.time()
        return end_time - start_time, graphens
    
    # Measure initialization time for both methods
    print("Measuring initialization time...")
    
    pickle_time, pickle_graphens = measure_init_time(use_memmap=False)
    print(f"Regular pickle-based embeddings: {pickle_time:.3f} seconds")
    
    memmap_time, memmap_graphens = measure_init_time(use_memmap=True)
    print(f"Memory-mapped embeddings: {memmap_time:.3f} seconds")
    
    speedup = pickle_time / memmap_time if memmap_time > 0 else float('inf')
    print(f"Speedup: {speedup:.1f}x faster initialization")
    
    # Measure memory usage
    import psutil
    process = psutil.Process(os.getpid())
    
    # Measure graph creation performance
    print("\nMeasuring graph creation performance...")
    
    # Example phenotype IDs (more complex case with 15 phenotypes)
    phenotype_ids = [
        "HP:0001250",  # Seizure
        "HP:0001251",  # Atonic seizure
        "HP:0001257",  # Spasms
        "HP:0002197",  # Generalized myoclonic seizures
        "HP:0007359",  # Focal motor seizure
        "HP:0002123",  # Generalized tonic-clonic seizures
        "HP:0011097",  # Epileptic encephalopathy
        "HP:0002133",  # Status epilepticus
        "HP:0002383",  # Atypical absence seizure
        "HP:0010818",  # Partial seizure
        "HP:0002121",  # Absence seizure
        "HP:0012469",  # Infantile spasms
        "HP:0001335",  # Photosensitive seizure
        "HP:0002384",  # Focal clonic seizure
        "HP:0002373"   # Febrile seizure
    ]
    
    # Measure graph creation with pickle embeddings
    start_time = time.time()
    pickle_graph = pickle_graphens.create_graph_from_phenotypes(phenotype_ids)
    pickle_graph_time = time.time() - start_time
    
    # Measure memory after pickle graph creation
    pickle_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Measure graph creation with memmap embeddings
    start_time = time.time()
    memmap_graph = memmap_graphens.create_graph_from_phenotypes(phenotype_ids)
    memmap_graph_time = time.time() - start_time
    
    # Measure memory after memmap graph creation
    memmap_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Print results
    print(f"Graph creation with pickle embeddings: {pickle_graph_time:.3f} seconds")
    print(f"Graph creation with memmap embeddings: {memmap_graph_time:.3f} seconds")
    
    graph_speedup = pickle_graph_time / memmap_graph_time if memmap_graph_time > 0 else float('inf')
    print(f"Speedup: {graph_speedup:.1f}x faster graph creation")
    
    print(f"\nMemory usage with pickle embeddings: {pickle_memory:.1f} MB")
    print(f"Memory usage with memmap embeddings: {memmap_memory:.1f} MB")
    
    memory_reduction = (pickle_memory - memmap_memory) / pickle_memory * 100
    print(f"Memory reduction: {memory_reduction:.1f}%")
    
    # Print graph details
    print(f"\nGraph details:")
    print(f"  Nodes: {len(pickle_graph.node_mapping)}")
    print(f"  Edges: {pickle_graph.edge_index.shape[1]}")
    print(f"  Feature dimension: {pickle_graph.node_features.shape[1]}")


def main():
    """Main function to demonstrate memory-mapped embeddings."""
    print("Memory-mapped Embeddings Example")
    print("=" * 60)
    
    # Create memory-mapped embedding files if needed
    create_memmap_if_needed()
    
    # Compare performance
    compare_performance()
    
    print("\nExample usage in code:")
    print("""
    # Using memory-mapped embeddings with GraPhens
    graphens = GraPhens(use_default_embeddings=False)
    graphens.with_memmap_embeddings()
    
    # Creating a graph
    graph = graphens.create_graph_from_phenotypes(phenotype_ids)
    """)


if __name__ == "__main__":
    main() 