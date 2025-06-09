#!/usr/bin/env python3
"""
Simple example demonstrating the new simplified GraPhens interface.
"""

import sys
import os
import time
import random
import psutil
import numpy as np
from typing import List, Dict
import logging

# Add the parent directory to the path to import GraPhens
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graphens import GraPhens

# Sample patient phenotype data - each with 10-15 phenotypes
PATIENT_PHENOTYPES = [
    # Patient 1: Seizure disorder with developmental delay
    ["HP:0001250", "HP:0001263", "HP:0000750", "HP:0002376", "HP:0002197", 
     "HP:0000252", "HP:0000709", "HP:0001249", "HP:0001268", "HP:0002069",
     "HP:0002121", "HP:0002373", "HP:0002133", "HP:0012469", "HP:0001335"],
    # Patient 2: Neurodevelopmental disorder
    # Patient 3: Muscular disorder
    ["HP:0003198", "HP:0001371", "HP:0003391", "HP:0003694", "HP:0001324", 
     "HP:0006785", "HP:0003701", "HP:0003690", "HP:0002905", "HP:0003387",
     "HP:0003547", "HP:0009046", "HP:0003325", "HP:0003323", "HP:0001270"],
    # Patient 4: Cardiac condition
    ["HP:0001631", "HP:0001657", "HP:0001680", "HP:0001712", "HP:0001633", 
     "HP:0004749", "HP:0011675", "HP:0001711", "HP:0001629", "HP:0001654",
     "HP:0031594", "HP:0001083", "HP:0001644", "HP:0001630", "HP:0004308"],
    # Patient 5: Immune deficiency
    ["HP:0002721", "HP:0002850", "HP:0001880", "HP:0001903", "HP:0001875", 
     "HP:0010976", "HP:0001744", "HP:0010701", "HP:0012145", 
     "HP:0002204", "HP:0002738", "HP:0002788", "HP:0001508", "HP:0002716"],
    # Patient 6: Metabolic disorder
    ["HP:0001942", "HP:0001943", "HP:0008151", "HP:0000789", "HP:0003155", 
     "HP:0001946", "HP:0001944", "HP:0006775", "HP:0006971", "HP:0031283",
     "HP:0030233", "HP:0008263", "HP:0003737", "HP:0001824", "HP:0006979"],
    # Patient 7: Skeletal dysplasia
    ["HP:0002650", "HP:0002814", "HP:0000924", "HP:0001510", "HP:0002104", 
     "HP:0010885", "HP:0003521", "HP:0000772", "HP:0003312", "HP:0011304",
     "HP:0001374", "HP:0002870", "HP:0003549", "HP:0002655", "HP:0000939"],
    # Patient 8: Eye disorder
    ["HP:0000478", "HP:0000496", "HP:0000505", "HP:0000518", "HP:0000587", 
     "HP:0000495", "HP:0007957", "HP:0000589", "HP:0000639", "HP:0000593",
     "HP:0000662", "HP:0000575", "HP:0000519", "HP:0000553", "HP:0001386"],
    # Patient 9: Liver disease
    ["HP:0001397", "HP:0001410", "HP:0001399", "HP:0002240", "HP:0001392", 
     "HP:0002910", "HP:0001406", "HP:0001409", "HP:0000958", "HP:0001394",
     "HP:0001396", "HP:0001401", "HP:0002244", "HP:0002014", "HP:0001395"],
    # Patient 10: Renal disorder
    ["HP:0000077", "HP:0000093", "HP:0000113", "HP:0000126", "HP:0012622",
     "HP:0000119", "HP:0000104", "HP:0012211", "HP:0000121", "HP:0000830",
     "HP:0004730", "HP:0000122", "HP:0012210", "HP:0000148", "HP:0004936"],
    # Additional batch of 5 more patients with more phenotypes
    # Patient 11: Mixed neurological 
    ["HP:0001331", "HP:0001315", "HP:0100021", "HP:0002120", "HP:0001256", 
     "HP:0007319", "HP:0001332", "HP:0001336", "HP:0002072", "HP:0002524",
     "HP:0007360", "HP:0012469", "HP:0005484", "HP:0002373", "HP:0007333"],
    # Patient 12: Complex syndrome
    ["HP:0000486", "HP:0000678", "HP:0001249", "HP:0001263", "HP:0000252", 
     "HP:0000337", "HP:0000347", "HP:0000494", "HP:0007598", "HP:0001290",
     "HP:0001999", "HP:0001631", "HP:0000478", "HP:0000194", "HP:0000175"],
    # Patient 13: Congenital disorder
    ["HP:0001098", "HP:0001100", "HP:0001518", "HP:0001537", "HP:0002088", 
     "HP:0004322", "HP:0004691", "HP:0001561", "HP:0100775", "HP:0000006",
     "HP:0001363", "HP:0000952", "HP:0000238", "HP:0001773", "HP:0001385"],
    # Patient 14: Genetic syndrome
    ["HP:0001627", "HP:0000407", "HP:0000403", "HP:0000175", "HP:0000369", 
     "HP:0000160", "HP:0000286", "HP:0000297", "HP:0000460", "HP:0000470",
     "HP:0000494", "HP:0001249", "HP:0001250", "HP:0001342", "HP:0001426"],
    # Patient 15: Endocrine disorder
    ["HP:0000819", "HP:0001513", "HP:0000822", "HP:0000831", "HP:0000873", 
     "HP:0000829", "HP:0008163", "HP:0000800", "HP:0000939", "HP:0000821",
     "HP:0000857", "HP:0003621", "HP:0001903", "HP:0000851", "HP:0000843"]
]

def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark_patients_batch(use_memmap: bool = True, run_large_test: bool = True) -> Dict:
    """
    Benchmark processing a batch of patients with multiple phenotypes.
    
    Args:
        use_memmap: Whether to use memory-mapped embeddings
        run_large_test: Whether to run the large test with all patients
        
    Returns:
        Dictionary with benchmark results
    """
    # Create GraPhens instance
    graphens = GraPhens(use_default_embeddings=False)
    
    if use_memmap:
        print(f"Using memory-mapped embeddings")
        graphens.with_memmap_embeddings()
    else:
        print(f"Using traditional pickle-based embeddings")
        # Get latest pickle file in data/embeddings directory
        embedding_files = [f for f in os.listdir("data/embeddings") if f.endswith(".pkl")]
        if embedding_files:
            latest_file = max(embedding_files, key=lambda x: os.path.getmtime(os.path.join("data/embeddings", x)))
            graphens.with_lookup_embeddings(os.path.join("data/embeddings", latest_file))
        else:
            print("No pickle embedding files found. Using pre-trained embeddings instead.")
            graphens.with_pretrained_embeddings()
    
    # Measure memory after initialization
    init_memory = get_memory_usage_mb()
    
    # Process each patient
    results = {
        "init_memory_mb": init_memory,
        "patient_times": [],
        "total_time": 0,
        "max_memory_mb": init_memory,
        "patient_count": 0,
        "total_nodes": 0,
        "total_edges": 0,
        "avg_nodes_per_patient": 0,
        "avg_edges_per_patient": 0
    }
    
    # Select patients based on test size
    patients_to_process = PATIENT_PHENOTYPES if run_large_test else PATIENT_PHENOTYPES[:5]
    
    total_start_time = time.time()
    total_nodes = 0
    total_edges = 0
    
    for i, phenotype_ids in enumerate(patients_to_process):
        # Measure time for individual patient
        start_time = time.time()
        graph = graphens.create_graph_from_phenotypes(phenotype_ids, show_progress=False)
        end_time = time.time()
        
        # Store results
        processing_time = end_time - start_time
        results["patient_times"].append(processing_time)
        
        # Track graph statistics
        total_nodes += len(graph.node_mapping)
        total_edges += graph.edge_index.shape[1]
        
        # Check memory
        current_memory = get_memory_usage_mb()
        results["max_memory_mb"] = max(results["max_memory_mb"], current_memory)
        
        print(f"Patient {i+1}: {len(phenotype_ids)} phenotypes - Graph created with {len(graph.node_mapping)} nodes, {graph.edge_index.shape[1]} edges in {processing_time:.3f}s")
    
    results["total_time"] = time.time() - total_start_time
    results["avg_time"] = sum(results["patient_times"]) / len(results["patient_times"])
    results["patient_count"] = len(patients_to_process)
    results["total_nodes"] = total_nodes
    results["total_edges"] = total_edges
    results["avg_nodes_per_patient"] = total_nodes / len(patients_to_process)
    results["avg_edges_per_patient"] = total_edges / len(patients_to_process)
    
    return results

def benchmark_multiple_patients(graphens, patient_phenotypes, use_batch_method=True):
    """
    Benchmark performance when processing multiple patients.
    
    Args:
        graphens: Initialized GraPhens instance
        patient_phenotypes: Dictionary of patient IDs to phenotype lists
        use_batch_method: Whether to use the new efficient batch method
        
    Returns:
        Processing time in seconds
    """
    start_time = time.time()
    
    if use_batch_method:
        # Use the new efficient method that processes all patients at once
        graphs = graphens.create_graphs_from_multiple_patients(patient_phenotypes)
    else:
        # Use the old approach of processing each patient separately
        graphs = {}
        for patient_id, phenotypes in patient_phenotypes.items():
            graphs[patient_id] = graphens.create_graph_from_phenotypes(phenotypes)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return processing_time, len(graphs)

def compare_batch_performance():
    print("\n=== COMPARING BATCH PROCESSING PERFORMANCE ===")
    
    # Get patient phenotypes
    patient_phenotypes = get_patient_data()
    
    # Initialize GraPhens
    graphens = GraPhens()
    
    # Benchmark with individual processing approach
    print("Processing patients individually...")
    individual_time, individual_count = benchmark_multiple_patients(
        graphens, patient_phenotypes, use_batch_method=False
    )
    
    # Benchmark with batch processing approach
    print("Processing patients with batch method...")
    batch_time, batch_count = benchmark_multiple_patients(
        graphens, patient_phenotypes, use_batch_method=True
    )
    
    # Calculate speedup
    speedup = individual_time / batch_time if batch_time > 0 else float('inf')
    
    print(f"\nResults for processing {len(patient_phenotypes)} patients:")
    print(f"Individual processing time: {individual_time:.2f} seconds")
    print(f"Batch processing time: {batch_time:.2f} seconds")
    print(f"Speedup factor: {speedup:.2f}x")
    print(f"Successfully processed: {batch_count} patients")
    
    return speedup

def get_patient_data():
    """
    Convert the list of patient phenotypes to a dictionary for batch processing.
    
    Returns:
        Dictionary mapping patient IDs to lists of phenotype IDs
    """
    patient_data = {}
    for i, phenotypes in enumerate(PATIENT_PHENOTYPES):
        patient_data[f"patient_{i+1}"] = phenotypes
    return patient_data

def main():
    """Main example function."""
    
    print("Example 1: Using GraPhens with memory-mapped embeddings")
    print("-----------------------------------------------")
    
    # Create GraPhens instance with memory-mapped embeddings
    graphens = GraPhens()  # Will automatically use memory-mapped embeddings if available
    
    # Sample phenotype IDs (seizure-related)
    phenotype_ids = ["HP:0001250", "HP:0001251"]
    
    # Measure time to create graph
    start_time = time.time()
    # Create a graph
    graph = graphens.create_graph_from_phenotypes(phenotype_ids, show_progress=False)
    end_time = time.time()
    print(f"Time taken to create graph: {end_time - start_time} seconds")
    
    # Print some information about the graph
    print(f"Graph created with {len(graph.node_mapping)} nodes and {graph.edge_index.shape[1]} edges")
    print()
    
    print("Example 2: Explicitly using memory-mapped embeddings")
    print("------------------------------------------------------")
    
    # Create GraPhens with explicit memory-mapped embeddings
    graphens = GraPhens(use_default_embeddings=False)
    graphens.with_memmap_embeddings()  # Will use the latest memory-mapped file
    
    # Create a graph with the memory-mapped embeddings
    start_time = time.time()
    graph = graphens.create_graph_from_phenotypes(phenotype_ids, show_progress=True)
    end_time = time.time()
    print(f"Time taken to create graph: {end_time - start_time} seconds")
    
    # Print information about the graph
    print(f"Graph created with {len(graph.node_mapping)} nodes and {graph.edge_index.shape[1]} edges")
    print()
    
    print("Example 3: Using specific memory-mapped model")
    print("--------------------------------")
    
    # Use a specific model's memory-mapped embeddings
    graphens = GraPhens(use_default_embeddings=False)
    graphens.with_memmap_embeddings("gsarti/biobert-nli")
    
    # Create a graph with biomedical embeddings
    graph = graphens.create_graph_from_phenotypes(phenotype_ids)
    print(f"Graph created with {len(graph.node_mapping)} nodes and {graph.edge_index.shape[1]} edges")
    
    # Search for phenotypes related to 'seizure'
    seizure_phenotypes = graphens.phenotype_lookup("seizure")
    
    # Print the first 5 matching phenotypes
    print(f"\nFound {len(seizure_phenotypes)} phenotypes matching 'seizure'")
    print("First 5 matches:")
    for i, phenotype in enumerate(seizure_phenotypes[:5]):
        print(f"  {i+1}. {phenotype.id}: {phenotype.name}")
    
    # Add batch processing benchmark
    print("\nExample 4: Batch Processing Performance Benchmark")
    print("------------------------------------------------")
    print("Testing processing performance with multiple patients (15 patients with 15-20 phenotypes each)\n")
    
    # Test with memmap embeddings
    print("\n=== Testing with memory-mapped embeddings ===")
    memmap_results = benchmark_patients_batch(use_memmap=True, run_large_test=True)
    
    # Test with regular embeddings
    print("\n=== Testing with traditional pickle-based embeddings ===")
    pickle_results = benchmark_patients_batch(use_memmap=False, run_large_test=True)
    
    # Show comparison
    print("\n=== Performance Comparison ===")
    print(f"Memory-mapped embeddings ({memmap_results['patient_count']} patients):")
    print(f"  - Initial memory usage: {memmap_results['init_memory_mb']:.1f} MB")
    print(f"  - Maximum memory usage: {memmap_results['max_memory_mb']:.1f} MB")
    print(f"  - Total processing time: {memmap_results['total_time']:.3f} seconds")
    print(f"  - Average time per patient: {memmap_results['avg_time']:.3f} seconds")
    print(f"  - Total nodes created: {memmap_results['total_nodes']}")
    print(f"  - Total edges created: {memmap_results['total_edges']}")
    print(f"  - Average nodes per patient: {memmap_results['avg_nodes_per_patient']:.1f}")
    print(f"  - Average edges per patient: {memmap_results['avg_edges_per_patient']:.1f}")
    
    print(f"\nPickle-based embeddings ({pickle_results['patient_count']} patients):")
    print(f"  - Initial memory usage: {pickle_results['init_memory_mb']:.1f} MB")
    print(f"  - Maximum memory usage: {pickle_results['max_memory_mb']:.1f} MB")
    print(f"  - Total processing time: {pickle_results['total_time']:.3f} seconds")
    print(f"  - Average time per patient: {pickle_results['avg_time']:.3f} seconds")
    print(f"  - Total nodes created: {pickle_results['total_nodes']}")
    print(f"  - Total edges created: {pickle_results['total_edges']}")
    print(f"  - Average nodes per patient: {pickle_results['avg_nodes_per_patient']:.1f}")
    print(f"  - Average edges per patient: {pickle_results['avg_edges_per_patient']:.1f}")
    
    # Calculate improvements
    memory_reduction = (pickle_results['max_memory_mb'] - memmap_results['max_memory_mb']) / pickle_results['max_memory_mb'] * 100
    time_improvement = (pickle_results['total_time'] - memmap_results['total_time']) / pickle_results['total_time'] * 100
    init_memory_reduction = (pickle_results['init_memory_mb'] - memmap_results['init_memory_mb']) / pickle_results['init_memory_mb'] * 100
    
    print(f"\nImprovements with memory-mapped embeddings:")
    print(f"  - Initial memory usage reduction: {init_memory_reduction:.1f}%")
    print(f"  - Maximum memory usage reduction: {memory_reduction:.1f}%")
    print(f"  - Processing time improvement: {time_improvement:.1f}%")
    
    # Compare batch processing performance
    compare_batch_performance()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run main example
    main()
    
    # Compare batch processing performance
    compare_batch_performance() 