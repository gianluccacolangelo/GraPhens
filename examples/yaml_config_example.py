#!/usr/bin/env python3
"""
Example demonstrating how to use GraPhens with YAML configuration.

This example shows how to:
1. Create a GraPhens instance with YAML configuration
2. Save a modified configuration
3. Process patients using the configured system
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import GraPhens
from src.graphens import GraPhens

# Demo phenotype IDs for testing
PATIENT_DATA = {
    "patient_1": ["HP:0001250", "HP:0002066", "HP:0100543"],  # Seizures, Gait ataxia, Cognitive impairment
    "patient_2": ["HP:0000407", "HP:0001263", "HP:0001290"],  # Sensorineural hearing loss, Developmental delay, Hypotonia
    "patient_3": ["HP:0000256", "HP:0000486", "HP:0000556"]   # Macrocephaly, Strabismus, Abnormal aortic valve
}

def main():
    """Main function demonstrating YAML configuration usage."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("yaml_config_example")
    
    # Ensure the config directory exists
    os.makedirs("config", exist_ok=True)
    
    # Create a default configuration file if it doesn't exist
    if not os.path.exists("config/default.yaml"):
        logger.info("Creating default configuration file")
        # Create a temporary GraPhens instance to generate the default config
        temp_graphens = GraPhens()
        temp_graphens.save_config_to_yaml("config/default.yaml")
    
    # Create GraPhens instance with default configuration
    logger.info("Initializing GraPhens with default configuration")
    graphens = GraPhens(config_path="config/default.yaml")
    
    # Modify the configuration
    logger.info("Customizing the configuration")
    graphens.with_memmap_embeddings()
    graphens.with_augmentation(include_ancestors=True, include_descendants=False)
    graphens.with_adjacency_settings(include_reverse_edges=True)
    
    # Save the modified configuration
    custom_config_path = "config/custom_config.yaml"
    logger.info(f"Saving modified configuration to {custom_config_path}")
    graphens.save_config_to_yaml(custom_config_path)
    
    # Create a new GraPhens instance with the custom configuration
    logger.info("Creating new GraPhens instance with custom configuration")
    custom_graphens = GraPhens(config_path=custom_config_path)
    
    # Process patients
    logger.info(f"Processing {len(PATIENT_DATA)} patients")
    start_time = time.time()
    patient_graphs = custom_graphens.create_graphs_from_multiple_patients(PATIENT_DATA)
    total_time = time.time() - start_time
    
    # Print results
    for patient_id, graph in patient_graphs.items():
        logger.info(f"{patient_id}: Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    
    # Compare with non-batch processing
    logger.info("Comparing with individual processing")
    start_time = time.time()
    for patient_id, phenotypes in PATIENT_DATA.items():
        graph = custom_graphens.create_graph_from_phenotypes(phenotypes)
        logger.info(f"{patient_id} (individual): Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    individual_time = time.time() - start_time
    
    logger.info(f"Individual processing time: {individual_time:.2f} seconds")
    logger.info(f"Speedup: {individual_time / total_time:.2f}x")

if __name__ == "__main__":
    main() 