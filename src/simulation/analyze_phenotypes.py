#!/usr/bin/env python3
"""
Analyze the distributions of phenotypes in the simulation data.

This script analyzes:
1. Distribution of the number of phenotypes per observation
2. Distribution of node depths for observed phenotypes in the HPO hierarchy
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter
from typing import List, Dict, Set, Tuple

# Add the parent directory to the sys.path to import GraPhens modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.graphens import GraPhens
from src.ontology.hpo_graph import HPOGraphProvider
from src.core.types import Phenotype

def load_phenotypes_data(file_path: str) -> Dict:
    """Load the phenotypes data from the JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_phenotype_counts(phenotypes_data: Dict) -> List[int]:
    """Calculate the number of phenotypes in each observation."""
    counts = []
    for gene, observations in phenotypes_data.items():
        for phenotype_list in observations:
            counts.append(len(phenotype_list))
    return counts

def map_phenotype_terms_to_hpo(graphens: GraPhens, phenotype_terms: Set[str]) -> Dict[str, str]:
    """
    Map phenotype terms to HPO IDs using the phenotype_lookup function.
    Returns a dictionary mapping original terms to HPO IDs.
    """
    term_to_hpo = {}
    print(f"Mapping {len(phenotype_terms)} unique phenotype terms to HPO IDs...")
    
    # Process in batches to show progress
    batch_size = 100
    terms_list = list(phenotype_terms)
    
    for i in range(0, len(terms_list), batch_size):
        batch = terms_list[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(terms_list) + batch_size - 1)//batch_size}...")
        
        for term in batch:
            clean_term = term.strip()
            if clean_term:
                try:
                    # Use the phenotype_lookup function to find the appropriate HPO ID
                    results = graphens.phenotype_lookup(clean_term)
                    if results and len(results) > 0:
                        # Take the first match as the best match
                        term_to_hpo[term] = results[0].id
                    else:
                        print(f"No HPO ID found for term: '{clean_term}'")
                        term_to_hpo[term] = None
                except Exception as e:
                    print(f"Error looking up term '{clean_term}': {str(e)}")
                    term_to_hpo[term] = None
    
    # Print some statistics
    mapped_count = sum(1 for hpo_id in term_to_hpo.values() if hpo_id is not None)
    print(f"Successfully mapped {mapped_count}/{len(phenotype_terms)} terms to HPO IDs")
    
    return term_to_hpo

def calculate_node_depths(hpo_provider: HPOGraphProvider, term_to_hpo: Dict[str, str]) -> Dict[str, int]:
    """
    Calculate the depth of each phenotype term in the HPO hierarchy.
    Depth is defined as the shortest path length from the root.
    """
    # The HPO root term
    root = "HP:0000001"  # Phenotypic abnormality root
    
    # Calculate depth for each term using networkx shortest_path_length
    depths = {}
    for term, hpo_id in term_to_hpo.items():
        if hpo_id is None:
            depths[term] = 0
            continue
            
        try:
            # Get shortest path from root to the term
            if hpo_id in hpo_provider.graph:
                try:
                    path_length = nx.shortest_path_length(hpo_provider.graph, root, hpo_id)
                    depths[term] = path_length
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    print(f"No path found from root to {hpo_id}")
                    depths[term] = 0
            else:
                print(f"Node {hpo_id} not found in HPO graph")
                depths[term] = 0
        except Exception as e:
            print(f"Error processing HPO ID {hpo_id}: {str(e)}")
            depths[term] = 0
            
    return depths

def plot_distributions(counts: List[int], depths: Dict[str, int], output_path: str = 'phenotype_distributions.png') -> None:
    """Plot the distributions of phenotype counts and node depths."""
    plt.figure(figsize=(15, 6))
    
    # Plot phenotype counts
    plt.subplot(1, 2, 1)
    plt.hist(counts, bins=range(1, max(counts) + 2), alpha=0.7)
    plt.title('Distribution of Phenotype Counts per Observation')
    plt.xlabel('Number of Phenotypes')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    # Plot node depths
    plt.subplot(1, 2, 2)
    depth_values = [d for d in depths.values() if d > 0]  # Only include non-zero depths
    if depth_values:
        plt.hist(depth_values, bins=range(min(depth_values), max(depth_values) + 2), alpha=0.7)
        plt.title('Distribution of Node Depths in HPO Hierarchy')
        plt.xlabel('Node Depth')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No valid depth data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Distributions saved to {output_path}")

def main():
    # Initialize GraPhens
    graphens = GraPhens()
    
    # Get the HPO provider
    hpo_provider = HPOGraphProvider.get_instance()
    
    # Load phenotypes data
    data_path = os.path.join(os.path.dirname(__file__), 'fenotipos.json')
    phenotypes_data = load_phenotypes_data(data_path)
    
    # Extract all unique phenotype terms
    unique_phenotypes = set()
    for gene, observations in phenotypes_data.items():
        for phenotype_list in observations:
            unique_phenotypes.update(term.strip() for term in phenotype_list)
    
    print(f"Total genes: {len(phenotypes_data)}")
    print(f"Total unique phenotypes: {len(unique_phenotypes)}")
    
    # Calculate phenotype counts per observation
    phenotype_counts = get_phenotype_counts(phenotypes_data)
    print(f"Average phenotypes per observation: {np.mean(phenotype_counts):.2f}")
    print(f"Median phenotypes per observation: {np.median(phenotype_counts):.2f}")
    
    # Map phenotype terms to HPO IDs
    term_to_hpo = map_phenotype_terms_to_hpo(graphens, unique_phenotypes)
    
    # Calculate node depths
    print("Calculating node depths...")
    node_depths = calculate_node_depths(hpo_provider, term_to_hpo)
    
    # Filter out zero depths for statistics
    non_zero_depths = [d for d in node_depths.values() if d > 0]
    if non_zero_depths:
        print(f"Average node depth (non-zero): {np.mean(non_zero_depths):.2f}")
        print(f"Median node depth (non-zero): {np.median(non_zero_depths):.2f}")
        print(f"Max node depth: {max(non_zero_depths)}")
    else:
        print("No valid node depths found")
    
    # Plot the distributions
    plot_distributions(phenotype_counts, node_depths)

if __name__ == "__main__":
    main() 