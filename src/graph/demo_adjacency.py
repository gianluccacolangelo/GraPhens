#!/usr/bin/env python3

import numpy as np
import graphviz
from src.core.types import Phenotype
from src.graph.adjacency import HPOAdjacencyListBuilder
from src.ontology.hpo_graph import HPOGraphProvider
from src.augmentation.hpo_augmentation import HPOAugmentationService

def main():
    """
    Demonstrate the HPOAdjacencyListBuilder with real HPO data.
    """
    # Initialize HPO graph provider
    hpo_provider = HPOGraphProvider(data_dir="data/ontology")
    
    # Create initial phenotypes from IDs
    initial_phenotypes = [
        Phenotype(id="HP:0002353", name=""),  # Will be populated by augmentation
        Phenotype(id="HP:0001639", name=""),
        Phenotype(id="HP:0003128", name=""),
        Phenotype(id="HP:0002093", name="")      # Will be populated by augmentation
    ]

    initial_phenotypes = [
        Phenotype(id="HP:0001251", name=""),
        Phenotype(id="HP:0001272", name=""),
        Phenotype(id="HP:0001260", name=""),
        Phenotype(id="HP:0001310", name="")
    ]

    
    # First, populate the metadata for initial phenotypes
    for p in initial_phenotypes:
        metadata = hpo_provider.get_metadata(p.id)
        p.name = metadata.get("name", "")
        p.description = metadata.get("definition", "")
    
    print("Initial phenotypes:")
    for p in initial_phenotypes:
        print(f"  - {p.id}: {p.name}")
        if p.description:
            print(f"    Description: {p.description}")
    
    # Initialize and use the HPOAugmentationService
    augmentation_service = HPOAugmentationService(data_dir="data/ontology", include_ancestors=True,include_descendants=False)
    phenotypes = augmentation_service.augment(initial_phenotypes)
    
    print("\nAugmented phenotypes:")
    for p in phenotypes:
        print(f"  - {p.id}: {p.name}")
        if p.description:
            print(f"    Description: {p.description}")
    
    # Build adjacency list (with and without reverse edges)
    builder_with_reverse = HPOAdjacencyListBuilder(hpo_provider, include_reverse_edges=True)
    builder_without_reverse = HPOAdjacencyListBuilder(hpo_provider, include_reverse_edges=False)
    
    adj_list_with_reverse = builder_with_reverse.build(phenotypes)
    adj_list_without_reverse = builder_without_reverse.build(phenotypes)
    
    # Print the resulting edge lists
    print("\nAdjacency List (with reverse edges):")
    print(f"  Shape: {adj_list_with_reverse.shape}")
    for i in range(adj_list_with_reverse.shape[1]):
        src = adj_list_with_reverse[0, i]
        tgt = adj_list_with_reverse[1, i]
        print(f"  {src} ({phenotypes[src].id}: {phenotypes[src].name}) -> "
              f"{tgt} ({phenotypes[tgt].id}: {phenotypes[tgt].name})")
    
    print("\nAdjacency List (without reverse edges):")
    print(f"  Shape: {adj_list_without_reverse.shape}")
    for i in range(adj_list_without_reverse.shape[1]):
        src = adj_list_without_reverse[0, i]
        tgt = adj_list_without_reverse[1, i]
        print(f"  {src} ({phenotypes[src].id}: {phenotypes[src].name}) -> "
              f"{tgt} ({phenotypes[tgt].id}: {phenotypes[tgt].name})")
    
    # Visualize the graph using graphviz
    try:
        # Create a new directed graph
        dot = graphviz.Digraph('HPO_Hierarchy', filename='hpo_phenotype_graph')
        dot.attr(rankdir='BT')  # Bottom to Top direction
        
        # Add nodes with styling
        for i, p in enumerate(phenotypes):
            # Highlight initial phenotypes in a different color
            is_initial = p.id in {ip.id for ip in initial_phenotypes}
            color = 'lightpink' if is_initial else 'lightblue2'
            
            # Create label with both name and ID
            label = f"{p.name}\\n({p.id})"
            
            dot.node(str(i), label, style='filled', color=color, shape='box')
        
        # Add edges
        for i in range(adj_list_without_reverse.shape[1]):
            src = str(adj_list_without_reverse[0, i])
            tgt = str(adj_list_without_reverse[1, i])
            dot.edge(src, tgt)
        
        # Save and render the graph
        dot.render('hpo_phenotype_graph', view=True, format='png', cleanup=True)
        print("\nHierarchical graph visualization saved to 'hpo_phenotype_graph.png'")
            
    except Exception as e:
        print(f"\nError during graph visualization: {str(e)}")

if __name__ == "__main__":
    main() 