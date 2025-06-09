import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.types import Phenotype, Graph
from src.graph.assembler import StandardGraphAssembler
from src.graph.adjacency import HPOAdjacencyListBuilder
from src.embedding.context import EmbeddingContext
from src.embedding.strategies import LookupEmbeddingStrategy, TFIDFEmbeddingStrategy
from src.embedding.vector_db.similarity import load_vector_db
from src.ontology.hpo_graph import HPOGraphProvider
from src.ontology.hpo_updater import HPOUpdater
from src.augmentation.hpo_augmentation import HPOAugmentationService
from src.visualization.graphviz import GraphvizVisualizer

def load_phenotypes(phenotype_ids: List[str], hpo_provider: HPOGraphProvider) -> List[Phenotype]:
    """
    Load phenotypes from their IDs.
    
    Args:
        phenotype_ids: List of HPO IDs
        hpo_provider: HPO graph provider for metadata
        
    Returns:
        List of Phenotype objects
    """
    # Ensure the HPO graph is loaded
    hpo_provider.load()
    
    phenotypes = []
    for hpo_id in phenotype_ids:
        # Normalize the HPO ID format
        hpo_id = hpo_provider._normalize_hpo_id(hpo_id)
        
        # Get metadata from the HPO provider
        metadata = hpo_provider.get_metadata(hpo_id)
        
        if not metadata:
            print(f"Warning: No metadata found for {hpo_id}")
            name = hpo_id
            description = None
        else:
            name = metadata.get('name', hpo_id)
            description = metadata.get('definition', None)
            
        # Create Phenotype object
        phenotypes.append(Phenotype(
            id=hpo_id,
            name=name,
            description=description
        ))
    
    return phenotypes

def main():
    """Run the graph assembler demonstration."""
    parser = argparse.ArgumentParser(description="Demonstrate the graph assembly process")
    parser.add_argument('--ontology-dir', type=str, default="data/ontology", help="Directory containing HPO ontology files")
    parser.add_argument('--embeddings-dir', type=str, default="data/embeddings", help="Directory containing embedding files")
    parser.add_argument('--vector-db', type=str, default=None, help="Specific vector database file to use")
    parser.add_argument('--update-ontology', action='store_true', help="Update HPO ontology before running the demo")
    parser.add_argument('--format', type=str, choices=['json', 'obo'], default='json', help="HPO ontology format to use")
    parser.add_argument('--augment', action='store_true', help="Augment phenotypes with related terms", default=True)
    parser.add_argument('--include-ancestors', action='store_true', help="Include ancestor terms in augmentation", default=True)
    parser.add_argument('--include-descendants', action='store_true', help="Include descendant terms in augmentation", default=False)
    parser.add_argument('--output-dir', type=str, default="visualizations", help="Directory to save visualizations")
    args = parser.parse_args()
    
    # Step 1: Optionally update the HPO ontology
    if args.update_ontology:
        print("Updating HPO ontology...")
        updater = HPOUpdater(data_dir=args.ontology_dir)
        updater.update(format_type=args.format)
    
    # Step 2: Initialize HPO graph provider
    print(f"Initializing HPO graph provider using {args.format} format...")
    hpo_provider = HPOGraphProvider(data_dir=args.ontology_dir)
    
    # Step 3: Define some example phenotypes
    example_phenotypes = [
        "HP:0001250",  # Seizure
        "HP:0001251",  # Atonic seizure
        "HP:0002360",  # Sleep disturbance
        "HP:0000708",  # Behavioral abnormality
        "HP:0000707",  # Abnormality of the nervous system
        "HP:0000118"   # Phenotypic abnormality (root)
    ]
    
    # Load initial phenotype objects
    print("Loading phenotype objects...")
    initial_phenotypes = load_phenotypes(example_phenotypes, hpo_provider)
    
    # Display the loaded phenotypes
    print("\nInitial phenotypes:")
    for p in initial_phenotypes:
        print(f"  - {p.id}: {p.name}")
    
    # Step 4: Augment phenotypes with related terms
    phenotypes = initial_phenotypes
    
    if args.augment:
        print("\nAugmenting phenotypes with related terms...")
        augmentation_service = HPOAugmentationService(
            data_dir=args.ontology_dir, 
            include_ancestors=args.include_ancestors, 
            include_descendants=args.include_descendants
        )
        phenotypes = augmentation_service.augment(initial_phenotypes)
        
        print(f"Added {len(phenotypes) - len(initial_phenotypes)} related phenotypes through augmentation")
        print(f"Total phenotypes: {len(phenotypes)}")
    
    # Step 5: Set up embedding strategy using vector database
    print("\nSetting up embedding strategies...")
    
    # Load vector database for lookup strategy
    embedding_dict, vector_dimension = load_vector_db(args.vector_db, args.embeddings_dir)
    
    # Create lookup embedding strategy
    lookup_strategy = LookupEmbeddingStrategy(embedding_dict, dim=vector_dimension)
    lookup_context = EmbeddingContext(lookup_strategy)
    
    # Also create a TFIDF strategy as an alternative
    tfidf_strategy = TFIDFEmbeddingStrategy(max_features=vector_dimension)
    tfidf_context = EmbeddingContext(tfidf_strategy)
    
    # Step 6: Create adjacency list builder
    print("\nCreating adjacency list builder...")
    adjacency_builder = HPOAdjacencyListBuilder(hpo_provider, include_reverse_edges=False)
    
    # Step 7: Build edge indices
    print("Building edge indices...")
    edge_index = adjacency_builder.build(phenotypes)
    print(f"Created edge index with shape {edge_index.shape}")
    
    # Step 8: Generate node embeddings using both strategies
    print("\nGenerating node embeddings...")
    
    # Using lookup strategy
    print("Using lookup embedding strategy...")
    lookup_node_features = lookup_context.embed_phenotypes(phenotypes)
    print(f"Generated node features with shape {lookup_node_features.shape}")
    
    # Using TFIDF strategy
    print("Using TFIDF embedding strategy...")
    tfidf_node_features = tfidf_context.embed_phenotypes(phenotypes)
    print(f"Generated node features with shape {tfidf_node_features.shape}")
    
    # Step 9: Assemble the graph using the StandardGraphAssembler
    print("\nAssembling graphs...")
    graph_assembler = StandardGraphAssembler()
    
    # Assemble with lookup embeddings
    lookup_graph = graph_assembler.assemble(
        phenotypes=phenotypes,
        node_features=lookup_node_features,
        edge_index=edge_index
    )
    
    # Assemble with TFIDF embeddings
    tfidf_graph = graph_assembler.assemble(
        phenotypes=phenotypes,
        node_features=tfidf_node_features,
        edge_index=edge_index
    )
    
    # Step 10: Visualize and compare the results
    print("\nGraph assembly complete!")
    print(f"Total nodes: {len(phenotypes)} ({len(initial_phenotypes)} initial + {len(phenotypes) - len(initial_phenotypes)} from augmentation)")
    print(f"Total edges: {edge_index.shape[1]}")
    
    # Print node mapping as a demonstration
    print("\nNode mapping (HPO ID to index):")
    for i, (hpo_id, idx) in enumerate(lookup_graph.node_mapping.items()):
        if i < 10 or hpo_id in [p.id for p in initial_phenotypes]:  # Show only first 10 and initial phenotypes
            print(f"  - {hpo_id} -> {idx}")
    if len(lookup_graph.node_mapping) > 10:
        print(f"  ... and {len(lookup_graph.node_mapping) - 10} more")
    
    # Initialize the visualizer
    visualizer = GraphvizVisualizer(output_dir=args.output_dir, hpo_provider=hpo_provider)
    
    # Step 11: Visualize the augmentation results
    if args.augment:
        print("\nVisualizing augmentation results...")
        vis_path = visualizer.visualize_augmentation_result(
            augmented_phenotypes=phenotypes,
            initial_phenotypes=initial_phenotypes,
            title="Phenotype Augmentation Result"
        )
        print(f"Augmentation visualization saved to: {vis_path}")
    
    # Step 12: Visualize the assembled graphs with different embeddings
    print("\nVisualizing graphs...")
    lookup_vis_path = visualizer.visualize_graph(
        graph=lookup_graph, 
        phenotypes=phenotypes,
        initial_phenotypes=initial_phenotypes,
        title="Phenotype Graph (Lookup Embeddings)"
    )
    print(f"Lookup embedding graph visualization saved to: {lookup_vis_path}")
    
    tfidf_vis_path = visualizer.visualize_graph(
        graph=tfidf_graph, 
        phenotypes=phenotypes,
        initial_phenotypes=initial_phenotypes,
        title="Phenotype Graph (TFIDF Embeddings)"
    )
    print(f"TFIDF embedding graph visualization saved to: {tfidf_vis_path}")
    
    print("\nDemo complete! Graphs have been visualized and saved as PNG files.")

if __name__ == "__main__":
    main()