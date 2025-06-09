#!/usr/bin/env python3
"""
Example script showing how to integrate the gene-phenotype database with other components.

This script demonstrates using the gene-phenotype database to retrieve phenotypes for a gene,
then using those phenotypes with the ontology, embedding, and graph components to create
a comprehensive analysis.
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.gene_phenotype_db import HPOAGenePhenotypeDatabase
from src.core.types import Phenotype, Graph
from src.augmentation.hpo_augmentation import HPOAugmentationService
from src.ontology.hpo_graph import HPOGraphProvider
from src.graph.adjacency import HPOAdjacencyListBuilder
from src.graph.assembler import StandardGraphAssembler
from src.embedding.context import EmbeddingContext
from src.embedding.strategies import TFIDFEmbeddingStrategy

def setup_logging(verbose=False):
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def analyze_gene_phenotypes(
    gene_symbol: str,
    database_path: str,
    ontology_dir: str,
    augment: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Analyze phenotypes associated with a gene.
    
    Args:
        gene_symbol: Gene symbol to analyze
        database_path: Path to the gene-phenotype database file
        ontology_dir: Directory containing HPO ontology files
        augment: Whether to augment phenotypes with related terms
        verbose: Enable verbose logging
        
    Returns:
        Dictionary containing analysis results
    """
    logger = setup_logging(verbose)
    logger.info(f"Analyzing phenotypes for gene: {gene_symbol}")
    
    # Step 1: Load phenotypes from the database
    logger.info("Loading gene-phenotype database...")
    db = HPOAGenePhenotypeDatabase(
        genes_to_phenotypes_path=database_path
    )
    
    phenotypes = db.get_phenotypes_for_gene(gene_symbol)
    logger.info(f"Found {len(phenotypes)} phenotypes for gene {gene_symbol}")
    
    if not phenotypes:
        logger.warning(f"No phenotypes found for gene {gene_symbol}")
        return {
            "gene_symbol": gene_symbol,
            "phenotype_count": 0,
            "graph": None,
            "success": False
        }
    
    # Step 2: Set up HPO graph provider
    logger.info("Loading HPO ontology...")
    hpo_provider = HPOGraphProvider(data_dir=ontology_dir)
    
    # Step 3: Optionally augment phenotypes
    if augment:
        logger.info("Augmenting phenotypes with related terms...")
        augmentation_service = HPOAugmentationService(
            data_dir=ontology_dir,
            include_ancestors=True,
            include_descendants=False
        )
        
        augmented_phenotypes = augmentation_service.augment(phenotypes)
        logger.info(f"Augmented to {len(augmented_phenotypes)} phenotypes")
        working_phenotypes = augmented_phenotypes
    else:
        logger.info("Skipping phenotype augmentation")
        working_phenotypes = phenotypes
    
    # Step 4: Create embeddings
    logger.info("Creating phenotype embeddings...")
    embedding_strategy = TFIDFEmbeddingStrategy(max_features=768)
    embedding_context = EmbeddingContext(embedding_strategy)
    
    node_features = embedding_context.embed_phenotypes(working_phenotypes)
    logger.info(f"Created embeddings with shape {node_features.shape}")
    
    # Step 5: Build adjacency list
    logger.info("Building graph adjacency list...")
    adjacency_builder = HPOAdjacencyListBuilder(
        hpo_provider=hpo_provider,
        include_reverse_edges=True
    )
    
    edge_index = adjacency_builder.build(working_phenotypes)
    logger.info(f"Created graph with {edge_index.shape[1]} edges")
    
    # Step 6: Assemble the graph
    logger.info("Assembling final graph...")
    graph_assembler = StandardGraphAssembler()
    
    graph = graph_assembler.assemble(
        phenotypes=working_phenotypes,
        node_features=node_features,
        edge_index=edge_index
    )
    
    # Return the results
    return {
        "gene_symbol": gene_symbol,
        "phenotype_count": {
            "original": len(phenotypes),
            "augmented": len(working_phenotypes) if augment else None
        },
        "graph": {
            "node_count": graph.node_features.shape[0],
            "edge_count": graph.edge_index.shape[1],
            "feature_dim": graph.node_features.shape[1]
        },
        "phenotypes": [p.id for p in working_phenotypes],
        "success": True
    }

def print_analysis_results(results: Dict[str, Any]):
    """Print the analysis results in a readable format."""
    print("\n" + "=" * 60)
    print(f"Analysis Results for Gene: {results['gene_symbol']}")
    print("=" * 60)
    
    if not results["success"]:
        print("No phenotypes found for this gene.")
        return
    
    # Print phenotype counts
    print("\nPhenotype Counts:")
    print(f"  Original: {results['phenotype_count']['original']}")
    if results['phenotype_count']['augmented'] is not None:
        print(f"  Augmented: {results['phenotype_count']['augmented']}")
    
    # Print graph details
    print("\nGraph Details:")
    print(f"  Nodes: {results['graph']['node_count']}")
    print(f"  Edges: {results['graph']['edge_count']}")
    print(f"  Feature dimensions: {results['graph']['feature_dim']}")
    
    # Print some of the phenotypes
    print("\nSample Phenotypes:")
    max_display = 10
    phenotypes_to_show = results["phenotypes"][:max_display]
    for i, p in enumerate(phenotypes_to_show, 1):
        print(f"  {i}. {p}")
    
    if len(results["phenotypes"]) > max_display:
        remaining = len(results["phenotypes"]) - max_display
        print(f"  ... and {remaining} more phenotypes")
    
    print("\nAnalysis complete.")

def main():
    """Main function for the script."""
    parser = argparse.ArgumentParser(description="Analyze phenotypes for a gene")
    
    parser.add_argument("gene_symbol", help="Gene symbol to analyze (e.g., AARS1)")
    parser.add_argument("--database-path", default="data/genes_to_phenotype.txt", 
                        help="Path to the gene-phenotype database file")
    parser.add_argument("--ontology-dir", default="data/ontology",
                        help="Directory containing HPO ontology files")
    parser.add_argument("--no-augment", action="store_true", 
                        help="Disable phenotype augmentation")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Check if database file exists
    if not os.path.exists(args.database_path):
        print(f"Error: Database file not found: {args.database_path}")
        print("Please run setup_database.py to create the data directory.")
        sys.exit(1)
    
    # Check if ontology directory exists
    if not os.path.exists(args.ontology_dir):
        print(f"Error: Ontology directory not found: {args.ontology_dir}")
        print("Please download the HPO ontology files first.")
        sys.exit(1)
    
    try:
        results = analyze_gene_phenotypes(
            gene_symbol=args.gene_symbol,
            database_path=args.database_path,
            ontology_dir=args.ontology_dir,
            augment=not args.no_augment,
            verbose=args.verbose
        )
        
        print_analysis_results(results)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 