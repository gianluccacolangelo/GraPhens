"""
Command-line interface for vector database functionality.

This module provides CLI commands for:
1. Building vector databases from phenotypes
2. Finding similar phenotypes using vector databases
3. Demonstrating phenotype similarity
"""

import os
import argparse
import sys
from typing import List, Dict, Optional

from .builder import build_phenotype_vector_db, build_vector_db_for_models
from .similarity import demo_similar_phenotypes, find_similar_phenotypes
from src.ontology.hpo_graph import HPOGraphProvider


def build_cli():
    """
    Command-line interface for building vector databases.
    """
    parser = argparse.ArgumentParser(
        description="Build a vector database of all phenotypes in the HPO ontology."
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="all-MiniLM-L6-v2", 
        help="Model name to use (default: all-MiniLM-L6-v2)"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "huggingface"],
        help="Type of model to use (default: sentence-transformers)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size for processing (default: 32)"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/embeddings", 
        help="Directory to store embedding data (default: data/embeddings)"
    )
    
    parser.add_argument(
        "--ontology-dir", 
        type=str, 
        default="data/ontology", 
        help="Directory where HPO ontology data is stored (default: data/ontology)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Path to save the vector database (default: auto-generated)"
    )
    
    parser.add_argument(
        "--no-progress", 
        action="store_true", 
        help="Disable progress bar"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Build vector database
    output_path = build_phenotype_vector_db(
        model_name=args.model,
        model_type=args.model_type,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        ontology_dir=args.ontology_dir,
        output_path=args.output,
        progress_bar=not args.no_progress
    )
    
    print(f"\nTo use this vector database with LookupEmbeddingStrategy:")
    print(f"```python")
    print(f"import pickle")
    print(f"from src.embedding.strategies import LookupEmbeddingStrategy")
    print(f"")
    print(f"# Load embedding dictionary")
    print(f"with open('{output_path}', 'rb') as f:")
    print(f"    embedding_dict = pickle.load(f)")
    print(f"")
    print(f"# Create lookup embedding strategy")
    print(f"embedding_strategy = LookupEmbeddingStrategy(embedding_dict)")
    print(f"```")


def build_all_cli():
    """
    Command-line interface for building vector databases for multiple models.
    """
    parser = argparse.ArgumentParser(
        description="Build vector databases for multiple biomedical models."
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size for processing (default: 32)"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/embeddings", 
        help="Directory to store embedding data (default: data/embeddings)"
    )
    
    parser.add_argument(
        "--ontology-dir", 
        type=str, 
        default="data/ontology", 
        help="Directory where HPO ontology data is stored (default: data/ontology)"
    )
    
    parser.add_argument(
        "--no-progress", 
        action="store_true", 
        help="Disable progress bar"
    )
    
    # Biomedical models to build databases for
    biomedical_models = {
        "sentence-transformers": [
            "gsarti/biobert-nli",
            "pritamdeka/S-PubMedBert-MS-MARCO"
        ],
        "huggingface": [
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "dmis-lab/biobert-base-cased-v1.2",
            "emilyalsentzer/Bio_ClinicalBERT"
        ]
    }
    
    # Parse arguments
    args = parser.parse_args()
    
    # Build vector databases
    output_paths = build_vector_db_for_models(
        models=biomedical_models,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        ontology_dir=args.ontology_dir,
        progress_bar=not args.no_progress
    )
    
    print(f"\nSuccessfully built {len(output_paths)} vector databases")


def similarity_cli():
    """
    Command-line interface for finding similar phenotypes.
    """
    parser = argparse.ArgumentParser(
        description="Find phenotypes similar to a query phenotype."
    )
    
    parser.add_argument(
        "query_id",
        type=str,
        help="HPO ID of the query phenotype (e.g., HP:0001250)"
    )
    
    parser.add_argument(
        "--vector-db", 
        type=str, 
        default=None, 
        help="Path to the vector database file (default: use latest)"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/embeddings", 
        help="Directory to search for vector databases (default: data/embeddings)"
    )
    
    parser.add_argument(
        "--ontology-dir", 
        type=str, 
        default="data/ontology", 
        help="Directory where HPO ontology data is stored (default: data/ontology)"
    )
    
    parser.add_argument(
        "--num-similar", 
        type=int, 
        default=10, 
        help="Number of similar phenotypes to find (default: 10)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize HPO graph provider
    hpo_provider = HPOGraphProvider(data_dir=args.ontology_dir)
    if not hpo_provider.load():
        print("Error: Failed to load HPO data")
        sys.exit(1)
    
    try:
        # Import here to avoid circular imports
        from .similarity import load_vector_db, find_similar_phenotypes
        
        # Load vector database
        embedding_dict, _ = load_vector_db(args.vector_db, args.data_dir)
        
        # Check if query ID exists
        if args.query_id not in embedding_dict:
            print(f"Error: Query ID {args.query_id} not found in vector database")
            sys.exit(1)
        
        # Get query metadata
        metadata = hpo_provider.get_metadata(args.query_id)
        print(f"\nQuery: {metadata.get('name', '')} ({args.query_id})")
        
        # Get description if available
        description = None
        if "definition" in metadata:
            description = metadata["definition"]
        elif "meta" in metadata and "definition" in metadata["meta"]:
            if isinstance(metadata["meta"]["definition"], dict):
                description = metadata["meta"]["definition"].get("val", "")
            else:
                description = str(metadata["meta"]["definition"])
                
        if description:
            print(f"Description: {description[:100]}..." if len(description) > 100 else f"Description: {description}")
        
        # Find similar phenotypes
        similar_results = find_similar_phenotypes(
            query_id=args.query_id,
            embedding_dict=embedding_dict,
            hpo_provider=hpo_provider,
            top_n=args.num_similar,
            exclude_self=True
        )
        
        # Display results
        print(f"\nTop {len(similar_results)} similar phenotypes:")
        for i, result in enumerate(similar_results):
            print(f"{i+1}. {result.name} ({result.id}) - Similarity: {result.similarity:.4f}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run build_phenotype_vector_db first, or specify a path with --vector-db")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def demo_cli():
    """
    Command-line interface for demonstrating phenotype similarity.
    """
    parser = argparse.ArgumentParser(
        description="Demonstrate finding similar phenotypes using a vector database."
    )
    
    parser.add_argument(
        "--vector-db", 
        type=str, 
        default=None, 
        help="Path to the vector database file (default: use latest)"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/embeddings", 
        help="Directory to search for vector databases (default: data/embeddings)"
    )
    
    parser.add_argument(
        "--ontology-dir", 
        type=str, 
        default="data/ontology", 
        help="Directory where HPO ontology data is stored (default: data/ontology)"
    )
    
    parser.add_argument(
        "--query-ids",
        type=str,
        nargs="+",
        help="HPO IDs to use as query phenotypes (e.g., HP:0001250 HP:0001638)"
    )
    
    parser.add_argument(
        "--num-examples", 
        type=int, 
        default=5, 
        help="Number of example phenotypes to use if query IDs not provided (default: 5)"
    )
    
    parser.add_argument(
        "--num-similar", 
        type=int, 
        default=5, 
        help="Number of similar phenotypes to find for each query (default: 5)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (press Enter to continue between examples)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run demo
    demo_similar_phenotypes(
        vector_db_path=args.vector_db,
        ontology_dir=args.ontology_dir,
        data_dir=args.data_dir,
        query_ids=args.query_ids,
        num_examples=args.num_examples,
        num_similar=args.num_similar,
        interactive=args.interactive
    ) 