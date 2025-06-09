#!/usr/bin/env python3
"""
Script to build a vector database of all phenotypes in the HPO ontology.

This script:
1. Loads all phenotypes from the HPO graph
2. Extracts their descriptions (or names if no description is available)
3. Generates embeddings using one of our embedding models
4. Saves the embeddings in a format compatible with the LookupEmbeddingStrategy
"""

import os
import argparse
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from src.core.types import Phenotype
from src.ontology.hpo_graph import HPOGraphProvider
from src.embedding.strategies import SentenceTransformerEmbeddingStrategy
from src.embedding.context import EmbeddingContext


def build_phenotype_vector_db(
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    data_dir: str = "data/embeddings",
    ontology_dir: str = "data/ontology",
    output_path: Optional[str] = None,
    progress_bar: bool = True
):
    """
    Build a vector database of all phenotypes in the HPO ontology.
    
    Args:
        model_name: Name of the SentenceTransformers model to use
        batch_size: Batch size for processing
        data_dir: Directory to store embedding data
        ontology_dir: Directory where HPO ontology data is stored
        output_path: Path to save the vector database
        progress_bar: Whether to show a progress bar
        
    Returns:
        Path to the created vector database
    """
    print(f"Building vector database with {model_name}...")
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize HPO graph provider
    hpo_provider = HPOGraphProvider(data_dir=ontology_dir)
    
    # Load HPO data
    print("Loading HPO data...")
    hpo_provider.load()
    
    # Get all term IDs
    all_terms = list(hpo_provider.terms.keys())
    print(f"Found {len(all_terms)} HPO terms")
    
    # Convert terms to phenotype objects
    phenotypes = []
    
    print("Converting terms to phenotype objects...")
    iterator = tqdm(all_terms) if progress_bar else all_terms
    
    for term_id in iterator:
        metadata = hpo_provider.get_metadata(term_id)
        
        # Extract description - it might be in different fields depending on the format
        description = None
        if "definition" in metadata:
            description = metadata["definition"]
        elif "meta" in metadata and "definition" in metadata["meta"]:
            if isinstance(metadata["meta"]["definition"], dict):
                description = metadata["meta"]["definition"].get("val", "")
            else:
                description = str(metadata["meta"]["definition"])
        
        # Clean up definition if available
        if isinstance(description, str):
            # Remove quotes and everything after [
            if '"' in description:
                description = description.split('"')[1] if len(description.split('"')) > 1 else description
            if '[' in description:
                description = description.split('[')[0].strip()
        
        # Use name if no description is available
        # This is the key difference from the evaluation script - we don't skip terms without descriptions
        if not description:
            description = metadata.get("name", "")
            
        # Create phenotype object
        phenotype = Phenotype(
            id=term_id,
            name=metadata.get("name", ""),
            description=description
        )
        phenotypes.append(phenotype)
    
    print(f"Processed {len(phenotypes)} phenotypes")
    
    # Initialize embedding strategy
    print(f"Initializing embedding model: {model_name}")
    embedding_strategy = SentenceTransformerEmbeddingStrategy(
        model_name=model_name,
        batch_size=batch_size
    )
    
    # Create embedding context
    embedding_context = EmbeddingContext(embedding_strategy)
    
    # Generate embeddings
    print("Generating embeddings...")
    start_time = datetime.now()
    
    embeddings = embedding_context.embed_phenotypes(phenotypes)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"Generated {embeddings.shape[0]} embeddings in {duration:.2f} seconds")
    
    # Create embedding dictionary that maps HPO IDs to embeddings
    embedding_dict = {}
    for i, phenotype in enumerate(phenotypes):
        embedding_dict[phenotype.id] = embeddings[i]
    
    print(f"Created embedding dictionary with {len(embedding_dict)} entries")
    
    # Set default output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(data_dir, f"hpo_embeddings_{model_name.replace('/', '_')}_{timestamp}.pkl")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save embedding dictionary
    print(f"Saving vector database to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(embedding_dict, f)
    
    # Also save metadata for reference
    metadata_path = output_path.replace('.pkl', '_metadata.pkl')
    metadata = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "num_phenotypes": len(phenotypes),
        "embedding_shape": embeddings.shape,
        "phenotype_ids": [p.id for p in phenotypes],
        "hpo_version": hpo_provider.version
    }
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Vector database successfully saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")
    
    return output_path


def main():
    """Parse arguments and run the vector database builder."""
    parser = argparse.ArgumentParser(
        description="Build a vector database of all phenotypes in the HPO ontology."
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="all-MiniLM-L6-v2", 
        help="SentenceTransformers model to use (default: all-MiniLM-L6-v2)"
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
    
    args = parser.parse_args()
    
    # Build vector database
    output_path = build_phenotype_vector_db(
        model_name=args.model,
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


if __name__ == "__main__":
    main() 