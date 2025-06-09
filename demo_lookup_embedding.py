#!/usr/bin/env python3
"""
Demo script showing how to use the LookupEmbeddingStrategy with a pre-built vector database.

This script:
1. Loads a pre-built vector database
2. Creates a LookupEmbeddingStrategy
3. Demonstrates embedding some example phenotypes
4. Shows how to find similar phenotypes
"""

import os
import argparse
import pickle
import numpy as np
from typing import List, Dict
import glob

from src.core.types import Phenotype
from src.embedding.strategies import LookupEmbeddingStrategy
from src.embedding.context import EmbeddingContext
from src.ontology.hpo_graph import HPOGraphProvider


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_latest_vector_db(data_dir: str = "data/embeddings") -> str:
    """Find the latest vector database file."""
    pattern = os.path.join(data_dir, "hpo_embeddings_*.pkl")
    files = glob.glob(pattern)
    
    # Filter out metadata files
    files = [f for f in files if not f.endswith("_metadata.pkl")]
    
    if not files:
        raise FileNotFoundError(f"No vector database files found in {data_dir}")
    
    # Sort by modification time (newest first)
    return sorted(files, key=os.path.getmtime, reverse=True)[0]


def demo_lookup_embedding(
    vector_db_path: str = None,
    ontology_dir: str = "data/ontology",
    num_examples: int = 5,
    num_similar: int = 5
):
    """
    Demonstrate using the LookupEmbeddingStrategy with a pre-built vector database.
    
    Args:
        vector_db_path: Path to the vector database file
        ontology_dir: Directory where HPO ontology data is stored
        num_examples: Number of example phenotypes to show
        num_similar: Number of similar phenotypes to find for each example
    """
    # Find latest vector database if not specified
    if vector_db_path is None:
        try:
            vector_db_path = find_latest_vector_db()
            print(f"Using latest vector database: {vector_db_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please run build_phenotype_vector_db.py first, or specify a path with --vector-db")
            return
    
    # Load embedding dictionary
    print(f"Loading vector database from {vector_db_path}...")
    with open(vector_db_path, 'rb') as f:
        embedding_dict = pickle.load(f)
    
    print(f"Loaded embedding dictionary with {len(embedding_dict)} entries")
    
    # Initialize HPO graph provider for term metadata
    hpo_provider = HPOGraphProvider(data_dir=ontology_dir)
    hpo_provider.load()
    
    # Create LookupEmbeddingStrategy
    vector_dimension = next(iter(embedding_dict.values())).shape[0]
    embedding_strategy = LookupEmbeddingStrategy(embedding_dict, dim=vector_dimension)
    
    # Create embedding context
    embedding_context = EmbeddingContext(embedding_strategy)
    
    # Select some examples (phenotypes from different categories)
    example_terms = [
        "HP:0001250",  # Seizure
        "HP:0001638",  # Cardiomyopathy
        "HP:0001513",  # Obesity
        "HP:0000486",  # Strabismus
        "HP:0001510"   # Growth delay
    ]
    
    # Keep only terms that exist in our embedding dictionary
    example_terms = [term for term in example_terms if term in embedding_dict]
    
    # If we don't have enough examples, take some random ones
    if len(example_terms) < num_examples:
        import random
        random.seed(42)  # For reproducibility
        additional_terms = random.sample(list(embedding_dict.keys()), num_examples - len(example_terms))
        example_terms.extend(additional_terms)
    
    # Limit to requested number
    example_terms = example_terms[:num_examples]
    
    # Create phenotype objects for examples
    example_phenotypes = []
    for term_id in example_terms:
        metadata = hpo_provider.get_metadata(term_id)
        phenotype = Phenotype(
            id=term_id,
            name=metadata.get("name", ""),
            description=metadata.get("definition", "")
        )
        example_phenotypes.append(phenotype)
    
    # Embed examples
    example_embeddings = embedding_context.embed_phenotypes(example_phenotypes)
    
    # For each example, find similar phenotypes
    print("\nExample phenotypes and their similar terms:")
    print("============================================")
    
    for i, phenotype in enumerate(example_phenotypes):
        print(f"\n{i+1}. {phenotype.name} ({phenotype.id})")
        if phenotype.description:
            print(f"   Description: {phenotype.description[:100]}...")
        
        # Get embedding
        embedding = example_embeddings[i]
        
        # Calculate similarities with all other terms
        similarities = []
        for term_id, term_embedding in embedding_dict.items():
            if term_id != phenotype.id:  # Skip self
                similarity = cosine_similarity(embedding, term_embedding)
                similarities.append((term_id, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Show top similar terms
        print(f"\n   Top {num_similar} similar phenotypes:")
        for j, (term_id, similarity) in enumerate(similarities[:num_similar]):
            metadata = hpo_provider.get_metadata(term_id)
            print(f"   {j+1}. {metadata.get('name', '')} ({term_id}) - Similarity: {similarity:.4f}")


def main():
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="Demonstrate using the LookupEmbeddingStrategy with a pre-built vector database."
    )
    
    parser.add_argument(
        "--vector-db", 
        type=str, 
        default=None, 
        help="Path to the vector database file (default: use latest)"
    )
    
    parser.add_argument(
        "--ontology-dir", 
        type=str, 
        default="data/ontology", 
        help="Directory where HPO ontology data is stored (default: data/ontology)"
    )
    
    parser.add_argument(
        "--num-examples", 
        type=int, 
        default=5, 
        help="Number of example phenotypes to show (default: 5)"
    )
    
    parser.add_argument(
        "--num-similar", 
        type=int, 
        default=5, 
        help="Number of similar phenotypes to find for each example (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Run demo
    demo_lookup_embedding(
        vector_db_path=args.vector_db,
        ontology_dir=args.ontology_dir,
        num_examples=args.num_examples,
        num_similar=args.num_similar
    )


if __name__ == "__main__":
    main() 