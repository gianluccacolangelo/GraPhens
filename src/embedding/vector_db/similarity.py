"""
Similarity module for finding and visualizing similar phenotypes.

This module provides functions to:
1. Find phenotypes similar to a query phenotype
2. Demonstrate phenotype similarity using pre-built vector databases
3. Visualize phenotype clusters and relationships
"""

import os
import pickle
import numpy as np
import glob
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass

from src.core.types import Phenotype
from src.embedding.strategies import LookupEmbeddingStrategy
from src.embedding.context import EmbeddingContext
from src.ontology.hpo_graph import HPOGraphProvider


@dataclass
class SimilarityResult:
    """Container for phenotype similarity results."""
    id: str
    name: str
    description: Optional[str]
    similarity: float


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_latest_vector_db(data_dir: str = "data/embeddings") -> str:
    """
    Find the latest vector database file.
    
    Args:
        data_dir: Directory to search for vector databases
        
    Returns:
        Path to the latest vector database file
        
    Raises:
        FileNotFoundError: If no vector database is found
    """
    pattern = os.path.join(data_dir, "hpo_embeddings_*.pkl")
    files = glob.glob(pattern)
    
    # Filter out metadata files
    files = [f for f in files if not f.endswith("_metadata.pkl")]
    
    if not files:
        raise FileNotFoundError(f"No vector database files found in {data_dir}")
    
    # Sort by modification time (newest first)
    return sorted(files, key=os.path.getmtime, reverse=True)[0]


def load_vector_db(
    vector_db_path: Optional[str] = None, 
    data_dir: str = "data/embeddings"
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Load a vector database from disk.
    
    Args:
        vector_db_path: Path to the vector database file
        data_dir: Directory to search for vector databases if path not provided
        
    Returns:
        Tuple of (embedding_dict, vector_dimension)
        
    Raises:
        FileNotFoundError: If no vector database is found
    """
    # Find latest vector database if not specified
    if vector_db_path is None:
        vector_db_path = find_latest_vector_db(data_dir)
        print(f"Using latest vector database: {vector_db_path}")
    
    # Load embedding dictionary
    print(f"Loading vector database from {vector_db_path}...")
    with open(vector_db_path, 'rb') as f:
        embedding_dict = pickle.load(f)
    
    print(f"Loaded embedding dictionary with {len(embedding_dict)} entries")
    
    # Determine vector dimension
    vector_dimension = next(iter(embedding_dict.values())).shape[0]
    
    return embedding_dict, vector_dimension


def find_similar_phenotypes(
    query_id: str,
    embedding_dict: Dict[str, np.ndarray],
    hpo_provider: HPOGraphProvider,
    top_n: int = 10,
    exclude_self: bool = True
) -> List[SimilarityResult]:
    """
    Find phenotypes similar to a query phenotype.
    
    Args:
        query_id: HPO ID of the query phenotype
        embedding_dict: Dictionary mapping HPO IDs to embeddings
        hpo_provider: HPO graph provider for accessing phenotype metadata
        top_n: Number of similar phenotypes to return
        exclude_self: Whether to exclude the query phenotype from results
        
    Returns:
        List of SimilarityResult objects sorted by similarity (highest first)
    """
    if query_id not in embedding_dict:
        raise ValueError(f"Query ID {query_id} not found in embedding dictionary")
    
    query_embedding = embedding_dict[query_id]
    
    # Calculate similarity to all other phenotypes
    similarities = []
    for term_id, term_embedding in embedding_dict.items():
        if exclude_self and term_id == query_id:
            continue
            
        similarity = cosine_similarity(query_embedding, term_embedding)
        similarities.append((term_id, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N results
    results = []
    for i, (term_id, similarity) in enumerate(similarities[:top_n]):
        metadata = hpo_provider.get_metadata(term_id)
        
        # Extract description
        description = None
        if "definition" in metadata:
            description = metadata["definition"]
        elif "meta" in metadata and "definition" in metadata["meta"]:
            if isinstance(metadata["meta"]["definition"], dict):
                description = metadata["meta"]["definition"].get("val", "")
            else:
                description = str(metadata["meta"]["definition"])
        
        results.append(SimilarityResult(
            id=term_id,
            name=metadata.get("name", ""),
            description=description,
            similarity=similarity
        ))
    
    return results


def demo_similar_phenotypes(
    vector_db_path: Optional[str] = None,
    ontology_dir: str = "data/ontology",
    data_dir: str = "data/embeddings",
    query_ids: Optional[List[str]] = None,
    num_examples: int = 5,
    num_similar: int = 5,
    interactive: bool = False
) -> Dict[str, List[SimilarityResult]]:
    """
    Demonstrate finding similar phenotypes using a vector database.
    
    Args:
        vector_db_path: Path to the vector database file
        ontology_dir: Directory where HPO ontology data is stored
        data_dir: Directory to search for vector databases if path not provided
        query_ids: List of HPO IDs to use as query phenotypes
        num_examples: Number of example phenotypes to use if query_ids not provided
        num_similar: Number of similar phenotypes to find for each query
        interactive: Whether to run in interactive mode (prompting for user input)
        
    Returns:
        Dictionary mapping query IDs to lists of similar phenotype results
    """
    try:
        # Load vector database
        embedding_dict, vector_dimension = load_vector_db(vector_db_path, data_dir)
        
        # Initialize HPO graph provider for term metadata
        hpo_provider = HPOGraphProvider(data_dir=ontology_dir)
        hpo_provider.load()
        
        # Create LookupEmbeddingStrategy
        embedding_strategy = LookupEmbeddingStrategy(embedding_dict, dim=vector_dimension)
        
        # Create embedding context (not strictly needed for this demo but good practice)
        embedding_context = EmbeddingContext(embedding_strategy)
        
        # If no query IDs provided, select some examples
        if not query_ids:
            # Default example terms from different categories
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
            query_ids = example_terms[:num_examples]
        
        # Store all results
        all_results = {}
        
        # Process each query
        print("\nPhenotype Similarity Results:")
        print("============================")
        
        for query_id in query_ids:
            if query_id not in embedding_dict:
                print(f"Warning: Query ID {query_id} not found in embedding dictionary. Skipping.")
                continue
                
            metadata = hpo_provider.get_metadata(query_id)
            print(f"\n{metadata.get('name', '')} ({query_id})")
            
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
                query_id=query_id,
                embedding_dict=embedding_dict,
                hpo_provider=hpo_provider,
                top_n=num_similar,
                exclude_self=True
            )
            
            # Store results
            all_results[query_id] = similar_results
            
            # Display results
            print(f"\nTop {len(similar_results)} similar phenotypes:")
            for i, result in enumerate(similar_results):
                print(f"{i+1}. {result.name} ({result.id}) - Similarity: {result.similarity:.4f}")
            
            # In interactive mode, wait for user to press Enter before continuing
            if interactive and query_id != query_ids[-1]:
                input("\nPress Enter to continue to the next example...")
        
        return all_results
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run build_phenotype_vector_db first, or specify a path with --vector-db")
        return {} 
