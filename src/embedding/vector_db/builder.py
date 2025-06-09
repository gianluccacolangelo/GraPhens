"""
Builder module for creating vector databases of phenotype embeddings.

This module provides functions to:
1. Load phenotypes from the HPO ontology
2. Generate embeddings using various models
3. Save the embeddings as vector databases for use with LookupEmbeddingStrategy
"""

import os
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from tqdm import tqdm

from src.core.types import Phenotype
from src.ontology.hpo_graph import HPOGraphProvider
from src.embedding.strategies import (
    SentenceTransformerEmbeddingStrategy, 
    HuggingFaceEmbeddingStrategy
)
from src.embedding.context import EmbeddingContext


def build_phenotype_vector_db(
    model_name: str = "all-MiniLM-L6-v2",
    model_type: str = "sentence-transformers",
    batch_size: int = 32,
    data_dir: str = "data/embeddings",
    ontology_dir: str = "data/ontology",
    output_path: Optional[str] = None,
    progress_bar: bool = True
) -> str:
    """
    Build a vector database of all phenotypes in the HPO ontology.
    
    Args:
        model_name: Name of the model to use
        model_type: Type of model ('sentence-transformers' or 'huggingface')
        batch_size: Batch size for processing
        data_dir: Directory to store embedding data
        ontology_dir: Directory where HPO ontology data is stored
        output_path: Path to save the vector database
        progress_bar: Whether to show a progress bar
        
    Returns:
        Path to the created vector database
    """
    print(f"Building vector database with {model_type}/{model_name}...")
    
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
        # We don't skip terms without descriptions
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
    if model_type == "sentence-transformers":
        embedding_strategy = SentenceTransformerEmbeddingStrategy(
            model_name=model_name,
            batch_size=batch_size
        )
    elif model_type == "huggingface":
        embedding_strategy = HuggingFaceEmbeddingStrategy(
            model_name_or_path=model_name,
            batch_size=batch_size
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
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
        model_str = model_name.replace('/', '_')
        output_path = os.path.join(data_dir, f"hpo_embeddings_{model_str}_{timestamp}.pkl")
    
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
        "model_type": model_type,
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


def build_vector_db_for_models(
    models: Union[List[str], Dict[str, List[str]]],
    data_dir: str = "data/embeddings",
    ontology_dir: str = "data/ontology",
    batch_size: int = 32,
    progress_bar: bool = True
) -> List[str]:
    """
    Build vector databases for multiple models.
    
    Args:
        models: Either a list of model names (assuming all are sentence-transformers)
               or a dictionary mapping model types to lists of model names
        data_dir: Directory to store embedding data
        ontology_dir: Directory where HPO ontology data is stored
        batch_size: Batch size for processing
        progress_bar: Whether to show a progress bar
        
    Returns:
        List of paths to the created vector databases
    """
    # Convert list to dictionary if necessary
    if isinstance(models, list):
        models_dict = {"sentence-transformers": models}
    else:
        models_dict = models
    
    # Create output directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate vector databases for each model
    output_paths = []
    
    for model_type, model_names in models_dict.items():
        for model_name in model_names:
            print(f"\n{'='*80}")
            print(f"Building vector database for {model_type}/{model_name}")
            print(f"{'='*80}\n")
            
            try:
                output_path = build_phenotype_vector_db(
                    model_name=model_name,
                    model_type=model_type,
                    batch_size=batch_size,
                    data_dir=data_dir,
                    ontology_dir=ontology_dir,
                    progress_bar=progress_bar
                )
                
                output_paths.append(output_path)
                print(f"Successfully built vector database for {model_name}")
                
            except Exception as e:
                print(f"Error building vector database for {model_name}: {e}")
    
    return output_paths 