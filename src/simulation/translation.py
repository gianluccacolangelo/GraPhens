from src.embedding.strategies import SentenceTransformerEmbeddingStrategy
from src.core.types import Phenotype
from src.ontology.hpo_graph import HPOGraphProvider
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import glob

@dataclass
class SimilarityMatch:
    id: str
    name: str
    description: Optional[str]
    similarity: float

def find_phenotype_matches(
    description: str,
    model_name: str = "all-MiniLM-L6-v2",
    embeddings_dir: str = "data/embeddings",
    top_n: int = 5
) -> List[SimilarityMatch]:
    """
    Find matching HPO terms for a natural language description using specified embedding model.
    
    Args:
        description: Natural language description of the phenotype
        model_name: Name of the embedding model to use (e.g., "all-MiniLM-L6-v2", "gsarti/biobert-nli")
        embeddings_dir: Directory containing pre-computed embeddings
        top_n: Number of similar phenotypes to return
    
    Returns:
        List of SimilarityMatch objects containing the most similar phenotypes
    """
    # 1. Find the correct embedding files based on model name
    model_files = glob.glob(f"{embeddings_dir}/hpo_embeddings_{model_name.replace('/', '_')}_*.pkl")
    if not model_files:
        raise ValueError(f"No pre-computed embeddings found for model {model_name}")
    
    # Get the latest embedding file and its metadata
    embedding_file = sorted(f for f in model_files if not f.endswith('metadata.pkl'))[-1]
    metadata_file = embedding_file.replace('.pkl', '_metadata.pkl')
    
    # 2. Load embeddings and metadata
    with open(embedding_file, 'rb') as f:
        embeddings_dict = pickle.load(f)
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
        
    # 3. Create embedding strategy based on the model
    if 'biobert' in model_name.lower():
        from src.embedding.strategies import HuggingFaceEmbeddingStrategy
        strategy = HuggingFaceEmbeddingStrategy(model_name_or_path=model_name)
    else:
        strategy = SentenceTransformerEmbeddingStrategy(model_name=model_name)
    
    # 4. Create embedding for the description
    query_phenotype = Phenotype(id="query", name=description)
    query_embedding = strategy.embed_batch([query_phenotype])[0]
    
    # 5. Compute cosine similarity with all pre-computed embeddings
    similarities = {}
    for hpo_id, embedding in embeddings_dict.items():
        similarity = cosine_similarity(query_embedding, embedding)
        similarities[hpo_id] = similarity
    
    # 6. Get top N most similar phenotypes
    top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # 7. Get phenotype metadata
    hpo_provider = HPOGraphProvider.get_instance()
    
    results = []
    for hpo_id, similarity in top_matches:
        metadata = hpo_provider.get_metadata(hpo_id)
        match = SimilarityMatch(
            id=hpo_id,
            name=metadata.get('name', ''),
            description=metadata.get('definition', None),
            similarity=float(similarity)
        )
        results.append(match)
    
    return results

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_embedding_resources(model_name: str = "all-MiniLM-L6-v2", embeddings_dir: str = "data/embeddings"):
    """
    Load embedding resources once to avoid repeated loading.
    
    Returns:
        Dictionary with loaded resources
    """
    # 1. Find the correct embedding files based on model name
    model_files = glob.glob(f"{embeddings_dir}/hpo_embeddings_{model_name.replace('/', '_')}_*.pkl")
    if not model_files:
        raise ValueError(f"No pre-computed embeddings found for model {model_name}")
    
    # Get the latest embedding file and its metadata
    embedding_file = sorted(f for f in model_files if not f.endswith('metadata.pkl'))[-1]
    metadata_file = embedding_file.replace('.pkl', '_metadata.pkl')
    
    print(f"Loading embeddings from {embedding_file}")
    # 2. Load embeddings and metadata
    with open(embedding_file, 'rb') as f:
        embeddings_dict = pickle.load(f)
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
        
    # 3. Create embedding strategy based on the model
    if 'biobert' in model_name.lower():
        from src.embedding.strategies import HuggingFaceEmbeddingStrategy
        strategy = HuggingFaceEmbeddingStrategy(model_name_or_path=model_name)
    else:
        strategy = SentenceTransformerEmbeddingStrategy(model_name=model_name)
    
    # 4. Get HPO provider
    hpo_provider = HPOGraphProvider.get_instance()
    
    return {
        'embeddings_dict': embeddings_dict,
        'metadata': metadata,
        'strategy': strategy,
        'hpo_provider': hpo_provider
    }
