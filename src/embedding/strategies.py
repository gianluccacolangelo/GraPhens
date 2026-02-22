import os
import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from pathlib import Path

# Import optional dependencies at the top level
try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = None
    AutoModel = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None

from src.core.interfaces import EmbeddingStrategy
from src.core.types import Phenotype

"""
This module provides concrete implementations of the EmbeddingStrategy interface, offering 
different ways to generate vector representations of phenotypes.

Architecturally, it follows the Strategy Pattern, with each class encapsulating a specific 
algorithm for embedding phenotypes. The LLMEmbeddingStrategy uses language models while 
LookupEmbeddingStrategy uses pre-computed embeddings.

For extensibility, adding new embedding methods is as simple as creating a new class that 
implements the EmbeddingStrategy interface. The private helper methods like 
_get_phenotype_text help maintain clean, maintainable code.

In the bigger picture, these strategies allow the system to leverage different embedding 
technologies based on availability and requirements, from simple lookups to sophisticated 
language models.
"""

class LLMEmbeddingStrategy(EmbeddingStrategy):
    """Embeds phenotypes using a language model."""
    
    def __init__(self, llm_service):
        """Initialize with a language model service."""
        self.llm_service = llm_service
    
    def embed_batch(self, phenotypes: List[Phenotype]) -> np.ndarray:
        """Embed a batch of phenotypes using the language model."""
        texts = [self._get_phenotype_text(p) for p in phenotypes]
        return self.llm_service.get_embeddings(texts)
    
    def _get_phenotype_text(self, phenotype: Phenotype) -> str:
        """Get the text representation of a phenotype."""
        text = phenotype.name
        if phenotype.description:
            text += f": {phenotype.description}"
        return text

class LookupEmbeddingStrategy(EmbeddingStrategy):
    """Uses precomputed embeddings for phenotypes."""
    
    def __init__(self, embedding_dict: Dict[str, np.ndarray], dim: int = 768):
        """Initialize with a dictionary mapping HPO IDs to embeddings."""
        self.embedding_dict = embedding_dict
        self.dim = dim
    
    def embed_batch(self, phenotypes: List[Phenotype]) -> np.ndarray:
        """Look up embeddings for a batch of phenotypes."""
        embeddings = []
        for p in phenotypes:
            if p.id in self.embedding_dict:
                embeddings.append(self.embedding_dict[p.id])
            else:
                embeddings.append(np.zeros(self.dim))
        return np.array(embeddings)

class MemmapEmbeddingStrategy(EmbeddingStrategy):
    """
    Uses memory-mapped embeddings for phenotypes.
    
    This strategy provides much more efficient memory usage and faster startup times
    compared to LookupEmbeddingStrategy, especially for large embedding collections,
    as it doesn't load the entire embedding matrix into memory at once.
    """
    
    def __init__(self, 
                memmap_data_path: str, 
                memmap_index_path: str, 
                dim: Optional[int] = None,
                dtype: str = 'float32'):
        """
        Initialize with memory-mapped embedding files.
        
        Args:
            memmap_data_path: Path to the memory-mapped data file
            memmap_index_path: Path to the index mapping file (JSON)
            dim: Embedding dimension (optional, will be determined from the memmap file)
            dtype: NumPy data type of the memory-mapped array
        """
        # Load the index mapping
        with open(memmap_index_path, 'r') as f:
            self.index_map = json.load(f)
        
        # Open the memory-mapped file in read-only mode
        self.memmap = np.memmap(
            memmap_data_path,
            dtype=dtype,
            mode='r'
        )
        
        # Determine shape from the memmap file
        num_vectors = len(self.index_map)
        
        # If dim is not provided, calculate it
        if dim is None:
            # Calculate vector dimension from file size and number of vectors
            file_size = self.memmap.nbytes
            vector_size = file_size // num_vectors
            dim = vector_size // np.dtype(dtype).itemsize
        
        self.dim = dim
        
        # Reshape the memmap array
        self.memmap = self.memmap.reshape(num_vectors, self.dim)
    
    def embed_batch(self, phenotypes: List[Phenotype]) -> np.ndarray:
        """
        Look up embeddings for a batch of phenotypes using memory-mapped files.
        
        This method only loads the specific vectors needed, not the entire database.
        """
        embeddings = np.zeros((len(phenotypes), self.dim))
        
        for i, p in enumerate(phenotypes):
            if p.id in self.index_map:
                # Get the index in the memory-mapped array
                idx = self.index_map[p.id]
                # Load just that vector
                embeddings[i] = self.memmap[idx]
        
        return embeddings
    
    @classmethod
    def from_latest(cls, data_dir: str = "data/embeddings", **kwargs):
        """
        Create a MemmapEmbeddingStrategy using the most recent memory-mapped file.
        
        Args:
            data_dir: Directory to search for memory-mapped files
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            MemmapEmbeddingStrategy instance
        """
        from src.embedding.vector_db.memmap import find_latest_memmap
        
        memmap_data_path, memmap_index_path = find_latest_memmap(data_dir)
        
        return cls(
            memmap_data_path=memmap_data_path,
            memmap_index_path=memmap_index_path,
            **kwargs
        )

class HuggingFaceEmbeddingStrategy(EmbeddingStrategy):
    """Embeds phenotypes using Hugging Face Transformers models.
    
    This is a flexible embedding strategy that works with any Hugging Face model
    that has a get_embeddings or encode method.
    """
    
    def __init__(self, model_name_or_path: str, max_length: int = 128, use_gpu: bool = False, batch_size: int = 32):
        """Initialize with a Hugging Face model name or path.
        
        Args:
            model_name_or_path: Name of a Hugging Face model or path to a saved model
            max_length: Maximum sequence length for the model
            use_gpu: Whether to use GPU for inference if available
            batch_size: Batch size for processing
        """
        if AutoTokenizer is None or AutoModel is None:
            raise ImportError("transformers package not found. Install with: pip install transformers")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Configure device
        if use_gpu:
            if torch is None:
                raise ImportError("torch package not found. Install with: pip install torch")
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'
            
        self.model.to(self.device)
        self.model.eval()
    
    def embed_batch(self, phenotypes: List[Phenotype]) -> np.ndarray:
        """Embed a batch of phenotypes using the Hugging Face model."""
        if torch is None:
            raise ImportError("torch package not found. Install with: pip install torch")
        
        texts = [self._get_phenotype_text(p) for p in phenotypes]
        all_embeddings = []
        
        # Process in batches to avoid OOM errors
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use [CLS] token embedding as the sentence embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
        
        # Combine all batches
        return np.vstack(all_embeddings)
    
    def _get_phenotype_text(self, phenotype: Phenotype) -> str:
        """Get the text representation of a phenotype."""
        text = phenotype.name
        if phenotype.description:
            text += f": {phenotype.description}"
        return text

class SentenceTransformerEmbeddingStrategy(EmbeddingStrategy):
    """Embeds phenotypes using SentenceTransformers models.
    
    SentenceTransformers is specialized for semantic similarity and sentence embeddings,
    making it ideal for creating phenotype vectors.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32):
        """Initialize with a SentenceTransformers model.
        
        Args:
            model_name: Name of a SentenceTransformers model (e.g., 'all-MiniLM-L6-v2', 'all-mpnet-base-v2')
            batch_size: Batch size for processing
        """
        if SentenceTransformer is None:
            raise ImportError("sentence_transformers package not found. Install with: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
    
    def embed_batch(self, phenotypes: List[Phenotype]) -> np.ndarray:
        """Embed a batch of phenotypes using SentenceTransformers."""
        texts = [self._get_phenotype_text(p) for p in phenotypes]
        embeddings = self.model.encode(texts, batch_size=self.batch_size)
        return embeddings
    
    def _get_phenotype_text(self, phenotype: Phenotype) -> str:
        """Get the text representation of a phenotype."""
        text = phenotype.name
        if phenotype.description:
            text += f": {phenotype.description}"
        return text

class TFIDFEmbeddingStrategy(EmbeddingStrategy):
    """Embeds phenotypes using TF-IDF vectorization.
    
    This is a lightweight, non-neural embedding strategy that can be useful for 
    baseline comparisons or when computational resources are limited.
    """
    
    def __init__(self, max_features: int = 768, ngram_range: tuple = (1, 2)):
        """Initialize with TF-IDF parameters.
        
        Args:
            max_features: Maximum number of features (dimensions) in the embedding
            ngram_range: Range of n-grams to consider
        """
        if TfidfVectorizer is None:
            raise ImportError("scikit-learn package not found. Install with: pip install scikit-learn")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.is_fitted = False
    
    def embed_batch(self, phenotypes: List[Phenotype]) -> np.ndarray:
        """Embed a batch of phenotypes using TF-IDF."""
        texts = [self._get_phenotype_text(p) for p in phenotypes]
        
        # Fit the vectorizer if not already fitted
        if not self.is_fitted:
            embeddings = self.vectorizer.fit_transform(texts).toarray()
            self.is_fitted = True
        else:
            embeddings = self.vectorizer.transform(texts).toarray()
            
        return embeddings
    
    def _get_phenotype_text(self, phenotype: Phenotype) -> str:
        """Get the text representation of a phenotype."""
        text = phenotype.name
        if phenotype.description:
            text += f": {phenotype.description}"
        return text

class OpenAIEmbeddingStrategy(EmbeddingStrategy):
    """Embeds phenotypes using OpenAI's embedding models.
    
    This strategy leverages OpenAI's powerful embedding models which are
    trained on diverse data and perform well for semantic similarity tasks.
    """
    
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 32, api_key: Optional[str] = None):
        """Initialize with OpenAI embedding model configuration.
        
        Args:
            model: OpenAI embedding model to use (e.g., "text-embedding-3-small", "text-embedding-3-large")
            batch_size: Batch size for API calls
            api_key: OpenAI API key (if None, will look in OPENAI_API_KEY environment variable)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not found. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
    
    def embed_batch(self, phenotypes: List[Phenotype]) -> np.ndarray:
        """Embed a batch of phenotypes using OpenAI's embedding API."""
        texts = [self._get_phenotype_text(p) for p in phenotypes]
        all_embeddings = []
        
        # Process in batches to comply with API limits
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=batch_texts
            )
            
            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def _get_phenotype_text(self, phenotype: Phenotype) -> str:
        """Get the text representation of a phenotype."""
        text = phenotype.name
        if phenotype.description:
            text += f": {phenotype.description}"
        return text
