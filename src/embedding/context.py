import numpy as np
from typing import List
from src.core.interfaces import EmbeddingStrategy
from src.core.types import Phenotype

"""
This module implements the Strategy Pattern through the EmbeddingContext class, allowing 
the system to dynamically change embedding strategies at runtime.

Architecturally, it serves as a bridge between the embedding strategy and the rest of the 
system, providing a stable interface regardless of which embedding approach is being used.

For extensibility, this design allows new embedding strategies to be integrated seamlessly, 
as the EmbeddingContext delegates the actual embedding work to the strategy object it holds.

In the bigger picture, this separation of concerns enables the system to experiment with 
different embedding techniques without changing how the embeddings are used throughout the 
pipeline.
"""


class EmbeddingContext:
    """Context for embedding phenotypes using a strategy pattern."""
    
    def __init__(self, embedding_strategy: EmbeddingStrategy):
        """Initialize with an embedding strategy."""
        self.embedding_strategy = embedding_strategy
    
    def embed_phenotypes(self, phenotypes: List[Phenotype]) -> np.ndarray:
        """Embed a list of phenotypes using the current strategy."""
        return self.embedding_strategy.embed_batch(phenotypes)
"""

"""

