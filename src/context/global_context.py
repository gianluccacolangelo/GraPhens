import numpy as np
from typing import List
from src.core.interfaces import GlobalContextProvider
from src.core.types import Phenotype

"""
This module provides mechanisms to generate global context vectors that capture information 
about the entire set of phenotypes.

Architecturally, it implements the GlobalContextProvider interface with two different 
strategies: AverageEmbeddingContextProvider (statistical approach) and HPODAGContextProvider 
(structural approach).

For extensibility, new context generation approaches can be added by implementing the 
GlobalContextProvider interface. The current implementations showcase how to leverage both 
embedding information and domain-specific structure.

In the bigger picture, these global context vectors enhance the final graph representation 
by providing additional information that isn't captured by node features or edges alone, 
potentially improving downstream tasks.
"""

class AverageEmbeddingContextProvider(GlobalContextProvider):
    """Provides global context by averaging node embeddings."""
    
    def provide_context(self, phenotypes: List[Phenotype], embeddings: np.ndarray) -> np.ndarray:
        """Generate a global context vector by averaging node embeddings."""
        if len(embeddings) == 0:
            return np.array([])
        return np.mean(embeddings, axis=0)

class HPODAGContextProvider(GlobalContextProvider):
    """Provides global context derived from the HPO DAG structure."""
    
    def __init__(self, hpo_context_provider):
        """Initialize with an HPO context provider."""
        self.hpo_context_provider = hpo_context_provider
    
    def provide_context(self, phenotypes: List[Phenotype]) -> np.ndarray:
        """Generate a global context vector based on HPO DAG properties."""
        phenotype_ids = [p.id for p in phenotypes]
        return self.hpo_context_provider.get_context_vector(phenotype_ids)
