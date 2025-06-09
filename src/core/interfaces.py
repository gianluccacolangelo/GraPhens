from abc import ABC, abstractmethod
from typing import List, Any, Optional
import numpy as np
from .types import Phenotype, Graph

"""
This module defines the core abstraction layer of the system using abstract base classes. 
It follows the Interface Segregation Principle by providing specialized interfaces for 
each major component of the pipeline.

Architecturally, these interfaces establish clear boundaries between components, 
enabling a pluggable architecture where implementations can be swapped without affecting 
the rest of the system. Each interface has a single responsibility, focusing on specific
aspects like augmentation, embedding, or graph assembly.

For extensibility, new strategies can be implemented by creating classes that inherit from 
these interfaces. For example, new embedding approaches can be added by implementing the 
EmbeddingStrategy interface, without modifying existing code.

In the bigger picture, these interfaces form the backbone of dependency inversion, allowing 
high-level modules (like the orchestrator) to depend on abstractions rather than concrete 
implementations.
"""

class AugmentationService(ABC):
    """Interface for augmenting phenotypes."""
    @abstractmethod
    def augment(self, observed: List[Phenotype]) -> List[Phenotype]:
        """Augment the list of observed phenotypes."""
        pass

class EmbeddingStrategy(ABC):
    """Interface for embedding phenotypes."""
    @abstractmethod
    def embed_batch(self, phenotypes: List[Phenotype]) -> np.ndarray:
        """Embed a batch of phenotypes into a matrix."""
        pass
    
    def embed(self, phenotype: Phenotype) -> np.ndarray:
        """Embed a single phenotype into a vector."""
        return self.embed_batch([phenotype])[0]

class AdjacencyListBuilder(ABC):
    """Interface for building adjacency lists."""
    @abstractmethod
    def build(self, phenotypes: List[Phenotype]) -> np.ndarray:
        """Build an adjacency list from the phenotypes."""
        pass

class GraphAssembler(ABC):
    """Interface for assembling graphs."""
    @abstractmethod
    def assemble(
        self,
        phenotypes: List[Phenotype],
        node_features: np.ndarray,
        edge_index: np.ndarray,
        global_context: Optional[np.ndarray] = None
    ) -> Graph:
        """Assemble a graph from components."""
        pass

class GlobalContextProvider(ABC):
    """Interface for providing global context."""
    @abstractmethod
    def provide_context(self, phenotypes: List[Phenotype]) -> np.ndarray:
        """Provide a global context vector based on phenotypes."""
        pass

class PhenotypeVisualizer(ABC):
    """Interface for visualizing phenotypes and phenotype graphs."""
    
    @abstractmethod
    def visualize_hierarchy(
        self, 
        phenotypes: List[Phenotype], 
        initial_phenotypes: Optional[List[Phenotype]] = None, 
        title: str = "Phenotype Hierarchy"
    ) -> Any:
        """
        Visualize phenotypes as a hierarchical structure.
        
        Args:
            phenotypes: List of all phenotypes to visualize
            initial_phenotypes: Original phenotypes (to highlight differently)
            title: Title for the visualization
            
        Returns:
            Implementation-specific visualization output
        """
        pass
        
    @abstractmethod
    def visualize_graph(
        self, 
        graph: Graph, 
        phenotypes: List[Phenotype], 
        title: str = "Phenotype Graph"
    ) -> Any:
        """
        Visualize a graph of phenotypes.
        
        Args:
            graph: Graph object to visualize
            phenotypes: List of phenotypes corresponding to graph nodes
            title: Title for the visualization
            
        Returns:
            Implementation-specific visualization output
        """
        pass

class PipelineOrchestrator(ABC):
    """Interface for orchestrating the pipeline."""
    @abstractmethod
    def build_graph(self, observed_phenotypes: List[str]) -> Graph:
        """Build a graph from observed phenotype IDs."""
        pass
