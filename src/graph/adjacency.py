import numpy as np
from typing import List, Dict, Set
from src.core.interfaces import AdjacencyListBuilder
from src.core.types import Phenotype
from src.ontology.hpo_graph import HPOGraphProvider

"""
This module builds adjacency lists that represent the connections between phenotypes 
based on the HPO DAG structure.

Architecturally, it implements the AdjacencyListBuilder interface, separating the logic 
of edge creation from the rest of the pipeline.

For extensibility, the implementation is parameterized to include or exclude reverse edges, 
and the helper method _collect_edges encapsulates the specific logic for gathering edges, 
making it easier to extend or modify.

In the bigger picture, this component translates the hierarchical relationships within the 
HPO ontology into a graph structure that can be processed by graph-based algorithms.
"""

class HPOAdjacencyListBuilder(AdjacencyListBuilder):
    """Builds adjacency lists using the HPO DAG structure."""
    
    def __init__(self, hpo_graph_provider: HPOGraphProvider, include_reverse_edges=True):
        """
        Initialize with an HPO graph provider.
        
        Args:
            hpo_graph_provider: Provider for accessing the HPO directed acyclic graph
            include_reverse_edges: Whether to include edges in both directions (default: True)
        """
        self.hpo_graph = hpo_graph_provider
        self.include_reverse_edges = include_reverse_edges
    
    def build(self, phenotypes: List[Phenotype]) -> np.ndarray:
        """
        Build an adjacency list from the phenotypes.
        
        Args:
            phenotypes: List of phenotypes to build the adjacency list for
            
        Returns:
            A 2xE edge index tensor representing the graph structure
        """
        # Create a mapping from HPO IDs to indices
        id_to_idx = {p.id: i for i, p in enumerate(phenotypes)}
        edges = self._collect_edges(phenotypes, id_to_idx)
        
        # Convert to a 2xE edge index tensor
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        
        return np.array(edges, dtype=np.int64).T
    
    def _get_direct_parents(self, term_id: str) -> Set[str]:
        """
        Get direct parent terms of a given HPO term.
        
        Args:
            term_id: HPO term ID
            
        Returns:
            Set of HPO term IDs representing direct parents
        """
        # First ensure the graph is loaded
        if not self.hpo_graph.load():
            return set()
            
        # Normalize the input ID
        term_id = self.hpo_graph._normalize_hpo_id(term_id)
        
        if term_id not in self.hpo_graph.graph:
            return set()
            
        # Get direct parents (successors in the graph since edges go from child to parent)
        return set(self.hpo_graph.graph.successors(term_id))
    
    def _collect_edges(self, phenotypes: List[Phenotype], id_to_idx: Dict[str, int]) -> List:
        """
        Collect edges based on HPO relationships.
        
        Args:
            phenotypes: List of phenotypes to process
            id_to_idx: Mapping from HPO IDs to indices
            
        Returns:
            List of edges as (source, target) tuples
        """
        edges = []
        for i, phenotype in enumerate(phenotypes):
            # Get direct parent terms ("is_a" relationships)
            parents = self._get_direct_parents(phenotype.id)
            for parent in parents:
                if parent in id_to_idx:  # Only include if parent is in our node set
                    # Add edge from child to parent (source -> target)
                    edges.append((i, id_to_idx[parent]))
                    
                    # Add reverse edge if configured
                    if self.include_reverse_edges:
                        edges.append((id_to_idx[parent], i))
        return edges
