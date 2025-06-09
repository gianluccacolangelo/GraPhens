import numpy as np
from typing import List, Optional
from src.core.interfaces import GraphAssembler
from src.core.types import Phenotype, Graph
from src.graph.validation import IndexAlignmentChecker, ValidationError

"""
This module combines node features, edge connections, and optional global context into a 
complete graph representation.

Architecturally, it implements the GraphAssembler interface, handling the final step of 
graph construction before the graph is returned to the client.

For extensibility, the StandardGraphAssembler provides a clean implementation that could be 
extended to support additional graph attributes or different graph formats if needed.

In the bigger picture, this component creates the final data structure that encapsulates 
all the information generated throughout the pipeline, ready for use in graph-based machine 
learning models.
"""

class StandardGraphAssembler(GraphAssembler):
    """Standard implementation of the graph assembler."""
    
    def __init__(self, validate: bool = True):
        """
        Initialize the graph assembler.
        
        Args:
            validate: Whether to validate alignment between components (default: True)
        """
        self.validate = validate
    
    def assemble(
        self,
        phenotypes: List[Phenotype],
        node_features: np.ndarray,
        edge_index: np.ndarray,
        global_context: Optional[np.ndarray] = None
    ) -> Graph:
        """
        Assemble a graph from components.
        
        Args:
            phenotypes: List of phenotypes that correspond to nodes
            node_features: Node feature matrix with shape (num_nodes, feature_dim)
            edge_index: Edge connectivity in COO format with shape (2, num_edges)
            global_context: Optional global context vector
            
        Returns:
            Assembled Graph object
            
        Raises:
            ValidationError: If validation is enabled and components are misaligned
        """
        # Optionally validate component alignment
        if self.validate:
            IndexAlignmentChecker.check_components(phenotypes, node_features, edge_index)
        
        # Create the node mapping dictionary
        node_mapping = {p.id: i for i, p in enumerate(phenotypes)}
        
        # Assemble the graph
        graph = Graph(
            node_features=node_features,
            edge_index=edge_index,
            node_mapping=node_mapping,
            global_context=global_context
        )
        
        # Optionally validate the assembled graph
        if self.validate:
            IndexAlignmentChecker.check_graph(graph, phenotypes)
        
        return graph
