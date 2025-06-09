import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from src.core.types import Phenotype, Graph

"""
This module provides validation utilities for the graph construction process.

It ensures data integrity by verifying alignment between the different components
of a graph, such as node features, edge indices, and phenotype lists.

The IndexAlignmentChecker class in particular validates that the indices used in
edge connections properly correspond to the indices in the node feature matrix,
which is critical for correct graph neural network operations.
"""

class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass

class IndexAlignmentChecker:
    """
    Validates alignment between node features, edge indices, and phenotypes.
    
    This checker ensures that the indices used in the edge_index correctly refer
    to the corresponding positions in the node_features matrix, and that these
    align with the phenotype list used to generate them.
    """
    
    @staticmethod
    def check_components(
        phenotypes: List[Phenotype],
        node_features: np.ndarray,
        edge_index: np.ndarray,
        original_phenotypes: Optional[List[Phenotype]] = None
    ) -> bool:
        """
        Verify alignment between components before assembly into a graph.
        
        Args:
            phenotypes: List of phenotypes
            node_features: Node feature matrix
            edge_index: Edge index tensor (2xE)
            original_phenotypes: If provided, checks if phenotype order has changed
            
        Returns:
            True if components are properly aligned
            
        Raises:
            ValidationError: If validation fails with details about the misalignment
        """
        # Check that we have the same number of phenotypes and node features
        if len(phenotypes) != node_features.shape[0]:
            raise ValidationError(
                f"Number of phenotypes ({len(phenotypes)}) doesn't match "
                f"number of node features ({node_features.shape[0]})"
            )
        
        # Get the maximum index referenced in edge_index
        if edge_index.size > 0:  # Only check if there are edges
            max_idx = np.max(edge_index)
            if max_idx >= len(phenotypes):
                raise ValidationError(
                    f"Edge index contains index {max_idx} which is out of bounds "
                    f"for phenotype list of length {len(phenotypes)}"
                )
        
        # Check that edge indices are non-negative
        if edge_index.size > 0 and np.min(edge_index) < 0:
            raise ValidationError(
                f"Edge index contains negative index {np.min(edge_index)}"
            )
            
        # Check if phenotype order has changed, if original_phenotypes is provided
        if original_phenotypes is not None and edge_index.size > 0:
            if len(original_phenotypes) != len(phenotypes):
                raise ValidationError(
                    f"Number of original phenotypes ({len(original_phenotypes)}) "
                    f"doesn't match current phenotypes ({len(phenotypes)})"
                )
            
            # Check if the order has changed by comparing phenotype IDs
            if [p.id for p in original_phenotypes] != [p.id for p in phenotypes]:
                # Get mappings from ID to index for both lists
                original_id_to_idx = {p.id: i for i, p in enumerate(original_phenotypes)}
                current_id_to_idx = {p.id: i for i, p in enumerate(phenotypes)}
                
                # Find a phenotype that has changed position
                for p_id, orig_idx in original_id_to_idx.items():
                    if p_id in current_id_to_idx and current_id_to_idx[p_id] != orig_idx:
                        raise ValidationError(
                            f"Phenotype order has changed. Phenotype {p_id} was at "
                            f"index {orig_idx} but is now at index {current_id_to_idx[p_id]}. "
                            f"Edge indices no longer correspond to the correct phenotypes."
                        )
        
        return True
    
    @staticmethod
    def check_consistent_phenotype_order(
        original_phenotypes: List[Phenotype],
        current_phenotypes: List[Phenotype]
    ) -> bool:
        """
        Check if the phenotype order is consistent between two phenotype lists.
        
        Args:
            original_phenotypes: Original phenotype list
            current_phenotypes: Current phenotype list
            
        Returns:
            True if the phenotype order is consistent
            
        Raises:
            ValidationError: If the phenotype order has changed
        """
        # Quick check: if the lists are identical, no need to continue
        if original_phenotypes == current_phenotypes:
            return True
        
        # Check if the lists have the same length
        if len(original_phenotypes) != len(current_phenotypes):
            raise ValidationError(
                f"Number of phenotypes doesn't match: original has {len(original_phenotypes)}, "
                f"current has {len(current_phenotypes)}"
            )
        
        # Check if the order has changed
        for i, (orig_p, curr_p) in enumerate(zip(original_phenotypes, current_phenotypes)):
            if orig_p.id != curr_p.id:
                raise ValidationError(
                    f"Phenotype order has changed at index {i}: "
                    f"original has {orig_p.id} ({orig_p.name}), "
                    f"current has {curr_p.id} ({curr_p.name})"
                )
        
        return True
    
    @staticmethod
    def check_graph(graph: Graph, phenotypes: List[Phenotype]) -> bool:
        """
        Verify alignment in an assembled graph object.
        
        Args:
            graph: Assembled graph object
            phenotypes: List of phenotypes used to create the graph
            
        Returns:
            True if graph is properly aligned
            
        Raises:
            ValidationError: If validation fails with details about the misalignment
        """
        # Check node count
        if len(phenotypes) != graph.node_features.shape[0]:
            raise ValidationError(
                f"Number of phenotypes ({len(phenotypes)}) doesn't match "
                f"number of node features in graph ({graph.node_features.shape[0]})"
            )
        
        # Check that all phenotype IDs are in the node_mapping
        for i, phenotype in enumerate(phenotypes):
            if phenotype.id not in graph.node_mapping:
                raise ValidationError(
                    f"Phenotype {phenotype.id} is missing from graph.node_mapping"
                )
            
            # Check that the mapping is correct
            if graph.node_mapping[phenotype.id] != i:
                raise ValidationError(
                    f"Phenotype {phenotype.id} is at index {i} in phenotype list "
                    f"but mapped to index {graph.node_mapping[phenotype.id]} in graph.node_mapping"
                )
        
        # Check edge indices
        if graph.edge_index.size > 0:
            max_idx = np.max(graph.edge_index)
            if max_idx >= graph.node_features.shape[0]:
                raise ValidationError(
                    f"Edge index in graph contains index {max_idx} which is out of bounds "
                    f"for node feature matrix of shape {graph.node_features.shape}"
                )
        
        return True
    
    @staticmethod
    def validate_embedding_adjacency_alignment(
        phenotypes: List[Phenotype],
        node_features: np.ndarray,
        edge_index: np.ndarray,
        original_phenotypes: Optional[List[Phenotype]] = None
    ) -> Tuple[bool, str]:
        """
        Comprehensive validation of alignment between embeddings and adjacency list.
        
        This is a more user-friendly method that returns a tuple of (success, message)
        rather than raising exceptions.
        
        Args:
            phenotypes: List of phenotypes
            node_features: Node feature matrix
            edge_index: Edge index tensor (2xE)
            original_phenotypes: If provided, checks if phenotype order has changed
            
        Returns:
            Tuple of (is_valid, message) where is_valid is a boolean and message is a
            description of the validation result or error
        """
        try:
            IndexAlignmentChecker.check_components(
                phenotypes, node_features, edge_index, original_phenotypes
            )
            return True, "Validation successful: node features and edge indices are properly aligned"
        except ValidationError as e:
            return False, str(e) 