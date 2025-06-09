import os
import logging
from typing import List, Optional, Any, Set, Union
import numpy as np

from src.core.interfaces import PhenotypeVisualizer
from src.core.types import Phenotype, Graph
from src.ontology.hpo_graph import HPOGraphProvider

class GraphvizVisualizer(PhenotypeVisualizer):
    """
    Implementation of PhenotypeVisualizer using Graphviz.
    
    This visualizer creates hierarchical directed graphs showing the relationships
    between phenotypes. It can visualize both plain phenotype lists (using the HPO 
    hierarchy) and full Graph objects (from the graph assembler).
    
    It distinguishes between initial phenotypes and those added through augmentation
    by using different colors.
    """
    
    def __init__(self, output_dir: str = ".", hpo_provider: Optional[HPOGraphProvider] = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory where visualization files will be saved
            hpo_provider: Optional HPOGraphProvider for hierarchy information
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.hpo_provider = hpo_provider
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_hierarchy(
        self, 
        phenotypes: List[Phenotype], 
        initial_phenotypes: Optional[List[Phenotype]] = None, 
        title: str = "Phenotype Hierarchy",
        hpo_provider: Optional[HPOGraphProvider] = None
    ) -> str:
        """
        Visualize phenotype hierarchy using Graphviz.
        
        Args:
            phenotypes: List of all phenotypes to visualize
            initial_phenotypes: Original phenotypes before augmentation (to highlight)
            title: Title for the visualization
            hpo_provider: HPOGraphProvider for hierarchy information (optional)
            
        Returns:
            Path to the generated image file
        """
        try:
            import graphviz
        except ImportError:
            self.logger.error("Graphviz not installed. Please install graphviz package.")
            return None
        
        # Use provided hpo_provider or fall back to the one from initialization
        provider = hpo_provider or self.hpo_provider
        if not provider:
            self.logger.warning("No HPO provider available. Will use metadata if available or skip edges.")
            
        # Ensure provider is loaded if available
        if provider:
            provider.load()
            
        # Create a new directed graph
        dot = graphviz.Digraph(title, filename=os.path.join(self.output_dir, title.replace(' ', '_').lower()))
        dot.attr(rankdir='BT')  # Bottom to Top direction
        
        # Get set of initial phenotype IDs for highlighting
        initial_ids: Set[str] = set()
        if initial_phenotypes:
            initial_ids = {p.id for p in initial_phenotypes}
        
        # Create a mapping from phenotype ID to index
        id_to_index = {p.id: i for i, p in enumerate(phenotypes)}
        
        # Add nodes with styling
        for i, phenotype in enumerate(phenotypes):
            # Highlight initial phenotypes in a different color
            is_initial = phenotype.id in initial_ids
            color = 'lightpink' if is_initial else 'lightblue2'
            
            # Create label with both name and ID
            label = f"{phenotype.name}\\n({phenotype.id})"
            
            dot.node(str(i), label, style='filled', color=color, shape='box')
        
        # Add edges
        edges_added = 0
        for i, source_phenotype in enumerate(phenotypes):
            # Get parent IDs
            parent_ids = []
            
            # Try to get parents from HPO provider first
            if provider:
                parent_ids = set(provider.graph.successors(source_phenotype.id))
            
            # Fall back to metadata if available
            elif hasattr(source_phenotype, 'metadata') and source_phenotype.metadata:
                parent_ids = source_phenotype.metadata.get('parents', [])
            
            # Add edges for each parent that's in our phenotype list
            for parent_id in parent_ids:
                if parent_id in id_to_index:
                    parent_idx = id_to_index[parent_id]
                    dot.edge(str(i), str(parent_idx))
                    edges_added += 1
        
        self.logger.info(f"Added {edges_added} edges to the hierarchy visualization")
        
        # Save and render the graph
        output_path = dot.render(format='png', cleanup=True)
        self.logger.info(f"Hierarchical graph visualization saved to '{output_path}'")
        
        return output_path
    
    def visualize_graph(
        self, 
        graph: Graph, 
        phenotypes: List[Phenotype], 
        title: str = "Phenotype Graph",
        initial_phenotypes: Optional[List[Phenotype]] = None
    ) -> str:
        """
        Visualize a graph of phenotypes using Graphviz.
        
        Args:
            graph: Graph object to visualize
            phenotypes: List of phenotypes corresponding to graph nodes
            title: Title for the visualization
            initial_phenotypes: Original phenotypes to highlight differently
            
        Returns:
            Path to the generated image file
        """
        try:
            import graphviz
        except ImportError:
            self.logger.error("Graphviz not installed. Please install graphviz package.")
            return None
        
        # Create a new directed graph
        dot = graphviz.Digraph(title, filename=os.path.join(self.output_dir, title.replace(' ', '_').lower()))
        dot.attr(rankdir='BT')  # Bottom to Top direction
        
        # Get the original node mapping from the graph
        node_mapping = graph.node_mapping
        
        # Get set of initial phenotype IDs for highlighting
        initial_ids: Set[str] = set()
        if initial_phenotypes:
            initial_ids = {p.id for p in initial_phenotypes}
        
        # Add nodes with styling
        for i, phenotype in enumerate(phenotypes):
            # Create label with both name and ID
            label = f"{phenotype.name}\\n({phenotype.id})"
            
            # Check if this is an initial phenotype
            is_initial = phenotype.id in initial_ids
            color = 'lightpink' if is_initial else 'lightblue2'
            
            dot.node(str(i), label, style='filled', color=color, shape='box')
        
        # Add edges from the graph
        edge_index = graph.edge_index
        for i in range(edge_index.shape[1]):
            src = str(edge_index[0, i])
            tgt = str(edge_index[1, i])
            dot.edge(src, tgt)
        
        # Save and render the graph
        output_path = dot.render(format='png', cleanup=True)
        self.logger.info(f"Graph visualization saved to '{output_path}'")
        
        return output_path
        
    def visualize_augmentation_result(
        self,
        augmented_phenotypes: List[Phenotype],
        initial_phenotypes: List[Phenotype],
        title: str = "Phenotype Augmentation Result",
        hpo_provider: Optional[HPOGraphProvider] = None
    ) -> str:
        """
        Convenience method to visualize the results of phenotype augmentation.
        
        Args:
            augmented_phenotypes: The full list of phenotypes after augmentation
            initial_phenotypes: The original list of phenotypes before augmentation
            title: Title for the visualization
            hpo_provider: Optional HPOGraphProvider for hierarchy information
            
        Returns:
            Path to the generated image file
        """
        return self.visualize_hierarchy(
            phenotypes=augmented_phenotypes,
            initial_phenotypes=initial_phenotypes,
            title=title,
            hpo_provider=hpo_provider
        ) 