from typing import List, Optional
from src.core.interfaces import (
    AugmentationService, 
    AdjacencyListBuilder, GraphAssembler, GlobalContextProvider,
    PipelineOrchestrator, PhenotypeVisualizer
)
from src.core.types import Graph, Phenotype
from src.embedding.context import EmbeddingContext
from src.ontology.hpo_graph import HPOGraphProvider

class StandardPipelineOrchestrator(PipelineOrchestrator):
    """Standard implementation of the pipeline orchestrator (Facade pattern)."""
    
    def __init__(
        self,
        hpo_provider: HPOGraphProvider,
        augmentation_service: AugmentationService,
        embedding_context: EmbeddingContext,
        adjacency_builder: AdjacencyListBuilder,
        graph_assembler: GraphAssembler,
        global_context_provider: Optional[GlobalContextProvider] = None,
        visualizer: Optional[PhenotypeVisualizer] = None
    ):
        """Initialize with all required components."""
        self.hpo_provider = hpo_provider
        self.augmentation_service = augmentation_service
        self.embedding_context = embedding_context
        self.adjacency_builder = adjacency_builder
        self.graph_assembler = graph_assembler
        self.global_context_provider = global_context_provider
        self.visualizer = visualizer
    
    def build_graph(self, observed_phenotype_ids: List[str]) -> Graph:
        """Build a graph from observed phenotype IDs."""
        # Load phenotypes directly using the HPO provider
        observed_phenotypes = [self._create_phenotype(hpo_id) for hpo_id in observed_phenotype_ids]
        
        # Augment phenotypes
        augmented_phenotypes = self.augmentation_service.augment(observed_phenotypes)
        
        # Embed phenotypes and build graph structure
        node_features = self.embedding_context.embed_phenotypes(augmented_phenotypes)
        edge_index = self.adjacency_builder.build(augmented_phenotypes)
        
        # Generate global context if provider exists
        global_context = None
        if self.global_context_provider:
            global_context = self.global_context_provider.provide_context(augmented_phenotypes)
        
        # Assemble and return the final graph
        return self.graph_assembler.assemble(
            augmented_phenotypes,
            node_features,
            edge_index,
            global_context
        )
    
    def _create_phenotype(self, hpo_id: str) -> Phenotype:
        """Create a phenotype object from an HPO ID."""
        metadata = self.hpo_provider.get_metadata(hpo_id)
        return Phenotype(
            id=hpo_id,
            name=metadata.get('name', ''),
            description=metadata.get('definition', None),
            metadata=metadata
        )
