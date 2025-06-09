from typing import Dict, Any, Optional
from src.core.interfaces import (
    AugmentationService, EmbeddingStrategy,
    AdjacencyListBuilder, GraphAssembler, GlobalContextProvider,
    PipelineOrchestrator, PhenotypeVisualizer
)
from src.augmentation.hpo_augmentation import HPOAugmentationService, APIAugmentationService
from src.embedding.strategies import (
    LLMEmbeddingStrategy, LookupEmbeddingStrategy, 
    HuggingFaceEmbeddingStrategy, SentenceTransformerEmbeddingStrategy,
    TFIDFEmbeddingStrategy, OpenAIEmbeddingStrategy, MemmapEmbeddingStrategy
)
from src.embedding.context import EmbeddingContext
from src.graph.adjacency import HPOAdjacencyListBuilder
from src.graph.assembler import StandardGraphAssembler
from src.context.global_context import AverageEmbeddingContextProvider, HPODAGContextProvider
from src.pipeline.orchestrator import StandardPipelineOrchestrator
from src.visualization.graphviz import GraphvizVisualizer
from src.ontology.hpo_graph import HPOGraphProvider
from src.ontology.hpo_updater import HPOUpdater

class ComponentFactory:
    """Factory for creating pipeline components."""
    
    @staticmethod
    def create_augmentation_service(config: Dict[str, Any]) -> AugmentationService:
        """Create an augmentation service based on configuration."""
        augmentation_type = config.get('type', 'local')
        
        if augmentation_type == 'local':
            return HPOAugmentationService(
                data_dir=config.get('data_dir', 'data/ontology'),
                include_ancestors=config.get('include_ancestors', True),
                include_descendants=config.get('include_descendants', False)
            )
        elif augmentation_type == 'api':
            return APIAugmentationService(
                api_base_url=config.get('api_base_url', 'https://ontology.jax.org/api/hp')
            )
        else:
            raise ValueError(f"Unknown augmentation service type: {augmentation_type}")
    
    @staticmethod
    def create_embedding_strategy(config: Dict[str, Any]) -> EmbeddingStrategy:
        """Create an embedding strategy based on configuration."""
        strategy_type = config.get('type', 'llm')
        
        if strategy_type == 'llm':
            return LLMEmbeddingStrategy(config.get('llm_service'))
        
        elif strategy_type == 'lookup':
            # Ensure embedding_dict is provided
            embedding_dict = config.get('embedding_dict')
            if embedding_dict is None:
                raise ValueError("For 'lookup' embedding strategy, 'embedding_dict' must be provided")
                
            return LookupEmbeddingStrategy(
                embedding_dict=embedding_dict,
                dim=config.get('dim', 768)
            )
            
        elif strategy_type == 'memmap':
            # For memory-mapped embeddings
            if 'memmap_data_path' in config and 'memmap_index_path' in config:
                # Explicit paths provided
                return MemmapEmbeddingStrategy(
                    memmap_data_path=config.get('memmap_data_path'),
                    memmap_index_path=config.get('memmap_index_path'),
                    dim=config.get('dim'),
                    dtype=config.get('dtype', 'float32')
                )
            else:
                # Use latest memmap file
                return MemmapEmbeddingStrategy.from_latest(
                    data_dir=config.get('data_dir', 'data/embeddings'),
                    dim=config.get('dim'),
                    dtype=config.get('dtype', 'float32')
                )
            
        elif strategy_type == 'huggingface':
            return HuggingFaceEmbeddingStrategy(
                model_name_or_path=config.get('model_name_or_path', 'bert-base-uncased'),
                max_length=config.get('max_length', 128),
                use_gpu=config.get('use_gpu', False),
                batch_size=config.get('batch_size', 32)
            )
            
        elif strategy_type == 'sentence_transformer':
            return SentenceTransformerEmbeddingStrategy(
                model_name=config.get('model_name', 'all-MiniLM-L6-v2'),
                batch_size=config.get('batch_size', 32)
            )
            
        elif strategy_type == 'tfidf':
            return TFIDFEmbeddingStrategy(
                max_features=config.get('max_features', 768),
                ngram_range=config.get('ngram_range', (1, 2))
            )
            
        elif strategy_type == 'openai':
            return OpenAIEmbeddingStrategy(
                model=config.get('model', 'text-embedding-3-small'),
                batch_size=config.get('batch_size', 32),
                api_key=config.get('api_key')
            )
            
        else:
            raise ValueError(f"Unknown embedding strategy type: {strategy_type}")
    
    @staticmethod
    def create_adjacency_builder(config: Dict[str, Any]) -> AdjacencyListBuilder:
        """Create an adjacency list builder based on configuration."""
        return HPOAdjacencyListBuilder(
            config.get('hpo_dag_provider'),
            include_reverse_edges=config.get('include_reverse_edges', True)
        )
    
    @staticmethod
    def create_graph_assembler(config: Dict[str, Any]) -> GraphAssembler:
        """Create a graph assembler based on configuration."""
        return StandardGraphAssembler(
            validate=config.get('validate', True)
        )
    
    @staticmethod
    def create_global_context_provider(config: Dict[str, Any]) -> Optional[GlobalContextProvider]:
        """Create a global context provider based on configuration."""
        provider_type = config.get('type')
        if not provider_type:
            return None
        
        if provider_type == 'average_embedding':
            return AverageEmbeddingContextProvider()
        elif provider_type == 'hpo_dag':
            return HPODAGContextProvider(config.get('hpo_context_provider'))
        else:
            raise ValueError(f"Unknown global context provider type: {provider_type}")
    
    @staticmethod
    def create_phenotype_visualizer(config: Dict[str, Any]) -> PhenotypeVisualizer:
        """Create a phenotype visualizer based on configuration."""
        visualizer_type = config.get('type', 'graphviz')
        
        if visualizer_type == 'graphviz':
            return GraphvizVisualizer(
                output_dir=config.get('output_dir', 'visualizations')
            )
        else:
            raise ValueError(f"Unknown phenotype visualizer type: {visualizer_type}")
    
    @classmethod
    def create_hpo_provider(cls, config: Dict[str, Any]) -> Any:
        """Create an HPO graph provider from configuration.
        
        Args:
            config: Dictionary with configuration options
            
        Returns:
            An initialized HPO graph provider
        """
        from src.ontology.hpo_graph import HPOGraphProvider
        
        data_dir = config.get("data_dir", "data/ontology")
        
        # Use the cached instance method instead of creating a new one
        return HPOGraphProvider.get_instance(data_dir=data_dir)
    
    @staticmethod
    def create_hpo_updater(config: Dict[str, Any]) -> HPOUpdater:
        """Create an HPO updater based on configuration."""
        return HPOUpdater(
            data_dir=config.get('data_dir', 'data/ontology'),
            check_interval_days=config.get('check_interval_days', 7),
            use_github=config.get('use_github', True)
        )
    
    @staticmethod
    def create_pipeline_orchestrator(config: Dict[str, Any]) -> PipelineOrchestrator:
        """Create a pipeline orchestrator with all necessary components."""
        # Get or create the HPO provider
        hpo_provider = config.get('hpo_provider')
        if not hpo_provider:
            hpo_provider = ComponentFactory.create_hpo_provider(config.get('hpo_config', {}))
            hpo_provider.load()
        
        augmentation_service = ComponentFactory.create_augmentation_service(config.get('augmentation', {}))
        embedding_strategy = ComponentFactory.create_embedding_strategy(config.get('embedding', {}))
        embedding_context = EmbeddingContext(embedding_strategy)
        adjacency_builder = ComponentFactory.create_adjacency_builder(config.get('adjacency', {}))
        graph_assembler = ComponentFactory.create_graph_assembler(config.get('assembler', {}))
        global_context_provider = ComponentFactory.create_global_context_provider(config.get('global_context', {}))
        
        # Create visualizer if specified in config
        visualizer = None
        if 'visualization' in config and config['visualization'] is not None:
            visualizer = ComponentFactory.create_phenotype_visualizer(config.get('visualization', {}))
        
        return StandardPipelineOrchestrator(
            hpo_provider,
            augmentation_service,
            embedding_context,
            adjacency_builder,
            graph_assembler,
            global_context_provider,
            visualizer
        )
