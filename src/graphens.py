"""
GraPhens: An Intuitive Facade for Phenotype-Based Graph Neural Networks

This module provides a user-friendly interface to the GraPhens system, 
following principles of intuitive design:

1. Self-explanatory interfaces - Methods have clear, meaningful names
2. Progressive disclosure - Simple by default, customizable when needed
3. Sensible defaults - Common configurations pre-selected
4. Meaningful feedback - Clear status updates and helpful error messages
5. Embedded guidance - Help and explanations built into the interface

Usage examples:
    
    # Quick start with defaults
    graphens = GraPhens()
    graph = graphens.create_graph_from_phenotypes(["HP:0001250", "HP:0001251"])
    
    # Configure with specific embedding model
    graphens = GraPhens().with_embedding_model("sentence-transformer", "all-MiniLM-L6-v2")
    graph = graphens.create_graph_from_phenotypes(["HP:0001250", "HP:0001251"])
    
    # Full customization with a fluent interface
    graphens = (GraPhens()
                .with_embedding_model("openai", "text-embedding-3-small")
                .with_augmentation(include_ancestors=True, include_descendants=False)
                .with_visualization(enabled=True, output_dir="visualizations"))
    
    # Process phenotypes with progress feedback
    graph = graphens.create_graph_from_phenotypes(
        ["HP:0001250", "HP:0001251"], 
        show_progress=True
    )
"""

import os
import json
from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# Add YAML support
import yaml

from src.core.types import Phenotype, Graph
from src.factory import ComponentFactory

class GraPhens:
    """
    A user-friendly facade for the GraPhens phenotype graph system.
    
    This class simplifies access to the system's functionality through
    an intuitive interface with sensible defaults.
    """
    
    def __init__(self, 
                data_dir: str = "data/ontology", 
                use_default_embeddings: bool = True,
                config_path: Optional[str] = None):
        """
        Initialize GraPhens with sensible defaults or from a configuration file.
        
        Args:
            data_dir: Directory where ontology data is stored
            use_default_embeddings: Automatically use the default pre-computed embeddings
            config_path: Path to a YAML configuration file (overrides other parameters)
        """
        # Initialize default configuration
        self.config = {
            "augmentation": {
                "type": "local",
                "data_dir": data_dir,
                "include_ancestors": True,
                "include_descendants": False
            },
            "embedding": {
                "type": "sentence_transformer",
                "model_name": "all-MiniLM-L6-v2",
                "batch_size": 32
            },
            "adjacency": {
                "include_reverse_edges": True
            },
            "assembler": {
                "validate": True
            },
            "global_context": {},  # No global context by default
            "visualization": None  # Visualization disabled by default
        }
        
        # Store the data directory for later use
        self.data_dir = data_dir
        
        # Initialize logger - IMPORTANT: Initialize logger before loading config
        self.logger = logging.getLogger("graphens")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        
        # If config path is provided, load from YAML
        if config_path:
            self.load_config_from_yaml(config_path)
            # Update data_dir from config after loading
            self.data_dir = self.config.get("augmentation", {}).get("data_dir", data_dir)
        
        # Ensure HPO data exists
        self._ensure_data_available(self.data_dir)
        
        # Internal state to track if HPO provider has been initialized
        self._hpo_provider = None
        
        # Cache for the orchestrator to avoid rebuilding it
        self._orchestrator_cache = None
        self._config_hash = None
        
        # Automatically use pre-computed embeddings if requested
        if use_default_embeddings:
            try:
                # First try to use memory-mapped embeddings for better performance
                from src.embedding.vector_db.memmap import list_memmap_files
                embeddings_dir = "data/embeddings"
                memmap_files = list_memmap_files(embeddings_dir)
                
                if memmap_files:
                    self.with_memmap_embeddings()
                    self.logger.info("Using memory-mapped embeddings for optimal performance")
                else:
                    # Fall back to regular embeddings if no memory-mapped files available
                    self.with_pretrained_embeddings()
                    self.logger.info(
                        "Using default pre-computed embeddings. For better performance, convert to memory-mapped format "
                        "with: python -m src.embedding.vector_db.scripts.convert_to_memmap --all"
                    )
            except Exception as e:
                self.logger.warning(f"Could not load default embeddings: {str(e)}. Using on-the-fly embeddings instead.")
    
    def load_config_from_yaml(self, config_path: str) -> "GraPhens":
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Self for method chaining
            
        Examples:
            graphens = GraPhens().load_config_from_yaml("config/custom_config.yaml")
        """
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                
            # Process the YAML configuration and update the internal config
            if yaml_config.get("data"):
                data_config = yaml_config["data"]
                # Update data directories
                if "ontology_dir" in data_config:
                    if "augmentation" not in self.config:
                        self.config["augmentation"] = {}
                    self.config["augmentation"]["data_dir"] = data_config["ontology_dir"]
            
            # Process augmentation settings
            if yaml_config.get("augmentation"):
                aug_config = yaml_config["augmentation"]
                if "augmentation" not in self.config:
                    self.config["augmentation"] = {}
                
                if "type" in aug_config:
                    self.config["augmentation"]["type"] = aug_config["type"]
                if "include_ancestors" in aug_config:
                    self.config["augmentation"]["include_ancestors"] = aug_config["include_ancestors"]
                if "include_descendants" in aug_config:
                    self.config["augmentation"]["include_descendants"] = aug_config["include_descendants"]
                if "api_base_url" in aug_config and aug_config["type"] == "api":
                    self.config["augmentation"]["api_base_url"] = aug_config["api_base_url"]
            
            # Process embedding settings
            if yaml_config.get("embedding"):
                emb_config = yaml_config["embedding"]
                emb_type = emb_config.get("type", "sentence_transformer")
                
                # Set the embedding type
                self.config["embedding"] = {"type": emb_type}
                
                # Add type-specific settings
                if emb_type == "memmap":
                    # Use with_memmap_embeddings later
                    pass
                elif emb_type == "lookup":
                    # Use with_pretrained_embeddings later
                    self.config["embedding"]["embedding_file"] = emb_config.get("lookup", {}).get("embedding_file")
                elif emb_type == "sentence_transformer":
                    st_config = emb_config.get("sentence_transformer", {})
                    self.config["embedding"]["model_name"] = st_config.get("model_name", "all-MiniLM-L6-v2")
                    self.config["embedding"]["batch_size"] = st_config.get("batch_size", 32)
                elif emb_type == "huggingface":
                    hf_config = emb_config.get("huggingface", {})
                    self.config["embedding"]["model_name_or_path"] = hf_config.get("model_name_or_path")
                    self.config["embedding"]["max_length"] = hf_config.get("max_length", 128)
                    self.config["embedding"]["use_gpu"] = hf_config.get("use_gpu", False)
                    self.config["embedding"]["batch_size"] = hf_config.get("batch_size", 32)
                elif emb_type == "tfidf":
                    tfidf_config = emb_config.get("tfidf", {})
                    self.config["embedding"]["max_features"] = tfidf_config.get("max_features", 768)
                    self.config["embedding"]["ngram_range"] = tfidf_config.get("ngram_range", [1, 2])
                elif emb_type == "openai":
                    openai_config = emb_config.get("openai", {})
                    self.config["embedding"]["model"] = openai_config.get("model", "text-embedding-3-small")
                    api_key = openai_config.get("api_key", "")
                    if not api_key:
                        api_key = os.environ.get("OPENAI_API_KEY", "")
                    self.config["embedding"]["api_key"] = api_key
            
            # Process graph settings
            if yaml_config.get("graph"):
                graph_config = yaml_config["graph"]
                
                # Adjacency settings
                if "adjacency" in graph_config:
                    self.config["adjacency"] = {
                        "include_reverse_edges": graph_config["adjacency"].get("include_reverse_edges", True)
                    }
                
                # Assembler settings
                if "assembler" in graph_config:
                    self.config["assembler"] = {
                        "validate": graph_config["assembler"].get("validate", True)
                    }
                
                # Global context settings
                if "global_context" in graph_config:
                    gc_config = graph_config["global_context"]
                    if gc_config.get("type"):
                        self.config["global_context"] = {
                            "type": gc_config["type"],
                            "include_root": gc_config.get("include_root", False)
                        }
                    else:
                        self.config["global_context"] = {}
            
            # Process visualization settings
            if yaml_config.get("visualization"):
                vis_config = yaml_config["visualization"]
                if vis_config.get("enabled", False):
                    self.config["visualization"] = {
                        "type": vis_config.get("type", "graphviz"),
                        "format": vis_config.get("format", "png"),
                        "output_dir": vis_config.get("output_dir", "visualizations"),
                        "limit_nodes": vis_config.get("limit_nodes", 100)
                    }
                else:
                    self.config["visualization"] = None
            
            # Apply embedding strategy based on the loaded configuration
            emb_type = self.config["embedding"].get("type")
            if emb_type == "memmap":
                self.with_memmap_embeddings()
            elif emb_type == "lookup":
                embedding_file = self.config["embedding"].get("embedding_file")
                self.with_pretrained_embeddings(embedding_file)
            
            self.logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            self.logger.error("Using default configuration instead")
        
        return self
    
    def save_config_to_yaml(self, config_path: str) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            config_path: Path where the configuration should be saved
            
        Examples:
            graphens.save_config_to_yaml("config/my_custom_config.yaml")
        """
        # Make sure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Convert internal config to YAML-friendly format
        yaml_config = {
            "data": {
                "ontology_dir": self.config.get("augmentation", {}).get("data_dir", "data/ontology"),
                "embeddings_dir": "data/embeddings",
                "visualizations_dir": self.config.get("visualization", {}).get("output_dir", "visualizations")
            },
            "augmentation": {
                "type": self.config.get("augmentation", {}).get("type", "local"),
                "include_ancestors": self.config.get("augmentation", {}).get("include_ancestors", True),
                "include_descendants": self.config.get("augmentation", {}).get("include_descendants", False)
            },
            "embedding": {
                "type": self.config.get("embedding", {}).get("type", "sentence_transformer")
            },
            "graph": {
                "adjacency": {
                    "include_reverse_edges": self.config.get("adjacency", {}).get("include_reverse_edges", True)
                },
                "assembler": {
                    "validate": self.config.get("assembler", {}).get("validate", True)
                }
            }
        }
        
        # Add embedding-specific settings
        emb_type = yaml_config["embedding"]["type"]
        if emb_type == "sentence_transformer":
            yaml_config["embedding"]["sentence_transformer"] = {
                "model_name": self.config.get("embedding", {}).get("model_name", "all-MiniLM-L6-v2"),
                "batch_size": self.config.get("embedding", {}).get("batch_size", 32)
            }
        elif emb_type == "huggingface":
            yaml_config["embedding"]["huggingface"] = {
                "model_name_or_path": self.config.get("embedding", {}).get("model_name_or_path"),
                "max_length": self.config.get("embedding", {}).get("max_length", 128),
                "use_gpu": self.config.get("embedding", {}).get("use_gpu", False),
                "batch_size": self.config.get("embedding", {}).get("batch_size", 32)
            }
        
        # Add global context settings if present
        if self.config.get("global_context", {}).get("type"):
            yaml_config["graph"]["global_context"] = {
                "type": self.config["global_context"]["type"],
                "include_root": self.config["global_context"].get("include_root", False)
            }
        else:
            yaml_config["graph"]["global_context"] = {"type": None}
        
        # Add visualization settings if present
        if self.config.get("visualization"):
            yaml_config["visualization"] = {
                "enabled": True,
                "type": self.config["visualization"].get("type", "graphviz"),
                "format": self.config["visualization"].get("format", "png"),
                "output_dir": self.config["visualization"].get("output_dir", "visualizations"),
                "limit_nodes": self.config["visualization"].get("limit_nodes", 100)
            }
        else:
            yaml_config["visualization"] = {"enabled": False}
        
        # Save YAML config
        with open(config_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
            
        self.logger.info(f"Configuration saved to {config_path}")
    
    def _ensure_data_available(self, data_dir: str) -> None:
        """
        Ensure that HPO data is available, downloading if needed.
        
        Args:
            data_dir: Directory where ontology data should be stored
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            self.logger.info(f"Creating data directory {data_dir}")
            data_path.mkdir(parents=True, exist_ok=True)
        
        # Check if HPO data files exist (.obo or .json format)
        obo_file = data_path / "hp.obo"
        json_file = data_path / "hp.json"
        
        # Download data if neither file exists
        if not obo_file.exists() and not json_file.exists():
            self.logger.info("HPO data not found. Downloading...")
            # Create an HPOUpdater to download the data
            updater_config = {"data_dir": data_dir, "use_github": True}
            updater = ComponentFactory.create_hpo_updater(updater_config)
            updater.update_if_needed(force=True)
            self.logger.info("HPO data downloaded successfully")
    
    def _get_hpo_provider(self):
        """
        Get or create the HPO provider.
        
        Returns:
            An initialized HPO graph provider
        """
        if self._hpo_provider is None:
            self._hpo_provider = ComponentFactory.create_hpo_provider({
                "data_dir": self.config["augmentation"]["data_dir"]
            })
            self._hpo_provider.load()
        return self._hpo_provider
    
    def _build_orchestrator(self):
        """
        Build the pipeline orchestrator with current configuration.
        
        Returns:
            A configured pipeline orchestrator
        """
        # Check if we can reuse the cached orchestrator
        # Create a hash of serializable parts of the config
        config_hash_dict = {k: v for k, v in self.config.items() 
                          if k not in ["hpo_provider", "adjacency", "global_context"]}
        
        # Handle special case for adjacency and global_context which may contain HPOGraphProvider
        if "adjacency" in self.config:
            adj_config = {k: v for k, v in self.config["adjacency"].items() 
                         if k != "hpo_dag_provider"}
            config_hash_dict["adjacency"] = adj_config
        
        if "global_context" in self.config and isinstance(self.config["global_context"], dict):
            global_config = {k: v for k, v in self.config["global_context"].items() 
                           if k != "hpo_context_provider"}
            config_hash_dict["global_context"] = global_config
        
        # Create a serializable representation of the config
        # Convert any numpy arrays or non-serializable objects to strings
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return f"ndarray:shape={obj.shape}:dtype={obj.dtype}"
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        serializable_config = make_serializable(config_hash_dict)
        current_config_hash = hash(json.dumps(serializable_config, sort_keys=True))
        
        if self._orchestrator_cache and current_config_hash == self._config_hash:
            return self._orchestrator_cache
            
        # Set up HPO provider for components that need it
        hpo_provider = self._get_hpo_provider()
        
        # Pass the HPO provider directly to the pipeline orchestrator
        self.config["hpo_provider"] = hpo_provider
        
        # Configure other components that need the HPO provider
        self.config["adjacency"]["hpo_dag_provider"] = hpo_provider
        
        if "type" in self.config.get("global_context", {}):
            if self.config["global_context"]["type"] == "hpo_dag":
                self.config["global_context"]["hpo_context_provider"] = hpo_provider
        
        # Create the orchestrator
        self._orchestrator_cache = ComponentFactory.create_pipeline_orchestrator(self.config)
        self._config_hash = current_config_hash
        
        return self._orchestrator_cache
    
    def with_embedding_model(self, embedding_type: str, model_name: Optional[str] = None, **kwargs) -> "GraPhens":
        """
        Configure the embedding model.
        
        Args:
            embedding_type: Type of embedding ('sentence_transformer', 'huggingface', 'openai', 'tfidf', etc.)
            model_name: Name of the model to use (if applicable)
            **kwargs: Additional configuration options
            
        Returns:
            Self for method chaining
        
        Examples:
            # Use Sentence Transformers
            graphens.with_embedding_model("sentence_transformer", "all-MiniLM-L6-v2")
            
            # Use HuggingFace model
            graphens.with_embedding_model("huggingface", "pritamdeka/S-PubMedBert-MS-MARCO")
            
            # Use OpenAI embeddings
            graphens.with_embedding_model("openai", "text-embedding-3-small", api_key="your-key")
            
            # Use TF-IDF with custom parameters
            graphens.with_embedding_model("tfidf", max_features=512, ngram_range=(1, 3))
        """
        self.config["embedding"] = {"type": embedding_type, **kwargs}
        if model_name:
            if embedding_type == "sentence_transformer":
                self.config["embedding"]["model_name"] = model_name
            elif embedding_type == "huggingface":
                self.config["embedding"]["model_name_or_path"] = model_name
            elif embedding_type == "openai":
                self.config["embedding"]["model"] = model_name
        
        return self
    
    def with_pretrained_embeddings(self, model_name: str = None) -> "GraPhens":
        """
        Configure the system to use pre-computed embeddings from the embeddings directory.
        
        This is a convenience method that makes it easy to use pre-computed embeddings
        without having to specify the full path or format.
        
        Args:
            model_name: Name of the model to use embeddings from:
                - "all-MiniLM-L6-v2" - General purpose embeddings
                - "pritamdeka/S-PubMedBert-MS-MARCO" - Biomedical embeddings
                - "gsarti/biobert-nli" - Biomedical NLI embeddings
                If None, uses the default "all-MiniLM-L6-v2" embeddings.
            
        Returns:
            Self for method chaining
            
        Examples:
            # Use default embeddings
            graphens = graphens.with_pretrained_embeddings()
            
            # Use PubMedBERT embeddings
            graphens = graphens.with_pretrained_embeddings("pritamdeka/S-PubMedBert-MS-MARCO")
        """
        import pickle
        import os
        
        # Default to all-MiniLM-L6-v2 if no model specified
        if model_name is None:
            model_name = "gsarti_biobert-nli"
        
        # Log which model we're using from config
        self.logger.info(f"Using embedding model: {model_name}")
        
        # Clean model name for file lookup
        clean_model_name = model_name.replace("/", "_")
        
        # Define embeddings directory
        embeddings_dir = "data/embeddings"
        
        # Search for matching embedding file
        found_file = None
        
        # First, try exact match with model name pattern
        for filename in os.listdir(embeddings_dir):
            if filename.endswith(".pkl") and not filename.endswith("_metadata.pkl"):
                if clean_model_name in filename:
                    found_file = os.path.join(embeddings_dir, filename)
                    break
        
        # If not found, try default patterns
        if not found_file:
            model_mapping = {
                "all-MiniLM-L6-v2": "hpo_embeddings_all-MiniLM-L6-v2",
                "pritamdeka/S-PubMedBert-MS-MARCO": "hpo_embeddings_pritamdeka_S-PubMedBert-MS-MARCO",
                "gsarti/biobert-nli": "hpo_embeddings_gsarti_biobert-nli"
            }
            
            if model_name in model_mapping:
                prefix = model_mapping[model_name]
                for filename in os.listdir(embeddings_dir):
                    if filename.startswith(prefix) and filename.endswith(".pkl") and not filename.endswith("_metadata.pkl"):
                        found_file = os.path.join(embeddings_dir, filename)
                        break
        
        if not found_file:
            raise ValueError(f"Could not find embedding file for model '{model_name}'. Available models: all-MiniLM-L6-v2, pritamdeka/S-PubMedBert-MS-MARCO, gsarti/biobert-nli")
        
        # After finding the file, add more detailed logging
        self.logger.info(f"Selected embedding file: {found_file}")
        
        # Load the embeddings
        self.logger.info(f"Loading embeddings from {found_file}")
        try:
            with open(found_file, 'rb') as f:
                embedding_dict = pickle.load(f)
            
            self.logger.info(f"Loaded embeddings for {len(embedding_dict)} phenotypes")
            
            # Get dimensionality from the first embedding
            first_key = next(iter(embedding_dict))
            dim = len(embedding_dict[first_key])
            
            # Configure the embedding model
            return self.with_embedding_model("lookup", embedding_dict=embedding_dict, dim=dim)
            
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {str(e)}")
            raise
    
    def with_memmap_embeddings(self, model_name: Optional[str] = None, data_dir: str = "data/embeddings") -> "GraPhens":
        """
        Configure the system to use memory-mapped embeddings.
        
        This is a high-performance alternative to with_pretrained_embeddings() that uses
        memory-mapped files for efficient access, significantly reducing memory usage
        and startup time.
        
        Args:
            model_name: Name of the model to use embeddings from (same options as with_pretrained_embeddings)
            data_dir: Directory containing embedding files
            
        Returns:
            Self for method chaining
            
        Examples:
            # Use default memory-mapped embeddings (latest file)
            graphens = graphens.with_memmap_embeddings()
            
            # Use specific model memory-mapped embeddings
            graphens = graphens.with_memmap_embeddings("pritamdeka/S-PubMedBert-MS-MARCO")
        """
        import os
        import json
        
        # If model_name is specified, look for a specific memory-mapped file
        if model_name:
            # Clean model name for file lookup
            clean_model_name = model_name.replace("/", "_")
            
            # Search for matching memory-mapped file
            memmap_data_path = None
            
            # First, try exact match with model name pattern
            for filename in os.listdir(data_dir):
                if filename.endswith(".mmap"):
                    if clean_model_name in filename:
                        memmap_data_path = os.path.join(data_dir, filename)
                        memmap_index_path = memmap_data_path.replace(".mmap", ".index.json")
                        if os.path.exists(memmap_index_path):
                            break
            
            # If not found, try to find it by checking the standard model names
            if not memmap_data_path:
                model_mapping = {
                    "all-MiniLM-L6-v2": "hpo_embeddings_all-MiniLM-L6-v2",
                    "pritamdeka/S-PubMedBert-MS-MARCO": "hpo_embeddings_pritamdeka_S-PubMedBert-MS-MARCO",
                    "gsarti/biobert-nli": "hpo_embeddings_gsarti_biobert-nli"
                }
                
                if model_name in model_mapping:
                    prefix = model_mapping[model_name]
                    for filename in os.listdir(data_dir):
                        if filename.startswith(prefix) and filename.endswith(".mmap"):
                            memmap_data_path = os.path.join(data_dir, filename)
                            memmap_index_path = memmap_data_path.replace(".mmap", ".index.json")
                            if os.path.exists(memmap_index_path):
                                break
            
            if not memmap_data_path:
                # If no memory-mapped file for this model, suggest conversion
                pkl_file = None
                for filename in os.listdir(data_dir):
                    if filename.endswith(".pkl") and not filename.endswith("_metadata.pkl"):
                        if clean_model_name in filename:
                            pkl_file = os.path.join(data_dir, filename)
                            break
                
                if pkl_file:
                    self.logger.warning(
                        f"No memory-mapped file found for model '{model_name}', but pickle file exists. "
                        f"Convert it with: python -m src.embedding.vector_db.scripts.convert_to_memmap --input {pkl_file}"
                    )
                    
                    # Fall back to pickle-based embeddings
                    return self.with_pretrained_embeddings(model_name)
                else:
                    raise ValueError(f"Could not find memory-mapped or pickle file for model '{model_name}'")
            
            # Extract model name from filename for clearer logging
            model_identifier = os.path.basename(memmap_data_path).split('_')[2:] 
            if len(model_identifier) >= 1:
                model_display_name = '_'.join(model_identifier).replace('.mmap', '')
                self.logger.info(f"Using embedding model: {model_display_name}")
            
            self.logger.info(f"Using memory-mapped embeddings from {memmap_data_path}")
            self.config["embedding"] = {
                "type": "memmap",
                "memmap_data_path": memmap_data_path,
                "memmap_index_path": memmap_index_path
            }
            
        else:
            # Use the latest memory-mapped file
            latest_mmap_file = None
            latest_timestamp = 0
            
            for filename in os.listdir(data_dir):
                if filename.endswith(".mmap"):
                    file_path = os.path.join(data_dir, filename)
                    timestamp = os.path.getmtime(file_path)
                    
                    if timestamp > latest_timestamp:
                        index_path = file_path.replace(".mmap", ".index.json")
                        if os.path.exists(index_path):
                            latest_mmap_file = file_path
                            latest_timestamp = timestamp
            
            if latest_mmap_file:
                # Extract model name from filename for clearer logging
                model_identifier = os.path.basename(latest_mmap_file).split('_')[2:] 
                if len(model_identifier) >= 1:
                    model_display_name = '_'.join(model_identifier).replace('.mmap', '')
                    self.logger.info(f"Using embedding model: {model_display_name}")
                    
                self.logger.info(f"Using latest memory-mapped embeddings: {latest_mmap_file}")
                
                self.config["embedding"] = {
                    "type": "memmap",
                    "memmap_data_path": latest_mmap_file,
                    "memmap_index_path": latest_mmap_file.replace(".mmap", ".index.json")
                }
            else:
                self.logger.warning(
                    "No memory-mapped embedding files found. Using default embeddings instead. "
                    "For optimal performance, convert embeddings with: "
                    "python -m src.embedding.vector_db.scripts.convert_to_memmap --all"
                )
                # Fall back to regular embeddings
                return self.with_pretrained_embeddings()
            
        return self
    
    def with_augmentation(self, 
                         strategy: str = "local", 
                         **kwargs) -> "GraPhens":
        """
        Configure the phenotype augmentation process.
        
        Args:
            strategy: The augmentation strategy to use. 
                      Options: 'local', 'api', 'siblings', 'n_hop'.
                      Defaults to 'local'.
            **kwargs: Additional configuration options for the chosen strategy.
                      For 'local': include_ancestors (bool), include_descendants (bool).
                      For 'api': api_base_url (str).
                      For 'n_hop': n_hops (int).
                      For backward compatibility, `use_api=True` is also accepted.
            
        Returns:
            Self for method chaining
            
        Examples:
            # Configure with ancestors (old way still works)
            graphens.with_augmentation(include_ancestors=True, include_descendants=False)
            
            # Use siblings augmentation
            graphens.with_augmentation(strategy="siblings")
            
            # Use N-hop augmentation
            graphens.with_augmentation(strategy="n_hop", n_hops=2)
            
            # Use API-based augmentation
            graphens.with_augmentation(strategy="api", api_base_url="https://example.org/api")
        """
        # For backward compatibility with `use_api=True`
        if kwargs.get("use_api"):
            effective_strategy = "api"
        else:
            effective_strategy = strategy
            
        # Get the existing data_dir or use the default
        data_dir = self.config.get("augmentation", {}).get("data_dir", self.data_dir)
        
        # Build the new configuration
        self.config["augmentation"] = {
            "type": effective_strategy,
            "data_dir": data_dir,
            **kwargs
        }
        
        # Preserve default behavior for 'local' strategy when called via old method signature
        if effective_strategy == 'local':
            if 'include_ancestors' not in self.config['augmentation']:
                self.config['augmentation']['include_ancestors'] = True
            if 'include_descendants' not in self.config['augmentation']:
                self.config['augmentation']['include_descendants'] = False
        
        return self
    
    def with_visualization(self, 
                          enabled: bool = True, 
                          output_dir: str = "visualizations", 
                          **kwargs) -> "GraPhens":
        """
        Configure visualization of phenotypes and graphs.
        
        Args:
            enabled: Whether visualization is enabled
            output_dir: Directory where visualizations should be saved
            **kwargs: Additional visualization options
            
        Returns:
            Self for method chaining
            
        Examples:
            # Enable visualization with default settings
            graphens.with_visualization(enabled=True)
            
            # Customize output directory
            graphens.with_visualization(output_dir="my_visualizations")
            
            # Disable visualization
            graphens.with_visualization(enabled=False)
        """
        if enabled:
            self.config["visualization"] = {
                "type": "graphviz",
                "output_dir": output_dir,
                **kwargs
            }
            
            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.config["visualization"] = None
        
        return self
    
    def with_global_context(self, context_type: Optional[str] = None, **kwargs) -> "GraPhens":
        """
        Configure global context for the graph.
        
        Args:
            context_type: Type of global context ('average_embedding', 'hpo_dag', or None)
            **kwargs: Additional configuration options
            
        Returns:
            Self for method chaining
            
        Examples:
            # Use average embedding context
            graphens.with_global_context("average_embedding")
            
            # Use HPO DAG context
            graphens.with_global_context("hpo_dag")
            
            # Disable global context
            graphens.with_global_context(None)
        """
        if context_type:
            self.config["global_context"] = {"type": context_type, **kwargs}
        else:
            self.config["global_context"] = {}
        
        return self
    
    def with_adjacency_settings(self, include_reverse_edges: bool = True, **kwargs) -> "GraPhens":
        """
        Configure adjacency settings for graph construction.
        
        Args:
            include_reverse_edges: Whether to include reverse edges in the graph
            **kwargs: Additional configuration options
            
        Returns:
            Self for method chaining
            
        Examples:
            # Include reverse edges
            graphens.with_adjacency_settings(include_reverse_edges=True)
            
            # Only use forward edges
            graphens.with_adjacency_settings(include_reverse_edges=False)
        """
        self.config["adjacency"] = {
            "include_reverse_edges": include_reverse_edges,
            **kwargs
        }
        
        return self
    
    def with_config_from_file(self, config_path: str) -> "GraPhens":
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Self for method chaining
            
        Examples:
            graphens.with_config_from_file("config/my_config.json")
        """
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            
        # Update config with file values, keeping defaults for unspecified options
        for section, values in file_config.items():
            if isinstance(values, dict):
                if section in self.config and isinstance(self.config[section], dict):
                    self.config[section].update(values)
                else:
                    self.config[section] = values
            else:
                self.config[section] = values
        
        return self
    
    def save_config(self, config_path: str) -> None:
        """
        Save the current configuration to a JSON file.
        
        Args:
            config_path: Path where the configuration should be saved
            
        Examples:
            graphens.save_config("config/my_custom_config.json")
        """
        # Make sure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
        self.logger.info(f"Configuration saved to {config_path}")
    
    def create_graph_from_phenotypes(self, 
                                    phenotype_ids: List[str], 
                                    show_progress: bool = False) -> Graph:
        """
        Create a graph from a list of phenotype IDs.
        
        This is the main entry point for graph creation. It processes the phenotypes
        through the entire pipeline and returns a graph suitable for GNN processing.
        
        Args:
            phenotype_ids: List of HPO IDs (e.g., ["HP:0001250", "HP:0001251"])
            show_progress: Whether to show progress indicators
            
        Returns:
            A Graph object ready for GNN processing
            
        Examples:
            # Create a graph from seizure-related phenotypes
            graph = graphens.create_graph_from_phenotypes([
                "HP:0001250",  # Seizure
                "HP:0001251"   # Atonic seizure
            ])
            
            # Create a graph with progress feedback
            graph = graphens.create_graph_from_phenotypes(
                phenotype_ids=["HP:0001250", "HP:0001251"],
                show_progress=True
            )
        """
        # Validate phenotype IDs
        self._validate_phenotype_ids(phenotype_ids)
        
        # Build orchestrator with current configuration (will reuse cached one if possible)
        orchestrator = self._build_orchestrator()
        
        if show_progress:
            self.logger.info(f"Creating graph from {len(phenotype_ids)} phenotypes")
            
            # Process each step with progress indicators
            # Note: In a real implementation, we would modify the orchestrator
            # to provide step-by-step progress. This is a simplified version.
            with tqdm(total=5, desc="Processing phenotypes") as pbar:
                pbar.set_description("Loading phenotypes")
                # Call the orchestrator's build_graph method
                graph = orchestrator.build_graph(phenotype_ids)
                pbar.update(5)  # Complete all steps
                
            self.logger.info(f"Graph created with {len(graph.node_mapping)} nodes and {graph.edge_index.shape[1]} edges")
        else:
            # Simply call the orchestrator without progress indicators
            graph = orchestrator.build_graph(phenotype_ids)
        
        return graph
    
    def _validate_phenotype_ids(self, phenotype_ids: List[str]) -> None:
        """
        Validate phenotype IDs to ensure they are properly formatted.
        
        Args:
            phenotype_ids: List of phenotype IDs to validate
            
        Raises:
            ValueError: If any ID is improperly formatted
        """
        for pid in phenotype_ids:
            if not pid.startswith("HP:") or not pid[3:].isdigit():
                raise ValueError(
                    f"Invalid phenotype ID: {pid}. Must be in format 'HP:0000000'. "
                    f"See https://hpo.jax.org/ for valid IDs."
                )
    
    def phenotype_lookup(self, term: str) -> List[Phenotype]:
        """
        Look up phenotype terms by name, description, or ID.
        
        This helper method finds phenotypes matching a search term,
        making it easier to discover relevant phenotype IDs.
        
        Args:
            term: Search term to find matching phenotypes
            
        Returns:
            List of matching phenotype objects
            
        Examples:
            # Search for seizure-related phenotypes
            seizure_phenotypes = graphens.phenotype_lookup("seizure")
            
            # Use the IDs from the search results
            graph = graphens.create_graph_from_phenotypes([p.id for p in seizure_phenotypes])
        """
        # Get the HPO provider
        hpo_provider = self._get_hpo_provider()
        
        # Search for matching terms
        matching_phenotypes = []
        search_term = term.lower()
        
        # HPOGraphProvider stores terms in its terms dictionary
        for phenotype_id, phenotype_data in hpo_provider.terms.items():
            # Check if the search term is in the ID, name, or definition
            name = phenotype_data.get("name", "").lower()
            definition = phenotype_data.get("definition", "") or ""
            definition = definition.lower()
            
            if (search_term in phenotype_id.lower() or 
                search_term in name or 
                search_term in definition):
                
                matching_phenotypes.append(Phenotype(
                    id=phenotype_id,
                    name=phenotype_data.get("name", ""),
                    description=phenotype_data.get("definition")
                ))
        
        return matching_phenotypes
    
    def visualize(self, 
                 graph: Graph, 
                 phenotypes: List[Phenotype],
                 initial_phenotypes: Optional[List[Phenotype]] = None,
                 output_path: Optional[str] = None,
                 title: str = "Phenotype Graph") -> Optional[str]:
        """
        Visualize a phenotype graph.
        
        Args:
            graph: The graph to visualize
            phenotypes: List of phenotypes in the graph
            initial_phenotypes: Original phenotypes before augmentation (highlighted differently)
            output_path: Path where the visualization should be saved
            title: Title for the visualization
            
        Returns:
            Path to the generated visualization file, if applicable
            
        Examples:
            # Create and visualize a graph
            graph = graphens.create_graph_from_phenotypes(["HP:0001250", "HP:0001251"])
            graphens.visualize(
                graph=graph, 
                phenotypes=graph.metadata["phenotypes"],
                title="Seizure Phenotype Graph"
            )
        """
        # Check if visualization is enabled
        if not self.config.get("visualization"):
            self.logger.warning("Visualization is not enabled. Use with_visualization() to enable it.")
            return None
        
        # Create visualizer
        visualizer = ComponentFactory.create_phenotype_visualizer(self.config["visualization"])
        
        # Set output path if provided
        if output_path:
            visualizer.output_dir = output_path
            
        # Visualize the graph
        return visualizer.visualize_graph(graph, phenotypes, title)
    
    def export_graph(self, graph: Union[Graph, Dict[str, Graph]], format: str = "pytorch", output_path: Optional[str] = None, batch: bool = False) -> Any:
        """
        Export a graph or dictionary of graphs to a format suitable for machine learning frameworks.
        
        Args:
            graph: The graph to export, or a dictionary mapping patient IDs to graphs
            format: Target format ('pytorch', 'tensorflow', 'networkx', 'numpy', 'json')
            output_path: Path where the exported graph should be saved (if applicable)
            batch: Whether to combine multiple graphs into a single batched graph (for formats that support it)
            
        Returns:
            The exported graph(s) in the requested format. If a dictionary of graphs was provided:
              - With batch=True: Returns a single batched graph (for supported formats)
              - With batch=False: Returns a dictionary of exported graphs
            
        Examples:
            # Export a single graph to PyTorch Geometric
            import torch_geometric as pyg
            pyg_graph = graphens.export_graph(graph, format="pytorch")
            
            # Export multiple graphs as a dictionary
            patient_graphs = graphens.create_graphs_from_multiple_patients(patient_data)
            exported_graphs = graphens.export_graph(patient_graphs, format="pytorch", batch=False)
            
            # Export multiple graphs as a single batched graph
            patient_graphs = graphens.create_graphs_from_multiple_patients(patient_data)
            batched_graph = graphens.export_graph(patient_graphs, format="pytorch", batch=True)
        """
        # Check if we have a single graph or a dictionary of graphs
        is_dict_of_graphs = isinstance(graph, dict) and all(isinstance(g, Graph) for g in graph.values())
        
        # Handle a single graph case
        if not is_dict_of_graphs:
            return self._export_single_graph(graph, format, output_path)
        
        # Handle dictionary of graphs
        if batch and format in ["pytorch", "tensorflow"]:
            # Batch the graphs into a single graph
            return self._export_batch_graphs(graph, format, output_path)
        else:
            # Process each graph separately
            result = {}
            for patient_id, patient_graph in graph.items():
                result[patient_id] = self._export_single_graph(patient_graph, format, 
                                                             output_path=f"{output_path}_{patient_id}" if output_path else None)
            return result

    def _export_single_graph(self, graph: Graph, format: str, output_path: Optional[str] = None) -> Any:
        """Export a single graph to the specified format."""
        # This contains the original export_graph implementation
        if format == "pytorch":
            try:
                import torch
                from torch_geometric.data import Data
                
                # Convert to PyTorch tensors
                x = torch.tensor(graph.node_features, dtype=torch.float)
                edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
                
                # Create PyG Data object
                pyg_graph = Data(x=x, edge_index=edge_index)
                
                # Add metadata
                pyg_graph.node_mapping = graph.node_mapping
                
                return pyg_graph
            except ImportError:
                raise ImportError("PyTorch and PyTorch Geometric are required for 'pytorch' format.")
        
        elif format == "tensorflow":
            try:
                import tensorflow as tf
                
                # Return as tensors
                return {
                    'node_features': tf.convert_to_tensor(graph.node_features, dtype=tf.float32),
                    'edge_index': tf.convert_to_tensor(graph.edge_index, dtype=tf.int32),
                    'node_mapping': graph.node_mapping
                }
            except ImportError:
                raise ImportError("TensorFlow is required for 'tensorflow' format.")
                
        elif format == "networkx":
            try:
                import networkx as nx
                
                # Create a directed graph
                G = nx.DiGraph()
                
                # Add nodes with features
                for node_id, idx in graph.node_mapping.items():
                    G.add_node(node_id, features=graph.node_features[idx])
                
                # Add edges
                for i in range(graph.edge_index.shape[1]):
                    src_idx = graph.edge_index[0, i]
                    dst_idx = graph.edge_index[1, i]
                    
                    # Convert indices to node IDs
                    src_id = next(k for k, v in graph.node_mapping.items() if v == src_idx)
                    dst_id = next(k for k, v in graph.node_mapping.items() if v == dst_idx)
                    
                    G.add_edge(src_id, dst_id)
                
                return G
            except ImportError:
                raise ImportError("NetworkX is required for 'networkx' format.")
                
        elif format == "numpy":
            # Just return the raw components
            return {
                'node_features': graph.node_features,
                'edge_index': graph.edge_index,
                'node_mapping': graph.node_mapping
            }
            
        elif format == "json":
            # Convert to JSON-serializable format
            json_graph = {
                'nodes': [],
                'edges': []
            }
            
            # Add nodes
            for node_id, idx in graph.node_mapping.items():
                json_graph['nodes'].append({
                    'id': node_id,
                    'features': graph.node_features[idx].tolist()
                })
            
            # Add edges
            for i in range(graph.edge_index.shape[1]):
                src_idx = graph.edge_index[0, i]
                dst_idx = graph.edge_index[1, i]
                
                # Convert indices to node IDs
                src_id = next(k for k, v in graph.node_mapping.items() if v == src_idx)
                dst_id = next(k for k, v in graph.node_mapping.items() if v == dst_idx)
                
                json_graph['edges'].append({
                    'source': src_id,
                    'target': dst_id
                })
            
            # Save to file if output_path is provided
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(json_graph, f, indent=2)
                self.logger.info(f"Graph exported to {output_path}")
            
            return json_graph
            
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_batch_graphs(self, graphs: Dict[str, Graph], format: str, output_path: Optional[str] = None) -> Any:
        """Export multiple graphs as a single batched graph for efficiency."""
        if format == "pytorch":
            try:
                import torch
                from torch_geometric.data import Data, Batch
                
                # Convert each graph to a PyG Data object
                pyg_graphs = []
                patient_ids = []
                node_mappings = {}
                
                for patient_id, graph in graphs.items():
                    # Convert to PyTorch tensors
                    x = torch.tensor(graph.node_features, dtype=torch.float)
                    edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
                    
                    # Create PyG Data object
                    pyg_graph = Data(x=x, edge_index=edge_index)
                    
                    # Store patient ID as additional attribute (as a tensor)
                    pyg_graph.patient_id = patient_id
                    
                    # Don't add node_mapping to PyG Data object - PyG can't batch dictionaries properly
                    # Store it separately instead
                    
                    pyg_graphs.append(pyg_graph)
                    patient_ids.append(patient_id)
                    node_mappings[patient_id] = graph.node_mapping
                
                # Batch all graphs into a single graph
                batched_graph = Batch.from_data_list(pyg_graphs)
                
                # Store patient IDs and node mappings as separate attributes
                batched_graph.patient_ids = patient_ids
                batched_graph.node_mappings = node_mappings
                
                return batched_graph
            
            except ImportError:
                raise ImportError("PyTorch and PyTorch Geometric are required for 'pytorch' format.")
        
        elif format == "tensorflow":
            try:
                import tensorflow as tf
                import numpy as np
                
                # Determine the maximum feature dimensions
                max_nodes = max(g.node_features.shape[0] for g in graphs.values())
                feature_dim = next(iter(graphs.values())).node_features.shape[1]
                
                # Initialize batch arrays
                batch_size = len(graphs)
                batch_features = np.zeros((batch_size, max_nodes, feature_dim), dtype=np.float32)
                batch_masks = np.zeros((batch_size, max_nodes), dtype=np.float32)
                batch_edge_indices = []
                batch_metadata = {}
                
                # Fill the arrays
                for i, (patient_id, graph) in enumerate(graphs.items()):
                    # Node features and mask
                    num_nodes = graph.node_features.shape[0]
                    batch_features[i, :num_nodes, :] = graph.node_features
                    batch_masks[i, :num_nodes] = 1.0
                    
                    # Edge indices with offset
                    edges = graph.edge_index.copy()
                    batch_edge_indices.append(edges)
                    
                    # Metadata
                    batch_metadata[patient_id] = {
                        'node_mapping': graph.node_mapping,
                        'num_nodes': num_nodes
                    }
                
                # Convert to TF tensors
                return {
                    'node_features': tf.convert_to_tensor(batch_features, dtype=tf.float32),
                    'masks': tf.convert_to_tensor(batch_masks, dtype=tf.float32),
                    'edge_indices': batch_edge_indices,  # List of edge indices per graph
                    'metadata': batch_metadata
                }
            
            except ImportError:
                raise ImportError("TensorFlow is required for 'tensorflow' format.")
        
        else:
            raise ValueError(f"Batch export not supported for format: {format}. Use batch=False instead.")
    
    def with_lookup_embeddings(self, embeddings_path: str, dim: int = 768) -> "GraPhens":
        """
        Configure the system to use pre-computed embeddings from a file.
        
        This is a convenience method that loads embeddings from a pickle file
        and configures the lookup embedding strategy.
        
        Note: For easier usage with standard models, consider using `with_pretrained_embeddings()` 
        instead, which automatically finds the right embedding file.
        
        Args:
            embeddings_path: Path to a pickle file containing embeddings dict
            dim: Dimensionality of the embeddings (default: 768)
            
        Returns:
            Self for method chaining
            
        Examples:
            # Use pre-computed embeddings
            graphens = graphens.with_lookup_embeddings("data/embeddings/hpo_embeddings.pkl")
            
            # Use with custom dimension
            graphens = graphens.with_lookup_embeddings(
                "data/biobert_embeddings.pkl", 
                dim=1024
            )
        """
        import pickle
        
        # Load the embeddings from the file
        self.logger.info(f"Loading embeddings from {embeddings_path}")
        try:
            with open(embeddings_path, 'rb') as f:
                embedding_dict = pickle.load(f)
            
            self.logger.info(f"Loaded embeddings for {len(embedding_dict)} phenotypes")
            
            # Configure the embedding model with the loaded dictionary
            return self.with_embedding_model("lookup", embedding_dict=embedding_dict, dim=dim)
            
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {str(e)}")
            raise
    
    @staticmethod
    def run_on_cluster(script_path: str, node: str = "nodo8"):
        """
        Run a script on the cluster using the deployment script.
        
        This is a convenience method for testing, following the rule in howtorunanything.mdc.
        Not meant to be included in production code.
        
        Args:
            script_path: Path to the script to run
            node: Cluster node to run on (default: nodo8)
            
        Examples:
            # Run a script on the cluster
            GraPhens.run_on_cluster("my_script.py")
            
            # Run on a specific node
            GraPhens.run_on_cluster("my_script.py", node="nodo9")
        """
        import subprocess
        import os
        
        # Make sure script path is relative or absolute
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script {script_path} not found")
        
        # Build the command
        command = f"./deploy_and_run.sh \"python3 {script_path}\" {node}"
        
        # Run the command
        print(f"Running command: {command}")
        subprocess.run(command, shell=True)

    def create_graphs_from_multiple_patients(self, 
                                          patient_phenotypes: Dict[str, List[str]],
                                          show_progress: bool = False) -> Dict[str, Graph]:
        """
        Process multiple patients' phenotypes efficiently, creating a graph for each.
        
        This method avoids rebuilding the orchestrator for each patient, making it
        significantly more efficient for batch processing compared to calling
        create_graph_from_phenotypes separately for each patient.
        
        Args:
            patient_phenotypes: Dictionary mapping patient IDs to lists of phenotype IDs
            show_progress: Whether to show a progress bar during processing
            
        Returns:
            Dictionary mapping patient IDs to their respective graphs
            
        Examples:
            # Process multiple patients at once
            patient_data = {
                "patient1": ["HP:0001250", "HP:0001251"],
                "patient2": ["HP:0002015", "HP:0002017"],
                "patient3": ["HP:0000252", "HP:0000316"]
            }
            graphs = graphens.create_graphs_from_multiple_patients(patient_data, show_progress=True)
            
            # Access individual patient graphs
            patient1_graph = graphs["patient1"]
        """
        # Build orchestrator once for all patients
        orchestrator = self._build_orchestrator()
        
        # Create a dictionary to hold results
        result_graphs = {}
        
        # Process each patient
        if show_progress:
            self.logger.info(f"Processing {len(patient_phenotypes)} patients")
            
            for patient_id, phenotype_ids in tqdm(patient_phenotypes.items(), desc="Processing patients"):
                # Validate phenotype IDs
                try:
                    self._validate_phenotype_ids(phenotype_ids)
                    
                    # Create graph for this patient
                    graph = orchestrator.build_graph(phenotype_ids)
                    result_graphs[patient_id] = graph
                    
                    # Log detailed information when in verbose mode
                    self.logger.debug(
                        f"Created graph for patient {patient_id} with "
                        f"{len(graph.node_mapping)} nodes and {graph.edge_index.shape[1]} edges"
                    )
                except Exception as e:
                    self.logger.error(f"Error processing patient {patient_id}: {str(e)}")
        else:
            # Process without progress bar
            for patient_id, phenotype_ids in patient_phenotypes.items():
                try:
                    self._validate_phenotype_ids(phenotype_ids)
                    result_graphs[patient_id] = orchestrator.build_graph(phenotype_ids)
                except Exception as e:
                    self.logger.error(f"Error processing patient {patient_id}: {str(e)}")
        
        # Log summary
        self.logger.info(f"Successfully processed {len(result_graphs)} out of {len(patient_phenotypes)} patients")
        
        return result_graphs 