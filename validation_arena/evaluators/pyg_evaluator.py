import importlib
import json
import logging
import torch
from pathlib import Path
from typing import Any, Dict

from torch_geometric.data import Batch # Type hint for predict

# Assuming the training dataset class is accessible for metadata loading
# Adjust the import path as necessary for your project structure
try:
    from training.datasets.lmdb_hpo_dataset import LMDBHPOGraphDataset 
except ImportError:
    # Add fallback or error handling if the import path might vary
    logger = logging.getLogger(__name__)
    logger.warning("Could not import LMDBHPOGraphDataset from training.datasets. Adjust path if needed.")
    # Define a dummy class or raise error if essential
    LMDBHPOGraphDataset = None 

from .base_evaluator import BaseModelEvaluator

# Use standard Python logging
logger = logging.getLogger(__name__)

class PyGModelEvaluator(BaseModelEvaluator):
    """Evaluator for PyTorch Geometric models."""

    def __init__(self, config: Dict[str, Any], name: str):
        """
        Initializes the PyG model evaluator.

        Requires config keys:
        - model_class_fqn: Fully qualified name (e.g., 'training.models.models.GenePhenAIv2_0')
        - checkpoint_path: Path to the model's .pt checkpoint file.
        - training_args_path: Path to the training run's args.json.
        - training_lmdb_dir: Path to the LMDB dataset used for training (for gene list/num_classes).
        - device: 'cuda' or 'cpu'.
        - num_node_features: Integer number of input node features. (Required temporarily, might be inferred later)
        """
        super().__init__(config, name)
        self.model_class_fqn = config['model_class_fqn']
        self.checkpoint_path = Path(config['checkpoint_path'])
        self.training_args_path = Path(config['training_args_path'])
        self.training_lmdb_dir = Path(config['training_lmdb_dir'])
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        # TODO: Ideally, infer num_node_features from training_args or dataset metadata if possible
        self.num_node_features = config['num_node_features'] 
        
        self.model = None
        self.num_classes = None
        self.training_args = None # Store loaded training args

        # Validate required config keys
        required_keys = ['model_class_fqn', 'checkpoint_path', 'training_args_path', 'training_lmdb_dir', 'num_node_features']
        if not all(key in config for key in required_keys):
            missing = [key for key in required_keys if key not in config]
            raise ValueError(f"PyGModelEvaluator '{name}' config missing required keys: {missing}")

        logger.info(f"PyGModelEvaluator '{self.name}' initialized.")
        logger.debug(f" Config: model_class={self.model_class_fqn}, checkpoint={self.checkpoint_path}, "
                     f"train_args={self.training_args_path}, train_lmdb={self.training_lmdb_dir}, "
                     f"device={self.device}, node_features={self.num_node_features}")


    def load(self) -> None:
        """Loads the model, training args, and determines class mapping."""
        if self._is_loaded:
            logger.info(f"Model '{self.name}' is already loaded.")
            return

        logger.info(f"Loading model '{self.name}'...")
        load_start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else time.time()

        # --- 1. Load Training Config ---
        try:
            with open(self.training_args_path, 'r') as f:
                self.training_args = json.load(f)
            logger.info(f" Loaded training arguments from {self.training_args_path}")
            hidden_channels = self.training_args.get('hidden_channels')
            if hidden_channels is None:
                logger.error("Could not find 'hidden_channels' in loaded training arguments.")
                raise ValueError("'hidden_channels' not found in training args")
        except FileNotFoundError:
            logger.error(f"Training arguments file not found: {self.training_args_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading training arguments from {self.training_args_path}: {e}")
            raise

        # --- 2. Load Gene Mapping / Determine Num Classes ---
        try:
            if LMDBHPOGraphDataset is None:
                 raise RuntimeError("LMDBHPOGraphDataset class not available for loading metadata.")
            # Instantiate dataset just to access metadata
            logger.info(f" Loading gene list from training LMDB metadata: {self.training_lmdb_dir}")
            training_dataset_meta = LMDBHPOGraphDataset(root_dir=self.training_lmdb_dir, readonly=True)
            training_genes = training_dataset_meta.metadata.get('genes')
            if not training_genes:
                logger.error(f"Could not find 'genes' list in metadata of training LMDB at {self.training_lmdb_dir}.")
                raise ValueError("'genes' list not found in training dataset metadata")
            gene_to_class_idx = {gene: i for i, gene in enumerate(training_genes)}
            self.num_classes = len(gene_to_class_idx)
            logger.info(f" Determined {self.num_classes} output classes based on training gene list.")
            # We don't need to keep the dataset instance in memory
            del training_dataset_meta 
        except FileNotFoundError:
             logger.error(f"Training LMDB dataset not found at {self.training_lmdb_dir}.")
             raise
        except Exception as e:
            logger.error(f"Error loading metadata or creating gene map from {self.training_lmdb_dir}: {e}")
            raise

        # --- 3. Dynamically Import and Instantiate Model ---
        try:
            module_name, class_name = self.model_class_fqn.rsplit('.', 1)
            ModelClass = getattr(importlib.import_module(module_name), class_name)
            
            self.model = ModelClass(
                in_channels=self.num_node_features,
                hidden_channels=hidden_channels, 
                out_channels=self.num_classes
            ).to(self.device)
            logger.info(f" Instantiated model '{self.model_class_fqn}' on device {self.device}.")
        except (ImportError, AttributeError, Exception) as e:
            logger.error(f"Failed to import or instantiate model class '{self.model_class_fqn}': {e}")
            raise

        # --- 4. Load Checkpoint ---
        try:
            logger.info(f" Loading checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Handle potential variations in checkpoint structure
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                 model_state = checkpoint['state_dict']
            else:
                 # Assume the checkpoint *is* the state dict
                 model_state = checkpoint 
                 logger.warning("Checkpoint does not contain 'model_state_dict' key. Assuming the checkpoint *is* the state dict.")

            self.model.load_state_dict(model_state)
            epoch = checkpoint.get('epoch', 'N/A')
            batch = checkpoint.get('batch', 'N/A') # Or maybe 'step'? Check training script save format
            logger.info(f" Successfully loaded model weights from checkpoint (Epoch {epoch}, Batch/Step {batch})")
        except FileNotFoundError:
            logger.error(f"Checkpoint file not found: {self.checkpoint_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading checkpoint state dict from {self.checkpoint_path}: {e}")
            # Consider logging details about mismatched keys if relevant
            # Mismatched keys: e.g. logging missing/unexpected keys
            raise

        # --- 5. Finalize ---
        self.model.eval()
        self._is_loaded = True
        
        if self.device.type == 'cuda':
            load_end_time = torch.cuda.Event(enable_timing=True)
            load_end_time.record()
            torch.cuda.synchronize()
            elapsed_ms = load_start_time.elapsed_time(load_end_time)
            logger.info(f"Model '{self.name}' loading complete. Took {elapsed_ms / 1000.0:.2f} seconds.")
        else:
            elapsed_s = time.time() - load_start_time
            logger.info(f"Model '{self.name}' loading complete. Took {elapsed_s:.2f} seconds.")


    @torch.no_grad() # Ensure no gradients are computed during prediction
    def predict(self, batch_data: Batch) -> torch.Tensor:
        """
        Performs inference on a PyG Batch object.

        Args:
            batch_data: A torch_geometric.data.Batch object.

        Returns:
            torch.Tensor: The model's output logits/scores [batch_size, num_classes].
        
        Raises:
            RuntimeError: If the evaluator is not loaded.
            TypeError: If batch_data is not a PyG Batch object.
            Exception: If the forward pass fails.
        """
        if not self._is_loaded or self.model is None:
            logger.error(f"Predict called on '{self.name}' before model was loaded.")
            raise RuntimeError(f"Model evaluator '{self.name}' is not loaded.")
        
        if not isinstance(batch_data, Batch):
             logger.error(f"Predict expected a PyG Batch object, but received {type(batch_data)}.")
             raise TypeError(f"Input must be a torch_geometric.data.Batch object for PyGModelEvaluator")

        # Move data to the correct device
        batch_on_device = batch_data.to(self.device)
        logger.debug(f"Moved batch data to {self.device} for prediction.")

        try:
            # Ensure all required inputs for the specific model are present
            # This assumes a common signature: x, edge_index, batch vector
            # Adapt if your models require different inputs
            if not hasattr(batch_on_device, 'x') or \
               not hasattr(batch_on_device, 'edge_index') or \
               not hasattr(batch_on_device, 'batch'):
                 logger.error("Batch object missing required attributes (x, edge_index, batch) for prediction.")
                 raise ValueError("Batch object missing required attributes for model forward pass.")

            out = self.model(batch_on_device.x, batch_on_device.edge_index, batch_on_device.batch)
            logger.debug(f"Prediction successful for batch of size {batch_data.num_graphs}.")
            # Return tensor on CPU for consistent handling downstream
            return out.cpu() 
        except Exception as e:
            logger.error(f"Error during model forward pass for evaluator '{self.name}': {e}")
            # Potentially log batch details for debugging
            # logger.error(f"Batch details: {batch_data}") 
            raise # Re-raise the exception

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the loaded PyG model and its configuration."""
        if not self.training_args:
            logger.warning(f"Metadata requested for '{self.name}' before loading, training_args unavailable.")

        return {
            "evaluator_name": self.name,
            "evaluator_type": "PyGModelEvaluator",
            "model_class": self.model_class_fqn,
            "checkpoint_path": str(self.checkpoint_path),
            "training_args_path": str(self.training_args_path),
            "training_lmdb_dir": str(self.training_lmdb_dir),
            "device": str(self.device),
            "num_node_features": self.num_node_features,
            "num_output_classes": self.num_classes,
            "loaded_training_args": self.training_args if self.training_args else "Not loaded",
            "is_loaded": self._is_loaded
        }

    def unload(self) -> None:
        """Releases the model from memory."""
        if self.model is not None:
             logger.info(f"Unloading model '{self.name}' from device {self.device}.")
             del self.model # Remove reference
             self.model = None
             if self.device.type == 'cuda':
                 torch.cuda.empty_cache() # Clear GPU cache
                 logger.debug(f"Cleared CUDA cache for device {self.device}.")
        super().unload() # Sets _is_loaded to False

    # __enter__ and __exit__ are inherited from BaseModelEvaluator
    # They will call self.load() and self.unload() automatically

