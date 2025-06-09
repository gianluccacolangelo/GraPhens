import abc
import logging
import torch
from typing import Any, Dict, Optional

# Use standard Python logging
logger = logging.getLogger(__name__)

class BaseModelEvaluator(abc.ABC):
    """Abstract Base Class for model evaluators in the validation arena."""

    def __init__(self, config: Dict[str, Any], name: str):
        """
        Initializes the evaluator with its specific configuration and name.

        Args:
            config: Dictionary containing evaluator-specific settings 
                    (e.g., paths, hyperparameters, image names).
            name: A unique name for this model instance being evaluated.
        """
        self.config = config
        self.name = name
        self._is_loaded = False
        logger.debug(f"BaseModelEvaluator '{self.name}' initialized with config: {config}")

    @abc.abstractmethod
    def load(self) -> None:
        """
        Loads the model resources. 
        This could involve loading weights, starting a container, connecting to an API, etc.
        Sets the internal state to loaded upon successful completion.
        
        Raises:
            Exception: If loading fails.
        """
        pass

    @abc.abstractmethod
    def predict(self, batch_data: Any) -> torch.Tensor:
        """
        Performs inference on a batch of data.

        Args:
            batch_data: The input data batch, format depends on the model type 
                       (e.g., PyG Batch object, list of dicts).

        Returns:
            torch.Tensor: The model's output predictions (e.g., logits or scores) 
                          in a standardized tensor format [batch_size, num_classes].
        
        Raises:
            RuntimeError: If the evaluator is not loaded (`load()` wasn't called or failed).
            Exception: If prediction fails.
        """
        pass

    @abc.abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns metadata about the model. 
        Useful for logging and reporting.
        
        Returns:
            Dict[str, Any]: Dictionary containing model metadata 
                            (e.g., model type, source, hyperparameters).
        """
        pass
    
    def is_loaded(self) -> bool:
        """Checks if the model resources are loaded."""
        return self._is_loaded

    def unload(self) -> None:
        """
        (Optional) Cleans up resources (e.g., stops container, releases GPU memory).
        Implement this in subclasses if necessary.
        """
        self._is_loaded = False
        logger.debug(f"Unloaded evaluator '{self.name}'.")
        # Default implementation does nothing, subclasses can override
        pass

    def __enter__(self):
        """Context manager entry: ensures model is loaded."""
        if not self.is_loaded():
            logger.info(f"Loading model '{self.name}' via context manager...")
            self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: ensures model is unloaded."""
        logger.info(f"Unloading model '{self.name}' via context manager...")
        self.unload()
        # Don't suppress exceptions
        return False