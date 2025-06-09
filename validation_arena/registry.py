import logging
from typing import Dict, Type, Callable, Any

# Use standard Python logging
logger = logging.getLogger(__name__)

# --- Evaluator Registry (Simple Factory) ---
# Maps evaluator type strings (from config) to their classes
EVALUATOR_REGISTRY: Dict[str, Type] = {}

def register_evaluator(name: str) -> Callable:
    """
    Decorator to register evaluator classes.
    
    Args:
        name: The name to register the evaluator class under
        
    Returns:
        A decorator function that registers the decorated class
    """
    def decorator(cls) -> Type:
        # We can't check for BaseModelEvaluator here directly to avoid circular imports
        # The check will happen in arena.py when loading evaluators
        EVALUATOR_REGISTRY[name] = cls
        logger.debug(f"Registered evaluator: '{name}' -> {cls.__name__}")
        return cls
    return decorator 