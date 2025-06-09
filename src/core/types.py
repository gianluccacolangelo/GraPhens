from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

"""
This module provides the foundational data structures used across the entire system as 
immutable dataclasses. It defines the Phenotype class representing HPO terms and the 
Graph class that holds the final graph representation.

Architecturally, these types serve as the data transfer objects between different 
components of the pipeline, ensuring consistency in how data is represented and manipulated.

From an extensibility perspective, the classes use sensible defaults and employ 
field(default_factory=dict) for mutable attributes to prevent shared reference issues. 
The optional fields provide flexibility to add new features without breaking existing code.

In the bigger picture, these data structures form the common language of the system, 
enabling different components to communicate clearly about what data they expect and produce.
"""

@dataclass
class Phenotype:
    """Represents a single phenotype from HPO."""
    id: str  # HPO ID (e.g., "HP:0000118")
    name: str  # Human-readable name
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Graph:
    """Represents a graph ready for GNN processing."""
    node_features: np.ndarray  # Node feature matrix
    edge_index: np.ndarray     # Edge connectivity (COO format)
    node_mapping: Dict[str, int]  # Maps HPO IDs to indices
    global_context: Optional[np.ndarray] = None  # Optional global context vector
    edge_attr: Optional[np.ndarray] = None  # Optional edge attributes
    metadata: Dict[str, Any] = field(default_factory=dict)
