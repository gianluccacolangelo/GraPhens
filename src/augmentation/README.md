# Phenotype Augmentation Module

Enriches phenotype datasets with related terms (ancestors/descendants) from the Human Phenotype Ontology (HPO), improving the completeness of graph representations.

## Overview

This module addresses incomplete clinical observations by adding semantically related HPO terms. It provides two implementations of the `AugmentationService` interface:

1.  **`HPOAugmentationService`**: Uses a local HPO graph (`HPOGraphProvider`) for fast, offline augmentation.
2.  **`APIAugmentationService`**: Uses the public JAX HPO REST API for serverless or environments without local HPO data.

## Key Features

- **Hierarchical Enrichment**: Adds HPO ancestors and/or descendants.
- **Configurable**: Control whether to include ancestors or descendants.
- **Robust**: Handles missing terms and API errors gracefully.
- **Standard Interface**: Pluggable implementations.

## Architectural Role

Augmentation typically occurs after initial phenotype loading and before embedding, enriching the data used for graph construction.

```
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ Phenotype Data  │───────▶│  Augmentation   │───────▶│  Embedding      │
└─────────────────┘        └─────────────────┘        └─────────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │  Graph Assembly │
                          └─────────────────┘
```

## Usage Example

```python
from src.augmentation.hpo_augmentation import HPOAugmentationService # Or APIAugmentationService
from src.core.types import Phenotype

# Use local HPO graph (requires data_dir setup via HPOGraphProvider/HPOUpdater)
augmentation_service = HPOAugmentationService(
    data_dir="data/ontology",
    include_ancestors=True,     # Default: True
    include_descendants=False   # Default: False
)

# # Or use the API service
# from src.augmentation.hpo_augmentation import APIAugmentationService
# augmentation_service = APIAugmentationService()

# Define initial phenotypes
initial_phenotypes = [
    Phenotype(id="HP:0001251", name="Atonic seizure") # Child of HP:0001250
]

# Augment the phenotype set (adds HP:0001250 Seizure, and its ancestors)
augmented_phenotypes = augmentation_service.augment(initial_phenotypes)

print(f"Original count: {len(initial_phenotypes)}, Augmented count: {len(augmented_phenotypes)}")
```

## Integration

- Depends on `HPOGraphProvider` (from `ontology` module) when using `HPOAugmentationService`.
- Produces an enriched list of `Phenotype` objects for downstream modules (Embedding, Graph, Visualization).

## Extending

Create custom augmentation logic by implementing the `AugmentationService` interface from `src.core.interfaces`. 