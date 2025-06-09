# Core Module

The Core Module provides the foundational abstractions (interfaces) and data structures (types) for the GraPhens system. It establishes the contracts between components, enabling a modular and extensible architecture.

## Overview

This module follows SOLID principles, defining:
- **Interfaces (`interfaces.py`):** Abstract base classes for key components (Augmentation, Embedding, Graph Building/Assembly, Visualization, etc.).
- **Types (`types.py`):** Core data classes (`Phenotype`, `Graph`) used throughout the pipeline.

## Key Features

- **Interface-Based Design**: Clear contracts for component interactions.
- **Type Safety**: Standardized data representation with type annotations.
- **Extensibility**: Designed for easy implementation of new component strategies.

## Architectural Role

The Core Module is central, defining the abstractions that other modules implement. It has no dependencies on other modules, ensuring a clean dependency structure.

```
                           ┌─────────────────────┐
                           │    Core Module      │
                           │ (Interfaces, Types) │
                           └─────────────────────┘
                                     │
                 ┌───────────────────┼───────────────────┐
                 │                   │                   │
    ┌────────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
    │   Graph Module      │ │ Embedding Module│ │ Ontology Module │
    └─────────────────────┘ └─────────────────┘ └─────────────────┘
                 │                   │                   │
    ┌────────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
    │ Augmentation Module │ │Visualization Mod│ │ Database Module │
    └─────────────────────┘ └─────────────────┘ └─────────────────┘
```

## Interfaces (`interfaces.py`)

Defines the required methods for:

- `AugmentationService`: Enriching phenotype lists.
- `EmbeddingStrategy`: Converting phenotypes to vectors.
- `AdjacencyListBuilder`: Creating graph edge structures.
- `GraphAssembler`: Combining components into a final `Graph` object.
- `GlobalContextProvider`: Providing global graph context.
- `PhenotypeVisualizer`: Creating visualizations.
- `PipelineOrchestrator`: Coordinating the overall process.

*(Refer to `interfaces.py` for specific method signatures)*

## Data Types (`types.py`)

Defines the primary data structures:

- `Phenotype`: Represents an HPO term (ID, name, description, metadata).
- `Graph`: Represents the final graph for GNNs (node features, edge index, mappings, metadata).

*(Refer to `types.py` for detailed class definitions)*

## Extending the System

New component implementations can be created by inheriting from the relevant interface in `interfaces.py` and implementing the required methods. These new implementations can then be used by the orchestrator or facade. 