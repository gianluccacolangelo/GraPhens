# GraPhens: Phenotype-Based Graph Neural Network System

## Overview

GraPhens is a comprehensive system designed to transform phenotypic data into structured graph representations suitable for downstream machine learning tasks, particularly Graph Neural Networks (GNNs). The system follows a modular, interface-based architecture that adheres to SOLID principles, allowing for flexible component selection and extension.

## System Architecture

GraPhens consists of seven core modules that work together in a pipeline fashion:

```
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ Ontology Module │───────▶│  Augmentation   │───────▶│  Embedding      │
└─────────────────┘        └─────────────────┘        └─────────────────┘
       │                          │                         │
       │                          │                         │
       ▼                          ▼                         ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ Database Module │───────▶│ Core Module     │◀───────│  Graph Module   │
└─────────────────┘        └─────────────────┘        └─────────────────┘
                                  ▲                         │
                                  │                         │
                                  │                         ▼
                                  └─────────────────────────│ Visualization Module │
                                                            └──────────────────────┘
```

### Module Responsibilities

- **Core Module**: Defines the foundational interfaces and data structures used throughout the system
- **Database Module**: Provides access to gene-phenotype relationship data from HPOA
- **Ontology Module**: Manages the Human Phenotype Ontology (HPO) data and provides traversal capabilities
- **Augmentation Module**: Enriches phenotype sets with related terms from the HPO hierarchy
- **Embedding Module**: Converts phenotypes into numerical vector representations
- **Graph Module**: Builds graph structures from phenotypes with their relationships and embeddings
- **Visualization Module**: Creates visual representations of phenotypes, hierarchies, and graphs

## Performance Optimizations

GraPhens implements several key performance optimizations:

- **Resource Caching:** Global caching for HPO providers and pipeline orchestrators avoids redundant work.
- **Efficient Batch Processing:** Optimized methods for processing multiple patients significantly speed up common workflows (40-70x faster).
- **Efficient Graph Export:** Batch-aware export to ML frameworks (like PyTorch Geometric) enables efficient training.
- **Memory-Optimized Embeddings:** Support for memory-mapped embeddings reduces memory footprint and improves access time.
- **Optimized Imports:** Lazy loading and conditional checks for optional dependencies improve startup time and runtime efficiency.

*(See specific module READMEs for detailed benchmarks and implementation details)*

## Data Flow

The typical data flow through the system is:

1. **Load Ontology**: The Ontology Module loads and manages the HPO data, providing an interface for traversing the hierarchy
2. **Initial Phenotypes**: A list of `Phenotype` objects is created or loaded from an external source (e.g., Database Module querying by gene)
3. **Augmentation**: The Augmentation Module enriches the initial phenotype set with related terms
4. **Embedding**: The Embedding Module converts each phenotype into a numerical vector representation
5. **Graph Construction**: The Graph Module builds adjacency lists representing relationships between phenotypes
6. **Graph Assembly**: Node features (embeddings) and edge indices are combined into a complete graph
7. **Visualization**: The Visualization Module creates visual representations of the results at various stages

## Module Integration

The modules integrate seamlessly through the interfaces defined in the Core Module. Key integrations include:

- **Database -> Augmentation:** Initial phenotypes retrieved from the database can be directly augmented.
- **Ontology -> Augmentation/Graph:** The HPO graph is used for adding related terms and building graph structure.
- **Augmentation -> Embedding -> Graph:** Augmented phenotypes are embedded and then assembled into the final graph structure.
- **Graph -> Visualization:** The final graph can be visualized.

*(See module READMEs for specific integration code examples)*

## Complete Pipeline Example

A complete example demonstrating the end-to-end pipeline can be found in the `examples/` directory.

## Extensibility & Design

The system's interface-based, modular design (following SOLID principles) makes it highly extensible. Key aspects include:

- **Interchangeable Components:** Easily swap implementations (e.g., different embedding strategies, visualizers).
- **Custom Implementations:** Add new functionality by implementing the core interfaces.
- **Configuration:** Customize behavior through component configuration (e.g., augmentation options, embedding models).

## Conclusion

GraPhens provides a flexible, modular system for transforming phenotypic data into graph representations suitable for GNN processing. The interface-based design allows for easy extension and component swapping, while the clear data flow makes the system easy to understand and adapt to different use cases.

By leveraging the Human Phenotype Ontology and various embedding techniques, the system creates rich graph representations that capture both the hierarchical relationships between phenotypes and their semantic similarity, enabling powerful downstream machine learning applications.