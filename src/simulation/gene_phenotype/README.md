# Gene-Phenotype Module (Simulation)

Provides an interface for accessing and potentially modifying gene-phenotype relationships from HPO database files (`genes_to_phenotype.txt`, `phenotype_to_genes.txt`).

**Note:** This module offers similar query capabilities to the main `src/database` module, but uses a `pandas`-based implementation and includes an update method. Clarify intended use case if distinct from `src/database`.

## Overview

Uses a Facade (`GenePhenotypeFacade`) over a pandas-based database (`GenePhenotypeDatabase`) to query gene-phenotype associations.

## Key Features / Methods (via Facade)

- `get_phenotypes_for_gene(gene)`
- `get_genes_for_phenotype(phenotype_id, include_ancestors)`
- `get_frequency_information(gene, phenotype_id, disease_id)`
- `get_diseases_for_gene(gene)`
- `update_gene_phenotype(gene, phenotype_id, frequency, disease_id)`: Adds/updates associations (in memory).
- `save()`: Saves changes back to the files.

## Usage Example

```python
from src.simulation.gene_phenotype import GenePhenotypeFacade

# Initialize facade (loads data lazily)
facade = GenePhenotypeFacade(data_dir="data/gene_phenotype")

# Get phenotypes for a gene
phenotypes = facade.get_phenotypes_for_gene("AARS1")
print(f"Phenotypes for AARS1: {[p.name for p in phenotypes]}")

# Add/update an association
success = facade.update_gene_phenotype("AARS1", "HP:0001939", frequency="HP:0040283")
if success:
    print("Association updated.")
    # facade.save() # Persist changes to files
```

## Running the Demo

See `src/simulation/gene_phenotype/demo.py` for more examples:

```bash
python -m src.simulation.gene_phenotype.demo
```

## Data Files

This module requires two data files from the HPO project:

1. `genes_to_phenotype.txt` - Maps genes to their most specific HPO phenotypes
2. `phenotype_to_genes.txt` - Maps HPO phenotypes to genes, including ancestors

These files should be placed in the configured data directory (default: `data/gene_phenotype/`).

## Usage Examples

### Basic Usage

```python
from src.simulation.gene_phenotype import GenePhenotypeFacade

# Create the facade
facade = GenePhenotypeFacade(data_dir="data/gene_phenotype")

# Get phenotypes for a gene
phenotypes = facade.get_phenotypes_for_gene("AARS1")
for phenotype in phenotypes:
    print(f"{phenotype.id} - {phenotype.name}")
    print(f"Frequency: {phenotype.metadata.get('frequency')}")
    print(f"Disease: {phenotype.metadata.get('disease_id')}")
    
# Get genes for a phenotype (without including ancestors)
genes = facade.get_genes_for_phenotype("HP:0001939", include_ancestors=False)
for gene in genes:
    print(f"{gene['gene_symbol']} - Disease: {gene['disease_id']}")
    
# Get genes for a phenotype (including ancestors)
genes_with_ancestors = facade.get_genes_for_phenotype("HP:0001939", include_ancestors=True)
print(f"Found {len(genes_with_ancestors)} genes including ancestors")

# Get frequency information
frequency_data = facade.get_frequency_information("AARS1")
for data in frequency_data:
    print(f"{data['hpo_id']} - {data['hpo_name']}")
    print(f"Frequency: {data['frequency']}")
    
# Get diseases associated with a gene
diseases = facade.get_diseases_for_gene("AARS1")
for disease in diseases:
    print(disease)
```

### Integration with HPO Graph Provider

```python
from src.simulation.gene_phenotype import GenePhenotypeFacade
from src.ontology.hpo_graph import HPOGraphProvider

# Create the facade and HPO graph provider
facade = GenePhenotypeFacade(data_dir="data/gene_phenotype")
hpo_provider = HPOGraphProvider(data_dir="data/ontology")

# Get phenotypes for a gene
phenotypes = facade.get_phenotypes_for_gene("AARS1")

# Get ancestors for each phenotype
for phenotype in phenotypes:
    ancestors = hpo_provider.get_ancestors(phenotype.id)
    print(f"{phenotype.id} has {len(ancestors)} ancestors")
```

### Integration with Augmentation Service

```python
from src.simulation.gene_phenotype import GenePhenotypeFacade
from src.augmentation.hpo_augmentation import HPOAugmentationService

# Create the facade and augmentation service
facade = GenePhenotypeFacade(data_dir="data/gene_phenotype")
augmentation_service = HPOAugmentationService(data_dir="data/ontology")

# Get phenotypes for a gene
phenotypes = facade.get_phenotypes_for_gene("AARS1")

# Augment the phenotypes
augmented_phenotypes = augmentation_service.augment(phenotypes)
print(f"Original: {len(phenotypes)}, Augmented: {len(augmented_phenotypes)}")
```

### Integration with Graph Assembly

```python
from src.simulation.gene_phenotype import GenePhenotypeFacade
from src.embedding.strategies import SentenceTransformerEmbeddingStrategy
from src.graph.adjacency import HPOAdjacencyListBuilder
from src.graph.assembler import StandardGraphAssembler
from src.ontology.hpo_graph import HPOGraphProvider

# Create components
facade = GenePhenotypeFacade(data_dir="data/gene_phenotype")
hpo_provider = HPOGraphProvider(data_dir="data/ontology")
embedding_strategy = SentenceTransformerEmbeddingStrategy(model_name="all-MiniLM-L6-v2")
adjacency_builder = HPOAdjacencyListBuilder(hpo_provider)
assembler = StandardGraphAssembler()

# Get phenotypes for a gene
phenotypes = facade.get_phenotypes_for_gene("AARS1")

# Create embeddings
node_features = embedding_strategy.embed_batch(phenotypes)

# Build adjacency list
edge_index = adjacency_builder.build(phenotypes)

# Assemble graph
graph = assembler.assemble(phenotypes, node_features, edge_index)

print(f"Created graph with {len(phenotypes)} nodes and {edge_index.shape[1]} edges")
```

### Updating Gene-Phenotype Associations

```python
from src.simulation.gene_phenotype import GenePhenotypeFacade

# Create the facade
facade = GenePhenotypeFacade(data_dir="data/gene_phenotype")

# Add a new gene-phenotype association
success = facade.update_gene_phenotype(
    gene="AARS1",
    phenotype_id="HP:0001939",
    frequency="HP:0040283",  # Very frequent (99-80%)
    disease_id="OMIM:601065"
)

if success:
    print("Association added successfully")
else:
    print("Failed to add association - phenotype does not exist in database")
```

## Running the Demo

To see the module in action, run the included demo script:

```bash
# Using default settings
python -m src.simulation.gene_phenotype.demo

# Specifying custom data directory and gene
python -m src.simulation.gene_phenotype.demo --data-dir /path/to/data --gene NAT2 --phenotype HP:0000007
```

## Implementation Details

The module consists of two main classes:

1. `GenePhenotypeDatabase` - Low-level database access with pandas 
2. `GenePhenotypeFacade` - High-level, user-friendly interface

The implementation follows the Facade design pattern to provide a simplified interface to the underlying database operations.

## Data Integration

This module can be used alongside other GraPhens components:

- `HPOGraphProvider` for hierarchical phenotype information
- `HPOAugmentationService` for phenotype augmentation
- Embedding strategies to convert phenotypes to vectors
- Graph assembly components to create complete graph structures

## Performance Considerations

- The database loads data files lazily on first query
- Pandas is used for efficient data filtering and querying
- The facade pattern keeps the interface simple while enabling advanced usage

## Future Extensions

Potential future enhancements include:

- Caching mechanisms for improved performance
- Support for additional data sources
- Integration with more HPO data files
- Statistical analysis of gene-phenotype relationships 