# Gene-Phenotype Database Module

Provides access to gene-phenotype relationship data, primarily from the Human Phenotype Ontology Annotation (HPOA) data files.

## Overview

This module allows querying associations between genes, phenotypes, and diseases using a standardized interface.

## Components

- **`GenePhenotypeDatabase` (Interface)**: Abstract base class defining the standard API for gene-phenotype data access.
- **`HPOAGenePhenotypeDatabase` (Implementation)**: Concrete implementation using HPOA formatted files (`phenotype.hpoa`).

## Key Features & Methods

The `GenePhenotypeDatabase` interface (implemented by `HPOAGenePhenotypeDatabase`) provides methods to:

- `get_phenotypes_for_gene(gene_symbol)`: Get `Phenotype` objects for a gene.
- `get_genes_for_phenotype(hpo_id)`: Get gene symbols for an HPO term.
- `get_frequency(gene_symbol, hpo_id, disease_id)`: Get phenotype frequency for a gene/disease.
- `get_diseases_for_gene(gene_symbol)`: Get associated disease IDs for a gene.
- `get_genes_for_disease(disease_id)`: Get associated gene symbols for a disease.

Phenotypes are returned as standard `Phenotype` objects (from `src.core.types`) with metadata including frequency and disease info.

## Usage Example

```python
from src.database.gene_phenotype_db import HPOAGenePhenotypeDatabase

# Initialize with path to HPOA data file
db = HPOAGenePhenotypeDatabase(
    genes_to_phenotypes_path="data/database/phenotype.hpoa"
)

# Query phenotypes for gene "AARS1"
phenotypes = db.get_phenotypes_for_gene("AARS1")
if phenotypes:
    print(f"Phenotypes for AARS1: {[p.name for p in phenotypes]}")
    print(f"Metadata for first phenotype ({phenotypes[0].id}): {phenotypes[0].metadata}")

# Query genes for phenotype "HP:0001939" (Anemia)
genes = db.get_genes_for_phenotype("HP:0001939")
print(f"Genes associated with Anemia: {genes}")
```

## Demo Script

See `src/database/test_gene_phenotype_db.py` for more usage examples:

```bash
python -m src.database.test_gene_phenotype_db
```

## Data File Format

The `HPOAGenePhenotypeDatabase` expects the HPOA `phenotype.hpoa` file (tab-delimited) containing columns like `gene_symbol`, `hpo_id`, `hpo_name`, `frequency`, `disease_id`.

## Extending

Create alternative database backends (e.g., SQL, API) by implementing the `GenePhenotypeDatabase` interface. 