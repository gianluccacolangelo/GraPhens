"""
Demonstration script for the gene-phenotype database.

This script shows how to use the GenePhenotypeFacade class to query gene-phenotype relationships.
"""

import os
import sys
import argparse
from src.simulation.gene_phenotype.facade import GenePhenotypeFacade


def demo_phenotypes_for_gene(facade, gene_symbol):
    """Demonstrate getting phenotypes for a gene."""
    print(f"\n=== Phenotypes for gene '{gene_symbol}' ===")
    phenotypes = facade.get_phenotypes_for_gene(gene_symbol)
    
    if not phenotypes:
        print(f"No phenotypes found for gene '{gene_symbol}'")
        return
        
    print(f"Found {len(phenotypes)} phenotypes:")
    for i, phenotype in enumerate(phenotypes, 1):
        frequency = phenotype.metadata.get('frequency', '-')
        disease = phenotype.metadata.get('disease_id', '-')
        print(f"{i}. {phenotype.id} - {phenotype.name}")
        print(f"   Frequency: {frequency}, Disease: {disease}")


def demo_genes_for_phenotype(facade, hpo_id, include_ancestors=False):
    """Demonstrate getting genes for a phenotype."""
    print(f"\n=== Genes for phenotype '{hpo_id}' (include_ancestors={include_ancestors}) ===")
    genes = facade.get_genes_for_phenotype(hpo_id, include_ancestors)
    
    if not genes:
        print(f"No genes found for phenotype '{hpo_id}'")
        return
        
    print(f"Found {len(genes)} genes:")
    for i, gene in enumerate(genes, 1):
        print(f"{i}. {gene['gene_symbol']} (NCBI ID: {gene['ncbi_gene_id']}, Disease: {gene['disease_id']})")


def demo_frequency_information(facade, gene_symbol, disease_id=None):
    """Demonstrate getting frequency information."""
    filter_text = f"disease '{disease_id}'" if disease_id else "all diseases"
    print(f"\n=== Frequency information for gene '{gene_symbol}' and {filter_text} ===")
    
    frequency_data = facade.get_frequency_information(gene_symbol, disease_id=disease_id)
    
    if not frequency_data:
        print(f"No frequency data found for gene '{gene_symbol}'")
        return
        
    print(f"Found {len(frequency_data)} entries:")
    for i, data in enumerate(frequency_data, 1):
        print(f"{i}. {data['hpo_id']} - {data['hpo_name']}")
        print(f"   Frequency: {data['frequency']}, Disease: {data['disease_id']}")


def demo_diseases_for_gene(facade, gene_symbol):
    """Demonstrate getting diseases for a gene."""
    print(f"\n=== Diseases associated with gene '{gene_symbol}' ===")
    diseases = facade.get_diseases_for_gene(gene_symbol)
    
    if not diseases:
        print(f"No diseases found for gene '{gene_symbol}'")
        return
        
    print(f"Found {len(diseases)} diseases:")
    for i, disease in enumerate(diseases, 1):
        print(f"{i}. {disease}")


def demo_update_gene_phenotype(facade, gene_symbol, hpo_id):
    """Demonstrate adding a new gene-phenotype association."""
    print(f"\n=== Adding new association: gene '{gene_symbol}' -> phenotype '{hpo_id}' ===")
    
    # Try to add the association
    success = facade.update_gene_phenotype(gene_symbol, hpo_id)
    
    if success:
        print(f"Successfully added association")
        # Show the updated phenotypes for this gene
        demo_phenotypes_for_gene(facade, gene_symbol)
    else:
        print(f"Failed to add association - phenotype '{hpo_id}' does not exist in database")


def main():
    parser = argparse.ArgumentParser(description="Demonstrate Gene-Phenotype Database")
    parser.add_argument("--data-dir", default="data/gene_phenotype", 
                        help="Directory containing gene-phenotype database files")
    parser.add_argument("--gene", default="AARS1", help="Gene symbol to demonstrate")
    parser.add_argument("--phenotype", default="HP:0002460", help="HPO ID to demonstrate")
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        print("Please download gene-phenotype data files and place them in this directory.")
        sys.exit(1)
        
    # Check if the required files exist
    required_files = ["genes_to_phenotype.txt", "phenotype_to_genes.txt"]
    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(args.data_dir, f))]
    if missing_files:
        print(f"Error: Missing required files in '{args.data_dir}':")
        for f in missing_files:
            print(f" - {f}")
        sys.exit(1)
    
    # Create the facade
    facade = GenePhenotypeFacade(data_dir=args.data_dir)
    
    # Run the demonstrations
    print(f"Using Gene-Phenotype database in '{args.data_dir}'")
    
    # Demonstrate phenotypes for a gene
    demo_phenotypes_for_gene(facade, args.gene)
    
    # Demonstrate genes for a phenotype (without ancestors)
    demo_genes_for_phenotype(facade, args.phenotype, include_ancestors=False)
    
    # Demonstrate genes for a phenotype (with ancestors)
    demo_genes_for_phenotype(facade, args.phenotype, include_ancestors=True)
    
    # Demonstrate frequency information
    demo_frequency_information(facade, args.gene)
    
    # If the gene has multiple diseases, demonstrate filtering by disease
    diseases = facade.get_diseases_for_gene(args.gene)
    if len(diseases) > 1:
        demo_frequency_information(facade, args.gene, disease_id=diseases[0])
    
    # Demonstrate diseases for a gene
    demo_diseases_for_gene(facade, args.gene)

    # Add demonstration of update functionality
    print("\n=== Testing Update Functionality ===")
    # Try with a valid phenotype ID
    demo_update_gene_phenotype(facade, args.gene, args.phenotype)
    # Try with an invalid phenotype ID
    demo_update_gene_phenotype(facade, args.gene, "HP:9999999")


if __name__ == "__main__":
    main() 