#!/usr/bin/env python3
"""
Test script for demonstrating the GenePhenotypeDatabase interface.
"""

import os
import sys
import logging
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.gene_phenotype_db import HPOAGenePhenotypeDatabase
from src.core.types import Phenotype

def setup_logging():
    """Configure logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def print_phenotype_list(phenotypes: List[Phenotype], title: str):
    """Print a formatted list of phenotypes."""
    print(f"\n{title} ({len(phenotypes)} phenotypes):")
    print("-" * (len(title) + 15))
    
    if not phenotypes:
        print("  No phenotypes found")
        return
    
    for i, p in enumerate(phenotypes, 1):
        freq = p.metadata.get("frequency", "-")
        disease = p.metadata.get("disease_id", "-")
        print(f"  {i}. {p.id} - {p.name}")
        print(f"     Frequency: {freq}")
        print(f"     Disease: {disease}")

def print_gene_list(genes: List[str], title: str):
    """Print a formatted list of genes."""
    print(f"\n{title} ({len(genes)} genes):")
    print("-" * (len(title) + 11))
    
    if not genes:
        print("  No genes found")
        return
    
    # Format into rows of 5 genes each
    for i in range(0, len(genes), 5):
        row = genes[i:i+5]
        print("  " + ", ".join(row))

def print_disease_list(diseases: List[str], title: str):
    """Print a formatted list of diseases."""
    print(f"\n{title} ({len(diseases)} diseases):")
    print("-" * (len(title) + 14))
    
    if not diseases:
        print("  No diseases found")
        return
    
    for i, disease in enumerate(diseases, 1):
        print(f"  {i}. {disease}")

def main():
    """Main function to demonstrate the GenePhenotypeDatabase usage."""
    setup_logging()
    
    # Path to the gene-phenotype files
    genes_to_phenotypes_path = "data/database/phenotype.hpoa"
    phenotype_to_genes_path = "data/database/phenotype_to_genes.txt"
    
    # Check if files exist
    if not os.path.exists(genes_to_phenotypes_path):
        print(f"Error: {genes_to_phenotypes_path} not found.")
        print("Please make sure the file is in the correct location or update the path in this script.")
        return
    
    # Initialize the database
    try:
        db = HPOAGenePhenotypeDatabase(
            genes_to_phenotypes_path=genes_to_phenotypes_path,
            phenotype_to_genes_path=phenotype_to_genes_path if os.path.exists(phenotype_to_genes_path) else None
        )
        print(f"Successfully loaded gene-phenotype database")
    except Exception as e:
        print(f"Error loading database: {str(e)}")
        return
    
    # Demonstrate querying phenotypes for a gene
    print("\n===== Example 1: Get phenotypes for a gene =====")
    gene_symbol = "AARS1"
    phenotypes = db.get_phenotypes_for_gene(gene_symbol)
    print_phenotype_list(phenotypes, f"Phenotypes for gene {gene_symbol}")
    
    # Demonstrate querying genes for a phenotype
    print("\n===== Example 2: Get genes for a phenotype =====")
    hpo_id = "HP:0001939"  # Abnormality of metabolism/homeostasis
    genes = db.get_genes_for_phenotype(hpo_id)
    print_gene_list(genes, f"Genes for phenotype {hpo_id}")
    
    # Demonstrate querying frequency
    print("\n===== Example 3: Get frequency for a gene-phenotype-disease combination =====")
    gene = "AARS1"
    phenotype = "HP:0002460"  # Distal muscle weakness
    disease = "OMIM:613287"
    frequency = db.get_frequency(gene, phenotype, disease)
    print(f"Frequency of {phenotype} for gene {gene} in disease {disease}: {frequency}")
    
    # Demonstrate querying diseases for a gene
    print("\n===== Example 4: Get diseases for a gene =====")
    gene_symbol = "AARS1"
    diseases = db.get_diseases_for_gene(gene_symbol)
    print_disease_list(diseases, f"Diseases for gene {gene_symbol}")
    
    # Demonstrate querying genes for a disease
    print("\n===== Example 5: Get genes for a disease =====")
    disease_id = "OMIM:613287"
    genes = db.get_genes_for_disease(disease_id)
    print_gene_list(genes, f"Genes for disease {disease_id}")
    
    print("\n===== Complete =====")

if __name__ == "__main__":
    main() 