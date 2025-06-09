#!/usr/bin/env python3
"""
Command-line interface for the Gene-Phenotype Database.
"""

import os
import sys
import argparse
import logging
from typing import List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.gene_phenotype_db import HPOAGenePhenotypeDatabase
from src.core.types import Phenotype

def setup_logging(verbose=False):
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
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

def query_phenotypes_for_gene(db, gene_symbol):
    """Query and display phenotypes for a gene."""
    phenotypes = db.get_phenotypes_for_gene(gene_symbol)
    print_phenotype_list(phenotypes, f"Phenotypes for gene {gene_symbol}")

def query_genes_for_phenotype(db, hpo_id):
    """Query and display genes for a phenotype."""
    genes = db.get_genes_for_phenotype(hpo_id)
    print_gene_list(genes, f"Genes for phenotype {hpo_id}")

def query_frequency(db, gene_symbol, hpo_id, disease_id=None):
    """Query and display the frequency for a gene-phenotype pair."""
    frequency = db.get_frequency(gene_symbol, hpo_id, disease_id)
    if disease_id:
        print(f"\nFrequency of {hpo_id} for gene {gene_symbol} in disease {disease_id}:")
    else:
        print(f"\nFrequency of {hpo_id} for gene {gene_symbol} (any disease):")
    print("-" * 60)
    
    if frequency:
        print(f"  {frequency}")
    else:
        print("  Frequency information not available")

def query_diseases_for_gene(db, gene_symbol):
    """Query and display diseases for a gene."""
    diseases = db.get_diseases_for_gene(gene_symbol)
    print_disease_list(diseases, f"Diseases for gene {gene_symbol}")

def query_genes_for_disease(db, disease_id):
    """Query and display genes for a disease."""
    genes = db.get_genes_for_disease(disease_id)
    print_gene_list(genes, f"Genes for disease {disease_id}")

def main():
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(description="Query the Gene-Phenotype Database")
    
    # Common arguments
    parser.add_argument("--database-path", default="data/database/phenotype.hpoa", 
                        help="Path to the genes_to_phenotypes.txt file")
    parser.add_argument("--phenotype-to-genes", default=None,
                        help="Path to the phenotype_to_genes.txt file (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    # Subparsers for different query types
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Query phenotypes for gene
    gene_parser = subparsers.add_parser("gene-phenotypes", help="Get phenotypes for a gene")
    gene_parser.add_argument("gene_symbol", help="Gene symbol (e.g., AARS1)")
    
    # Query genes for phenotype
    phenotype_parser = subparsers.add_parser("phenotype-genes", help="Get genes for a phenotype")
    phenotype_parser.add_argument("hpo_id", help="HPO ID (e.g., HP:0001939)")
    
    # Query frequency
    frequency_parser = subparsers.add_parser("frequency", help="Get frequency for a gene-phenotype pair")
    frequency_parser.add_argument("gene_symbol", help="Gene symbol (e.g., AARS1)")
    frequency_parser.add_argument("hpo_id", help="HPO ID (e.g., HP:0002460)")
    frequency_parser.add_argument("--disease-id", help="Disease ID (e.g., OMIM:613287)")
    
    # Query diseases for gene
    gene_diseases_parser = subparsers.add_parser("gene-diseases", help="Get diseases for a gene")
    gene_diseases_parser.add_argument("gene_symbol", help="Gene symbol (e.g., AARS1)")
    
    # Query genes for disease
    disease_genes_parser = subparsers.add_parser("disease-genes", help="Get genes for a disease")
    disease_genes_parser.add_argument("disease_id", help="Disease ID (e.g., OMIM:613287)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Check if database file exists
    if not os.path.exists(args.database_path):
        print(f"Error: Database file not found: {args.database_path}")
        print("Please make sure the file is in the correct location or update the path.")
        sys.exit(1)
    
    # Initialize the database
    try:
        db = HPOAGenePhenotypeDatabase(
            genes_to_phenotypes_path=args.database_path,
            phenotype_to_genes_path=args.phenotype_to_genes
        )
    except Exception as e:
        print(f"Error loading database: {str(e)}")
        sys.exit(1)
    
    # Execute the requested command
    if args.command == "gene-phenotypes":
        query_phenotypes_for_gene(db, args.gene_symbol)
    elif args.command == "phenotype-genes":
        query_genes_for_phenotype(db, args.hpo_id)
    elif args.command == "frequency":
        query_frequency(db, args.gene_symbol, args.hpo_id, args.disease_id)
    elif args.command == "gene-diseases":
        query_diseases_for_gene(db, args.gene_symbol)
    elif args.command == "disease-genes":
        query_genes_for_disease(db, args.disease_id)

if __name__ == "__main__":
    main() 