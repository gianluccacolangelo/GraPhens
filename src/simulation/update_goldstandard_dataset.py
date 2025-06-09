"""
This script updates a gene-phenotype database with new phenotype associations from a JSON file of cases.

The script performs the following key functions:

1. Database Update:
   - Takes a JSON or TSV file containing gene-phenotype associations from clinical cases
   - Updates an existing gene-phenotype database with new associations
   - Validates phenotype IDs against existing database entries
   - Performs efficient batch updates to minimize database operations

2. Data Validation:
   - Checks that input files and directories exist
   - Validates phenotype IDs against the existing database
   - Skips invalid phenotype associations while continuing processing

3. Statistics Tracking:
   - Tracks number of genes, cases and associations processed
   - Records skipped associations due to invalid phenotypes
   - Counts new associations added to the database

4. Command Line Interface:
   - Accepts arguments for data directory and input file paths
   - Supports both JSON and TSV input formats
   - Provides a dry-run option for testing without making changes
   - Outputs detailed statistics about the update process

Usage:
    python update_goldstandard_dataset.py [--data-dir DATA_DIR] [--cases-file CASES_FILE] [--dry-run]

Arguments:
    --data-dir: Directory containing gene-phenotype database files (default: "data/")
    --cases-file: Path to JSON or TSV file with case data 
                  (default: "src/simulation/fenotipos_cleaned_processed.json")
    --dry-run: Run without making actual changes to database

The input file should be either:
- JSON: A dictionary where keys are gene symbols and values are lists of phenotype lists
- TSV: A table with columns for ID, Gene, and Phenotype (with Phenotype containing comma-separated HPO IDs)
"""

import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List
from src.simulation.gene_phenotype.facade import GenePhenotypeFacade
import pandas as pd
from tqdm import tqdm
import os

def load_phenotypes_json(file_path: str) -> Dict[str, List[List[str]]]:
    """
    Load the cleaned and processed phenotypes JSON file.
    
    Returns:
        Dictionary with gene symbols as keys and lists of phenotype lists as values
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_phenotypes_tsv(file_path: str) -> Dict[str, List[List[str]]]:
    """
    Load phenotypes from a TSV file with ID, Gene, Phenotype columns.
    
    Returns:
        Dictionary with gene symbols as keys and lists of phenotype lists as values
    """
    gene_cases = {}
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            gene = row['Gene']
            # Convert comma-separated phenotype string to list
            phenotypes = row['Phenotype'].split(',')
            
            # Initialize gene in dictionary if not present
            if gene not in gene_cases:
                gene_cases[gene] = []
                
            # Add this phenotype set to the gene
            gene_cases[gene].append(phenotypes)
    
    return gene_cases


def update_database_from_cases(facade: GenePhenotypeFacade, 
                             gene_cases: Dict[str, List[List[str]]]) -> Dict[str, Dict[str, int]]:
    """
    Update the gene-phenotype database with cases from the JSON file.
    
    Args:
        facade: GenePhenotypeFacade instance
        gene_cases: Dictionary with gene symbols as keys and lists of phenotype lists as values
        
    Returns:
        Dictionary with statistics about the update process
    """
    stats = {
        'processed': {'genes': 0, 'cases': 0, 'associations': 0},
        'skipped': {'associations': 0},
        'added': {'associations': 0}
    }
    
    # Get existing phenotypes for validation
    valid_phenotypes = set(facade.get_available_phenotypes())
    
    # Pre-load the database to avoid loading it multiple times
    # This ensures database.load() is called just once
    facade.database._genes_to_phenotype  # This triggers the load() method
    
    # Collect all valid updates in a batch
    update_batch = []
    
    # Process each gene
    for gene, cases in gene_cases.items():
        stats['processed']['genes'] += 1
        
        # Process each case for this gene
        for phenotype_list in cases:
            stats['processed']['cases'] += 1
            
            # Process each phenotype in the case
            for phenotype in phenotype_list:
                stats['processed']['associations'] += 1
                
                # Skip if phenotype doesn't exist in database
                if phenotype not in valid_phenotypes:
                    print(f'phenotype {phenotype} not in database for gene {gene}')
                    stats['skipped']['associations'] += 1
                    continue
                
                # Add to batch instead of updating immediately
                update_batch.append((gene, phenotype))
    
    # Now process the batch of updates efficiently
    if update_batch:
        print(f"Processing {len(update_batch)} updates in batch...")
        # Add a batch update method to the database class and call it here
        added_count = batch_update_gene_phenotypes(facade.database, update_batch)
        stats['added']['associations'] = added_count
    
    return stats


def batch_update_gene_phenotypes(database, update_batch):
    """
    Perform batch update of gene-phenotype associations for better performance.
    
    Args:
        database: GenePhenotypeDatabase instance
        update_batch: List of (gene_symbol, phenotype_id) tuples to add
        
    Returns:
        Number of associations successfully added
    """
    # Ensure the database is loaded
    if database._genes_to_phenotype is None:
        database.load()
    
    # Create dataframes for new rows
    genes_to_phenotype_rows = []
    phenotype_to_genes_rows = []
    
    added_count = 0
    
    # Create lookup structures outside the loop
    valid_hpo_ids = set(database._genes_to_phenotype['hpo_id'])
    hpo_id_to_name = dict(zip(
        database._genes_to_phenotype['hpo_id'],
        database._genes_to_phenotype['hpo_name']
    ))
    gene_to_ncbi = {}
    for _, row in database._genes_to_phenotype.drop_duplicates(subset=['gene_symbol']).iterrows():
        gene_to_ncbi[row['gene_symbol']] = row['ncbi_gene_id']
    
    # Process each update in the batch
    for gene_symbol, hpo_id in tqdm(update_batch, desc="Processing gene-phenotype pairs"):
        # Use set membership (O(1) operation)
        if hpo_id not in valid_hpo_ids:
            continue
            
        # Dictionary lookup instead of dataframe filtering
        hpo_name = hpo_id_to_name[hpo_id]
        
        # Dictionary lookup for NCBI gene ID
        ncbi_gene_id = gene_to_ncbi.get(gene_symbol, gene_symbol)
        
        # Add to genes_to_phenotype rows
        genes_to_phenotype_rows.append({
            'ncbi_gene_id': ncbi_gene_id,
            'gene_symbol': gene_symbol,
            'hpo_id': hpo_id,
            'hpo_name': hpo_name,
            'frequency': '',
            'disease_id': ''
        })
        
        # Add to phenotype_to_genes rows
        phenotype_to_genes_rows.append({
            'hpo_id': hpo_id,
            'hpo_name': hpo_name,
            'ncbi_gene_id': ncbi_gene_id,
            'gene_symbol': gene_symbol,
            'disease_id': ''
        })
        
        added_count += 1
    
    # Now add all rows at once if we have any
    if genes_to_phenotype_rows:
        genes_df = pd.DataFrame(genes_to_phenotype_rows)
        database._genes_to_phenotype = pd.concat(
            [database._genes_to_phenotype, genes_df], ignore_index=True
        )
        
        phenotypes_df = pd.DataFrame(phenotype_to_genes_rows)
        database._phenotype_to_genes = pd.concat(
            [database._phenotype_to_genes, phenotypes_df], ignore_index=True
        )
    return added_count


def main():
    parser = argparse.ArgumentParser(description="Update gene-phenotype database from cases")
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Directory containing gene-phenotype database files"
    )
    parser.add_argument(
        "--cases-file",
        default="src/simulation/fenotipos_cleaned_processed.json",
        help="Path to the processed cases file (JSON or TSV format)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making actual changes to database"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    data_dir = Path(args.data_dir)
    cases_file = Path(args.cases_file)
    
    if not data_dir.is_dir():
        raise ValueError(f"Data directory not found: {data_dir}")
    if not cases_file.is_file():
        raise ValueError(f"Cases file not found: {cases_file}")
        
    # Load cases data based on file extension
    print(f"Loading cases from {cases_file}")
    file_extension = cases_file.suffix.lower()
    
    if file_extension == '.json':
        gene_cases = load_phenotypes_json(str(cases_file))
    elif file_extension in ['.tsv', '.tab']:
        gene_cases = load_phenotypes_tsv(str(cases_file))
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported formats are .json and .tsv")
    
    print(f"Loaded data for {len(gene_cases)} genes")
    
    # Create facade
    facade = GenePhenotypeFacade(data_dir=str(data_dir))
    
    # Process cases
    if args.dry_run:
        print("DRY RUN - No changes will be made to database")
        
    print("\nUpdating database...")
    stats = update_database_from_cases(facade, gene_cases)
    
    # Save the updated database if not in dry run mode
    if not args.dry_run:
        print("Saving changes to database files...")
        # Get the paths of the files in data_dir
        file_paths = [
            os.path.join(str(data_dir), "genes_to_phenotype.txt"),
            os.path.join(str(data_dir), "phenotype_to_genes.txt")
        ]
        
        # Create backups before saving
        for file_path in file_paths:
            if os.path.exists(file_path):
                backup_path = f"{file_path}.bak"
                print(f" - Creating backup: {file_path} → {backup_path}")
                os.rename(file_path, backup_path)
                
        # Save and log the updated files
        saved_files = facade.save()
        for file_path in saved_files:
            print(f" - Updated file: {file_path}")
        
        print("Database files updated successfully.")
    
    # Print statistics
    print("\nUpdate Statistics:")
    print(f"Processed {stats['processed']['genes']} genes")
    print(f"Processed {stats['processed']['cases']} cases")
    print(f"\nAssociations:")
    print(f"- Processed: {stats['processed']['associations']}")
    print(f"- Skipped: {stats['skipped']['associations']} (phenotype not in database)")
    print(f"- Added: {stats['added']['associations']}")
    
    if args.dry_run:
        print("\nThis was a dry run - no changes were made to database")


if __name__ == "__main__":
    main()
