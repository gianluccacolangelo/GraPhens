#!/usr/bin/env python3
"""
Setup script to create the necessary directory structure for database files.
"""

import os
import sys
import shutil
import argparse

def setup_database_directories(database_path="data/database"):
    """
    Create the necessary directory structure for database files.
    
    Args:
        database_path: Base path for database files
    """
    # Create the database directory if it doesn't exist
    if not os.path.exists(database_path):
        os.makedirs(database_path)
        print(f"Created directory: {database_path}")
    else:
        print(f"Directory already exists: {database_path}")
    
    # Create placeholder files with instructions
    placeholder_content = """# Gene-Phenotype Database Files

Place your database files here:

1. `phenotype.hpoa` - Map of genes to phenotypes
2. `phenotype_to_genes.txt` - Map of phenotypes to genes (optional)

You can download these files from the Human Phenotype Ontology (HPO) project:
https://hpo.jax.org/app/download/annotation
"""
    
    placeholder_path = os.path.join(database_path, "README.txt")
    with open(placeholder_path, 'w') as f:
        f.write(placeholder_content)
    
    print(f"Created placeholder file with instructions: {placeholder_path}")
    
    # Create a simple example database file for testing
    example_content = """ncbi_gene_id\tgene_symbol\thpo_id\thpo_name\tfrequency\tdisease_id
10\tNAT2\tHP:0000007\tAutosomal recessive inheritance\t-\tOMIM:243400
10\tNAT2\tHP:0001939\tAbnormality of metabolism/homeostasis\t-\tOMIM:243400
16\tAARS1\tHP:0002460\tDistal muscle weakness\t15/15\tOMIM:613287
16\tAARS1\tHP:0002451\tLimb dystonia\t3/3\tOMIM:616339
16\tAARS1\tHP:0008619\tBilateral sensorineural hearing impairment\t-\tOMIM:613287
"""
    
    example_path = os.path.join(database_path, "example.hpoa")
    with open(example_path, 'w') as f:
        f.write(example_content)
    
    print(f"Created example database file: {example_path}")
    print("\nTo test the database module with the example file:")
    print(f"1. Copy {example_path} to {os.path.join(database_path, 'phenotype.hpoa')}")
    print("2. Run: python -m src.database.test_gene_phenotype_db")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup database directories for GraPhens")
    parser.add_argument("--database-path", default="data/database", 
                        help="Path where database files should be stored")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    setup_database_directories(args.database_path)
    print("\nSetup complete!")

if __name__ == "__main__":
    main() 