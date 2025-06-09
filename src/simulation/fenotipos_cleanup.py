import json
import os
from typing import Dict, List
from tqdm import tqdm

def clean_phenotypes_file(
    input_file: str,
    output_file: str
) -> None:
    """
    Clean a processed phenotypes JSON file by removing entries that aren't valid HPO IDs.
    
    Args:
        input_file: Path to input JSON file (processed phenotypes)
        output_file: Path to output JSON file (cleaned phenotypes)
    """
    # Load input file
    print(f"Loading processed phenotypes from {input_file}")
    with open(input_file, 'r') as f:
        phenotypes_data = json.load(f)
    
    # Process each gene
    cleaned_data = {}
    total_genes = len(phenotypes_data)
    
    print(f"Cleaning phenotypes for {total_genes} genes...")
    for gene in tqdm(phenotypes_data.keys()):
        phenotype_lists = phenotypes_data[gene]
        cleaned_lists = []
        
        # Process each list of phenotypes for the gene
        for phenotype_list in phenotype_lists:
            # Keep only valid HPO IDs (starting with HP:)
            cleaned_phenotypes = [p for p in phenotype_list if p.startswith("HP:")]
            
            # Only add the list if it's not empty after cleaning
            if cleaned_phenotypes:
                cleaned_lists.append(cleaned_phenotypes)
        
        # Only add the gene if it has at least one non-empty phenotype list
        if cleaned_lists:
            cleaned_data[gene] = cleaned_lists
    
    # Save cleaned data
    print(f"\nSaving cleaned phenotypes to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    # Print statistics
    print_cleaning_stats(phenotypes_data, cleaned_data)

def print_cleaning_stats(original_data: Dict, cleaned_data: Dict) -> None:
    """Print statistics about the cleaning results."""
    original_genes = len(original_data)
    cleaned_genes = len(cleaned_data)
    
    original_phenotypes = 0
    kept_phenotypes = 0
    removed_phenotypes = 0
    
    # Count original phenotypes and categorize them
    for gene, phenotype_lists in original_data.items():
        for phenotype_list in phenotype_lists:
            for phenotype in phenotype_list:
                original_phenotypes += 1
                if phenotype.startswith("HP:"):
                    kept_phenotypes += 1
                else:
                    removed_phenotypes += 1
    
    # Calculate how many genes were completely removed
    removed_genes = original_genes - cleaned_genes
    
    print("\nCleaning Statistics:")
    print(f"Original genes: {original_genes}")
    print(f"Genes after cleaning: {cleaned_genes}")
    print(f"Genes removed completely: {removed_genes} ({removed_genes/original_genes*100:.1f}% of genes)")
    print(f"Original phenotypes: {original_phenotypes}")
    print(f"Phenotypes kept: {kept_phenotypes} ({kept_phenotypes/original_phenotypes*100:.1f}%)")
    print(f"Phenotypes removed: {removed_phenotypes} ({removed_phenotypes/original_phenotypes*100:.1f}%)")

if __name__ == "__main__":
    # You can adjust these file paths as needed
    input_file = "src/simulation/updated_fenotipos_processed_all-MiniLM-L6-v2_fast.json"
    output_file = "src/simulation/fenotipos_cleaned_processed.json"
    
    clean_phenotypes_file(
        input_file=input_file,
        output_file=output_file
    ) 