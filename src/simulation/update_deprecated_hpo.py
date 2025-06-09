#!/usr/bin/env python3
import os
import json
import logging
import pandas as pd
from typing import Dict, List, Set, Any, Tuple
from pathlib import Path

from src.ontology.check_deprecated import CheckDeprecated
from src.simulation.gene_phenotype.facade import GenePhenotypeFacade
from src.simulation.gene_phenotype.database import GenePhenotypeDatabase

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HPOTermUpdater:
    """
    Updates deprecated HPO terms in gene-phenotype data files.
    
    This class checks gene-phenotype datasets for deprecated HPO terms
    and replaces them with their updated versions.
    """
    
    def __init__(self, 
                gene_phenotype_dir: str = "data/gene_phenotype_specific/",
                fenotipos_file: str = "src/simulation/fenotipos_cleaned_processed.json",
                tsv_file: str = None):
        """
        Initialize the HPO term updater.
        
        Args:
            gene_phenotype_dir: Directory containing gene-phenotype database files
            fenotipos_file: Path to the fenotipos JSON file
            tsv_file: Path to a TSV file with format: ID, Gene, Phenotype
        """
        self.gene_phenotype_dir = gene_phenotype_dir
        self.fenotipos_file = fenotipos_file
        self.tsv_file = tsv_file
        
        # Initialize facade and deprecated term checker
        self.facade = GenePhenotypeFacade(data_dir=gene_phenotype_dir)
        # Get direct access to the database for advanced operations
        self.database = self.facade.database
        self.checker = CheckDeprecated()
        
        # Track stats
        self.db_terms_checked = 0
        self.db_terms_replaced = 0
        self.json_terms_checked = 0
        self.json_terms_replaced = 0
        self.tsv_terms_checked = 0
        self.tsv_terms_replaced = 0
        
        # Alternative ID mapping cache
        self.alternative_id_map = {}
        self._load_alternative_ids()
        
    def _load_alternative_ids(self):
        """
        Load alternative IDs from the HPO ontology file.
        Builds a mapping from alternative ID to current ID.
        """
        logger.info("Loading alternative HPO IDs from ontology...")
        try:
            # Get path to the HPO json file
            json_file = os.path.join(self.gene_phenotype_dir, "ontology", "hp.json")
            if not os.path.exists(json_file):
                logger.warning(f"HPO JSON file not found at {json_file}, alternative ID detection disabled")
                return
                
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Build alternative ID mapping
            alt_ids_found = 0
            
            # Handle the graph-based JSON structure
            if "graphs" in data:
                for graph in data["graphs"]:
                    for node in graph.get("nodes", []):
                        # Get current ID
                        if "id" not in node:
                            continue
                            
                        current_id = node["id"]
                        if "/" in current_id:
                            current_id = current_id.split("/")[-1]
                        current_id = current_id.replace("_", ":", 1) if "_" in current_id else current_id
                        
                        # Skip non-HPO terms
                        if not current_id.startswith("HP:"):
                            continue
                            
                        # Look for alternative IDs in meta data
                        meta = node.get("meta", {})
                        for basic_value in meta.get("basicPropertyValues", []):
                            if basic_value.get("pred") == "http://www.geneontology.org/formats/oboInOwl#hasAlternativeId":
                                alt_id = basic_value.get("val")
                                if alt_id and alt_id.startswith("HP:"):
                                    self.alternative_id_map[alt_id] = current_id
                                    alt_ids_found += 1
            
            logger.info(f"Loaded {alt_ids_found} alternative HPO IDs")
            
        except Exception as e:
            logger.error(f"Error loading alternative HPO IDs: {str(e)}")
    
    def check_and_replace_all(self, phenotype_id: str) -> Tuple[bool, str]:
        """
        Comprehensive check for both deprecated terms and alternative IDs.
        
        Args:
            phenotype_id: HPO term ID to check
            
        Returns:
            Tuple of (needs_replacing, replacement_id)
            If needs_replacing is False, replacement_id will be the same as phenotype_id
        """
        # First check if it's a deprecated term
        is_deprecated, replacement = self.checker.check_and_replace(phenotype_id)
        
        if is_deprecated and replacement:
            return True, replacement
            
        # If not deprecated, check if it's an alternative ID
        if phenotype_id in self.alternative_id_map:
            logger.debug(f"Found alternative ID: {phenotype_id} → {self.alternative_id_map[phenotype_id]}")
            return True, self.alternative_id_map[phenotype_id]
            
        # No replacement needed
        return False, phenotype_id
        
    def update_database(self) -> None:
        """
        Update deprecated HPO terms in the gene-phenotype database.
        
        This method processes all genes in the database, checks if their
        associated phenotypes are deprecated, and replaces them if needed.
        """
        logger.info(f"Updating HPO terms in gene-phenotype database at {self.gene_phenotype_dir}")
        
        # Make sure the database is loaded
        if self.database._genes_to_phenotype is None:
            self.database.load()
            
        # First, identify all deprecated terms and their replacements
        replacements = {}
        phenotype_ids = self.database.get_all_phenotype_ids()
        logger.info(f"Checking {len(phenotype_ids)} unique phenotype IDs for deprecation")
        
        for phenotype_id in phenotype_ids:
            self.db_terms_checked += 1
            # Use enhanced check method to get both deprecated and alternative IDs
            needs_replacing, replacement = self.check_and_replace_all(phenotype_id)
            
            if needs_replacing:
                self.db_terms_replaced += 1
                replacements[phenotype_id] = replacement
                logger.debug(f"Found term {phenotype_id} to be replaced with {replacement}")
        
        logger.info(f"Found {len(replacements)} terms that need to be replaced")
        
        if not replacements:
            logger.info("No deprecated or alternative terms found in the database. No changes needed.")
            return
            
        # Process each replacement
        for old_id, new_id in replacements.items():
            logger.info(f"Replacing term {old_id} with {new_id}")
            
            # Get all genes that have the deprecated term
            genes_with_term = []
            for _, row in self.database._genes_to_phenotype[self.database._genes_to_phenotype['hpo_id'] == old_id].iterrows():
                genes_with_term.append({
                    'gene_symbol': row['gene_symbol'],
                    'ncbi_gene_id': row['ncbi_gene_id'],
                    'frequency': row['frequency'],
                    'disease_id': row['disease_id'],
                    'hpo_name': row['hpo_name']
                })
                
            # Get the name of the new term if it exists in the database
            new_term_name = None
            if new_id in self.database._genes_to_phenotype['hpo_id'].values:
                new_term_name = self.database._genes_to_phenotype[
                    self.database._genes_to_phenotype['hpo_id'] == new_id
                ]['hpo_name'].iloc[0]
            else:
                # Get the name from the checker's terms dictionary
                if new_id in self.checker.terms:
                    new_term_name = self.checker.terms[new_id].get("label", "Unknown")
                else:
                    new_term_name = f"Replacement for {old_id}"
                    
            # Remove all occurrences of the deprecated term
            self.database._genes_to_phenotype = self.database._genes_to_phenotype[
                self.database._genes_to_phenotype['hpo_id'] != old_id
            ]
            
            self.database._phenotype_to_genes = self.database._phenotype_to_genes[
                self.database._phenotype_to_genes['hpo_id'] != old_id
            ]
            
            # Add the replacement term for each gene
            for gene_info in genes_with_term:
                # Create new row for genes_to_phenotype
                new_row_g2p = pd.DataFrame({
                    'ncbi_gene_id': [gene_info['ncbi_gene_id']],
                    'gene_symbol': [gene_info['gene_symbol']],
                    'hpo_id': [new_id],
                    'hpo_name': [new_term_name],
                    'frequency': [gene_info['frequency']],
                    'disease_id': [gene_info['disease_id']]
                })
                
                # Create new row for phenotype_to_genes
                new_row_p2g = pd.DataFrame({
                    'hpo_id': [new_id],
                    'hpo_name': [new_term_name],
                    'ncbi_gene_id': [gene_info['ncbi_gene_id']],
                    'gene_symbol': [gene_info['gene_symbol']],
                    'disease_id': [gene_info['disease_id']]
                })
                
                # Append to both dataframes
                self.database._genes_to_phenotype = pd.concat(
                    [self.database._genes_to_phenotype, new_row_g2p], ignore_index=True
                )
                
                self.database._phenotype_to_genes = pd.concat(
                    [self.database._phenotype_to_genes, new_row_p2g], ignore_index=True
                )
        
        # Save changes to the database
        logger.info(f"Saving updated gene-phenotype database...")
        
        # First create backup of original files
        for filename in ["genes_to_phenotype.txt", "phenotype_to_genes.txt"]:
            original_file = os.path.join(self.gene_phenotype_dir, filename)
            backup_file = os.path.join(self.gene_phenotype_dir, f"{filename}.bak")
            
            if os.path.exists(original_file):
                logger.info(f"Creating backup of {original_file} → {backup_file}")
                os.rename(original_file, backup_file)
        
        # Save changes and log the files that were updated
        saved_files = self.database.save()
        for file_path in saved_files:
            logger.info(f"Updated database file: {file_path}")
            
        logger.info(f"Database updated successfully. Checked {self.db_terms_checked} terms, replaced {self.db_terms_replaced} deprecated terms.")
        
    def update_fenotipos_json(self) -> None:
        """
        Update deprecated HPO terms in the fenotipos JSON file.
        
        This method processes the fenotipos_cleaned_processed.json file,
        checks if any HPO terms are deprecated, and replaces them if needed.
        """
        logger.info(f"Updating HPO terms in fenotipos file at {self.fenotipos_file}")
        
        # Check if the file exists
        if not os.path.exists(self.fenotipos_file):
            logger.error(f"Fenotipos file not found: {self.fenotipos_file}")
            return
            
        try:
            # Load the JSON file
            with open(self.fenotipos_file, 'r') as f:
                data = json.load(f)
                
            # Track if any changes were made
            changes_made = False
            
            # Process each gene
            for gene, phenotype_lists in data.items():
                # Process each list of phenotypes
                for i, phenotype_list in enumerate(phenotype_lists):
                    # Process each phenotype ID in the list
                    for j, phenotype_id in enumerate(phenotype_list):
                        self.json_terms_checked += 1
                        
                        # Check if deprecated or alternative ID
                        needs_replacing, replacement = self.check_and_replace_all(phenotype_id)
                        
                        if needs_replacing:
                            self.json_terms_replaced += 1
                            logger.debug(f"Replacing term {phenotype_id} with {replacement} for gene {gene}")
                            
                            # Update the phenotype ID
                            data[gene][i][j] = replacement
                            changes_made = True
            
            # Save the updated JSON file
            if changes_made:
                # Create backup of original file
                backup_path = f"{self.fenotipos_file}.bak"
                logger.info(f"Creating backup of original file: {self.fenotipos_file} → {backup_path}")
                os.rename(self.fenotipos_file, backup_path)
                
                # Save updated data
                logger.info(f"Saving updated fenotipos file to: {self.fenotipos_file}")
                with open(self.fenotipos_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                logger.info(f"Fenotipos file updated successfully.")
            else:
                logger.info(f"No deprecated or alternative terms found in fenotipos file. No changes made.")
                
            logger.info(f"Fenotipos file processing complete. Checked {self.json_terms_checked} terms, replaced {self.json_terms_replaced} terms.")
                
        except Exception as e:
            logger.error(f"Error updating fenotipos file: {str(e)}")
            
    def update_tsv_file(self) -> None:
        """
        Update deprecated HPO terms in a TSV file with format: ID, Gene, Phenotype.
        
        This method processes each row of the TSV file, checks if any HPO terms 
        in the Phenotype column are deprecated, and replaces them if needed.
        """
        if not self.tsv_file:
            logger.info("No TSV file provided. Skipping TSV update.")
            return
            
        logger.info(f"Updating HPO terms in TSV file at {self.tsv_file}")
        
        # Check if the file exists
        if not os.path.exists(self.tsv_file):
            logger.error(f"TSV file not found: {self.tsv_file}")
            return
            
        try:
            # Load the TSV file
            df = pd.read_csv(self.tsv_file, sep='\t')
            
            # Check if required columns exist
            required_columns = ['ID', 'Gene', 'Phenotype']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Required column '{col}' not found in TSV file")
                    return
                    
            # Track if any changes were made
            changes_made = False
            
            # Process each row
            for idx, row in df.iterrows():
                # Get the comma-separated phenotype list
                phenotypes_str = row['Phenotype']
                phenotypes = phenotypes_str.split(',')
                
                # Process each phenotype ID
                updated_phenotypes = []
                for phenotype_id in phenotypes:
                    phenotype_id = phenotype_id.strip()
                    self.tsv_terms_checked += 1
                    
                    # Check if deprecated or alternative ID
                    needs_replacing, replacement = self.check_and_replace_all(phenotype_id)
                    
                    if needs_replacing:
                        self.tsv_terms_replaced += 1
                        logger.debug(f"Replacing term {phenotype_id} with {replacement} for {row['ID']}")
                        updated_phenotypes.append(replacement)
                        changes_made = True
                    else:
                        updated_phenotypes.append(phenotype_id)
                
                # Update the row if changes were made
                if phenotypes != updated_phenotypes:
                    df.at[idx, 'Phenotype'] = ','.join(updated_phenotypes)
            
            # Save the updated TSV file
            if changes_made:
                # Create backup of original file
                backup_path = f"{self.tsv_file}.bak"
                logger.info(f"Creating backup of original file: {self.tsv_file} → {backup_path}")
                os.rename(self.tsv_file, backup_path)
                
                # Save updated data
                logger.info(f"Saving updated TSV file to: {self.tsv_file}")
                df.to_csv(self.tsv_file, sep='\t', index=False)
                
                logger.info(f"TSV file updated successfully.")
            else:
                logger.info(f"No deprecated or alternative terms found in TSV file. No changes made.")
                
            logger.info(f"TSV file processing complete. Checked {self.tsv_terms_checked} terms, replaced {self.tsv_terms_replaced} terms.")
                
        except Exception as e:
            logger.error(f"Error updating TSV file: {str(e)}")
            
    def run(self) -> None:
        """Run the complete HPO term update process on all files."""
        logger.info("Starting HPO term update process")
        
        # Update the gene-phenotype database
        self.update_database()
        
        # Update the fenotipos JSON file
        self.update_fenotipos_json()
        
        # Update the TSV file if provided
        if self.tsv_file:
            self.update_tsv_file()
        
        # Print summary
        logger.info("HPO term update process complete!")
        logger.info(f"Database: Checked {self.db_terms_checked} terms, replaced {self.db_terms_replaced} terms")
        logger.info(f"JSON file: Checked {self.json_terms_checked} terms, replaced {self.json_terms_replaced} terms")
        if self.tsv_file:
            logger.info(f"TSV file: Checked {self.tsv_terms_checked} terms, replaced {self.tsv_terms_replaced} terms")
        total_checked = self.db_terms_checked + self.json_terms_checked + self.tsv_terms_checked
        total_replaced = self.db_terms_replaced + self.json_terms_replaced + self.tsv_terms_replaced
        logger.info(f"Total: Checked {total_checked} terms, replaced {total_replaced} terms")
        
def main():
    """Run the HPO term update process."""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Update deprecated HPO terms in data files")
    parser.add_argument("--gene-phenotype-dir", type=str, default="data/",
                      help="Directory containing gene-phenotype database files")
    parser.add_argument("--fenotipos-file", type=str, 
                      default="src/simulation/fenotipos_cleaned_processed.json",
                      help="Path to the fenotipos JSON file")
    parser.add_argument("--tsv-file", type=str, default=None,
                      help="Path to a TSV file with format: ID, Gene, Phenotype")
    
    args = parser.parse_args()
    
    updater = HPOTermUpdater(
        gene_phenotype_dir=args.gene_phenotype_dir,
        fenotipos_file=args.fenotipos_file,
        tsv_file=args.tsv_file
    )
    updater.run()
    
if __name__ == "__main__":
    main() 