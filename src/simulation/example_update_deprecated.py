#!/usr/bin/env python3
import logging
import argparse
from update_deprecated_hpo import HPOTermUpdater

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Run the HPO term update process with command-line arguments.
    
    This script demonstrates how to use the HPOTermUpdater to check for
    and replace deprecated HPO terms in the gene-phenotype database and
    the fenotipos JSON file.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Update deprecated HPO terms in datasets')
    
    parser.add_argument('--gene-phenotype-dir', 
                       default='data/gene_phenotype_specific',
                       help='Directory containing gene-phenotype database files')
    
    parser.add_argument('--fenotipos-file',
                       default='src/simulation/fenotipos_cleaned_processed.json',
                       help='Path to the fenotipos JSON file')
    
    parser.add_argument('--skip-database', action='store_true',
                       help='Skip updating the gene-phenotype database')
    
    parser.add_argument('--skip-json', action='store_true',
                       help='Skip updating the fenotipos JSON file')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose debug logging')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Initialize the updater
    updater = HPOTermUpdater(
        gene_phenotype_dir=args.gene_phenotype_dir,
        fenotipos_file=args.fenotipos_file
    )
    
    logger.info("Starting HPO term update process")
    
    # Update the gene-phenotype database unless skipped
    if not args.skip_database:
        updater.update_database()
    else:
        logger.info("Skipping gene-phenotype database update")
    
    # Update the fenotipos JSON file unless skipped
    if not args.skip_json:
        updater.update_fenotipos_json()
    else:
        logger.info("Skipping fenotipos JSON file update")
    
    # Print summary
    logger.info("HPO term update process complete!")
    
    if not args.skip_database:
        logger.info(f"Database: Checked {updater.db_terms_checked} terms, replaced {updater.db_terms_replaced} deprecated terms")
    
    if not args.skip_json:
        logger.info(f"JSON file: Checked {updater.json_terms_checked} terms, replaced {updater.json_terms_replaced} deprecated terms")
    
    total_checked = (0 if args.skip_database else updater.db_terms_checked) + \
                   (0 if args.skip_json else updater.json_terms_checked)
    
    total_replaced = (0 if args.skip_database else updater.db_terms_replaced) + \
                    (0 if args.skip_json else updater.json_terms_replaced)
    
    logger.info(f"Total: Checked {total_checked} terms, replaced {total_replaced} deprecated terms")

if __name__ == "__main__":
    main() 