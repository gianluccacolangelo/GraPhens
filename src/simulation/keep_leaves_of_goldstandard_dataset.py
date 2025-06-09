import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd

from src.simulation.gene_phenotype.facade import GenePhenotypeFacade
from src.ontology.hpo_graph import HPOGraphProvider
from src.augmentation.hpo_augmentation import HPOAugmentationService
from src.core.types import Phenotype


def setup_logging(verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def get_leaf_phenotypes_for_gene(gene: str, facade: GenePhenotypeFacade, 
                               hpo_provider: HPOGraphProvider,
                               augmentation_service: HPOAugmentationService) -> Set[str]:
    """
    Get only the most specific (leaf) phenotypes for a gene in its augmented subgraph.
    
    The process works as follows:
    1. Get all phenotypes associated with the gene from the database
    2. Use the augmentation service to build a connected subgraph by adding ancestors
       of these phenotypes (this creates the gene-specific HPO subgraph)
    3. Within this augmented subgraph, identify which of the original phenotypes 
       are leaves by checking if they have any children that are also in the 
       original set
    
    A phenotype is considered a leaf if none of its children in the augmented
    subgraph are present in the original set of phenotypes. This ensures we keep
    only the most specific phenotypes that were actually observed/reported for
    the gene, even if they're not leaves in the full HPO graph.
    
    Args:
        gene: Gene symbol
        facade: GenePhenotypeFacade instance
        hpo_provider: HPOGraphProvider instance
        augmentation_service: HPOAugmentationService instance
        
    Returns:
        Set of HPO IDs representing the most specific phenotypes for this gene
        within its augmented subgraph
    """
    # Get all phenotypes for this gene
    all_phenotypes = facade.get_phenotypes_for_gene(gene)
    
    # If no phenotypes, return empty set
    if not all_phenotypes:
        return set()
        
    # Extract phenotype IDs
    phenotype_ids = {p.id for p in all_phenotypes}
    
    # For each phenotype, check if any of its HPO descendants are also in our phenotype set
    # If none are, then it's a leaf in the gene-specific subgraph
    leaf_phenotypes = set()
    for phenotype_id in phenotype_ids:
        # Get all descendants of this phenotype in the entire HPO graph
        all_descendants = hpo_provider.get_descendants(phenotype_id)
        
        # Check if any of these descendants are in our gene-specific phenotype set
        # If there's no intersection, this is a leaf for this gene
        if not all_descendants.intersection(phenotype_ids):
            leaf_phenotypes.add(phenotype_id)
    
    return leaf_phenotypes


def remove_non_leaf_phenotypes(gene: str, leaf_phenotypes: Set[str], 
                             database) -> int:
    """
    Remove non-leaf phenotypes from the database for a given gene.
    
    Args:
        gene: Gene symbol
        leaf_phenotypes: Set of leaf phenotype IDs to keep
        database: GenePhenotypeDatabase instance
        
    Returns:
        Number of phenotypes removed
    """
    # Ensure the database is loaded
    if database._genes_to_phenotype is None:
        database.load()
    
    # Get current gene rows
    gene_rows = database._genes_to_phenotype[
        database._genes_to_phenotype['gene_symbol'] == gene
    ]
    
    # Count phenotypes to be removed
    to_remove = gene_rows[~gene_rows['hpo_id'].isin(leaf_phenotypes)]
    removed_count = len(to_remove)
    
    if removed_count > 0:
        # Filter out the non-leaf phenotypes
        database._genes_to_phenotype = database._genes_to_phenotype[
            ~((database._genes_to_phenotype['gene_symbol'] == gene) & 
              (~database._genes_to_phenotype['hpo_id'].isin(leaf_phenotypes)))
        ]
        
        # Do the same for phenotype_to_genes
        database._phenotype_to_genes = database._phenotype_to_genes[
            ~((database._phenotype_to_genes['gene_symbol'] == gene) & 
              (~database._phenotype_to_genes['hpo_id'].isin(leaf_phenotypes)))
        ]
    
    return removed_count


def process_all_genes(facade: GenePhenotypeFacade, 
                     hpo_provider: HPOGraphProvider,
                     augmentation_service: HPOAugmentationService,
                     logger) -> Dict[str, int]:
    """
    Process all genes in the database to keep only the most specific phenotypes.
    
    Args:
        facade: GenePhenotypeFacade instance
        hpo_provider: HPOGraphProvider instance
        logger: Logger instance
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'genes_processed': 0,
        'genes_with_phenotypes': 0,
        'phenotypes_before': 0,
        'phenotypes_after': 0,
        'phenotypes_removed': 0
    }
    
    # Pre-load the database
    facade.database._genes_to_phenotype
    
    # Get all available genes
    genes = facade.get_available_genes()
    total_genes = len(genes)
    logger.info(f"Processing {total_genes} genes")
    
    # Process each gene
    for i, gene in enumerate(genes, 1):
        # Get all current phenotypes for this gene
        current_phenotypes = facade.get_phenotypes_for_gene(gene)
        current_count = len(current_phenotypes)
        stats['phenotypes_before'] += current_count
        
        if current_count > 0:
            stats['genes_with_phenotypes'] += 1
            
            # Get only the most specific (leaf) phenotypes for this gene
            leaf_phenotypes = get_leaf_phenotypes_for_gene(gene, facade, hpo_provider, augmentation_service)
            
            # Log detailed info in verbose mode
            if logger.level == logging.DEBUG:
                phenotype_names = {p.id: p.name for p in current_phenotypes}
                logger.debug(f"\nGene: {gene}")
                logger.debug(f"Total phenotypes: {current_count}")
                logger.debug(f"Leaf phenotypes: {len(leaf_phenotypes)}")
                logger.debug("Removed phenotypes:")
                for p_id in set(p.id for p in current_phenotypes) - leaf_phenotypes:
                    logger.debug(f"  - {p_id}: {phenotype_names.get(p_id, 'Unknown')}")
            
            # Remove non-leaf phenotypes
            removed = remove_non_leaf_phenotypes(gene, leaf_phenotypes, facade.database)
            stats['phenotypes_removed'] += removed
            stats['phenotypes_after'] += (current_count - removed)
        
        stats['genes_processed'] += 1
        
        # Log progress periodically
        if i % 10 == 0 or i == total_genes:
            logger.info(f"Processed {i}/{total_genes} genes")
    
    return stats


def save_database(database, output_dir: Path, logger):
    """
    Save the updated database to output files.
    
    Args:
        database: GenePhenotypeDatabase instance
        output_dir: Output directory path
        logger: Logger instance
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save genes_to_phenotype.txt
    genes_to_phenotype_path = output_dir / "genes_to_phenotype.txt"
    database._genes_to_phenotype.to_csv(
        genes_to_phenotype_path, 
        sep='\t', 
        index=False
    )
    logger.info(f"Saved genes to phenotype data to {genes_to_phenotype_path}")
    
    # Save phenotype_to_genes.txt
    phenotype_to_genes_path = output_dir / "phenotype_to_genes.txt"
    database._phenotype_to_genes.to_csv(
        phenotype_to_genes_path, 
        sep='\t', 
        index=False
    )
    logger.info(f"Saved phenotype to genes data to {phenotype_to_genes_path}")


def main():
    parser = argparse.ArgumentParser(description="Keep only the most specific phenotypes for each gene")
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Directory containing gene-phenotype database files"
    )
    parser.add_argument(
        "--ontology-dir",
        default="data/ontology",
        help="Directory containing HPO ontology files"
    )
    parser.add_argument(
        "--output-dir",
        default="data/gene_phenotype_specific",
        help="Output directory for refined database files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without saving changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    # Validate input directories
    data_dir = Path(args.data_dir)
    ontology_dir = Path(args.ontology_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.is_dir():
        raise ValueError(f"Data directory not found: {data_dir}")
    if not ontology_dir.is_dir():
        raise ValueError(f"Ontology directory not found: {ontology_dir}")
    
    # Initialize components
    logger.info("Initializing components")
    facade = GenePhenotypeFacade(data_dir=str(data_dir))
    hpo_provider = HPOGraphProvider(data_dir=str(ontology_dir))
    
    # Initialize the augmentation service
    augmentation_service = HPOAugmentationService(
        data_dir=str(ontology_dir),
        include_ancestors=True,
        include_descendants=False
    )
    
    # Make sure HPO graph is loaded
    if not hpo_provider.load():
        raise ValueError("Failed to load HPO ontology graph")
    
    # Process all genes
    logger.info("Processing genes to keep only the most specific phenotypes")
    stats = process_all_genes(facade, hpo_provider, augmentation_service, logger)
    
    # Print statistics
    logger.info("\nRefinement Statistics:")
    logger.info(f"Genes processed: {stats['genes_processed']}")
    logger.info(f"Genes with phenotypes: {stats['genes_with_phenotypes']}")
    logger.info(f"Phenotypes before: {stats['phenotypes_before']}")
    logger.info(f"Phenotypes after: {stats['phenotypes_after']}")
    
    if stats['phenotypes_before'] > 0:
        removal_percentage = (stats['phenotypes_removed'] / stats['phenotypes_before']) * 100
        logger.info(f"Phenotypes removed: {stats['phenotypes_removed']} ({removal_percentage:.1f}%)")
    
    # Save the updated database
    if not args.dry_run:
        logger.info(f"Saving refined database to {output_dir}")
        save_database(facade.database, output_dir, logger)
    else:
        logger.info("Dry run - no changes were saved")


if __name__ == "__main__":
    main()
