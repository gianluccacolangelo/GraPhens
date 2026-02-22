#!/usr/bin/env python
"""Demo script for the phenotype simulation module."""
import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Any, Tuple
import time  # Import time

from src.simulation.phenotype_simulation.factory import SimulationFactory
from src.simulation.phenotype_simulation.data_loader import PhenotypeDistributionDataLoader
from src.core.types import Phenotype
from src.simulation.gene_phenotype import GenePhenotypeFacade
from src.ontology.hpo_graph import HPOGraphProvider
from src.ontology.hpo_updater import HPOUpdater


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('phenotype_simulation_demo')


def setup_directories(args: argparse.Namespace) -> Dict[str, str]:
    """Set up the necessary directories for the demo.
    
    Args:
        args: Command-line arguments with directory paths
        
    Returns:
        Dictionary mapping directory names to paths
    """
    dirs = {
        'simulation': args.data_dir_simulation,
        'hpo': args.data_dir_hpo,
        'gene_phenotype': args.data_dir_gene_phenotype,
        'output': args.output_dir
    }
    
    # Create directories if they don't exist
    for dir_path in dirs.values():
        if not os.path.exists(dir_path):
            logger.info(f"Creating directory: {dir_path}")
            os.makedirs(dir_path)
    
    return dirs


def update_hpo_if_needed(hpo_dir: str, force: bool = False) -> None:
    """Update HPO data if needed.
    
    Args:
        hpo_dir: Directory for HPO data
        force: Whether to force an update
    """
    updater = HPOUpdater(data_dir=hpo_dir)
    
    if force or updater.check_for_updates():
        logger.info("Updating HPO data...")
        updater.update(format_type="json")
    else:
        logger.info("HPO data is up-to-date")


def prepare_simulation_data(dirs: Dict[str, str], args: argparse.Namespace) -> None:
    """Prepare simulation data if example files don't exist.
    
    Creates example phenotype_counts.csv and phenotype_distances.csv
    files if they don't exist.
    
    Args:
        dirs: Directory paths
        args: Command-line arguments
    """
    # Check if files exist
    count_file = os.path.join(dirs['simulation'], 'phenotype_counts.csv')
    distance_file = os.path.join(dirs['simulation'], 'phenotype_distances.csv')
    
    # Generate example count data if file doesn't exist
    if not os.path.exists(count_file) or args.regenerate_data:
        logger.info(f"Generating example count data: {count_file}")
        
        # Create a sample distribution centered around 5 (mean)
        np.random.seed(42)  # For reproducibility
        counts = np.random.poisson(lam=5, size=1000)
        counts = np.clip(counts, 1, None)  # Ensure at least 1 phenotype
        
        # Save to CSV
        pd.DataFrame({'count': counts}).to_csv(count_file, index=False)
        logger.info(f"Saved example count data with {len(counts)} samples (mean: {counts.mean():.2f})")
    
    # Generate example distance data if file doesn't exist
    if not os.path.exists(distance_file) or args.regenerate_data:
        logger.info(f"Generating example distance data: {distance_file}")
        
        # Create a sample distribution with most weight at 0, then decaying
        np.random.seed(42)  # For reproducibility
        probabilities = [0.6, 0.25, 0.1, 0.05]  # 0, 1, 2, 3
        distances = np.random.choice(range(len(probabilities)), size=5000, p=probabilities)
        
        # Save to CSV
        pd.DataFrame({'distance': distances}).to_csv(distance_file, index=False)
        logger.info(f"Saved example distance data with {len(distances)} samples")


def simulate_and_save_patients(
    dirs: Dict[str, str],
    genes_to_simulate: List[str],
    args: argparse.Namespace
) -> Tuple[Dict[str, List[List[str]]], Dict[str, float]]:
    """Simulate patient phenotypes and save the results.
    
    Args:
        dirs: Directory paths
        genes_to_simulate: List of gene symbols to simulate for
        args: Command-line arguments
        
    Returns:
        Tuple containing:
            - Dictionary mapping gene symbols to lists of patient phenotype ID lists
            - Dictionary containing timing information for different stages (fitting, generation, saving, total)
    """
    timings = {}
    start_total_time = time.time()
    
    # --- Setup and Fitting ---
    fit_start_time = time.time()
    # Create the simulation components
    factory = SimulationFactory()
    
    # Create a complete simulator with data from CSV files
    simulator = factory.create_complete_simulator(
        data_dir_simulation=dirs['simulation'],
        data_dir_hpo=dirs['hpo'],
        data_dir_gene_phenotype=dirs['gene_phenotype'],
        distribution_type="empirical",
        count_file=args.count_file,
        distance_file=args.distance_file,
        count_column=args.count_column,
        distance_column=args.distance_column,
        selector_type="hpo_distance",
        selector_kwargs={
            "allow_duplicates": args.allow_duplicates,
            "max_attempts": args.max_attempts
        }
    )
    timings['fitting'] = time.time() - fit_start_time
    logger.info(f"Simulator setup and fitting took {timings['fitting']:.2f} seconds")
    
    # --- Patient Generation ---
    generation_start_time = time.time()
    # Create gene-to-count mapping for simulation
    gene_to_count = {gene: args.patients_per_gene for gene in genes_to_simulate}
    
    # Simulate patients
    total_patients_to_simulate = sum(gene_to_count.values())
    logger.info(f"Simulating {total_patients_to_simulate} patients for {len(genes_to_simulate)} genes ({args.patients_per_gene} per gene)")
    gene_patients_phenotypes = simulator.generate_patients(gene_to_count, use_tqdm=True)
    timings['generation'] = time.time() - generation_start_time
    logger.info(f"Patient generation took {timings['generation']:.2f} seconds")
    
    # --- Convert and Save Results ---
    save_start_time = time.time()
    # Convert to the desired output format (dict of lists of phenotype IDs)
    result = {}
    for gene, patients in gene_patients_phenotypes.items():
        gene_data = []
        for patient_phenotypes in patients:
            # Extract just the phenotype IDs
            phenotype_ids = [p.id for p in patient_phenotypes]
            gene_data.append(phenotype_ids)
        result[gene] = gene_data
    
    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gene_identifier = 'all' if args.all_genes else '-'.join(genes_to_simulate)
    output_file = os.path.join(
        dirs['output'],
        f"simulated_patients_{gene_identifier}_{timestamp}.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    timings['saving'] = time.time() - save_start_time
    logger.info(f"Saved simulated patient data (phenotype IDs) to {output_file} (took {timings['saving']:.2f}s)")
    
    timings['total'] = time.time() - start_total_time
    logger.info(f"Total process completed in {timings['total']:.2f} seconds")
    
    return result, timings


def print_simulation_stats(result: Dict[str, List[List[str]]], timings: Dict[str, float]) -> None:
    """Print statistics about the simulation results.
    
    Args:
        result: Dictionary mapping gene symbols to lists of patient phenotype ID lists
        timings: Dictionary containing timing information for different stages
    """
    total_patients = 0
    all_phenotype_counts = []
    
    logger.info("\n--- Simulation Summary ---")
    # Limit printing individual gene stats if there are too many
    max_genes_to_print = 10
    genes_printed = 0
    for gene, patients in result.items():
        patient_count = len(patients)
        phenotype_counts = [len(phenotype_ids) for phenotype_ids in patients]
        all_phenotype_counts.extend(phenotype_counts)
        total_patients += patient_count
        
        if genes_printed < max_genes_to_print:
            logger.info(f"Gene {gene}:")
            logger.info(f"  Number of simulated patients: {patient_count}")
            if phenotype_counts:
                logger.info(f"  Phenotypes per patient: min={min(phenotype_counts)}, max={max(phenotype_counts)}, mean={np.mean(phenotype_counts):.2f}")
            else:
                logger.info("  Phenotypes per patient: N/A (no patients generated)")
            genes_printed += 1
        elif genes_printed == max_genes_to_print:
            logger.info(f"... (stats for remaining {len(result) - max_genes_to_print} genes omitted)")
            genes_printed += 1
    
    # Overall statistics
    logger.info("\n--- Overall Performance ---")
    logger.info(f"Total genes processed: {len(result)}")
    logger.info(f"Total patients simulated: {total_patients}")
    if all_phenotype_counts:
        logger.info(f"Overall phenotypes per patient: mean={np.mean(all_phenotype_counts):.2f}, std={np.std(all_phenotype_counts):.2f}")
    
    logger.info(f"\n--- Timing Breakdown ---")
    logger.info(f"Setup & Fitting: {timings.get('fitting', 0):.2f} seconds")
    logger.info(f"Patient Generation: {timings.get('generation', 0):.2f} seconds")
    logger.info(f"Result Saving: {timings.get('saving', 0):.2f} seconds")
    logger.info(f"-------------------------")
    logger.info(f"Total Time: {timings.get('total', 0):.2f} seconds")
    
    if total_patients > 0 and timings.get('total', 0) > 0:
        avg_time_per_patient = timings['total'] / total_patients
        logger.info(f"Average time per patient (overall): {avg_time_per_patient:.4f} seconds")
    if total_patients > 0 and timings.get('generation', 0) > 0:
        avg_gen_time_per_patient = timings['generation'] / total_patients
        logger.info(f"Average time per patient (generation only): {avg_gen_time_per_patient:.4f} seconds")


def main():
    """Run the phenotype simulation demo."""
    parser = argparse.ArgumentParser(description="Phenotype Simulation Demo")
    
    # Directory arguments
    parser.add_argument("--data-dir-simulation", type=str, default="results/",
                        help="Directory for simulation data files")
    parser.add_argument("--data-dir-hpo", type=str, default="data/ontology",
                        help="Directory for HPO data")
    parser.add_argument("--data-dir-gene-phenotype", type=str, default="data/",
                        help="Directory for gene-phenotype data")
    parser.add_argument("--output-dir", type=str, default="data/simulation/output",
                        help="Directory for output files")
    
    # Simulation arguments
    gene_group = parser.add_mutually_exclusive_group(required=True)
    gene_group.add_argument("--genes", type=str, nargs="+", default=None,
                        help="Gene symbols to simulate patients for")
    gene_group.add_argument("--all-genes", action="store_true",
                        help="Simulate patients for all available genes in the database")
    parser.add_argument("--patients-per-gene", type=int, default=10,
                        help="Number of patients to simulate per gene")
    parser.add_argument("--allow-duplicates", action="store_true",
                        help="Allow the same phenotype to be selected multiple times")
    parser.add_argument("--max-attempts", type=int, default=10,
                        help="Maximum attempts to find phenotypes at a specific distance")
    
    # Setup arguments
    parser.add_argument("--update-hpo", action="store_true",
                        help="Force update of HPO data")
    parser.add_argument("--regenerate-data", action="store_true",
                        help="Regenerate example data files")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set the logging level (e.g., DEBUG for performance tracing)")

    # Distribution arguments
    parser.add_argument("--count-file", type=str, default=None,
                        help="Path to custom phenotype count distribution CSV file")
    parser.add_argument("--distance-file", type=str, default=None,
                        help="Path to custom phenotype distance distribution CSV file")
    parser.add_argument("--count-column", type=str, default="count",
                        help="Column name for phenotype counts in the CSV file")
    parser.add_argument("--distance-column", type=str, default="distance",
                        help="Column name for phenotype distances in the CSV file")
    
    args = parser.parse_args()
    
    # Update root logger level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    logger.info("Starting phenotype simulation demo")
    logger.info(f"Log level set to: {args.log_level}")
    
    # Set up directories
    dirs = setup_directories(args)
    
    # Update HPO data if needed
    update_hpo_if_needed(dirs['hpo'], args.update_hpo)
    
    # Prepare simulation data
    prepare_simulation_data(dirs, args)
    
    # Determine which genes to simulate
    if args.all_genes:
        logger.info("Fetching all available genes from the database...")
        facade = GenePhenotypeFacade(data_dir=dirs['gene_phenotype'])
        genes_to_simulate = facade.get_available_genes()
        if not genes_to_simulate:
            logger.error("Could not retrieve any genes from the database. Exiting.")
            sys.exit(1)
        logger.info(f"Found {len(genes_to_simulate)} genes to simulate.")
    else:
        # Use the provided --genes list (which is now required if --all-genes is not set)
        genes_to_simulate = args.genes
    
    # Simulate patients
    result, timings = simulate_and_save_patients(dirs, genes_to_simulate, args)
    
    # Print statistics
    print_simulation_stats(result, timings)
    
    logger.info("Demo completed successfully")


if __name__ == "__main__":
    main() 