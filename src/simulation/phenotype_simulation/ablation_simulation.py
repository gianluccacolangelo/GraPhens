#!/usr/bin/env python
"""
This script demonstrates the composition of different simulation strategies 
to perform an ablation study on the effect of phenotype distances.

It defines and runs two simulation scenarios:
1. Uniform Count + Empirical Distances: Simulates patients with a uniform
   distribution of phenotype counts and empirically-derived phenotype distances.
2. Uniform Count + Specific Phenotypes Only: Simulates patients with a uniform
   distribution of phenotype counts but only selects the most specific
   phenotypes for a gene (distance 0).
"""
import logging
import sys
import os
import argparse
import json
import gc
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

from src.core.types import Phenotype
from src.simulation.phenotype_simulation.simulator import StandardPhenotypeSimulator
from src.simulation.phenotype_simulation.data_loader import PhenotypeDistributionDataLoader
from src.simulation.phenotype_simulation.selector import HPODistancePhenotypeSelector
from src.simulation.phenotype_simulation.distributions import (
    DistributionStrategy,
    UniformCountDistribution,
    UniformCountSpecificPhenotypesDistribution,
)
from src.ontology.hpo_graph import HPOGraphProvider
from src.simulation.gene_phenotype import GenePhenotypeFacade

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger('ablation_simulation')

# --- Configuration ---
# Default directories
DEFAULT_DATA_DIR_SIMULATION = "data/simulation"
DEFAULT_DATA_DIR_HPO = "data/ontology"
DEFAULT_DATA_DIR_GENE_PHENOTYPE = "data/"
DEFAULT_OUTPUT_DIR = "data/simulation/output"


def simulate_and_save(
    scenario_name: str,
    simulator: StandardPhenotypeSimulator,
    gene_to_count: Dict[str, int],
    output_dir: str,
    genes_to_simulate: List[str]
) -> Dict[str, List[List[str]]]:
    """Runs the simulation for a given scenario and saves the results."""
    
    logger.info(f"Simulating {sum(gene_to_count.values())} patients for {len(gene_to_count)} genes...")
    gene_patients_phenotypes = simulator.generate_patients(gene_to_count, use_tqdm=True)
    
    # Convert to JSON-serializable format (phenotype IDs)
    result = {
        gene: [[p.id for p in patient] for patient in patients]
        for gene, patients in gene_patients_phenotypes.items()
    }
    
    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gene_identifier = 'all' if len(genes_to_simulate) > 10 else '-'.join(genes_to_simulate)
    output_file = os.path.join(
        output_dir,
        f"simulated_patients_{scenario_name}_{gene_identifier}_{timestamp}.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved simulation results to {output_file}")
    
    return result


def print_simulation_stats(result: Dict[str, List[List[str]]]) -> None:
    """Prints summary statistics for the simulation results."""
    total_patients = 0
    all_phenotype_counts = []
    
    logger.info("\n--- Simulation Summary ---")
    for gene, patients in result.items():
        patient_count = len(patients)
        phenotype_counts = [len(p_ids) for p_ids in patients]
        all_phenotype_counts.extend(phenotype_counts)
        total_patients += patient_count

    logger.info(f"Total genes processed: {len(result)}")
    logger.info(f"Total patients simulated: {total_patients}")
    if all_phenotype_counts:
        logger.info(f"Overall phenotypes per patient: mean={np.mean(all_phenotype_counts):.2f}, std={np.std(all_phenotype_counts):.2f}")
    logger.info("-" * 26)


def main():
    """Main function to run the ablation simulation."""
    parser = argparse.ArgumentParser(description="Phenotype Simulation Ablation Study")
    
    # Directory arguments
    parser.add_argument("--data-dir-simulation", type=str, default=DEFAULT_DATA_DIR_SIMULATION)
    parser.add_argument("--data-dir-hpo", type=str, default=DEFAULT_DATA_DIR_HPO)
    parser.add_argument("--data-dir-gene-phenotype", type=str, default=DEFAULT_DATA_DIR_GENE_PHENOTYPE)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    
    # Simulation arguments
    gene_group = parser.add_mutually_exclusive_group(required=True)
    gene_group.add_argument("--genes", type=str, nargs="+", default=None, help="Gene symbols to simulate for")
    gene_group.add_argument("--all-genes", action="store_true", help="Simulate for all available genes")
    parser.add_argument("--patients-per-gene", type=int, default=100, help="Number of patients to simulate per gene")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--scenario", type=int, choices=[1, 2], default=None, help="Which scenario to run (1 or 2). Runs both if not specified.")

    args = parser.parse_args()
    
    # Update logger level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # --- Shared Components ---
    logger.info("Setting up shared simulation components...")
    data_loader = PhenotypeDistributionDataLoader(data_dir=args.data_dir_simulation)
    try:
        distribution_data = data_loader.load_distribution_data(
            count_file=None, distance_file='data/simulation/phenotype_distances.csv'
        )
    except FileNotFoundError:
        logger.error(f"Error: 'phenotype_distances.csv' not found in '{args.data_dir_simulation}'.")
        sys.exit(1)

    hpo_provider = HPOGraphProvider.get_instance(data_dir=args.data_dir_hpo)
    gene_phenotype_facade = GenePhenotypeFacade(data_dir=args.data_dir_gene_phenotype)
    phenotype_selector = HPODistancePhenotypeSelector(
        gene_phenotype_facade=gene_phenotype_facade, hpo_provider=hpo_provider
    )
    
    # Determine genes to simulate
    if args.all_genes:
        logger.info("Fetching all available genes...")
        genes_to_simulate = gene_phenotype_facade.get_available_genes()
        logger.info(f"Found {len(genes_to_simulate)} genes.")
    else:
        genes_to_simulate = args.genes
        
    gene_to_count = {gene: args.patients_per_gene for gene in genes_to_simulate}

    if args.scenario is None or args.scenario == 1:
        # --- Scenario 1: Uniform Distribution + Empirical Distances ---
        logger.info("\n=== Running Scenario 1: Uniform Count with Empirical Distances ===")
        dist_strat_1 = UniformCountDistribution(min_count=1, max_count=10)
        dist_strat_1.fit(distribution_data)
        simulator_1 = StandardPhenotypeSimulator(dist_strat_1, phenotype_selector)
        results_1 = simulate_and_save("uniform_with_distances", simulator_1, gene_to_count, args.output_dir, genes_to_simulate)
        print_simulation_stats(results_1)

    if args.scenario is None:
        # --- Memory Cleanup ---
        logger.info("\n=== Cleaning up memory before Scenario 2 ===")
        if 'results_1' in locals(): del results_1
        if 'simulator_1' in locals(): del simulator_1
        if 'dist_strat_1' in locals(): del dist_strat_1
        gc.collect()
        logger.info("Memory cleanup completed. Freed memory from Scenario 1 results.")

    if args.scenario is None or args.scenario == 2:
        # --- Scenario 2: Uniform Distribution + Specific Phenotypes Only ---
        logger.info("\n=== Running Scenario 2: Uniform Count with Specific Phenotypes Only ===")
        dist_strat_2 = UniformCountSpecificPhenotypesDistribution(min_count=1, max_count=10)
        # No fitting required for this strategy
        simulator_2 = StandardPhenotypeSimulator(dist_strat_2, phenotype_selector)
        results_2 = simulate_and_save("uniform_specific_only", simulator_2, gene_to_count, args.output_dir, genes_to_simulate)
        print_simulation_stats(results_2)
    
    # --- Final Cleanup ---
    logger.info("\n=== Final memory cleanup ===")
    if 'results_2' in locals(): del results_2
    if 'simulator_2' in locals(): del simulator_2
    if 'dist_strat_2' in locals(): del dist_strat_2
    gc.collect()
    
    logger.info("\nAblation simulation finished successfully.")

if __name__ == "__main__":
    main()
