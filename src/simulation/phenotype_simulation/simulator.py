"""Main phenotype simulator implementation."""
import logging
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm import tqdm
import time  # Import time module

from src.core.types import Phenotype
from src.simulation.phenotype_simulation.interfaces import (
    PhenotypeSimulator, 
    DistributionStrategy, 
    PhenotypeSelector
)


logger = logging.getLogger(__name__)


class StandardPhenotypeSimulator(PhenotypeSimulator):
    """Standard implementation of the phenotype simulator.
    
    Combines a distribution strategy for sampling phenotype counts and distances,
    with a phenotype selector for choosing specific phenotypes.
    """
    
    def __init__(
        self, 
        distribution_strategy: DistributionStrategy, 
        phenotype_selector: PhenotypeSelector,
    ):
        """Initialize the standard phenotype simulator.
        
        Args:
            distribution_strategy: Strategy for sampling phenotype counts and distances
            phenotype_selector: Strategy for selecting phenotypes based on distances
        """
        self.distribution = distribution_strategy
        self.selector = phenotype_selector
    
    def fit(self, data: Dict[str, np.ndarray]) -> None:
        """Fit the simulator's distribution model to historical data.
        
        Args:
            data: Dictionary containing data arrays to fit the distribution
        """
        start_time = time.time()
        
        self.distribution.fit(data)
        
        total_time = time.time() - start_time
        logger.info("Fitted phenotype simulator distribution model")
        logger.debug(f"StandardPhenotypeSimulator.fit completed in {total_time:.4f} seconds")
        
        # Log distribution summaries
        if hasattr(self.distribution, 'count_summary'):
            logger.info(f"Count distribution summary: {self.distribution.count_summary}")
        if hasattr(self.distribution, 'distance_summary'):
            logger.info(f"Distance distribution summary: {self.distribution.distance_summary}")
    
    def generate_patient(self, gene: str) -> List[Phenotype]:
        """Generate a simulated patient phenotype set for a given gene.
        
        Args:
            gene: Gene symbol to simulate phenotypes for
            
        Returns:
            List of simulated Phenotype objects
        """
        start_time = time.time()
        
        # Sample the number of phenotypes
        count_start = time.time()
        phenotype_count = self.distribution.sample_phenotype_count()
        logger.debug(f"Sampled phenotype count: {phenotype_count} (took {time.time() - count_start:.4f}s)")
        
        # Sample distances for each phenotype
        distances_start = time.time()
        distances = self.distribution.sample_distances(phenotype_count)
        logger.debug(f"Sampled distances: {distances} (took {time.time() - distances_start:.4f}s)")
        
        # Select phenotypes based on the gene and distances
        select_start = time.time()
        phenotypes = self.selector.select_phenotypes(gene, distances)
        logger.debug(f"Selected {len(phenotypes)} phenotypes for gene {gene} (took {time.time() - select_start:.4f}s)")
        
        # Log total patient generation time
        total_time = time.time() - start_time
        logger.debug(f"generate_patient({gene}) completed in {total_time:.4f} seconds")
        
        return phenotypes
    
    def generate_patients(self, gene_to_count: Dict[str, int], use_tqdm: bool = False) -> Dict[str, List[List[Phenotype]]]:
        """Generate multiple simulated patients for multiple genes.
        
        Args:
            gene_to_count: Dictionary mapping gene symbols to number of patients to generate
            use_tqdm: Whether to display a tqdm progress bar (default: False)
            
        Returns:
            Dictionary mapping gene symbols to lists of patient phenotype lists
        """
        start_time = time.time()
        result = {}
        total_patients_to_simulate = sum(gene_to_count.values())

        pbar = None
        if use_tqdm:
            pbar = tqdm(total=total_patients_to_simulate, desc="Generating patients", unit="patient")

        try:
            for gene, count in gene_to_count.items():
                gene_start = time.time()
                log_msg = f"Generating {count} patients for gene {gene}"
                if pbar:
                    pbar.write(log_msg)
                else:
                    logger.info(log_msg)

                gene_patients = []

                # Generate patients for this gene with progress tracking
                for i in range(count):
                    gene_patients.append(self.generate_patient(gene))
                    if pbar:
                        pbar.update(1)

                result[gene] = gene_patients
                gene_time = time.time() - gene_start
                logger.debug(f"Generated {count} patients for gene {gene} in {gene_time:.4f} seconds ({gene_time / count:.4f}s/patient)")

                if pbar:
                    pbar.set_description(f"Completed gene {gene}")

        finally:
            if pbar:
                pbar.close()

        total_time = time.time() - start_time
        logger.info(f"generate_patients completed in {total_time:.4f} seconds")

        return result 
