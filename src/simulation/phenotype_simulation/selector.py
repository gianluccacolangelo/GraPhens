"""Phenotype selector strategies for simulation."""
import logging
import random
from typing import Dict, List, Set, Optional
from collections import defaultdict
import time  # Import time module

from src.core.types import Phenotype
from src.simulation.phenotype_simulation.interfaces import PhenotypeSelector
from src.simulation.gene_phenotype import GenePhenotypeFacade
from src.ontology.hpo_graph import HPOGraphProvider


logger = logging.getLogger(__name__)


class HPODistancePhenotypeSelector(PhenotypeSelector):
    """Selects phenotypes for a gene based on HPO graph distances.
    
    Uses the gene-phenotype database to find specific phenotypes for a gene,
    then uses the HPO graph to find phenotypes at specific distances.
    """
    
    def __init__(
        self, 
        gene_phenotype_facade: GenePhenotypeFacade,
        hpo_provider: HPOGraphProvider,
        max_attempts: int = 10,
        allow_duplicates: bool = False
    ):
        """Initialize the HPO distance-based phenotype selector.
        
        Args:
            gene_phenotype_facade: Facade for accessing gene-phenotype associations
            hpo_provider: Provider for HPO ontology graph
            max_attempts: Maximum number of attempts to find phenotypes
                at a specific distance (default: 10)
            allow_duplicates: Whether to allow the same phenotype to be selected
                multiple times (default: False)
        """
        self.gene_phenotype_facade = gene_phenotype_facade
        self.hpo_provider = hpo_provider
        self.max_attempts = max_attempts
        self.allow_duplicates = allow_duplicates
        self._distance_cache = {}
    
    def _get_phenotype_pools_for_gene(self, gene: str) -> Dict[int, List[Phenotype]]:
        """Get pools of phenotypes at different distances for a gene.
        
        Args:
            gene: Gene symbol
            
        Returns:
            Dictionary mapping distances to lists of phenotypes
                0 -> specific phenotypes
                1 -> parent phenotypes (distance 1)
                2 -> grandparent phenotypes (distance 2)
                etc.
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = gene
        if cache_key in self._distance_cache:
            cache_hit_time = time.time() - start_time
            logger.debug(f"Cache hit for gene {gene}, returning pools in {cache_hit_time:.4f} seconds")
            return self._distance_cache[cache_key]
        
        logger.debug(f"Building phenotype pools for gene {gene}")
        
        # Get specific phenotypes for the gene (distance 0)
        specific_start = time.time()
        specific_phenotypes = self.gene_phenotype_facade.get_phenotypes_for_gene(gene)
        logger.debug(f"Fetching specific phenotypes took {time.time() - specific_start:.4f} seconds")
        
        if not specific_phenotypes:
            logger.warning(f"No specific phenotypes found for gene {gene}")
            # Cache empty result
            self._distance_cache[cache_key] = {0: []}
            return {0: []}
        
        # Initialize pools with specific phenotypes
        pools = {0: specific_phenotypes}
        
        # Build sets of phenotype IDs at each distance to avoid duplicates
        seen_ids = {p.id for p in specific_phenotypes}
        
        # Find the maximum possible distance in the HPO graph (to avoid excessive iterations)
        max_distance = 10  # Reasonable upper limit
        
        # Build pools for each distance level
        build_pools_start = time.time()
        for distance in range(1, max_distance + 1):
            distance_start = time.time()
            distance_phenotypes = []
            distance_ids = set()
            
            # For each phenotype at the previous distance
            for prev_phenotype in pools[distance - 1]:
                # Get its parents
                try:
                    parents = self.hpo_provider.get_direct_parents(prev_phenotype.id)
                    for parent_id in parents:
                        # Skip if already seen
                        if parent_id in seen_ids:
                            continue
                        
                        # Skip if already in this distance pool
                        if parent_id in distance_ids:
                            continue
                        
                        # Get metadata for the parent
                        metadata = self.hpo_provider.get_metadata(parent_id)
                        if metadata:
                            # Create a Phenotype object
                            parent_phenotype = Phenotype(
                                id=parent_id,
                                name=metadata.get('name', ''),
                                description=metadata.get('definition', ''),
                                metadata={'distance': distance} 
                            )
                            distance_phenotypes.append(parent_phenotype)
                            distance_ids.add(parent_id)
                            seen_ids.add(parent_id)
                except Exception as e:
                    logger.warning(f"Error getting parents for {prev_phenotype.id}: {e}")
            
            logger.debug(f"Building pool for distance {distance} took {time.time() - distance_start:.4f} seconds")
            
            # If no phenotypes at this distance, we've reached the top of the hierarchy
            if not distance_phenotypes:
                break
                
            # Add this distance pool
            pools[distance] = distance_phenotypes
            
        logger.debug(f"Total pool building took {time.time() - build_pools_start:.4f} seconds")
        
        # Cache the result
        self._distance_cache[cache_key] = pools
        
        # Log total pool generation time
        total_time = time.time() - start_time
        logger.debug(f"_get_phenotype_pools_for_gene({gene}) completed in {total_time:.4f} seconds")
        
        return pools
    
    def select_phenotypes(self, gene: str, distances: List[int]) -> List[Phenotype]:
        """Select phenotypes for a gene based on specified distances.
        
        Args:
            gene: Gene symbol to select phenotypes for
            distances: List of distance values (0 = specific, 1 = parent, etc.)
            
        Returns:
            List of selected Phenotype objects
        """
        start_time = time.time()
        
        # Get pools of phenotypes at each distance
        pools = self._get_phenotype_pools_for_gene(gene)
        
        # Track selected phenotypes to avoid duplicates
        selected_ids = set()
        selected_phenotypes = []
        
        selection_loop_start = time.time()
        # For each requested distance
        for distance in distances:
            # Skip if no pool exists for this distance
            if distance not in pools or not pools[distance]:
                logger.debug(f"No phenotypes available at distance {distance} for gene {gene}, skipping")
                continue
            
            # Get candidates (exclude already selected if duplicates not allowed)
            candidates = pools[distance]
            if not self.allow_duplicates:
                candidates = [p for p in candidates if p.id not in selected_ids]
            
            # If no candidates available, try to use a nearby distance
            attempts = 0
            while not candidates and attempts < self.max_attempts:
                attempts += 1
                # Try distance + 1, then distance - 1, then distance + 2, etc.
                offset = attempts // 2 + 1
                sign = 1 if attempts % 2 == 1 else -1
                alt_distance = distance + sign * offset
                
                if alt_distance in pools and pools[alt_distance]:
                    alt_candidates = pools[alt_distance]
                    if not self.allow_duplicates:
                        alt_candidates = [p for p in alt_candidates if p.id not in selected_ids]
                    
                    if alt_candidates:
                        logger.debug(f"Using distance {alt_distance} instead of {distance} (attempt {attempts})")
                        candidates = alt_candidates
                        break
            
            # If still no candidates, skip this distance
            if not candidates:
                logger.debug(f"Could not find candidates for distance {distance} after {attempts} attempts")
                continue
            
            # Randomly select a phenotype
            selected = random.choice(candidates)
            selected_phenotypes.append(selected)
            selected_ids.add(selected.id)
            
        logger.debug(f"Phenotype selection loop took {time.time() - selection_loop_start:.4f} seconds")
            
        # Log total selection time
        total_time = time.time() - start_time
        logger.debug(f"select_phenotypes({gene}, distances={len(distances)}) completed in {total_time:.4f} seconds")
        
        return selected_phenotypes 