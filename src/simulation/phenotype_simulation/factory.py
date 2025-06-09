"""Factory module for creating simulation components."""
import logging
from typing import Dict, Any, Optional

from src.simulation.phenotype_simulation.interfaces import (
    DistributionStrategy, 
    PhenotypeSelector, 
    PhenotypeSimulator
)
from src.simulation.phenotype_simulation.distributions import EmpiricalDistribution
from src.simulation.phenotype_simulation.selector import HPODistancePhenotypeSelector
from src.simulation.phenotype_simulation.simulator import StandardPhenotypeSimulator
from src.simulation.phenotype_simulation.data_loader import PhenotypeDistributionDataLoader

from src.simulation.gene_phenotype import GenePhenotypeFacade
from src.ontology.hpo_graph import HPOGraphProvider


logger = logging.getLogger(__name__)


class SimulationFactory:
    """Factory for creating simulation components."""
    
    @classmethod
    def create_distribution_strategy(
        cls, 
        strategy_type: str = "empirical",
        **kwargs
    ) -> DistributionStrategy:
        """Create a distribution strategy.
        
        Args:
            strategy_type: Type of distribution strategy to create
                - "empirical": EmpiricalDistribution
            **kwargs: Additional keyword arguments for the strategy
            
        Returns:
            Distribution strategy instance
            
        Raises:
            ValueError: If the strategy type is not supported
        """
        if strategy_type.lower() == "empirical":
            return EmpiricalDistribution(**kwargs)
        else:
            raise ValueError(f"Unsupported distribution strategy type: {strategy_type}")
    
    @classmethod
    def create_phenotype_selector(
        cls,
        selector_type: str = "hpo_distance",
        gene_phenotype_facade: Optional[GenePhenotypeFacade] = None,
        hpo_provider: Optional[HPOGraphProvider] = None,
        data_dir_hpo: str = "data/ontology",
        data_dir_gene_phenotype: str = "data/gene_phenotype",
        **kwargs
    ) -> PhenotypeSelector:
        """Create a phenotype selector.
        
        Args:
            selector_type: Type of phenotype selector to create
                - "hpo_distance": HPODistancePhenotypeSelector
            gene_phenotype_facade: Optional GenePhenotypeFacade instance
            hpo_provider: Optional HPOGraphProvider instance
            data_dir_hpo: Directory for HPO data (used if hpo_provider is None)
            data_dir_gene_phenotype: Directory for gene-phenotype data (used if gene_phenotype_facade is None)
            **kwargs: Additional keyword arguments for the selector
            
        Returns:
            Phenotype selector instance
            
        Raises:
            ValueError: If the selector type is not supported
        """
        if selector_type.lower() == "hpo_distance":
            # Create or use provided HPOGraphProvider
            if hpo_provider is None:
                hpo_provider = HPOGraphProvider.get_instance(data_dir=data_dir_hpo)
            
            # Create or use provided GenePhenotypeFacade
            if gene_phenotype_facade is None:
                from src.simulation.gene_phenotype import GenePhenotypeFacade
                gene_phenotype_facade = GenePhenotypeFacade(data_dir=data_dir_gene_phenotype)
            
            return HPODistancePhenotypeSelector(
                gene_phenotype_facade=gene_phenotype_facade,
                hpo_provider=hpo_provider,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported phenotype selector type: {selector_type}")
    
    @classmethod
    def create_simulator(
        cls,
        distribution_strategy: Optional[DistributionStrategy] = None,
        phenotype_selector: Optional[PhenotypeSelector] = None,
        distribution_type: str = "empirical",
        selector_type: str = "hpo_distance",
        **kwargs
    ) -> PhenotypeSimulator:
        """Create a phenotype simulator.
        
        Args:
            distribution_strategy: Optional distribution strategy instance
            phenotype_selector: Optional phenotype selector instance
            distribution_type: Type of distribution strategy to create if not provided
            selector_type: Type of phenotype selector to create if not provided
            **kwargs: Additional keyword arguments for the components
            
        Returns:
            Phenotype simulator instance
        """
        # Create distribution strategy if not provided
        if distribution_strategy is None:
            distribution_strategy = cls.create_distribution_strategy(
                strategy_type=distribution_type,
                **kwargs.get("distribution_kwargs", {})
            )
        
        # Create phenotype selector if not provided
        if phenotype_selector is None:
            phenotype_selector = cls.create_phenotype_selector(
                selector_type=selector_type,
                **kwargs.get("selector_kwargs", {})
            )
        
        # Create the simulator
        return StandardPhenotypeSimulator(
            distribution_strategy=distribution_strategy,
            phenotype_selector=phenotype_selector,
            **kwargs.get("simulator_kwargs", {})
        )
    
    @classmethod
    def create_data_loader(
        cls,
        data_dir: str = "data/simulation",
        **kwargs
    ) -> PhenotypeDistributionDataLoader:
        """Create a phenotype distribution data loader.
        
        Args:
            data_dir: Directory where CSV files are stored
            **kwargs: Additional keyword arguments for the data loader
            
        Returns:
            PhenotypeDistributionDataLoader instance
        """
        return PhenotypeDistributionDataLoader(data_dir=data_dir, **kwargs)
    
    @classmethod
    def create_complete_simulator(
        cls,
        count_file: Optional[str] = None,
        distance_file: Optional[str] = None,
        data_dir_simulation: str = "data/simulation",
        data_dir_hpo: str = "data/ontology",
        data_dir_gene_phenotype: str = "data/gene_phenotype",
        distribution_type: str = "empirical",
        selector_type: str = "hpo_distance",
        **kwargs
    ) -> PhenotypeSimulator:
        """Create a complete simulator with data loaded from CSV files.
        
        This is a convenience method that:
        1. Creates a data loader
        2. Loads distribution data from CSV files
        3. Creates a distribution strategy and fits it with the data
        4. Creates a phenotype selector
        5. Creates and returns a simulator
        
        Args:
            count_file: Path to the phenotype count CSV file (optional)
            distance_file: Path to the phenotype distance CSV file (optional)
            data_dir_simulation: Directory for simulation data files
            data_dir_hpo: Directory for HPO data
            data_dir_gene_phenotype: Directory for gene-phenotype data
            distribution_type: Type of distribution strategy to create
            selector_type: Type of phenotype selector to create
            **kwargs: Additional keyword arguments
            
        Returns:
            Fully configured phenotype simulator
        """
        # Create data loader
        data_loader = cls.create_data_loader(data_dir=data_dir_simulation)
        
        # Load distribution data
        data = data_loader.load_distribution_data(
            count_file=count_file,
            distance_file=distance_file,
            count_column=kwargs.get("count_column", "count"),
            distance_column=kwargs.get("distance_column", "distance")
        )
        
        # Create distribution strategy and fit it
        distribution_strategy = cls.create_distribution_strategy(
            strategy_type=distribution_type,
            **kwargs.get("distribution_kwargs", {})
        )
        distribution_strategy.fit(data)
        
        # Create phenotype selector
        phenotype_selector = cls.create_phenotype_selector(
            selector_type=selector_type,
            data_dir_hpo=data_dir_hpo,
            data_dir_gene_phenotype=data_dir_gene_phenotype,
            **kwargs.get("selector_kwargs", {})
        )
        
        # Create and return simulator
        return cls.create_simulator(
            distribution_strategy=distribution_strategy,
            phenotype_selector=phenotype_selector,
            **kwargs.get("simulator_kwargs", {})
        ) 