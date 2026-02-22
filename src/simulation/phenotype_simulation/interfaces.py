"""Interfaces for phenotype simulation components."""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

from src.core.types import Phenotype


class DistributionStrategy(ABC):
    """Interface for phenotype distribution sampling strategies."""

    @abstractmethod
    def fit(self, data: Dict[str, np.ndarray]) -> None:
        """Fit the distribution strategy to historical data.
        
        Args:
            data: Dictionary containing data arrays to fit the distribution
                 (e.g., {'phenotype_counts': [...], 'distance_values': [...]})
        """
        pass
    
    @abstractmethod
    def sample_phenotype_count(self) -> int:
        """Sample the number of phenotypes to generate for a patient.
        
        Returns:
            Number of phenotypes to generate
        """
        pass
    
    @abstractmethod
    def sample_distances(self, count: int) -> List[int]:
        """Sample the distances from specific phenotypes for a given count.
        
        Args:
            count: Number of phenotypes to generate
            
        Returns:
            List of distance values (0 = specific, 1 = parent, etc.)
        """
        pass


class PhenotypeSelector(ABC):
    """Interface for selecting phenotypes based on gene and distances."""
    
    @abstractmethod
    def select_phenotypes(self, gene: str, distances: List[int]) -> List[Phenotype]:
        """Select phenotypes for a gene based on specified distances.
        
        Args:
            gene: Gene symbol to select phenotypes for
            distances: List of distance values (0 = specific, 1 = parent, etc.)
            
        Returns:
            List of selected Phenotype objects
        """
        pass


class PhenotypeSimulator(ABC):
    """Interface for the overall phenotype simulation process."""
    
    @abstractmethod
    def fit(self, data: Dict[str, np.ndarray]) -> None:
        """Fit the simulator's distribution model to historical data.
        
        Args:
            data: Dictionary containing data arrays to fit the distribution
        """
        pass
    
    @abstractmethod
    def generate_patient(self, gene: str) -> List[Phenotype]:
        """Generate a simulated patient phenotype set for a given gene.
        
        Args:
            gene: Gene symbol to simulate phenotypes for
            
        Returns:
            List of simulated Phenotype objects
        """
        pass
    
    @abstractmethod
    def generate_patients(self, gene_to_count: Dict[str, int]) -> Dict[str, List[List[Phenotype]]]:
        """Generate multiple simulated patients for multiple genes.
        
        Args:
            gene_to_count: Dictionary mapping gene symbols to number of patients to generate
            
        Returns:
            Dictionary mapping gene symbols to lists of patient phenotype lists
        """
        pass 
