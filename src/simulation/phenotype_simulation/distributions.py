"""Distribution strategies for phenotype simulation."""
import numpy as np
from typing import Dict, List, Optional
import logging
import time  # Import time module

from src.simulation.phenotype_simulation.interfaces import DistributionStrategy


logger = logging.getLogger(__name__)


class EmpiricalDistribution(DistributionStrategy):
    """Empirical distribution strategy based on observed frequency data.
    
    This strategy creates probability mass functions directly from observed data
    frequencies without fitting parametric models.
    """
    
    def __init__(self):
        """Initialize the empirical distribution strategy."""
        self._count_probabilities = None
        self._distance_probabilities = None
        self._count_values = None
        self._distance_values = None
        self._fitted = False
    
    def fit(self, data: Dict[str, np.ndarray]) -> None:
        """Fit the empirical distribution to historical data.
        
        Args:
            data: Dictionary with keys:
                - 'phenotype_counts': Array of phenotype counts per patient
                - 'distance_values': Array of phenotype distances
        """
        start_time = time.time()
        
        if 'phenotype_counts' not in data or 'distance_values' not in data:
            raise ValueError("Data must contain 'phenotype_counts' and 'distance_values' keys")
        
        # Handle phenotype counts (N)
        counts_start = time.time()
        counts = data['phenotype_counts']
        unique_counts, count_frequencies = np.unique(counts, return_counts=True)
        count_probabilities = count_frequencies / np.sum(count_frequencies)
        logger.debug(f"Count processing took {time.time() - counts_start:.4f} seconds")
        
        # Store count PMF
        self._count_values = unique_counts
        self._count_probabilities = count_probabilities
        
        # Handle distance values (D)
        distances_start = time.time()
        distances = data['distance_values']
        unique_distances, distance_frequencies = np.unique(distances, return_counts=True)
        distance_probabilities = distance_frequencies / np.sum(distance_frequencies)
        logger.debug(f"Distance processing took {time.time() - distances_start:.4f} seconds")
        
        # Store distance PMF
        self._distance_values = unique_distances
        self._distance_probabilities = distance_probabilities
        
        self._fitted = True
        
        logger.info(f"Fitted empirical distribution with {len(unique_counts)} count values and {len(unique_distances)} distance values")
        logger.debug(f"Count values: {unique_counts}, probabilities: {count_probabilities}")
        logger.debug(f"Distance values: {unique_distances}, probabilities: {distance_probabilities}")
        
        # Log total fit time
        total_time = time.time() - start_time
        logger.debug(f"EmpiricalDistribution.fit completed in {total_time:.4f} seconds")
    
    def sample_phenotype_count(self) -> int:
        """Sample the number of phenotypes to generate for a patient.
        
        Returns:
            Number of phenotypes to generate
        
        Raises:
            RuntimeError: If the distribution has not been fitted
        """
        start_time = time.time()
        
        if not self._fitted:
            raise RuntimeError("Distribution must be fitted before sampling")
        
        # Sample from the count PMF
        count_idx = np.random.choice(
            len(self._count_values), 
            p=self._count_probabilities
        )
        result = int(self._count_values[count_idx])
        
        # Log sampling time
        total_time = time.time() - start_time
        logger.debug(f"sample_phenotype_count completed in {total_time:.4f} seconds")
        
        return result
    
    def sample_distances(self, count: int) -> List[int]:
        """Sample the distances from specific phenotypes for a given count.
        
        Args:
            count: Number of phenotypes to generate
            
        Returns:
            List of distance values (0 = specific, 1 = parent, etc.)
            
        Raises:
            RuntimeError: If the distribution has not been fitted
        """
        start_time = time.time()
        
        if not self._fitted:
            raise RuntimeError("Distribution must be fitted before sampling")
        
        # Sample from the distance PMF 'count' times
        distance_indices = np.random.choice(
            len(self._distance_values), 
            size=count,
            p=self._distance_probabilities
        )
        result = [int(self._distance_values[idx]) for idx in distance_indices]
        
        # Log sampling time
        total_time = time.time() - start_time
        logger.debug(f"sample_distances (count={count}) completed in {total_time:.4f} seconds")
        
        return result
    
    @property
    def is_fitted(self) -> bool:
        """Check if the distribution has been fitted.
        
        Returns:
            True if the distribution has been fitted, False otherwise
        """
        return self._fitted
    
    @property
    def count_summary(self) -> Dict[str, float]:
        """Get summary statistics for the fitted phenotype count distribution.
        
        Returns:
            Dictionary with count statistics (min, max, mean, median)
            
        Raises:
            RuntimeError: If the distribution has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Distribution must be fitted before accessing summary")
        
        return {
            "min": np.min(self._count_values),
            "max": np.max(self._count_values),
            "mean": np.sum(self._count_values * self._count_probabilities),
            "median": np.median(self._count_values)
        }
    
    @property
    def distance_summary(self) -> Dict[str, float]:
        """Get summary statistics for the fitted distance distribution.
        
        Returns:
            Dictionary with distance statistics (min, max, mean, median)
            
        Raises:
            RuntimeError: If the distribution has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Distribution must be fitted before accessing summary")
        
        return {
            "min": np.min(self._distance_values),
            "max": np.max(self._distance_values),
            "mean": np.sum(self._distance_values * self._distance_probabilities),
            "median": np.median(self._distance_values)
        }


class UniformCountDistribution(EmpiricalDistribution):
    """Distribution strategy that samples phenotype count from a uniform distribution.
    
    This strategy overrides the phenotype count sampling to use a uniform
    distribution between a specified min and max.
    
    It still inherits from EmpiricalDistribution to use its method for sampling
    phenotype distances from an empirical dataset. Therefore, it must be fitted
    on data containing 'distance_values'.
    """
    
    def __init__(self, min_count: int = 1, max_count: int = 10):
        """Initialize the uniform count distribution strategy.
        
        Args:
            min_count: The minimum number of phenotypes (inclusive).
            max_count: The maximum number of phenotypes (inclusive).
        """
        super().__init__()
        if min_count > max_count:
            raise ValueError("min_count cannot be greater than max_count")
        self.min_count = min_count
        self.max_count = max_count
        
    def sample_phenotype_count(self) -> int:
        """Sample the number of phenotypes from a uniform distribution.
        
        Returns:
            A random integer between min_count and max_count.
        """
        start_time = time.time()
        
        if not self._fitted:
            raise RuntimeError("Distribution must be fitted before sampling")
        
        result = np.random.randint(self.min_count, self.max_count + 1)
        
        total_time = time.time() - start_time
        logger.debug(f"UniformCountDistribution.sample_phenotype_count completed in {total_time:.4f} seconds")
        
        return int(result)


class GaussianCountDistribution(EmpiricalDistribution):
    """Distribution strategy that samples phenotype count from a Gaussian distribution.
    
    This strategy overrides the phenotype count sampling to use a Gaussian
    (normal) distribution with a specified mean and standard deviation.
    
    It still inherits from EmpiricalDistribution to use its method for sampling
    phenotype distances from an empirical dataset. Therefore, it must be fitted
    on data containing 'distance_values'.
    """
    
    def __init__(self, mean: float = 5.0, std: float = 2.0):
        """Initialize the Gaussian count distribution strategy.
        
        Args:
            mean: The mean of the Gaussian distribution.
            std: The standard deviation of the Gaussian distribution.
        """
        super().__init__()
        self.mean = mean
        self.std = std
        
    def sample_phenotype_count(self) -> int:
        """Sample the number of phenotypes from a Gaussian distribution.
        
        The result is clipped to ensure at least one phenotype is returned.
        
        Returns:
            A random integer sampled from the specified Gaussian distribution.
        """
        start_time = time.time()
        
        if not self._fitted:
            raise RuntimeError("Distribution must be fitted before sampling")
        
        count = np.random.normal(self.mean, self.std)
        result = max(1, int(round(count))) # Ensure at least 1 phenotype
        
        total_time = time.time() - start_time
        logger.debug(f"GaussianCountDistribution.sample_phenotype_count completed in {total_time:.4f} seconds")
        
        return result


class UniformCountSpecificPhenotypesDistribution(UniformCountDistribution):
    """Samples phenotype count uniformly and always returns distance 0.
    
    This ensures that only the most specific phenotypes for a gene are selected.
    This strategy does not need to be fitted on any data because it does not
    sample distances from an empirical distribution.
    """

    def __init__(self, min_count: int = 1, max_count: int = 10):
        """Initialize the strategy.
        
        Args:
            min_count: The minimum number of phenotypes (inclusive).
            max_count: The maximum number of phenotypes (inclusive).
        """
        super().__init__(min_count, max_count)
        self._fitted = True  # Ready to use without fitting

    def fit(self, data: Dict[str, np.ndarray]) -> None:
        """This strategy does not require fitting. This method is a no-op."""
        logger.info("UniformCountSpecificPhenotypesDistribution does not require fitting. Skipping.")
        pass

    def sample_distances(self, count: int) -> List[int]:
        """Return a list of zeros, corresponding to specific phenotypes.
        
        Args:
            count: Number of distances to return.
            
        Returns:
            A list containing 'count' zeros.
        """
        start_time = time.time()
        result = [0] * count
        total_time = time.time() - start_time
        logger.debug(f"UniformCountSpecificPhenotypesDistribution.sample_distances (count={count}) completed in {total_time:.4f} seconds")
        return result 