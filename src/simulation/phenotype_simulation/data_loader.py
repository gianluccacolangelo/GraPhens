"""Data loaders for phenotype simulation."""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import time  # Import time module

logger = logging.getLogger(__name__)


class PhenotypeDistributionDataLoader:
    """Loads phenotype distribution data from CSV files.
    
    This class handles loading and preprocessing distribution data for:
    - Phenotype counts per patient
    - Phenotype distances from specific phenotypes
    """
    
    def __init__(self, data_dir: str = "data/simulation"):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory where CSV files are stored
        """
        self.data_dir = data_dir
    
    def load_count_distribution(
        self, 
        file_path: Optional[str] = None,
        column_name: str = "count"
    ) -> np.ndarray:
        """Load the phenotype count distribution from a CSV file.
        
        Args:
            file_path: Path to the CSV file (default: data_dir/phenotype_counts.csv)
            column_name: Name of the column containing counts
            
        Returns:
            numpy array of phenotype counts
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the data format is invalid
        """
        start_time = time.time()
        
        if file_path is None:
            file_path = os.path.join(self.data_dir, "phenotype_counts.csv")
        
        logger.info(f"Loading phenotype count distribution from {file_path}")
        
        try:
            # Load the CSV
            load_start = time.time()
            df = pd.read_csv(file_path)
            logger.debug(f"CSV loading took {time.time() - load_start:.4f} seconds")
            
            # Validate the data
            validation_start = time.time()
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in {file_path}")
            
            # Extract the counts
            counts = df[column_name].values
            
            # Validate counts are non-negative integers
            if not np.all(np.isfinite(counts)) or np.any(counts < 0) or not np.all(np.equal(np.mod(counts, 1), 0)):
                raise ValueError(f"Invalid count values in {file_path}: {counts[:10]}...")
            
            logger.debug(f"Data validation took {time.time() - validation_start:.4f} seconds")
            
            logger.info(f"Loaded {len(counts)} count values")
            logger.debug(f"Count statistics: min={np.min(counts)}, max={np.max(counts)}, mean={np.mean(counts):.2f}")
            
            # Log total time
            total_time = time.time() - start_time
            logger.debug(f"load_count_distribution completed in {total_time:.4f} seconds")
            
            return counts.astype(int)
            
        except Exception as e:
            logger.error(f"Error loading count distribution: {e}")
            raise
    
    def load_distance_distribution(
        self, 
        file_path: Optional[str] = None,
        column_name: str = "distance"
    ) -> np.ndarray:
        """Load the phenotype distance distribution from a CSV file.
        
        Args:
            file_path: Path to the CSV file (default: data_dir/phenotype_distances.csv)
            column_name: Name of the column containing distances
            
        Returns:
            numpy array of phenotype distances
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the data format is invalid
        """
        start_time = time.time()
        
        if file_path is None:
            file_path = os.path.join(self.data_dir, "phenotype_distances.csv")
        
        logger.info(f"Loading phenotype distance distribution from {file_path}")
        
        try:
            # Load the CSV
            load_start = time.time()
            df = pd.read_csv(file_path)
            logger.debug(f"CSV loading took {time.time() - load_start:.4f} seconds")
            
            # Validate the data
            validation_start = time.time()
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in {file_path}")
            
            # Extract the distances
            distances = df[column_name].values
            
            # Validate distances are non-negative integers
            if not np.all(np.isfinite(distances)) or np.any(distances < 0) or not np.all(np.equal(np.mod(distances, 1), 0)):
                raise ValueError(f"Invalid distance values in {file_path}: {distances[:10]}...")
            
            logger.debug(f"Data validation took {time.time() - validation_start:.4f} seconds")
                
            logger.info(f"Loaded {len(distances)} distance values")
            logger.debug(f"Distance statistics: min={np.min(distances)}, max={np.max(distances)}, mean={np.mean(distances):.2f}")
            
            # Log total time
            total_time = time.time() - start_time
            logger.debug(f"load_distance_distribution completed in {total_time:.4f} seconds")
            
            return distances.astype(int)
            
        except Exception as e:
            logger.error(f"Error loading distance distribution: {e}")
            raise
    
    def load_distribution_data(
        self,
        count_file: Optional[str] = None,
        distance_file: Optional[str] = None,
        count_column: str = "count",
        distance_column: str = "distance"
    ) -> Dict[str, np.ndarray]:
        """Load both distributions and return them in a dictionary.
        
        Args:
            count_file: Path to the count CSV file (default: data_dir/phenotype_counts.csv)
            distance_file: Path to the distance CSV file (default: data_dir/phenotype_distances.csv)
            count_column: Name of the column containing counts
            distance_column: Name of the column containing distances
            
        Returns:
            Dictionary with keys 'phenotype_counts' and 'distance_values'
        """
        start_time = time.time()
        
        counts = self.load_count_distribution(count_file, count_column)
        distances = self.load_distance_distribution(distance_file, distance_column)
        
        total_time = time.time() - start_time
        logger.debug(f"load_distribution_data completed in {total_time:.4f} seconds")
        
        return {
            'phenotype_counts': counts,
            'distance_values': distances
        } 