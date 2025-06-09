import os
import json
import subprocess
import tempfile
import logging
import time
import torch
from pathlib import Path
from typing import Any, Dict, List

from torch_geometric.data import Batch

from .base_evaluator import BaseModelEvaluator
from validation_arena.registry import register_evaluator

# Use standard Python logging
logger = logging.getLogger(__name__)

@register_evaluator("Phen2Gene")
class Phen2GeneEvaluator(BaseModelEvaluator):
    """Evaluator for Phen2Gene, a phenotype-driven gene prioritization tool."""

    def __init__(self, config: Dict[str, Any], name: str):
        """
        Initializes the Phen2Gene evaluator.
        
        Requires config keys:
        - docker_image: Name of the Phen2Gene docker image (default: 'genomicslab/phen2gene')
        - weighting_method: Weighting method to use ('sk', 'w', 'ic', 'u') (default: 'sk')
        - max_genes: Maximum number of genes to return in results (default: 1000)
        - training_lmdb_dir: Path to the LMDB dataset used for training (for gene list/mapping)
        - gene_enum_path: Optional path to gene enumeration JSON mapping gene names to indices
        - gene_list: Optional list of genes in order of indices (used if gene_enum_path not provided)
        - output_dir: Directory to store intermediate outputs (default: 'phen2gene_output')
        - verbose: Verbose Phen2Gene output (default: False)
        """
        super().__init__(config, name)
        
        # Docker configuration
        self.docker_image = config.get('docker_image', 'genomicslab/phen2gene')
        
        # Phen2Gene options
        self.weighting_method = config.get('weighting_method', 'sk')
        self.max_genes = config.get('max_genes', 1000)
        self.verbose = config.get('verbose', False)
        
        # Output handling
        self.output_dir = Path(config.get('output_dir', 'phen2gene_output'))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Gene mapping options
        self.training_lmdb_dir = config.get('training_lmdb_dir')  # For loading gene mapping from training metadata
        self.gene_enum_path = config.get('gene_enum_path')        # Direct path to gene mapping JSON
        self.gene_list = config.get('gene_list')                 # Explicit gene list
        self.gene_to_idx = None
        
        # Validate required gene mapping options
        if not self.training_lmdb_dir and not self.gene_enum_path and not self.gene_list:
            raise ValueError("Either 'training_lmdb_dir', 'gene_enum_path', or 'gene_list' must be provided")
        
        # Temp directory for HPO term files
        self.temp_dir = None
        
        logger.info(f"Phen2Gene evaluator '{self.name}' initialized")
        logger.debug(f"  Docker image: {self.docker_image}")
        logger.debug(f"  Weighting method: {self.weighting_method}")
        logger.debug(f"  Max genes: {self.max_genes}")
        logger.debug(f"  Output directory: {self.output_dir}")
                
    def load(self) -> None:
        """
        Loads evaluator resources.
        
        For Phen2Gene:
        1. Checks if Docker is available
        2. Pulls the Docker image if needed
        3. Loads/creates gene mapping
        4. Creates temporary directory for input files
        """
        if self._is_loaded:
            logger.info(f"Phen2Gene '{self.name}' already loaded")
            return

        loading_start = time.time()
        logger.info(f"Loading Phen2Gene '{self.name}'...")

        # --- 1. Check if Docker is available ---
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            logger.info(f"Docker available: {result.stdout.strip()}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error(f"Docker is not available: {e}")
            raise RuntimeError("Docker is required for Phen2Gene evaluator but is not available")

        # --- 2. Pull the Docker image if needed ---
        try:
            # Check if image exists locally
            result = subprocess.run(
                ["docker", "image", "ls", self.docker_image, "--format", "{{.Repository}}"],
                capture_output=True,
                text=True
            )
            if self.docker_image not in result.stdout:
                logger.info(f"Pulling Docker image: {self.docker_image}")
                pull_result = subprocess.run(
                    ["docker", "pull", self.docker_image],
                    capture_output=True,
                    text=True
                )
                if pull_result.returncode != 0:
                    logger.error(f"Failed to pull Docker image: {pull_result.stderr}")
                    raise RuntimeError(f"Failed to pull Docker image: {self.docker_image}")
                logger.info(f"Successfully pulled Docker image: {self.docker_image}")
            else:
                logger.info(f"Docker image already available: {self.docker_image}")
        except subprocess.SubprocessError as e:
            logger.error(f"Error checking/pulling Docker image: {e}")
            raise RuntimeError(f"Error interacting with Docker: {e}")

        # --- 3. Load/create gene mapping ---
        try:
            # Priority: 1) gene_enum_path, 2) training_lmdb_dir, 3) gene_list
            if self.gene_enum_path:
                logger.info(f"Loading gene enumeration from: {self.gene_enum_path}")
                with open(self.gene_enum_path, 'r') as f:
                    self.gene_to_idx = json.load(f)
                logger.info(f"Loaded mapping for {len(self.gene_to_idx)} genes")
            
            elif self.training_lmdb_dir:
                # Import here to avoid circular imports
                try:
                    from training.datasets.lmdb_hpo_dataset import LMDBHPOGraphDataset
                    logger.info(f"Loading gene list from training LMDB metadata: {self.training_lmdb_dir}")
                    
                    # Instantiate dataset just to access metadata
                    training_dataset_meta = LMDBHPOGraphDataset(root_dir=self.training_lmdb_dir, readonly=True)
                    training_genes = training_dataset_meta.metadata.get('genes')
                    
                    if not training_genes:
                        logger.error(f"Could not find 'genes' list in metadata of training LMDB at {self.training_lmdb_dir}")
                        raise ValueError("'genes' list not found in training dataset metadata")
                    
                    self.gene_to_idx = {gene: i for i, gene in enumerate(training_genes)}
                    logger.info(f"Created gene-to-index mapping for {len(self.gene_to_idx)} genes from LMDB metadata")
                    
                    # Save the mapping for inspection/reuse
                    enum_path = self.output_dir / 'gene_enum.json'
                    with open(enum_path, 'w') as f:
                        json.dump(self.gene_to_idx, f, indent=2)
                    logger.info(f"Saved gene enumeration to: {enum_path}")
                    
                    # We don't need to keep the dataset instance in memory
                    del training_dataset_meta
                
                except ImportError:
                    logger.error("Could not import LMDBHPOGraphDataset. Cannot load gene mapping from LMDB.")
                    raise ValueError("Failed to import LMDBHPOGraphDataset for gene mapping")
                except Exception as e:
                    logger.error(f"Error loading gene mapping from LMDB: {e}")
                    raise
            
            elif self.gene_list:
                logger.info(f"Creating gene enumeration from provided gene list ({len(self.gene_list)} genes)")
                self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_list)}
                
                # Save the mapping for inspection/reuse
                enum_path = self.output_dir / 'gene_enum.json'
                with open(enum_path, 'w') as f:
                    json.dump(self.gene_to_idx, f, indent=2)
                logger.info(f"Saved gene enumeration to: {enum_path}")
            
            else:
                # This should never happen due to validation in __init__, but just in case
                raise ValueError("No source for gene mapping available")
                
        except Exception as e:
            logger.error(f"Error loading/creating gene mapping: {e}")
            raise RuntimeError(f"Error setting up gene mapping: {e}")

        # --- 4. Create temporary directory for input HPO files ---
        self.temp_dir = tempfile.TemporaryDirectory(prefix="phen2gene_")
        logger.debug(f"Created temporary directory: {self.temp_dir.name}")

        self._is_loaded = True
        loading_time = time.time() - loading_start
        logger.info(f"Phen2Gene '{self.name}' loaded successfully in {loading_time:.2f}s")

    def predict(self, batch_data: Batch) -> torch.Tensor:
        """
        Predicts gene rankings using Phen2Gene for a batch of samples.
        
        Args:
            batch_data: PyG Batch object containing graph data with the 'phenotype_ids' attribute
        
        Returns:
            torch.Tensor: Model output as scores in shape [batch_size, num_classes]
        """
        if not self._is_loaded:
            raise RuntimeError(f"Phen2Gene '{self.name}' not loaded. Call load() first.")
        
        if not isinstance(batch_data, Batch):
            raise TypeError(f"Input must be a PyG Batch object for Phen2GeneEvaluator, got {type(batch_data)}")
        
        start_time = time.time()
        logger.info(f"Running Phen2Gene prediction on batch of {batch_data.num_graphs} samples")
        
        # Extract phenotype IDs from each graph in the batch
        phenotype_lists = self._extract_phenotypes(batch_data)
        if not phenotype_lists:
            raise ValueError("No valid phenotype IDs found in batch data")
            
        # Run Phen2Gene for each sample
        results = self._run_phen2gene_batch(phenotype_lists)
        
        # Convert results to tensor in proper format [batch_size, num_classes] 
        output_tensor = self._convert_results_to_tensor(results, num_classes=len(self.gene_to_idx))
        
        prediction_time = time.time() - start_time
        logger.info(f"Phen2Gene prediction completed in {prediction_time:.2f}s")
        
        return output_tensor
        
    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the Phen2Gene evaluator."""
        return {
            "evaluator_name": self.name,
            "evaluator_type": "Phen2GeneEvaluator",
            "docker_image": self.docker_image,
            "weighting_method": self.weighting_method,
            "max_genes": self.max_genes,
            "num_output_classes": len(self.gene_to_idx) if self.gene_to_idx else 0,
            "output_dir": str(self.output_dir),
            "is_loaded": self._is_loaded
        }
    
    def unload(self) -> None:
        """
        Cleans up resources for the Phen2Gene evaluator.
        """
        if self.temp_dir:
            logger.debug(f"Cleaning up temporary directory: {self.temp_dir.name}")
            self.temp_dir.cleanup()
            self.temp_dir = None
            
        super().unload()  # Sets _is_loaded to False
        logger.info(f"Phen2Gene '{self.name}' unloaded")
        
    def _extract_phenotypes(self, batch_data: Batch) -> List[List[str]]:
        """
        Extracts phenotype IDs from a batch of graph data objects.
        
        Returns a list of lists, where each inner list contains the HPO IDs for one sample.
        """
        phenotype_lists = []
        
        # Get a list of the individual Data objects from the batch
        graph_list = batch_data.to_data_list()
        
        for i, graph in enumerate(graph_list):
            if hasattr(graph, 'phenotype_ids') and graph.phenotype_ids:
                # Ensure all HPO IDs are strings with HP: prefix
                hpo_ids = [
                    pid if pid.startswith("HP:") else f"HP:{pid}" 
                    for pid in graph.phenotype_ids
                    if pid  # Skip empty IDs
                ]
                
                if hpo_ids:
                    phenotype_lists.append(hpo_ids)
                else:
                    logger.warning(f"Sample {i} has phenotype_ids attribute but no valid HPO terms")
            else:
                logger.warning(f"Sample {i} missing phenotype_ids attribute or it's empty")
        
        return phenotype_lists
    
    def _run_phen2gene_batch(self, phenotype_lists: List[List[str]]) -> List[Dict[str, float]]:
        """
        Runs Phen2Gene for each list of phenotypes in the batch.
        
        Args:
            phenotype_lists: List of lists, where each inner list contains HPO IDs for one sample
            
        Returns:
            List of dicts mapping gene names to scores (one dict per sample)
        """
        if not self.temp_dir:
            raise RuntimeError("Temporary directory not created. Evaluator may not be loaded.")
            
        results = []
        
        for i, hpo_terms in enumerate(phenotype_lists):
            # Create HPO file for this sample
            input_file = os.path.join(self.temp_dir.name, f"sample_{i}.txt")
            with open(input_file, 'w') as f:
                f.write('\n'.join(hpo_terms))
                
            # Create output directory for this sample
            sample_output_dir = self.output_dir / f"sample_{i}"
            os.makedirs(sample_output_dir, exist_ok=True)
            
            # Construct the Docker command
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{self.temp_dir.name}:/input",
                "-v", f"{sample_output_dir.absolute()}:/output",
                self.docker_image,
                "-f", f"/input/sample_{i}.txt",
                "-out", "/output",
                "-w", self.weighting_method
            ]
            
            if self.verbose:
                cmd.append("-v")
            
            # Run Phen2Gene for this sample
            logger.debug(f"Running Phen2Gene for sample {i} with {len(hpo_terms)} HPO terms")
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Parse the output file to get gene scores
                output_file = sample_output_dir / "output_file.associated_gene_list"
                if not output_file.exists():
                    logger.error(f"Output file not found for sample {i}: {output_file}")
                    results.append({})  # Empty result for this sample
                    continue
                    
                # Parse the output file (tab-separated with header)
                gene_scores = {}
                with open(output_file, 'r') as f:
                    # Skip header line
                    next(f)
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 4:  # Rank, Gene, ID, Score
                            gene = parts[1]
                            score = float(parts[3])
                            gene_scores[gene] = score
                
                results.append(gene_scores)
                logger.debug(f"Sample {i}: Found scores for {len(gene_scores)} genes")
                
            except subprocess.SubprocessError as e:
                logger.error(f"Error running Phen2Gene for sample {i}: {e}")
                logger.error(f"Command: {' '.join(cmd)}")
                if hasattr(e, 'stderr'):
                    logger.error(f"Stderr: {e.stderr}")
                results.append({})  # Empty result for this sample
        
        return results
    
    def _convert_results_to_tensor(self, results: List[Dict[str, float]], num_classes: int) -> torch.Tensor:
        """
        Converts the Phen2Gene results to a tensor matching the format expected by the metrics calculator.
        
        Args:
            results: List of dicts mapping gene names to scores (one dict per sample)
            num_classes: Number of possible gene classes
            
        Returns:
            torch.Tensor: Tensor of shape [batch_size, num_classes] with gene scores
        """
        batch_size = len(results)
        output = torch.zeros((batch_size, num_classes), dtype=torch.float32)
        
        for i, gene_scores in enumerate(results):
            for gene, score in gene_scores.items():
                if gene in self.gene_to_idx:
                    idx = self.gene_to_idx[gene]
                    output[i, idx] = score
        
        # Normalize scores to prevent extremely large differences affecting metrics
        # We use softmax here to ensure the scores are comparable and in a reasonable range
        for i in range(batch_size):
            # Only apply softmax if the row isn't all zeros
            if torch.sum(output[i]) > 0:
                output[i] = torch.nn.functional.softmax(output[i] * 10, dim=0)  # Scaling factor of 10 to create sharper distribution
        
        return output 