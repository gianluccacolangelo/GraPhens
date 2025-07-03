import os
import json
import torch
import numpy as np
import logging
import time  # Import time for logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split

# Import GraPhens for graph creation
from src.graphens import GraPhens

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HPOGraphDataset(Dataset):
    """
    PyTorch Dataset for HPO graphs created from simulated patient data.
    
    This dataset:
    1. Loads a preprocessed index file
    2. On-demand loads individual graph objects
    3. Supports shuffling and random access
    """
    
    def __init__(self, root_dir: str, transform=None, pre_transform=None):
        """
        Initialize HPO Graph Dataset.
        
        Args:
            root_dir: Directory containing processed dataset
            transform: Transform to apply at load time
            pre_transform: Transform to apply during preprocessing
        """
        # Store our own copy of the paths that we know are Path objects
        self.root_path = Path(root_dir)  
        self.processed_path = self.root_path 
        self.index_path = self.processed_path / "index.pt"
        
        # Now initialize the Dataset parent class with the string version
        # This avoids path operation conflicts with the parent class implementation
        super().__init__(str(root_dir), transform, pre_transform)
        
        # Load index file
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found. Please run preprocessing first: {self.index_path}")
        
        self.index = torch.load(self.index_path)
        self.transform = transform
        self.pre_transform = pre_transform
        
        logger.info(f"Loaded HPO graph dataset with {len(self.index)} samples")
    
    def len(self):
        return len(self.index)
    
    def get(self, idx):
        """Load a graph by index."""
        gene, case_id, file_idx = self.index[idx]
        # Use our own Path objects for file operations
        data_path = self.processed_path / f"batch_{file_idx}.pt"
        
        # Load the specific batch file that contains our graph
        batch_data = torch.load(data_path)
        
        # Get the specific graph from the batch
        data = batch_data[f"{gene}_{case_id}"]
        
        # Apply transform if provided
        if self.transform:
            data = self.transform(data)
            
        return data
    
    @staticmethod
    def get_gene_mask(dataset, gene):
        """Get a boolean mask for all samples from a specific gene."""
        mask = []
        for idx in range(len(dataset)):
            gene_name, _, _ = dataset.index[idx]
            mask.append(gene_name == gene)
        return torch.tensor(mask)


def convert_to_dataset(
    input_json: str,
    output_dir: str,
    batch_size: int = 1000,
    max_samples_per_gene: Optional[int] = None,
    embedding_lookup_path: str = "data/embeddings/hpo_embeddings_latest.pkl",
    include_ancestors: bool = True,
    include_reverse_edges: bool = True,
    clear_existing: bool = False
):
    """
    Convert simulated patient data from JSON to PyTorch Geometric dataset.
    
    Args:
        input_json: Path to input JSON file with simulated patient data
        output_dir: Directory to save processed dataset
        batch_size: Number of samples to process at once
        max_samples_per_gene: Maximum samples to use per gene (useful for testing)
        embedding_lookup_path: Path to pre-computed HPO embeddings
        include_ancestors: Whether to augment with ancestor HPO terms
        include_reverse_edges: Whether to include reverse edges in the graph
        clear_existing: Whether to clear existing processed files
    """
    total_start_time = time.time()
    
    # Create output directories
    start_time = time.time()
    output_dir = Path(output_dir)
    processed_dir = output_dir 
    os.makedirs(processed_dir, exist_ok=True)
    logger.info(f"Directory setup took {time.time() - start_time:.2f} seconds")
    
    if clear_existing:
        start_time = time.time()
        logger.info(f"Clearing existing processed files in {processed_dir}")
        for f in processed_dir.glob("*.pt"):
            os.remove(f)
        logger.info(f"Clearing existing files took {time.time() - start_time:.2f} seconds")
            
    # Initialize GraPhens with configuration for our task
    start_time = time.time()
    logger.info("Initializing GraPhens...")
    graphens = (GraPhens()
                .with_lookup_embeddings(embedding_lookup_path)
                .with_augmentation(strategy=[
                    {"type": "siblings"},
                    {"type": "local", "include_ancestors": True, "include_descendants": False}
                ])
                .with_adjacency_settings(include_reverse_edges=include_reverse_edges))
    logger.info(f"GraPhens initialization took {time.time() - start_time:.2f} seconds")
                
    # Load data 
    start_time = time.time()
    logger.info(f"Loading JSON file: {input_json}")
    with open(input_json, 'r') as f:
        # Load only the basic structure first
        logger.info("Loading gene structure from JSON...")
        gene_data = json.load(f)
    json_load_time = time.time() - start_time
    logger.info(f"Loading JSON structure took {json_load_time:.2f} seconds")
    
    # Build index and process batches
    start_processing_time = time.time()
    index = []
    batch_num = 0
    current_batch = {}  # Store current batch of graphs
    current_batch_count = 0
    
    # Process each gene
    for gene_idx, (gene, cases) in enumerate(tqdm(gene_data.items(), desc="Processing genes")):
        gene_start_time = time.time()
        
        # Apply sample limit if specified
        if max_samples_per_gene:
            cases = cases[:max_samples_per_gene]
        
        # Collect all phenotype lists for the current gene
        patient_data_for_gene = {}
        for case_idx, phenotype_ids in enumerate(cases):
             # Check if phenotype_ids is not empty and contains valid HPO terms
            if phenotype_ids and all(isinstance(pid, str) and pid.startswith("HP:") for pid in phenotype_ids):
                patient_data_for_gene[f"{gene}_{case_idx}"] = phenotype_ids
            else:
                logger.warning(f"Skipping invalid or empty phenotype list for {gene} case {case_idx}: {phenotype_ids}")
        
        num_valid_cases = len(patient_data_for_gene)
        if num_valid_cases == 0:
            logger.warning(f"No valid cases found for gene {gene}. Skipping.")
            continue
            
        logger.info(f"Processing {num_valid_cases} valid cases for gene {gene}...")
        
        # Create graphs for all valid cases of the current gene in batch
        try:
            batch_graph_creation_start_time = time.time()
            graphs_for_gene = graphens.create_graphs_from_multiple_patients(
                patient_data_for_gene, 
                show_progress=False # Use outer tqdm progress
            )
            batch_graph_creation_time = time.time() - batch_graph_creation_start_time
            logger.info(f"Batch graph creation for {gene} ({num_valid_cases} cases) took {batch_graph_creation_time:.2f} seconds")

            # Iterate through the created graphs and add to batch/index
            for patient_key, graph in graphs_for_gene.items():
                # Extract gene and case_idx from patient_key
                _, case_id_str = patient_key.split('_', 1)
                case_idx = int(case_id_str)

                # Convert to PyTorch Geometric Data
                data = Data(
                    x=torch.tensor(graph.node_features, dtype=torch.float),
                    edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
                    # Retrieve original phenotype IDs from the input data
                    phenotype_ids=patient_data_for_gene[patient_key], 
                    gene=gene
                )
                
                # Add to current batch
                current_batch[patient_key] = data
                current_batch_count += 1
                
                # Add to index
                index.append((gene, case_idx, batch_num))
                
                # Save the batch if it reaches the batch size
                if current_batch_count >= batch_size:
                    batch_save_start_time = time.time()
                    batch_path = processed_dir / f"batch_{batch_num}.pt"
                    torch.save(current_batch, batch_path)
                    batch_save_time = time.time() - batch_save_start_time
                    logger.info(f"Saved batch {batch_num} with {current_batch_count} graphs. Took {batch_save_time:.2f} seconds")
                    
                    # Reset for next batch
                    current_batch = {}
                    current_batch_count = 0
                    batch_num += 1
                        
        except Exception as e:
            logger.error(f"Error processing gene {gene} with batch method: {e}")
            # Optionally, add fallback to single processing or skip the gene
            continue
            
        gene_processing_time = time.time() - gene_start_time
        logger.info(f"Processing gene {gene} took {gene_processing_time:.2f} seconds")
        
    processing_duration = time.time() - start_processing_time
    logger.info(f"Total data processing loop took {processing_duration:.2f} seconds")
    
    # Save any remaining graphs in the last batch
    if current_batch_count > 0:
        batch_save_start_time = time.time()
        batch_path = processed_dir / f"batch_{batch_num}.pt"
        torch.save(current_batch, batch_path)
        batch_save_time = time.time() - batch_save_start_time
        logger.info(f"Saved final batch {batch_num} with {current_batch_count} graphs. Took {batch_save_time:.2f} seconds")
        
    # Save the index
    start_time = time.time()
    index_path = processed_dir / "index.pt"
    torch.save(index, index_path)
    logger.info(f"Saved index with {len(index)} total samples. Took {time.time() - start_time:.2f} seconds")
    
    # Create metadata file
    start_time = time.time()
    metadata = {
        'num_samples': len(index),
        'num_batches': batch_num + 1,
        'batch_size': batch_size,
        'genes': list(gene_data.keys()),
        'samples_per_gene': {gene: len(cases) if not max_samples_per_gene 
                            else min(len(cases), max_samples_per_gene) 
                            for gene, cases in gene_data.items()},
        'embedding_lookup': embedding_lookup_path,
        'include_ancestors': include_ancestors,
        'include_reverse_edges': include_reverse_edges
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata. Took {time.time() - start_time:.2f} seconds")
        
    logger.info(f"Dataset creation complete. Saved to {output_dir}")
    total_duration = time.time() - total_start_time
    logger.info(f"Total script execution time: {total_duration:.2f} seconds")
    
    return metadata


def create_train_val_test_splits(
    dataset_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Create train/validation/test splits for the dataset.
    
    Args:
        dataset_dir: Directory containing the processed dataset
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
    """
    total_split_start_time = time.time()
    logger.info("Starting dataset split creation...")
    
    dataset_dir = Path(dataset_dir)
    splits_dir = dataset_dir / "splits"
    os.makedirs(splits_dir, exist_ok=True)
    
    # Load the dataset
    start_time = time.time()
    dataset = HPOGraphDataset(root_dir=dataset_dir)
    logger.info(f"Dataset loading for splits took {time.time() - start_time:.2f} seconds")
    
    # Calculate split sizes
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val
    
    # Create splits
    start_time = time.time()
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test]
    )
    logger.info(f"Random split creation took {time.time() - start_time:.2f} seconds")
    
    # Save the split indices
    start_time = time.time()
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices
    
    torch.save(train_indices, splits_dir / "train_indices.pt")
    torch.save(val_indices, splits_dir / "val_indices.pt")
    torch.save(test_indices, splits_dir / "test_indices.pt")
    logger.info(f"Saving standard split indices took {time.time() - start_time:.2f} seconds")
    
    # Create a gene-wise stratified split too
    start_time = time.time()
    gene_splits = {}
    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    for gene in tqdm(metadata['genes'], desc="Creating gene-wise splits"):
        # Get indices for this gene
        gene_mask = HPOGraphDataset.get_gene_mask(dataset, gene)
        gene_indices_tensor = torch.nonzero(gene_mask).squeeze()
        
        # Handle case where gene might have 0 or 1 sample after filtering
        if gene_indices_tensor.numel() == 0:
            logger.warning(f"Gene {gene} has no samples in the dataset. Skipping for gene-wise split.")
            continue
        elif gene_indices_tensor.numel() == 1:
             # Convert scalar tensor to list
            gene_indices = [gene_indices_tensor.item()]
        else:
            gene_indices = gene_indices_tensor.tolist()

        # Ensure it's a list
        if not isinstance(gene_indices, list):
             gene_indices = [gene_indices] # Wrap scalar in list if needed

        if len(gene_indices) == 0: # Double check after potential conversion issues
            logger.warning(f"Gene {gene} resulted in zero indices. Skipping.")
            continue

        # Shuffle the indices
        np.random.seed(seed)
        np.random.shuffle(gene_indices)
        
        # Calculate split sizes
        n_gene_total = len(gene_indices)
        # Ensure at least one sample per split if possible, handle small counts
        n_gene_train = max(1, int(train_ratio * n_gene_total)) if n_gene_total > 2 else n_gene_total
        n_gene_val = max(1, int(val_ratio * n_gene_total)) if n_gene_total > n_gene_train else 0
        n_gene_test = max(0, n_gene_total - n_gene_train - n_gene_val)

        # Adjust if splits exceed total (due to min 1 requirement)
        if n_gene_train + n_gene_val + n_gene_test > n_gene_total:
             # Prioritize train, then val, then test
             if n_gene_train > n_gene_total: n_gene_train = n_gene_total
             n_gene_val = min(n_gene_val, n_gene_total - n_gene_train)
             n_gene_test = n_gene_total - n_gene_train - n_gene_val


        # Create the splits
        gene_train = gene_indices[:n_gene_train]
        gene_val = gene_indices[n_gene_train : n_gene_train + n_gene_val]
        gene_test = gene_indices[n_gene_train + n_gene_val :]
        
        gene_splits[gene] = {
            'train': gene_train,
            'val': gene_val,
            'test': gene_test
        }
    logger.info(f"Gene-wise split calculation took {time.time() - start_time:.2f} seconds")
    
    # Save gene-wise splits
    start_time = time.time()
    with open(splits_dir / "gene_splits.json", 'w') as f:
        # Convert numpy types to native types for JSON serialization
        serializable_splits = {}
        for gene, split in gene_splits.items():
            serializable_splits[gene] = {
                k: [int(i) for i in v] for k, v in split.items()
            }
        json.dump(serializable_splits, f, indent=2)
    logger.info(f"Saving gene-wise splits took {time.time() - start_time:.2f} seconds")
    
    logger.info(f"Created dataset splits in {splits_dir}")
    logger.info(f"Train: {n_train}, Validation: {n_val}, Test: {n_test}")
    total_split_duration = time.time() - total_split_start_time
    logger.info(f"Total split creation time: {total_split_duration:.2f} seconds")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert simulated patient data to PyTorch Geometric dataset")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with simulated patient data")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for dataset")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of samples to process at once")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples per gene (for testing)")
    parser.add_argument("--embeddings", type=str, default="data/embeddings/hpo_embeddings_gsarti_biobert-nli_20250317_162424.pkl", 
                        help="Path to pre-computed HPO embeddings")
    parser.add_argument("--no-ancestors", action="store_true", help="Don't include ancestor terms")
    parser.add_argument("--no-reverse-edges", action="store_true", help="Don't include reverse edges")
    parser.add_argument("--clear-existing", action="store_true", help="Clear existing processed files")
    parser.add_argument("--create-splits", action="store_true", help="Create train/val/test splits")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training data ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation data ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    
    args = parser.parse_args()
    
    # Convert dataset
    metadata = convert_to_dataset(
        input_json=args.input,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples_per_gene=args.max_samples,
        embedding_lookup_path=args.embeddings,
        include_ancestors=not args.no_ancestors,
        include_reverse_edges=not args.no_reverse_edges,
        clear_existing=args.clear_existing
    )
    
    # Create splits if requested
    if args.create_splits:
        create_train_val_test_splits(
            dataset_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=1.0 - args.train_ratio - args.val_ratio,
            seed=args.seed
        )
