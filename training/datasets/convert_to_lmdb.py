#!/usr/bin/env python
"""
Convert existing HPO dataset .pt batch files to LMDB format.

Usage:
  python -m training.datasets.convert_to_lmdb /path/to/dataset

This script will create an LMDB database in the /path/to/dataset/hpo_lmdb directory
containing all the graph samples from the processed .pt batch files.

For large datasets, this can take some time, but it's a one-time cost that will
dramatically improve data loading performance, especially for random access.
"""

import os
import sys
import torch
import lmdb
import pickle
import argparse
import multiprocessing as mp
import time
import logging
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import io
from collections import Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import the serialization worker function
from training.datasets.lmdb_hpo_dataset import _worker_serialize_batch

# Add a constant for samples per batch file (adjust if needed)
EXPECTED_SAMPLES_PER_BATCH = 5000

def get_arg_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert HPO dataset to LMDB format')
    parser.add_argument('dataset_root', type=str, help='Path to the dataset root directory')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count(),
                       help='Number of worker processes (default: number of CPU cores)')
    parser.add_argument('--map-size', type=int, default=1099511627776*3,
                       help='LMDB map size in bytes (default: 1TB)')
    parser.add_argument('--batch-commit', type=int, default=5000,
                       help='Commit to LMDB every N samples (default: 50)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip conversion if LMDB already exists')
    return parser

def find_batch_files(processed_dir: Path) -> List[Path]:
    """Find all batch files in the processed directory."""
    batch_files = sorted([f for f in processed_dir.glob('batch_*.pt')])
    if not batch_files:
        raise FileNotFoundError(f"No batch_*.pt files found in {processed_dir}")
    return batch_files

def load_index_file(processed_dir: Path) -> List[Tuple[str, int, int]]:
    """Load the index file containing (gene, case_id, file_idx) tuples."""
    index_path = processed_dir / 'index.pt'
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found at {index_path}")
    
    try:
        index_data = torch.load(index_path)
        logger.info(f"Loaded index with {len(index_data)} entries")
        return index_data
    except Exception as e:
        logger.error(f"Failed to load index file: {e}")
        raise

def build_gene_index_map(index_data: List[Tuple[str, int, int]]) -> Dict[str, List[int]]:
    """Build a mapping from gene names to dataset indices."""
    gene_idx_map = {}
    for idx, (gene, _, _) in enumerate(index_data):
        if gene not in gene_idx_map:
            gene_idx_map[gene] = []
        gene_idx_map[gene].append(idx)
    
    logger.info(f"Built gene index map with {len(gene_idx_map)} genes")
    return gene_idx_map

def convert_to_lmdb(args):
    """Convert the dataset to LMDB format."""
    dataset_root = Path(args.dataset_root).resolve()
    processed_dir = dataset_root / 'processed' / 'processed' 
    lmdb_dir = dataset_root / 'hpo_lmdb'
    
    # Check if directories exist
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root directory not found: {dataset_root}")
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")
    
    # Check if LMDB already exists
    if lmdb_dir.exists() and (lmdb_dir / 'data.mdb').exists() and args.skip_existing:
        logger.info(f"LMDB already exists at {lmdb_dir} and --skip-existing is set. Skipping conversion.")
        return
    
    # Create LMDB directory if it doesn't exist
    lmdb_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all batch files
    batch_files = find_batch_files(processed_dir)
    logger.info(f"Found {len(batch_files)} batch files to convert")
    
    # Load the index file
    index_data = load_index_file(processed_dir)
    
    # Use the length of the loaded index as the definitive expected count
    expected_total_samples = len(index_data)
    logger.info(f"Definitive expected total samples based on index file: {expected_total_samples}")

    # --- NEW: Create mapping from original keys to global indices ---
    logger.info("Creating lookup dictionary from original keys to global indices...")
    original_key_to_global_idx = {}
    for idx, (gene, case_id, _) in enumerate(index_data):
        original_key = (gene, case_id)
        original_key_to_global_idx[original_key] = idx
    
    logger.info(f"Created lookup map with {len(original_key_to_global_idx)} entries.")
    
    # Verify no duplicate keys in the mapping
    if len(original_key_to_global_idx) != expected_total_samples:
        logger.warning(f"POTENTIAL DATA ISSUE: Lookup map has {len(original_key_to_global_idx)} entries "
                     f"but index file has {expected_total_samples} entries. "
                     f"This suggests duplicate (gene, case_id) pairs in the index.")
    # --- END NEW ---

    # Build gene index map for faster gene masks
    gene_idx_map = build_gene_index_map(index_data)
    
    # Create LMDB environment
    env = lmdb.open(str(lmdb_dir), map_size=args.map_size, max_dbs=1)
    
    # Start conversion
    logger.info(f"Starting conversion to LMDB using {args.num_workers} workers")
    start_time = time.time()
    
    # Prepare batch arguments for parallel processing
    batch_args = []
    logger.info("Preparing arguments for parallel processing...")
    # Directly create args without loading files
    for file_idx, batch_file in enumerate(tqdm(batch_files, desc="Preparing batch arguments")):
        batch_args.append((batch_file, lmdb_dir))
        # We're no longer calculating or tracking indices here

    # Process batches in parallel
    total_samples_written = 0
    total_samples_processed = 0
    missing_key_count = 0
    problematic_batches = []  # Track batch files with issues
    results_chunk = [] # Temporary list to hold results for batch commit

    logger.info("Starting batch processing with multiprocessing pool...")
    # Use a try/finally block to ensure pool is closed even on error
    pool = mp.Pool(args.num_workers)
    try:
        # Use imap_unordered for potentially better performance
        # Wrap with tqdm for progress bar
        for batch_idx, batch_results in enumerate(tqdm(
            pool.imap_unordered(_worker_serialize_batch, batch_args),
            total=len(batch_args),
            desc="Converting batches"
        )):
            if batch_results: # Only process if the worker returned results (didn't fail)
                total_samples_processed += len(batch_results)
                
                # --- CHANGED: Use index map to get global indices ---
                # Process each result and add to chunk if the key exists in our index map
                batch_missing_keys = 0  # Count missing keys in this batch
                batch_file_name = None  # Will be set from the first item
                
                for orig_key, serialized, gene, batch_file_name in batch_results:
                    # Save batch_file_name for potential inspection - This assignment was incorrect
                    # if batch_file_name and not batch_file_name:  
                    #     batch_file_name = batch_file_name
                    # The batch_file_name comes directly from the worker result tuple now
                    
                    # --- NEW: Key Format Conversion --- 
                    lookup_key = None
                    if isinstance(orig_key, str) and '_' in orig_key:
                        parts = orig_key.split('_', 1)
                        if len(parts) == 2:
                            key_gene, case_id_str = parts
                            try:
                                case_id = int(case_id_str)
                                lookup_key = (key_gene, case_id)
                            except ValueError:
                                logger.warning(f"Could not parse integer case_id from key '{orig_key}' in {batch_file_name}")
                    elif isinstance(orig_key, tuple): 
                        # Assume it's already in the correct format if it's a tuple
                        # Add validation if necessary (e.g., check length, types)
                        lookup_key = orig_key 
                    else:
                         logger.warning(f"Unrecognized key format '{orig_key}' (type: {type(orig_key)}) in {batch_file_name}")
                    # --- END NEW --- 

                    # Look up the global index for this key using the CONVERTED lookup_key
                    global_idx = None
                    if lookup_key is not None:
                        global_idx = original_key_to_global_idx.get(lookup_key)
                    
                    if global_idx is not None:
                        # Add to chunk with the correct index from the index.pt file
                        results_chunk.append((global_idx, serialized, gene, orig_key))
                    else:
                        # Key doesn't exist in index.pt - Critical data inconsistency
                        batch_missing_keys += 1
                        missing_key_count += 1
                        if missing_key_count <= 10:  # Limit logging to avoid spam
                            # Log the ORIGINAL key and the key we TRIED to look up
                            logger.error(f"MISSING KEY ERROR: Original key '{orig_key}' (lookup attempted as: {lookup_key}) from {batch_file_name} not found in index.")
                        elif missing_key_count == 11:
                            logger.error("Additional missing keys detected. Further warnings suppressed.")
                
                # If this batch had a significant number of missing keys, add to problematic batches
                # Note: This check might trigger less now if the key conversion works
                if batch_missing_keys > 0 and batch_file_name:
                    if batch_missing_keys / len(batch_results) > 0.1:  # More than 10% missing
                        logger.warning(f"Batch {batch_file_name} has {batch_missing_keys}/{len(batch_results)} missing keys AFTER format conversion attempt.")
                        problematic_batches.append(batch_file_name)
                        
                        # Removed automatic inspection call
                        # if len(problematic_batches) <= 3: ...
                # --- END CHANGED ---

                # Check if the chunk is large enough to commit
                if len(results_chunk) >= args.batch_commit:
                    logger.debug(f"Accumulated {len(results_chunk)} results. Writing batch commit...")
                    # Write the current chunk to LMDB
                    try:
                        with env.begin(write=True) as txn:
                             # Sort chunk by index before writing (optional, might help locality)
                             results_chunk.sort(key=lambda x: x[0])
                             for idx, serialized, gene, orig_key in results_chunk:
                                 txn.put(f"{idx}".encode('ascii'), serialized)
                                 total_samples_written += 1
                        logger.debug(f"Committed batch. Total written so far: {total_samples_written}")
                        results_chunk = [] # Clear the chunk
                    except lmdb.Error as e:
                        logger.error(f"LMDB Error during batch commit: {e}. Stopping conversion.")
                        pool.terminate() # Stop workers
                        raise
                    except Exception as e:
                        logger.error(f"Unexpected error during batch commit: {e}. Stopping conversion.")
                        pool.terminate() # Stop workers
                        raise
            else:
                # Worker returned empty results, likely due to an error
                logger.warning(f"Worker returned empty results for batch {batch_idx}, potentially due to an error.")
                # Try to identify which batch file had the issue
                if batch_idx < len(batch_args):
                    failed_batch_file = str(batch_args[batch_idx][0]) # Get path as string
                    logger.error(f"Possible failed batch file: {failed_batch_file}")
                    problematic_batches.append(failed_batch_file)
                    
                    # Removed automatic inspection call
                    # if Path(failed_batch_file).exists(): ...
    finally:
        pool.close()
        pool.join()
        logger.info("Multiprocessing pool closed.")

    # --- NEW: Log statistics about missing keys and problematic batches ---
    logger.info(f"Total samples processed from batch files: {total_samples_processed}")
    if missing_key_count > 0:
        logger.warning(f"Found {missing_key_count} samples in batch files that weren't in the index.pt file.")
    
    if problematic_batches:
        logger.warning(f"Identified {len(problematic_batches)} problematic batch files.")
        if len(problematic_batches) <= 10:
            logger.warning(f"Problematic batch files: {problematic_batches}")
        else:
            logger.warning(f"First 10 problematic batch files: {problematic_batches[:10]}")
    # --- END NEW ---

    # --- Write any remaining results in the last chunk --- 
    if results_chunk:
        logger.info(f"Writing final chunk of {len(results_chunk)} results to LMDB...")
        try:
            with env.begin(write=True) as txn:
                # Sort chunk by index before writing
                results_chunk.sort(key=lambda x: x[0])
                for idx, serialized, gene, orig_key in results_chunk:
                    txn.put(f"{idx}".encode('ascii'), serialized)
                    total_samples_written += 1
            logger.info(f"Committed final chunk. Total written: {total_samples_written}")
            results_chunk = [] # Clear chunk just in case
        except lmdb.Error as e:
             logger.error(f"LMDB Error during final batch commit: {e}. Dataset may be incomplete.")
             raise
        except Exception as e:
            logger.error(f"Unexpected error during final batch commit: {e}. Dataset may be incomplete.")
            raise
    else:
        logger.info("No remaining results in the final chunk to write.")

    # --- Write Metadata (Length and Gene Map) --- 
    logger.info("Writing metadata (__len__, __gene_idx_map__) to LMDB...")
    try:
        with env.begin(write=True) as txn:
            # Store the total length (use the count of successfully written samples)
            logger.info(f"Storing final sample count ({total_samples_written}) under key '__len__'")
            txn.put(b'__len__', str(total_samples_written).encode('ascii'))
            
            # Store the gene index map
            logger.info("Storing gene index map under key '__gene_idx_map__'")
            txn.put(b'__gene_idx_map__', pickle.dumps(gene_idx_map))

            logger.info("Finalizing metadata transaction...")
    except lmdb.Error as e:
        logger.error(f"LMDB Error writing metadata: {e}. Dataset metadata might be missing or incomplete.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error writing metadata: {e}")
        raise

    # --- Added: Final Count Verification --- 
    logger.info("--- Verification Step --- ")
    logger.info(f"Expected samples based on index file: {expected_total_samples}")
    logger.info(f"Samples actually written to LMDB:      {total_samples_written}")
    if total_samples_written != expected_total_samples:
         logger.critical(f"CRITICAL MISMATCH: Number of samples written ({total_samples_written}) does NOT match the number expected from the index file ({expected_total_samples})!")
         logger.critical("The LMDB dataset is incomplete or corrupted. Please investigate the conversion logs.")
         # Optionally raise an error or return a non-zero exit code from main
    else:
         logger.info("SUCCESS: Number of samples written matches the number expected.")
    # ----------------------------------------

    # Log completion
    elapsed = time.time() - start_time
    logger.info(f"Conversion completed in {elapsed:.2f} seconds")
    logger.info(f"Converted {total_samples_written} samples to LMDB format")
    logger.info(f"LMDB stored at: {lmdb_dir}")
    
    # Update metadata if it exists
    metadata_path = dataset_root / 'metadata.json'
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Add LMDB metadata
            metadata['num_samples'] = total_samples_written
            metadata['lmdb_created'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Write updated metadata
            with open(metadata_path, 'w') as f: json.dump(metadata, f, indent=2)
            
            logger.info(f"Updated metadata with LMDB information")
        except Exception as e:
            logger.warning(f"Failed to update metadata: {e}")

def main():
    """Main entry point."""
    parser = get_arg_parser()
    args = parser.parse_args()
    
    try:
        convert_to_lmdb(args)
        return 0
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 