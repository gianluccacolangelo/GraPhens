"""
Memory-mapped embedding utilities for efficient vector database access.

This module provides tools for working with memory-mapped embedding files,
which allow efficient random access to large embedding datasets without
loading the entire file into memory.

Key benefits of memory-mapped files include:
- Faster startup time (no need to load entire files)
- Lower memory usage (only accessed parts are loaded)
- File persistence (OS manages memory, survives program restarts)
- Shared memory (multiple processes can use the same mapped file)
"""

import os
import pickle
import numpy as np
import json
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path
import time

def convert_embedding_to_memmap(
    input_path: str,
    output_dir: Optional[str] = None,
    dtype: str = 'float32'
) -> Tuple[str, str]:
    """
    Convert a pickle-based embedding dictionary to memory-mapped format.
    
    Args:
        input_path: Path to the input pickle file containing embeddings
        output_dir: Directory to store memory-mapped files (defaults to same dir as input)
        dtype: NumPy data type for the memory-mapped array (default: float32)
    
    Returns:
        Tuple of (data_path, index_path) for the created memory-mapped files
    """
    print(f"Converting {input_path} to memory-mapped format...")
    start_time = time.time()
    
    # Load the original embedding dictionary
    with open(input_path, 'rb') as f:
        embedding_dict = pickle.load(f)
    
    # Extract embedding dimensions
    first_key = next(iter(embedding_dict))
    embedding_dim = len(embedding_dict[first_key])
    num_embeddings = len(embedding_dict)
    
    print(f"Converting {num_embeddings} embeddings with dimension {embedding_dim}")
    
    # Determine output paths
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    memmap_data_path = os.path.join(output_dir, f"{base_name}.mmap")
    memmap_index_path = os.path.join(output_dir, f"{base_name}.index.json")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create memory-mapped array
    memmap = np.memmap(
        memmap_data_path,
        dtype=dtype,
        mode='w+',
        shape=(num_embeddings, embedding_dim)
    )
    
    # Create index mapping
    index_map = {}
    
    # Populate memory-mapped array and index mapping
    for i, (hpo_id, embedding) in enumerate(embedding_dict.items()):
        # Store the vector in memmap file
        memmap[i] = embedding.astype(dtype)
        # Store the index mapping
        index_map[hpo_id] = i
    
    # Flush changes to disk
    memmap.flush()
    
    # Save index mapping as JSON
    with open(memmap_index_path, 'w') as f:
        json.dump(index_map, f)
    
    end_time = time.time()
    print(f"Conversion completed in {end_time - start_time:.2f} seconds")
    print(f"Memory-mapped data saved to {memmap_data_path}")
    print(f"Index mapping saved to {memmap_index_path}")
    
    # Verify sizes
    memmap_size = os.path.getsize(memmap_data_path) / (1024 * 1024)
    index_size = os.path.getsize(memmap_index_path) / (1024 * 1024)
    print(f"Memory-mapped data size: {memmap_size:.2f} MB")
    print(f"Index mapping size: {index_size:.2f} MB")
    
    return memmap_data_path, memmap_index_path

def list_memmap_files(data_dir: str = "data/embeddings") -> List[Tuple[str, str]]:
    """
    List all memory-mapped embedding files in the specified directory.
    
    Args:
        data_dir: Directory to search for memory-mapped files
        
    Returns:
        List of tuples (data_path, index_path) for each memory-mapped file
    """
    result = []
    
    for file in os.listdir(data_dir):
        if file.endswith('.mmap'):
            base_path = os.path.join(data_dir, file)
            index_path = base_path.replace('.mmap', '.index.json')
            if os.path.exists(index_path):
                result.append((base_path, index_path))
    
    return result

def convert_all_embeddings(
    data_dir: str = "data/embeddings",
    output_dir: Optional[str] = None,
    dtype: str = 'float32'
) -> List[Tuple[str, str]]:
    """
    Convert all pickle-based embedding files in a directory to memory-mapped format.
    
    Args:
        data_dir: Directory containing embedding pickle files
        output_dir: Directory to store memory-mapped files (defaults to same as data_dir)
        dtype: NumPy data type for the memory-mapped arrays
        
    Returns:
        List of tuples (data_path, index_path) for each created memory-mapped file
    """
    if output_dir is None:
        output_dir = data_dir
        
    os.makedirs(output_dir, exist_ok=True)
    
    result = []
    
    for file in os.listdir(data_dir):
        if file.endswith('.pkl') and not file.endswith('_metadata.pkl'):
            input_path = os.path.join(data_dir, file)
            try:
                data_path, index_path = convert_embedding_to_memmap(
                    input_path=input_path,
                    output_dir=output_dir,
                    dtype=dtype
                )
                result.append((data_path, index_path))
            except Exception as e:
                print(f"Error converting {input_path}: {e}")
    
    return result

def find_latest_memmap(data_dir: str = "data/embeddings") -> Tuple[str, str]:
    """
    Find the most recently created memory-mapped embedding file.
    
    Args:
        data_dir: Directory to search for memory-mapped files
        
    Returns:
        Tuple of (data_path, index_path) for the most recent memory-mapped file
        
    Raises:
        FileNotFoundError: If no memory-mapped files are found
    """
    memmap_files = list_memmap_files(data_dir)
    
    if not memmap_files:
        raise FileNotFoundError(
            f"No memory-mapped embedding files found in {data_dir}. "
            f"Run convert_embedding_to_memmap to create memory-mapped files."
        )
    
    # Sort by file modification time (most recent first)
    memmap_files.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
    
    return memmap_files[0] 