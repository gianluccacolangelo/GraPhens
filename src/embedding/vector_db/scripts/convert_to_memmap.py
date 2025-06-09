#!/usr/bin/env python3
"""
Convert embeddings from pickle format to memory-mapped format.

This script converts pickle-based embedding files to memory-mapped format for
more efficient access. Memory-mapped files allow the system to access embeddings
without loading the entire file into memory, significantly improving performance.

Example usage:
    # Convert a specific embedding file
    python -m src.embedding.vector_db.scripts.convert_to_memmap \
        --input data/embeddings/hpo_embeddings_pritamdeka_S-PubMedBert-MS-MARCO_20250317_155952.pkl
        
    # Convert all embedding files in the directory
    python -m src.embedding.vector_db.scripts.convert_to_memmap --all
    
    # Convert with a custom output directory
    python -m src.embedding.vector_db.scripts.convert_to_memmap \
        --all --output-dir data/embeddings/memmap
"""

import os
import argparse
import sys
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).resolve().parents[4]))

from src.embedding.vector_db.memmap import (
    convert_embedding_to_memmap,
    convert_all_embeddings,
    list_memmap_files
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert embedding files to memory-mapped format"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        help="Path to the input pickle file containing embeddings"
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Convert all embedding files in the directory"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Directory to store memory-mapped files (defaults to same as input)"
    )
    
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="NumPy data type for the memory-mapped array (default: float32)"
    )
    
    parser.add_argument(
        "--data-dir",
        default="data/embeddings",
        help="Directory containing embedding files (used with --all)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing memory-mapped files"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # List existing memory-mapped files
    if args.list:
        memmap_files = list_memmap_files(args.data_dir)
        if memmap_files:
            print(f"Found {len(memmap_files)} memory-mapped files:")
            for i, (data_path, index_path) in enumerate(memmap_files):
                data_size = os.path.getsize(data_path) / (1024 * 1024)
                index_size = os.path.getsize(index_path) / (1024 * 1024)
                print(f"{i+1}. {data_path} ({data_size:.2f} MB)")
                print(f"   Index: {index_path} ({index_size:.2f} MB)")
        else:
            print(f"No memory-mapped files found in {args.data_dir}")
        return
    
    # Start timer
    start_time = time.time()
    
    # Convert a single file
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist")
            return
        
        convert_embedding_to_memmap(
            input_path=args.input,
            output_dir=args.output_dir,
            dtype=args.dtype
        )
    
    # Convert all files
    elif args.all:
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory {args.data_dir} does not exist")
            return
        
        memmap_files = convert_all_embeddings(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            dtype=args.dtype
        )
        
        print(f"Converted {len(memmap_files)} files to memory-mapped format")
    
    # End timer
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main() 