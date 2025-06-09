#!/usr/bin/env python3
"""
Script to run the biomedical embedding model evaluation on phenotype descriptions.

This script samples phenotypes from the HPO ontology, generates embeddings with
different biomedical models, and analyzes the distributions of pairwise similarities.
"""

import os
import argparse
from datetime import datetime
from src.embedding.evaluation import run_evaluator

def main():
    """Run the embedding evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate biomedical embedding models on phenotype descriptions."
    )
    
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=1000, 
        help="Number of phenotypes to sample"
    )
    
    parser.add_argument(
        "--sample-pool-size", 
        type=int, 
        default=9600, 
        help="Size of the pool to sample from"
    )
    
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/embeddings", 
        help="Directory to store embedding data"
    )
    
    parser.add_argument(
        "--ontology-dir", 
        type=str, 
        default="data/ontology", 
        help="Directory where HPO ontology data is stored"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None, 
        help="Directory to save results (default: results/embedding_evaluation_YYYYMMDD_HHMMSS)"
    )
    
    parser.add_argument(
        "--no-vector-db", 
        action="store_true", 
        help="Do not save embeddings to a vector database"
    )
    
    args = parser.parse_args()
    
    # Generate timestamp-based output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/embedding_evaluation_{timestamp}"
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting evaluation with {args.sample_size} phenotypes...")
    print(f"Results will be saved to {args.output_dir}")
    
    # Run evaluation
    run_evaluator(
        sample_size=args.sample_size,
        sample_pool_size=args.sample_pool_size,
        random_seed=args.random_seed,
        data_dir=args.data_dir,
        ontology_dir=args.ontology_dir,
        output_dir=args.output_dir,
        save_vector_db=not args.no_vector_db
    )
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")
    print(f"Check the histograms and summary statistics to identify the models with the most variance.")

if __name__ == "__main__":
    main() 