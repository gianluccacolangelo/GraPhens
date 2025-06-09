#!/usr/bin/env python3
"""
Build vector databases for all biomedical models evaluated in our report.

This script builds vector databases for:
- gsarti/biobert-nli
- emilyalsentzer/Bio_ClinicalBERT
- dmis-lab/biobert-base-cased-v1.2
- microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

Note: For pritamdeka/S-PubMedBert-MS-MARCO, please check if it's already built.
"""

import os
import argparse
from src.embedding.vector_db import build_vector_db_for_models


def main():
    """Build vector databases for all biomedical models."""
    parser = argparse.ArgumentParser(
        description="Build vector databases for all biomedical models evaluated in our report."
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size for processing (default: 32)"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/embeddings", 
        help="Directory to store embedding data (default: data/embeddings)"
    )
    
    parser.add_argument(
        "--ontology-dir", 
        type=str, 
        default="data/ontology", 
        help="Directory where HPO ontology data is stored (default: data/ontology)"
    )
    
    parser.add_argument(
        "--no-progress", 
        action="store_true", 
        help="Disable progress bar"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have vector databases"
    )
    
    args = parser.parse_args()
    
    # Define biomedical models to build databases for
    biomedical_models = {
        "sentence-transformers": [
            "gsarti/biobert-nli",
            "pritamdeka/S-PubMedBert-MS-MARCO"
        ],
        "huggingface": [
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "dmis-lab/biobert-base-cased-v1.2",
            "emilyalsentzer/Bio_ClinicalBERT"
        ]
    }
    
    # Check which models already have vector databases
    if args.skip_existing:
        print("Checking for existing vector databases...")
        for model_type, model_names in biomedical_models.items():
            for model_name in list(model_names):
                model_str = model_name.replace('/', '_')
                pattern = os.path.join(args.data_dir, f"hpo_embeddings_{model_str}_*.pkl")
                import glob
                files = glob.glob(pattern)
                if files:
                    print(f"Found existing vector database for {model_name}: {files[0]}")
                    biomedical_models[model_type].remove(model_name)
    
    # Build vector databases
    print(f"Building vector databases for {sum(len(models) for models in biomedical_models.values())} models...")
    output_paths = build_vector_db_for_models(
        models=biomedical_models,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        ontology_dir=args.ontology_dir,
        progress_bar=not args.no_progress
    )
    
    print(f"\nSuccessfully built {len(output_paths)} vector databases")
    
    # Print paths to all vector databases
    print("\nVector databases:")
    for path in output_paths:
        print(f"- {path}")
    
    print("\nTo use these vector databases with LookupEmbeddingStrategy, see the documentation in src/embedding/README.md")
    print("\nTo find similar phenotypes using these vector databases:")
    print("python -m src.embedding.vector_db.scripts.find_similar_phenotypes HP:0001250 --vector-db <path_to_vector_db>")


if __name__ == "__main__":
    main() 