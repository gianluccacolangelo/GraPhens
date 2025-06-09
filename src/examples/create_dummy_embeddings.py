#!/usr/bin/env python
"""
Create Dummy Embeddings for Testing

This script creates a simple dictionary of dummy embeddings for testing GraPhens.
The embeddings are just random vectors, not useful for actual machine learning,
but perfect for testing the infrastructure.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Common HPO terms related to seizures and neurological conditions
HPO_TERMS = [
    "HP:0001250",  # Seizure
    "HP:0001251",  # Atonic seizure
    "HP:0002121",  # Generalized non-motor (absence) seizure
    "HP:0002123",  # Generalized myoclonic seizure
    "HP:0002069",  # Bilateral tonic-clonic seizure
    "HP:0002197",  # Generalized seizure
    "HP:0002373",  # Febrile seizure
    "HP:0002266",  # Focal seizure
    "HP:0010818",  # Central apnea
    "HP:0007270",  # Atypical absence seizure
    "HP:0010819",  # Convulsions in neonatal period
    "HP:0011097",  # Epileptiform EEG discharges
    "HP:0012469",  # Infantile spasms
    "HP:0010851",  # Focal clonic seizure
    "HP:0011153",  # Focal impaired awareness seizure
]

def main():
    # Create embeddings dir if it doesn't exist
    embeddings_dir = Path("/home/brainy/GraPhens/data/embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Create a dictionary of dummy embeddings
    embedding_dict = {}
    embedding_dim = 768
    
    # Generate a random embedding for each HPO term
    for term in HPO_TERMS:
        # Create a random vector
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        embedding_dict[term] = embedding
    
    # Save the embeddings to a pickle file
    output_path = embeddings_dir / "dummy_embeddings.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(embedding_dict, f)
    
    print(f"Created dummy embeddings for {len(embedding_dict)} HPO terms")
    print(f"Embeddings saved to {output_path}")
    print(f"Embedding dimension: {embedding_dim}")

if __name__ == "__main__":
    main() 