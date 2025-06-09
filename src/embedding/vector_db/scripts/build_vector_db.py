#!/usr/bin/env python3
"""
Build a vector database of all phenotypes in the HPO ontology.

This script loads all phenotypes from the HPO graph, generates embeddings using
the specified model, and saves them to a vector database for use with LookupEmbeddingStrategy.
"""

from src.embedding.vector_db.cli import build_cli

if __name__ == "__main__":
    build_cli() 