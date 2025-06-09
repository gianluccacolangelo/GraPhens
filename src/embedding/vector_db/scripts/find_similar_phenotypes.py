#!/usr/bin/env python3
"""
Find phenotypes similar to a query phenotype.

This script finds phenotypes similar to a query phenotype based on their
vector embeddings, using a pre-built vector database.
"""

from src.embedding.vector_db.cli import similarity_cli

if __name__ == "__main__":
    similarity_cli() 