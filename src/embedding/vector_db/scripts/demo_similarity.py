#!/usr/bin/env python3
"""
Demonstrate phenotype similarity using vector embeddings.

This script demonstrates how to find similar phenotypes for a set of example
phenotypes or user-specified queries, using a pre-built vector database.
"""

from src.embedding.vector_db.cli import demo_cli

if __name__ == "__main__":
    demo_cli() 