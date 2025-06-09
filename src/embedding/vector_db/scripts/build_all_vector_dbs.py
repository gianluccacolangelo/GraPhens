#!/usr/bin/env python3
"""
Build vector databases for multiple biomedical models.

This script builds vector databases for the biomedical models used in our evaluation,
which can then be used with LookupEmbeddingStrategy for efficient phenotype embedding.
"""

from src.embedding.vector_db.cli import build_all_cli

if __name__ == "__main__":
    build_all_cli() 