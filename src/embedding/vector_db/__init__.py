"""
Vector database module for working with phenotype embeddings.

This module provides utilities for:
1. Building vector databases of phenotype embeddings
2. Using pre-built vector databases with LookupEmbeddingStrategy
3. Finding similar phenotypes based on vector similarity
4. Creating and using memory-mapped embedding files for optimal performance

The vector databases created by this module are compatible with the 
LookupEmbeddingStrategy, allowing for efficient reuse of pre-computed embeddings.

Memory-mapped files provide significant performance improvements by:
- Not loading the entire embedding database into memory
- Using operating system virtual memory features for efficient access
- Only loading the parts of the file that are actually accessed
"""

from .builder import build_phenotype_vector_db, build_vector_db_for_models
from .similarity import find_similar_phenotypes, demo_similar_phenotypes
from .memmap import (
    convert_embedding_to_memmap,
    convert_all_embeddings,
    list_memmap_files,
    find_latest_memmap
)

__all__ = [
    'build_phenotype_vector_db',
    'build_vector_db_for_models',
    'find_similar_phenotypes',
    'demo_similar_phenotypes',
    'convert_embedding_to_memmap',
    'convert_all_embeddings',
    'list_memmap_files',
    'find_latest_memmap'
] 