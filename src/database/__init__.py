"""
Database module for accessing gene-phenotype relationship data.

This module provides interfaces and implementations for querying gene-phenotype
associations, frequencies, and disease relationships.
"""

from .gene_phenotype_db import GenePhenotypeDatabase, HPOAGenePhenotypeDatabase
from .gene_phenotype_analysis import analyze_gene_phenotypes

__all__ = [
    'GenePhenotypeDatabase', 
    'HPOAGenePhenotypeDatabase',
    'analyze_gene_phenotypes'
] 