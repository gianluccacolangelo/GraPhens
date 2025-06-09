"""
GraPhens Ontology Package

This package provides tools for working with ontologies, particularly the
Human Phenotype Ontology (HPO).
"""

from .check_deprecated import CheckDeprecated
from .hpo_graph import HPOGraphProvider
from .hpo_updater import HPOUpdater

__all__ = ['CheckDeprecated', 'HPOGraphProvider', 'HPOUpdater'] 