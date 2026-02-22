"""Whitelist for symbols intentionally kept in Phase 1 dead-code refactor."""

from src.core.interfaces import EmbeddingStrategy
from src.core.types import Graph
from src.graphens import GraPhens
from src.simulation.phenotype_simulation.integration_demo import SimpleGNN
from training.models.models import GenePhenAIv2_0, GenePhenAIv2_5, GenePhenAIv3_0

GraPhens.save_config_to_yaml
GraPhens.with_global_context
GraPhens.run_on_cluster

EmbeddingStrategy.embed
Graph.edge_attr

GenePhenAIv2_0.forward
GenePhenAIv2_5.forward
GenePhenAIv3_0.forward
SimpleGNN.forward
