"""Writers for Keras+JAX graph datasets stored as NPZ shards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from src.simulation.phenotype_simulation.jax_dataset_schema import (
    SCHEMA_VERSION,
    validate_manifest,
    validate_shard_arrays,
)


@dataclass
class GraphSample:
    """In-memory sample before shard serialization."""

    x: np.ndarray
    edge_index: np.ndarray
    y: int
    gene_idx: int
    case_idx: int


class JAXNPZShardWriter:
    """Writes padded graph tensors into compressed NPZ shards."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        shards_subdir: str | Path = "shards",
        manifest_path: str | Path = "manifest.json",
        source_json: str,
        feature_dim: int,
        max_nodes: int,
        max_edges: int,
        genes: List[str],
        gene_to_idx: Dict[str, int],
        config: Dict[str, Any],
    ) -> None:
        self.output_dir = Path(output_dir)
        shards_subdir_path = Path(shards_subdir)
        if shards_subdir_path.is_absolute():
            raise ValueError("shards_subdir must be a relative path inside output_dir.")
        manifest_path_path = Path(manifest_path)
        if manifest_path_path.is_absolute():
            raise ValueError("manifest_path must be a relative path inside output_dir.")

        self.shards_dir = self.output_dir / shards_subdir_path
        self.manifest_path = self.output_dir / manifest_path_path

        self.source_json = source_json
        self.feature_dim = int(feature_dim)
        self.max_nodes = int(max_nodes)
        self.max_edges = int(max_edges)
        self.genes = list(genes)
        self.gene_to_idx = dict(gene_to_idx)
        self.config = dict(config)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        self.shards: List[Dict[str, Any]] = []
        self.total_samples = 0
        self._shard_counter = 0

    def write_samples(self, samples: Iterable[GraphSample]) -> Dict[str, Any] | None:
        """Write a set of samples into a single shard."""
        sample_list = list(samples)
        if not sample_list:
            return None

        shard_size = len(sample_list)
        x = np.zeros((shard_size, self.max_nodes, self.feature_dim), dtype=np.float32)
        node_mask = np.zeros((shard_size, self.max_nodes), dtype=np.bool_)
        edge_index = np.full((shard_size, 2, self.max_edges), -1, dtype=np.int32)
        edge_mask = np.zeros((shard_size, self.max_edges), dtype=np.bool_)
        y = np.zeros((shard_size,), dtype=np.int32)
        gene_idx = np.zeros((shard_size,), dtype=np.int32)
        case_idx = np.zeros((shard_size,), dtype=np.int32)

        for row, sample in enumerate(sample_list):
            node_features = np.asarray(sample.x, dtype=np.float32)
            edges = np.asarray(sample.edge_index, dtype=np.int32)

            if node_features.ndim != 2:
                raise ValueError(f"Sample node features must be rank-2, got {node_features.shape}")
            if edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError(f"Sample edge_index must have shape (2, E), got {edges.shape}")
            if node_features.shape[1] != self.feature_dim:
                raise ValueError(
                    f"Feature dimension mismatch. Expected {self.feature_dim}, "
                    f"got {node_features.shape[1]}."
                )

            n_nodes = node_features.shape[0]
            n_edges = edges.shape[1]
            if n_nodes > self.max_nodes:
                raise ValueError(
                    f"Sample has {n_nodes} nodes, exceeding max_nodes={self.max_nodes}."
                )
            if n_edges > self.max_edges:
                raise ValueError(
                    f"Sample has {n_edges} edges, exceeding max_edges={self.max_edges}."
                )

            x[row, :n_nodes, :] = node_features
            node_mask[row, :n_nodes] = True

            if n_edges:
                edge_index[row, :, :n_edges] = edges
                edge_mask[row, :n_edges] = True

            y[row] = int(sample.y)
            gene_idx[row] = int(sample.gene_idx)
            case_idx[row] = int(sample.case_idx)

        validate_shard_arrays(
            x=x,
            node_mask=node_mask,
            edge_index=edge_index,
            edge_mask=edge_mask,
            y=y,
            gene_idx=gene_idx,
            case_idx=case_idx,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
            feature_dim=self.feature_dim,
        )

        file_name = f"shard_{self._shard_counter:05d}.npz"
        output_path = self.shards_dir / file_name
        np.savez_compressed(
            output_path,
            x=x,
            node_mask=node_mask,
            edge_index=edge_index,
            edge_mask=edge_mask,
            y=y,
            gene_idx=gene_idx,
            case_idx=case_idx,
        )

        relative_output_path = output_path.relative_to(self.output_dir).as_posix()
        index_start = self.total_samples
        index_end = index_start + shard_size - 1
        shard_entry = {
            "file": relative_output_path,
            "num_samples": shard_size,
            "index_start": index_start,
            "index_end": index_end,
        }
        self.shards.append(shard_entry)
        self.total_samples += shard_size
        self._shard_counter += 1
        return shard_entry

    def finalize(self) -> Dict[str, Any]:
        """Write and return manifest metadata."""
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_json": self.source_json,
            "num_samples": self.total_samples,
            "feature_dim": self.feature_dim,
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "genes": self.genes,
            "gene_to_idx": self.gene_to_idx,
            "shards": self.shards,
            "config": self.config,
        }
        validate_manifest(manifest)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=False)
        return manifest
