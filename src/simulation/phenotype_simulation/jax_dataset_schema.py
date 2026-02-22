"""Schema and validation helpers for Keras+JAX graph datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

SCHEMA_VERSION = "1.0.0"
NPZ_SHARD_KEYS = (
    "x",
    "node_mask",
    "edge_index",
    "edge_mask",
    "y",
    "gene_idx",
    "case_idx",
)


def _require_keys(payload: Dict[str, Any], required_keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")


def _assert_int_like(value: Any, field_name: str, min_value: int = 0) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Field '{field_name}' must be an int, got {type(value)}")
    if value < min_value:
        raise ValueError(f"Field '{field_name}' must be >= {min_value}, got {value}")
    return value


def validate_manifest(manifest: Dict[str, Any]) -> None:
    """Validate manifest structure and consistency."""
    _require_keys(
        manifest,
        (
            "schema_version",
            "created_at",
            "source_json",
            "num_samples",
            "feature_dim",
            "max_nodes",
            "max_edges",
            "genes",
            "gene_to_idx",
            "shards",
            "config",
        ),
        "manifest",
    )

    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version '{manifest['schema_version']}'. "
            f"Expected '{SCHEMA_VERSION}'."
        )

    num_samples = _assert_int_like(manifest["num_samples"], "num_samples")
    _assert_int_like(manifest["feature_dim"], "feature_dim", min_value=1)
    _assert_int_like(manifest["max_nodes"], "max_nodes", min_value=1)
    _assert_int_like(manifest["max_edges"], "max_edges")

    genes = manifest["genes"]
    if not isinstance(genes, list) or not all(isinstance(g, str) for g in genes):
        raise ValueError("Field 'genes' must be a list[str].")
    if len(set(genes)) != len(genes):
        raise ValueError("Field 'genes' contains duplicates.")

    gene_to_idx = manifest["gene_to_idx"]
    if not isinstance(gene_to_idx, dict):
        raise ValueError("Field 'gene_to_idx' must be a dictionary.")
    if set(gene_to_idx.keys()) != set(genes):
        raise ValueError("Field 'gene_to_idx' keys must match 'genes'.")
    expected_gene_indices = set(range(len(genes)))
    observed_gene_indices = set(gene_to_idx.values())
    if observed_gene_indices != expected_gene_indices:
        raise ValueError(
            "Field 'gene_to_idx' values must be contiguous and cover "
            f"{sorted(expected_gene_indices)}; got {sorted(observed_gene_indices)}."
        )

    shards = manifest["shards"]
    if not isinstance(shards, list):
        raise ValueError("Field 'shards' must be a list.")

    expected_start = 0
    accumulated_samples = 0
    for shard in shards:
        _require_keys(
            shard,
            ("file", "num_samples", "index_start", "index_end"),
            "manifest.shards entry",
        )
        shard_samples = _assert_int_like(shard["num_samples"], "shards[].num_samples")
        start = _assert_int_like(shard["index_start"], "shards[].index_start")
        end = _assert_int_like(shard["index_end"], "shards[].index_end")

        if not isinstance(shard["file"], str):
            raise ValueError("Field 'shards[].file' must be a string.")
        if shard_samples == 0:
            raise ValueError("Shard sample count must be > 0.")
        if start != expected_start:
            raise ValueError(
                f"Shard index_start mismatch. Expected {expected_start}, got {start}."
            )
        if end != start + shard_samples - 1:
            raise ValueError(
                f"Shard index range mismatch for '{shard['file']}': "
                f"start={start}, end={end}, samples={shard_samples}."
            )

        expected_start = end + 1
        accumulated_samples += shard_samples

    if accumulated_samples != num_samples:
        raise ValueError(
            f"Total samples from shards ({accumulated_samples}) does not match "
            f"manifest num_samples ({num_samples})."
        )

    if not isinstance(manifest["config"], dict):
        raise ValueError("Field 'config' must be a dictionary.")


def validate_shard_arrays(
    *,
    x: np.ndarray,
    node_mask: np.ndarray,
    edge_index: np.ndarray,
    edge_mask: np.ndarray,
    y: np.ndarray,
    gene_idx: np.ndarray,
    case_idx: np.ndarray,
    max_nodes: int,
    max_edges: int,
    feature_dim: int,
) -> None:
    """Validate shard array shapes, dtypes, and padding semantics."""
    if x.dtype != np.float32:
        raise ValueError(f"Array 'x' must be float32, got {x.dtype}.")
    if node_mask.dtype != np.bool_:
        raise ValueError(f"Array 'node_mask' must be bool, got {node_mask.dtype}.")
    if edge_index.dtype != np.int32:
        raise ValueError(f"Array 'edge_index' must be int32, got {edge_index.dtype}.")
    if edge_mask.dtype != np.bool_:
        raise ValueError(f"Array 'edge_mask' must be bool, got {edge_mask.dtype}.")
    if y.dtype != np.int32:
        raise ValueError(f"Array 'y' must be int32, got {y.dtype}.")
    if gene_idx.dtype != np.int32:
        raise ValueError(f"Array 'gene_idx' must be int32, got {gene_idx.dtype}.")
    if case_idx.dtype != np.int32:
        raise ValueError(f"Array 'case_idx' must be int32, got {case_idx.dtype}.")

    if x.ndim != 3:
        raise ValueError(f"Array 'x' must be rank-3, got shape {x.shape}.")
    if edge_index.ndim != 3:
        raise ValueError(
            f"Array 'edge_index' must be rank-3, got shape {edge_index.shape}."
        )
    if node_mask.ndim != 2:
        raise ValueError(
            f"Array 'node_mask' must be rank-2, got shape {node_mask.shape}."
        )
    if edge_mask.ndim != 2:
        raise ValueError(
            f"Array 'edge_mask' must be rank-2, got shape {edge_mask.shape}."
        )
    if y.ndim != 1 or gene_idx.ndim != 1 or case_idx.ndim != 1:
        raise ValueError("Arrays 'y', 'gene_idx', and 'case_idx' must be rank-1.")

    samples = x.shape[0]
    if node_mask.shape != (samples, max_nodes):
        raise ValueError(
            f"Array 'node_mask' shape mismatch. Expected {(samples, max_nodes)}, "
            f"got {node_mask.shape}."
        )
    if edge_index.shape != (samples, 2, max_edges):
        raise ValueError(
            f"Array 'edge_index' shape mismatch. Expected {(samples, 2, max_edges)}, "
            f"got {edge_index.shape}."
        )
    if edge_mask.shape != (samples, max_edges):
        raise ValueError(
            f"Array 'edge_mask' shape mismatch. Expected {(samples, max_edges)}, "
            f"got {edge_mask.shape}."
        )
    if x.shape[1:] != (max_nodes, feature_dim):
        raise ValueError(
            f"Array 'x' trailing shape mismatch. Expected {(max_nodes, feature_dim)}, "
            f"got {x.shape[1:]}."
        )
    if y.shape[0] != samples or gene_idx.shape[0] != samples or case_idx.shape[0] != samples:
        raise ValueError("Label/trace arrays length must match number of samples.")

    if np.any(x[~node_mask] != 0):
        raise ValueError("Padded node feature positions must be zero-filled.")

    edge_mask_3d = np.broadcast_to(edge_mask[:, None, :], edge_index.shape)
    edge_padding_values = edge_index[~edge_mask_3d]
    if edge_padding_values.size and not np.all(edge_padding_values == -1):
        raise ValueError("Padded edge positions must be filled with -1.")

    valid_edge_values = edge_index[edge_mask_3d]
    if valid_edge_values.size:
        if np.any(valid_edge_values < 0):
            raise ValueError("Valid edges contain negative indices.")
        if np.any(valid_edge_values >= max_nodes):
            raise ValueError("Valid edges reference node indices beyond max_nodes.")

    node_counts = node_mask.sum(axis=1).astype(np.int32)
    for sample_idx in range(samples):
        sample_edge_mask = edge_mask[sample_idx]
        if not np.any(sample_edge_mask):
            continue
        sample_edges = edge_index[sample_idx, :, sample_edge_mask]
        if np.any(sample_edges >= node_counts[sample_idx]):
            raise ValueError(
                "Valid edges reference padded node positions in sample "
                f"{sample_idx} (node_count={node_counts[sample_idx]})."
            )


def load_and_validate_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load manifest JSON and validate it."""
    import json

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    validate_manifest(manifest)
    return manifest


def validate_npz_shard_against_manifest(
    *,
    shard_path: Path,
    shard_entry: Dict[str, Any],
    manifest: Dict[str, Any],
) -> None:
    """Validate a shard file against manifest expectations."""
    with np.load(shard_path) as data:
        payload = {key: data[key] for key in NPZ_SHARD_KEYS}
        validate_shard_arrays(
            x=payload["x"],
            node_mask=payload["node_mask"],
            edge_index=payload["edge_index"],
            edge_mask=payload["edge_mask"],
            y=payload["y"],
            gene_idx=payload["gene_idx"],
            case_idx=payload["case_idx"],
            max_nodes=manifest["max_nodes"],
            max_edges=manifest["max_edges"],
            feature_dim=manifest["feature_dim"],
        )

        expected = shard_entry["num_samples"]
        observed = payload["x"].shape[0]
        if observed != expected:
            raise ValueError(
                f"Shard sample count mismatch for '{shard_path}'. "
                f"Expected {expected}, got {observed}."
            )

        num_classes = len(manifest["genes"])
        if np.any(payload["y"] < 0) or np.any(payload["y"] >= num_classes):
            raise ValueError(
                f"Shard '{shard_path}' contains out-of-range class labels in 'y'."
            )
        if np.any(payload["gene_idx"] < 0) or np.any(payload["gene_idx"] >= num_classes):
            raise ValueError(
                f"Shard '{shard_path}' contains out-of-range class labels in 'gene_idx'."
            )
        if not np.array_equal(payload["y"], payload["gene_idx"]):
            raise ValueError(
                f"Shard '{shard_path}' has mismatched 'y' and 'gene_idx' values."
            )
        if np.any(payload["case_idx"] < 0):
            raise ValueError(
                f"Shard '{shard_path}' contains negative 'case_idx' values."
            )
