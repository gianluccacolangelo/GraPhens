"""Convert simulated patient JSON data into Keras+JAX NPZ graph shards."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from src.graphens import GraPhens
from src.simulation.phenotype_simulation.jax_dataset_schema import (
    load_and_validate_manifest,
    validate_npz_shard_against_manifest,
)
from src.simulation.phenotype_simulation.jax_npz_writer import (
    GraphSample,
    JAXNPZShardWriter,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_case_idx(patient_key: str) -> int:
    """Extract case index from patient key formatted as '{gene}_{case_idx}'."""
    try:
        _, case_idx = patient_key.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid patient key format '{patient_key}'.") from exc
    return int(case_idx)


def _is_valid_phenotype_list(phenotype_ids: List[str]) -> bool:
    return bool(phenotype_ids) and all(
        isinstance(pid, str) and pid.startswith("HP:") for pid in phenotype_ids
    )


def _prepare_patient_data_for_gene(
    *,
    gene: str,
    cases: List[List[str]],
    max_samples_per_gene: Optional[int],
) -> Dict[str, List[str]]:
    selected_cases = cases[:max_samples_per_gene] if max_samples_per_gene else cases
    patient_data: Dict[str, List[str]] = {}
    for case_idx, phenotype_ids in enumerate(selected_cases):
        if _is_valid_phenotype_list(phenotype_ids):
            patient_data[f"{gene}_{case_idx}"] = phenotype_ids
        else:
            logger.warning(
                "Skipping invalid or empty phenotype list for %s case %s: %s",
                gene,
                case_idx,
                phenotype_ids,
            )
    return patient_data


def _build_graphens_pipeline(
    *,
    embedding_lookup_path: str,
    include_ancestors: bool,
    include_reverse_edges: bool,
) -> GraPhens:
    return (
        GraPhens()
        .with_lookup_embeddings(embedding_lookup_path)
        .with_augmentation(
            strategy=[
                {"type": "siblings"},
                {
                    "type": "local",
                    "include_ancestors": include_ancestors,
                    "include_descendants": False,
                },
            ]
        )
        .with_adjacency_settings(include_reverse_edges=include_reverse_edges)
    )


def _load_input_json(input_json: str) -> Dict[str, List[List[str]]]:
    logger.info("Loading JSON file: %s", input_json)
    with open(input_json, "r", encoding="utf-8") as f:
        gene_data = json.load(f)
    if not isinstance(gene_data, dict):
        raise ValueError("Input JSON must be a dictionary mapping genes to cases.")
    return gene_data


def _collect_dataset_stats(
    *,
    gene_data: Dict[str, List[List[str]]],
    graphens: GraPhens,
    max_samples_per_gene: Optional[int],
) -> Dict[str, object]:
    """Pass 1: compute global limits and sample counts."""
    max_nodes = 0
    max_edges = 0
    feature_dim: Optional[int] = None
    total_samples = 0
    samples_per_gene: Dict[str, int] = {}

    for gene in tqdm(sorted(gene_data.keys()), desc="Pass 1 - collecting stats"):
        patient_data = _prepare_patient_data_for_gene(
            gene=gene,
            cases=gene_data[gene],
            max_samples_per_gene=max_samples_per_gene,
        )
        if not patient_data:
            continue

        graphs_for_gene = graphens.create_graphs_from_multiple_patients(
            patient_data, show_progress=False
        )
        if not graphs_for_gene:
            continue

        valid_samples = 0
        for patient_key in sorted(graphs_for_gene.keys(), key=_parse_case_idx):
            graph = graphs_for_gene[patient_key]
            node_features = np.asarray(graph.node_features)
            edge_index = np.asarray(graph.edge_index)

            if node_features.ndim != 2:
                raise ValueError(
                    f"Node features for '{patient_key}' must be rank-2, got {node_features.shape}."
                )
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(
                    f"edge_index for '{patient_key}' must have shape (2, E), got {edge_index.shape}."
                )

            this_dim = int(node_features.shape[1])
            if feature_dim is None:
                feature_dim = this_dim
            elif feature_dim != this_dim:
                raise ValueError(
                    f"Inconsistent feature dimensions detected: {feature_dim} vs {this_dim}."
                )

            max_nodes = max(max_nodes, int(node_features.shape[0]))
            max_edges = max(max_edges, int(edge_index.shape[1]))
            valid_samples += 1

        if valid_samples:
            samples_per_gene[gene] = valid_samples
            total_samples += valid_samples

    if total_samples == 0:
        raise ValueError("No valid samples were produced from input JSON.")
    if feature_dim is None:
        raise ValueError("Could not infer feature dimension from generated graphs.")

    return {
        "max_nodes": max_nodes,
        "max_edges": max_edges,
        "feature_dim": feature_dim,
        "total_samples": total_samples,
        "samples_per_gene": samples_per_gene,
    }


def _resolve_padding_limit(
    *,
    observed: int,
    override: Optional[int],
    field_name: str,
) -> int:
    if override is None:
        return observed
    if override < observed:
        raise ValueError(
            f"{field_name} override ({override}) cannot be smaller than observed maximum ({observed})."
        )
    return override


def _graph_to_sample(
    *,
    graph,
    gene_idx_value: int,
    case_idx_value: int,
) -> GraphSample:
    node_features = np.asarray(graph.node_features, dtype=np.float32)
    edge_index = np.asarray(graph.edge_index, dtype=np.int32)
    return GraphSample(
        x=node_features,
        edge_index=edge_index,
        y=gene_idx_value,
        gene_idx=gene_idx_value,
        case_idx=case_idx_value,
    )


def _write_dataset_shards(
    *,
    gene_data: Dict[str, List[List[str]]],
    graphens: GraPhens,
    max_samples_per_gene: Optional[int],
    shard_size: int,
    gene_to_idx: Dict[str, int],
    writer: JAXNPZShardWriter,
    expected_total_samples: int,
) -> int:
    """Pass 2: generate graphs again and write padded NPZ shards."""
    if shard_size <= 0:
        raise ValueError(f"shard_size must be > 0, got {shard_size}.")

    pending_samples: List[GraphSample] = []
    total_written = 0

    for gene in tqdm(sorted(gene_data.keys()), desc="Pass 2 - writing shards"):
        if gene not in gene_to_idx:
            continue

        patient_data = _prepare_patient_data_for_gene(
            gene=gene,
            cases=gene_data[gene],
            max_samples_per_gene=max_samples_per_gene,
        )
        if not patient_data:
            continue

        graphs_for_gene = graphens.create_graphs_from_multiple_patients(
            patient_data, show_progress=False
        )
        if not graphs_for_gene:
            continue

        for patient_key in sorted(graphs_for_gene.keys(), key=_parse_case_idx):
            case_idx = _parse_case_idx(patient_key)
            sample = _graph_to_sample(
                graph=graphs_for_gene[patient_key],
                gene_idx_value=gene_to_idx[gene],
                case_idx_value=case_idx,
            )
            pending_samples.append(sample)

            if len(pending_samples) >= shard_size:
                writer.write_samples(pending_samples)
                total_written += len(pending_samples)
                pending_samples = []

    if pending_samples:
        writer.write_samples(pending_samples)
        total_written += len(pending_samples)

    if total_written != expected_total_samples:
        raise RuntimeError(
            f"Pass 2 produced {total_written} samples, expected {expected_total_samples}."
        )

    return total_written


def _clear_existing_outputs(output_dir: Path) -> None:
    """Delete generated files from a previous run."""
    generated_dirs = [output_dir / "shards", output_dir / "splits", output_dir / "processed"]
    generated_files = [output_dir / "manifest.json", output_dir / "index.pt", output_dir / "metadata.json"]

    for directory in generated_dirs:
        if directory.exists():
            shutil.rmtree(directory)

    for file_path in generated_files:
        if file_path.exists():
            file_path.unlink()

    for legacy_file in output_dir.glob("batch_*.pt"):
        legacy_file.unlink()


def convert_to_dataset(
    input_json: str,
    output_dir: str,
    shard_size: int = 1000,
    max_samples_per_gene: Optional[int] = None,
    embedding_lookup_path: str = "data/embeddings/hpo_embeddings_latest.pkl",
    include_ancestors: bool = True,
    include_reverse_edges: bool = True,
    clear_existing: bool = False,
    max_nodes: Optional[int] = None,
    max_edges: Optional[int] = None,
    seed: int = 42,
):
    """
    Convert simulated patient JSON data to NPZ shard dataset for Keras+JAX.

    The conversion uses a two-pass strategy:
    1. Pass 1 computes global shape limits and sample counts.
    2. Pass 2 writes fixed-shape NPZ shards with padding and masks.
    """
    total_start = time.time()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        logger.info("Clearing existing generated outputs in %s", output_path)
        _clear_existing_outputs(output_path)

    graphens = _build_graphens_pipeline(
        embedding_lookup_path=embedding_lookup_path,
        include_ancestors=include_ancestors,
        include_reverse_edges=include_reverse_edges,
    )
    gene_data = _load_input_json(input_json)

    pass1_start = time.time()
    stats = _collect_dataset_stats(
        gene_data=gene_data,
        graphens=graphens,
        max_samples_per_gene=max_samples_per_gene,
    )
    logger.info("Pass 1 completed in %.2fs", time.time() - pass1_start)

    observed_max_nodes = int(stats["max_nodes"])
    observed_max_edges = int(stats["max_edges"])
    feature_dim = int(stats["feature_dim"])
    total_samples = int(stats["total_samples"])
    samples_per_gene = dict(stats["samples_per_gene"])
    genes = sorted(samples_per_gene.keys())
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}

    resolved_max_nodes = _resolve_padding_limit(
        observed=observed_max_nodes,
        override=max_nodes,
        field_name="max_nodes",
    )
    resolved_max_edges = _resolve_padding_limit(
        observed=observed_max_edges,
        override=max_edges,
        field_name="max_edges",
    )

    config = {
        "embedding_lookup": embedding_lookup_path,
        "include_ancestors": include_ancestors,
        "include_reverse_edges": include_reverse_edges,
        "max_samples_per_gene": max_samples_per_gene,
        "seed": seed,
        "shard_size": shard_size,
        "observed_max_nodes": observed_max_nodes,
        "observed_max_edges": observed_max_edges,
        "samples_per_gene": samples_per_gene,
    }

    writer = JAXNPZShardWriter(
        output_dir=output_path,
        source_json=input_json,
        feature_dim=feature_dim,
        max_nodes=resolved_max_nodes,
        max_edges=resolved_max_edges,
        genes=genes,
        gene_to_idx=gene_to_idx,
        config=config,
    )

    pass2_start = time.time()
    _write_dataset_shards(
        gene_data=gene_data,
        graphens=graphens,
        max_samples_per_gene=max_samples_per_gene,
        shard_size=shard_size,
        gene_to_idx=gene_to_idx,
        writer=writer,
        expected_total_samples=total_samples,
    )
    logger.info("Pass 2 completed in %.2fs", time.time() - pass2_start)

    manifest = writer.finalize()
    _validate_written_dataset(output_dir=output_path, manifest=manifest)
    logger.info(
        "Dataset creation complete: %s samples, %s shards, max_nodes=%s, max_edges=%s.",
        manifest["num_samples"],
        len(manifest["shards"]),
        manifest["max_nodes"],
        manifest["max_edges"],
    )
    logger.info("Manifest written to %s", output_path / "manifest.json")
    logger.info("Total conversion time: %.2fs", time.time() - total_start)
    return manifest


def _load_gene_idx_vector(*, dataset_dir: Path, manifest: Dict[str, object]) -> np.ndarray:
    total_samples = int(manifest["num_samples"])
    gene_idx_vector = np.empty((total_samples,), dtype=np.int32)

    for shard in manifest["shards"]:
        shard_path = dataset_dir / shard["file"]
        with np.load(shard_path) as data:
            shard_gene_idx = np.asarray(data["gene_idx"], dtype=np.int32)

        index_start = int(shard["index_start"])
        index_end = int(shard["index_end"])
        expected_size = index_end - index_start + 1
        if shard_gene_idx.shape[0] != expected_size:
            raise ValueError(
                f"Shard gene_idx length mismatch for {shard_path}: "
                f"expected {expected_size}, got {shard_gene_idx.shape[0]}."
            )
        gene_idx_vector[index_start : index_end + 1] = shard_gene_idx

    return gene_idx_vector


def _validate_written_dataset(*, output_dir: Path, manifest: Dict[str, object]) -> None:
    for shard_entry in manifest["shards"]:
        shard_path = output_dir / shard_entry["file"]
        validate_npz_shard_against_manifest(
            shard_path=shard_path,
            shard_entry=shard_entry,
            manifest=manifest,
        )


def create_train_val_test_splits(
    dataset_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """Create deterministic train/val/test split index files as .npy arrays."""
    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        raise ValueError("train_ratio, val_ratio, and test_ratio must each be between 0 and 1.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )

    dataset_path = Path(dataset_dir)
    manifest = load_and_validate_manifest(dataset_path / "manifest.json")
    total_samples = int(manifest["num_samples"])
    if total_samples == 0:
        raise ValueError("Cannot create splits for an empty dataset.")

    splits_dir = dataset_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    indices = np.arange(total_samples, dtype=np.int32)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = int(train_ratio * total_samples)
    n_val = int(val_ratio * total_samples)
    n_test = total_samples - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    np.save(splits_dir / "train_indices.npy", train_indices)
    np.save(splits_dir / "val_indices.npy", val_indices)
    np.save(splits_dir / "test_indices.npy", test_indices)

    gene_idx_vector = _load_gene_idx_vector(dataset_dir=dataset_path, manifest=manifest)
    gene_splits: Dict[str, Dict[str, List[int]]] = {}

    for gene in manifest["genes"]:
        idx = int(manifest["gene_to_idx"][gene])
        gene_indices = np.where(gene_idx_vector == idx)[0].astype(np.int32)
        if gene_indices.size == 0:
            logger.warning("Gene %s has no samples. Skipping gene-wise split.", gene)
            continue

        gene_rng = np.random.default_rng(seed + idx)
        gene_rng.shuffle(gene_indices)

        n_gene_total = int(gene_indices.size)
        n_gene_train = max(1, int(train_ratio * n_gene_total)) if n_gene_total > 2 else n_gene_total
        n_gene_val = max(1, int(val_ratio * n_gene_total)) if n_gene_total > n_gene_train else 0
        n_gene_test = max(0, n_gene_total - n_gene_train - n_gene_val)

        if n_gene_train + n_gene_val + n_gene_test > n_gene_total:
            if n_gene_train > n_gene_total:
                n_gene_train = n_gene_total
            n_gene_val = min(n_gene_val, n_gene_total - n_gene_train)
            n_gene_test = n_gene_total - n_gene_train - n_gene_val

        gene_train = gene_indices[:n_gene_train].tolist()
        gene_val = gene_indices[n_gene_train : n_gene_train + n_gene_val].tolist()
        gene_test = gene_indices[n_gene_train + n_gene_val :].tolist()
        gene_splits[gene] = {"train": gene_train, "val": gene_val, "test": gene_test}

    with (splits_dir / "gene_splits.json").open("w", encoding="utf-8") as f:
        json.dump(gene_splits, f, indent=2)

    logger.info("Created splits in %s", splits_dir)
    logger.info("Train: %s, Validation: %s, Test: %s", n_train, n_val, n_test)
    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "gene_splits": gene_splits,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert simulated patient JSON data into Keras+JAX NPZ dataset shards."
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with simulated patient data")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for NPZ dataset shards")
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Number of samples per NPZ shard file",
    )
    parser.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples per gene (for testing)")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/embeddings/hpo_embeddings_gsarti_biobert-nli_20250317_162424.pkl",
        help="Path to pre-computed HPO embeddings",
    )
    parser.add_argument("--no-ancestors", action="store_true", help="Don't include ancestor terms")
    parser.add_argument("--no-reverse-edges", action="store_true", help="Don't include reverse edges")
    parser.add_argument("--clear-existing", action="store_true", help="Clear existing generated files")
    parser.add_argument("--create-splits", action="store_true", help="Create train/val/test splits")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training data ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation data ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    parser.add_argument("--max-nodes", type=int, default=None, help="Optional override for padded max nodes")
    parser.add_argument("--max-edges", type=int, default=None, help="Optional override for padded max edges")
    args = parser.parse_args()

    effective_shard_size = args.shard_size
    if args.batch_size is not None:
        if "--shard-size" in sys.argv:
            logger.warning(
                "Ignoring deprecated --batch-size because --shard-size was explicitly provided."
            )
        else:
            logger.warning("Argument --batch-size is deprecated. Use --shard-size instead.")
            effective_shard_size = args.batch_size

    convert_to_dataset(
        input_json=args.input,
        output_dir=args.output_dir,
        shard_size=effective_shard_size,
        max_samples_per_gene=args.max_samples,
        embedding_lookup_path=args.embeddings,
        include_ancestors=not args.no_ancestors,
        include_reverse_edges=not args.no_reverse_edges,
        clear_existing=args.clear_existing,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        seed=args.seed,
    )

    if args.create_splits:
        create_train_val_test_splits(
            dataset_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=1.0 - args.train_ratio - args.val_ratio,
            seed=args.seed,
        )
