#!/usr/bin/env python
"""Create a precomputed dense dataset for Keras FNN training."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

LOGGER = logging.getLogger("create_fnn_dataset")


@dataclass
class BuildStats:
    total_cases: int = 0
    total_genes: int = 0
    total_missing_embedding_terms: int = 0
    total_empty_cases: int = 0


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_simulation(path: Path) -> Dict[str, List[List[str]]]:
    LOGGER.info("Loading simulation JSON: %s", path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be a dictionary mapping genes to case lists.")
    LOGGER.info("Loaded simulation with %d genes", len(data))
    return data


def load_embeddings(path: Path) -> Tuple[Dict[str, np.ndarray], int]:
    LOGGER.info("Loading embedding lookup: %s", path)
    with path.open("rb") as f:
        embedding_dict = pickle.load(f)
    if not isinstance(embedding_dict, dict) or not embedding_dict:
        raise ValueError(f"Embedding dictionary is empty or invalid: {path}")
    dim = int(next(iter(embedding_dict.values())).shape[0])
    LOGGER.info("Loaded %d embeddings with dimension %d", len(embedding_dict), dim)
    return embedding_dict, dim


def mean_pooled_feature(
    hpo_ids: List[str],
    embedding_dict: Dict[str, np.ndarray],
    dim: int,
) -> Tuple[np.ndarray, int]:
    vectors: List[np.ndarray] = []
    missing = 0
    for hpo_id in hpo_ids:
        vec = embedding_dict.get(hpo_id)
        if vec is None:
            missing += 1
            continue
        vectors.append(vec)

    if not vectors:
        return np.zeros(dim, dtype=np.float32), missing

    stacked = np.vstack(vectors).astype(np.float32, copy=False)
    return stacked.mean(axis=0), missing


def build_stratified_splits(
    *,
    gene_to_count: Dict[str, int],
    gene_to_idx: Dict[str, int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    running_offset = 0

    genes_sorted = sorted(gene_to_idx, key=lambda g: gene_to_idx[g])
    for gene in genes_sorted:
        n = int(gene_to_count[gene])
        local = np.arange(running_offset, running_offset + n, dtype=np.int64)
        rng.shuffle(local)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_test <= 0:
            raise ValueError(
                f"Split ratios produce empty test split for gene {gene} (n={n})."
            )

        train.append(local[:n_train])
        val.append(local[n_train : n_train + n_val])
        test.append(local[n_train + n_val :])
        running_offset += n

    return {
        "train": np.concatenate(train, axis=0),
        "val": np.concatenate(val, axis=0),
        "test": np.concatenate(test, axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create FNN-ready dense dataset from simulated patients and lookup "
            "embeddings using mean pooling."
        )
    )
    parser.add_argument("--input-json", type=str, required=True, help="Input simulated patients JSON.")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/embeddings/hpo_embeddings_gsarti_biobert-nli_20250317_162424.pkl",
        help="Lookup embedding dictionary (.pkl).",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for memmap dataset files.")
    parser.add_argument("--drop-gene-dash", action="store_true", help="Drop placeholder '-' gene if present.")
    parser.add_argument("--create-splits", action="store_true", help="Also create stratified train/val/test indices.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training data ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation data ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    input_json = Path(args.input_json)
    embeddings_path = Path(args.embeddings)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = load_simulation(input_json)
    if args.drop_gene_dash and "-" in sim:
        LOGGER.info("Dropping placeholder gene '-'")
        sim.pop("-", None)

    embedding_dict, emb_dim = load_embeddings(embeddings_path)

    genes = sorted(sim.keys())
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    idx_to_gene = {i: g for g, i in gene_to_idx.items()}
    gene_to_count = {g: len(sim[g]) for g in genes}
    total_cases = int(sum(gene_to_count.values()))
    feature_dim = emb_dim

    LOGGER.info("Preparing dataset with %d genes and %d cases", len(genes), total_cases)
    LOGGER.info("Feature dimension: %d (mean pooled embeddings)", feature_dim)

    x_path = output_dir / "X.float32.memmap"
    y_path = output_dir / "y.int32.memmap"
    x_memmap = np.memmap(x_path, mode="w+", dtype=np.float32, shape=(total_cases, feature_dim))
    y_memmap = np.memmap(y_path, mode="w+", dtype=np.int32, shape=(total_cases,))

    stats = BuildStats(total_cases=total_cases, total_genes=len(genes))

    row = 0
    for gene in genes:
        label = gene_to_idx[gene]
        for case in sim[gene]:
            if not isinstance(case, list):
                raise ValueError(f"Case for gene '{gene}' must be a list of HPO IDs.")
            pooled, missing = mean_pooled_feature(case, embedding_dict, emb_dim)
            if len(case) == 0:
                stats.total_empty_cases += 1
            stats.total_missing_embedding_terms += missing
            x_memmap[row] = pooled
            y_memmap[row] = label
            row += 1

    x_memmap.flush()
    y_memmap.flush()

    with (output_dir / "gene_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump(gene_to_idx, f, indent=2)
    with (output_dir / "idx_to_gene.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx_to_gene.items()}, f, indent=2)

    if args.create_splits:
        splits = build_stratified_splits(
            gene_to_count=gene_to_count,
            gene_to_idx=gene_to_idx,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        splits_dir = output_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        np.save(splits_dir / "train_indices.npy", splits["train"])
        np.save(splits_dir / "val_indices.npy", splits["val"])
        np.save(splits_dir / "test_indices.npy", splits["test"])
        LOGGER.info(
            "Saved splits: train=%d val=%d test=%d",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )

    metadata = {
        "schema_version": "1.0.0",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_json": str(input_json),
        "embeddings_path": str(embeddings_path),
        "pooling": "mean",
        "num_genes": len(genes),
        "num_samples": total_cases,
        "feature_dim": feature_dim,
        "embedding_dim": emb_dim,
        "x_memmap_path": str(x_path),
        "y_memmap_path": str(y_path),
        "missing_embedding_terms_total": stats.total_missing_embedding_terms,
        "empty_cases_total": stats.total_empty_cases,
        "train_ratio": args.train_ratio if args.create_splits else None,
        "val_ratio": args.val_ratio if args.create_splits else None,
        "seed": args.seed if args.create_splits else None,
        "drop_gene_dash": args.drop_gene_dash,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    LOGGER.info("Done. Dataset written to: %s", output_dir)


if __name__ == "__main__":
    main()
