from __future__ import annotations

import numpy as np

from src.simulation.phenotype_simulation.create_hpo_dataset import create_train_val_test_splits
from src.simulation.phenotype_simulation.jax_dataset_schema import (
    load_and_validate_manifest,
    validate_npz_shard_against_manifest,
)
from src.simulation.phenotype_simulation.jax_npz_writer import GraphSample, JAXNPZShardWriter
from training.datasets.jax_npz_graph_dataset import JAXNPZGraphDataset


def _sample(
    *,
    num_nodes: int,
    feature_dim: int,
    edges: list[tuple[int, int]],
    gene_idx: int,
    case_idx: int,
) -> GraphSample:
    x = np.arange(num_nodes * feature_dim, dtype=np.float32).reshape(num_nodes, feature_dim)
    if edges:
        edge_index = np.asarray(edges, dtype=np.int32).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int32)
    return GraphSample(
        x=x,
        edge_index=edge_index,
        y=gene_idx,
        gene_idx=gene_idx,
        case_idx=case_idx,
    )


def _build_dataset(tmp_path, *, total_samples: int = 10, shard_size: int = 4):
    genes = ["GENE_A", "GENE_B"]
    gene_to_idx = {"GENE_A": 0, "GENE_B": 1}

    writer = JAXNPZShardWriter(
        output_dir=tmp_path,
        source_json="dummy.json",
        feature_dim=3,
        max_nodes=5,
        max_edges=6,
        genes=genes,
        gene_to_idx=gene_to_idx,
        config={"seed": 42},
    )

    samples: list[GraphSample] = []
    for idx in range(total_samples):
        gene_idx = idx % 2
        samples.append(
            _sample(
                num_nodes=2 + (idx % 4),
                feature_dim=3,
                edges=[(0, 1), (1, 0)] if idx % 3 else [(0, 1)],
                gene_idx=gene_idx,
                case_idx=idx,
            )
        )

    for start in range(0, len(samples), shard_size):
        writer.write_samples(samples[start : start + shard_size])

    return writer.finalize()


def test_padding_and_masks_are_correct(tmp_path):
    manifest = _build_dataset(tmp_path, total_samples=3, shard_size=3)
    shard = manifest["shards"][0]
    shard_path = tmp_path / shard["file"]

    with np.load(shard_path) as payload:
        x = payload["x"]
        node_mask = payload["node_mask"]
        edge_index = payload["edge_index"]
        edge_mask = payload["edge_mask"]

    assert x.dtype == np.float32
    assert node_mask.dtype == np.bool_
    assert edge_index.dtype == np.int32
    assert edge_mask.dtype == np.bool_

    assert x.shape == (3, 5, 3)
    assert node_mask.shape == (3, 5)
    assert edge_index.shape == (3, 2, 6)
    assert edge_mask.shape == (3, 6)

    # Sample 0 had 2 nodes and 1 edge.
    assert np.all(node_mask[0, :2])
    assert not np.any(node_mask[0, 2:])
    assert np.all(x[0, 2:, :] == 0.0)
    assert np.all(edge_mask[0, :1])
    assert not np.any(edge_mask[0, 1:])
    assert np.all(edge_index[0, :, 1:] == -1)

    loaded_manifest = load_and_validate_manifest(tmp_path / "manifest.json")
    validate_npz_shard_against_manifest(
        shard_path=shard_path,
        shard_entry=shard,
        manifest=loaded_manifest,
    )


def test_split_creation_is_deterministic(tmp_path):
    _build_dataset(tmp_path, total_samples=12, shard_size=5)

    create_train_val_test_splits(
        dataset_dir=str(tmp_path),
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=7,
    )
    train_a = np.load(tmp_path / "splits" / "train_indices.npy")
    val_a = np.load(tmp_path / "splits" / "val_indices.npy")
    test_a = np.load(tmp_path / "splits" / "test_indices.npy")

    create_train_val_test_splits(
        dataset_dir=str(tmp_path),
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=7,
    )
    train_b = np.load(tmp_path / "splits" / "train_indices.npy")
    val_b = np.load(tmp_path / "splits" / "val_indices.npy")
    test_b = np.load(tmp_path / "splits" / "test_indices.npy")

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert np.array_equal(test_a, test_b)


def test_jax_loader_batch_shapes_and_shuffle_contract(tmp_path):
    _build_dataset(tmp_path, total_samples=10, shard_size=4)

    ds_a = JAXNPZGraphDataset(
        dataset_root=tmp_path,
        split="all",
        batch_size=4,
        shuffle=True,
        seed=99,
        include_trace_fields=True,
    )
    ds_b = JAXNPZGraphDataset(
        dataset_root=tmp_path,
        split="all",
        batch_size=4,
        shuffle=True,
        seed=99,
        include_trace_fields=True,
    )

    ds_a.set_epoch(0)
    ds_b.set_epoch(0)

    first_batch = next(iter(ds_a))
    assert set(first_batch.keys()) == {
        "x",
        "node_mask",
        "edge_index",
        "edge_mask",
        "y",
        "gene_idx",
        "case_idx",
    }
    assert first_batch["x"].shape == (4, 5, 3)
    assert first_batch["node_mask"].shape == (4, 5)
    assert first_batch["edge_index"].shape == (4, 2, 6)
    assert first_batch["edge_mask"].shape == (4, 6)
    assert first_batch["y"].shape == (4,)

    order_a_epoch0 = np.concatenate([batch["case_idx"] for batch in ds_a])
    order_b_epoch0 = np.concatenate([batch["case_idx"] for batch in ds_b])
    assert np.array_equal(order_a_epoch0, order_b_epoch0)

    ds_a.set_epoch(1)
    order_a_epoch1 = np.concatenate([batch["case_idx"] for batch in ds_a])
    assert not np.array_equal(order_a_epoch0, order_a_epoch1)
