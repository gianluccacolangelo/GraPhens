from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from src.simulation.phenotype_simulation.jax_npz_writer import GraphSample, JAXNPZShardWriter
from training.datasets.keras_npz_sequence import resolve_split_indices
from training.training import compute_ranking_metrics

def _create_tiny_npz_dataset(dataset_root: Path, *, num_samples: int = 12) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    genes = ["GENE_A", "GENE_B", "GENE_C"]
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}

    writer = JAXNPZShardWriter(
        output_dir=dataset_root,
        source_json="dummy_simulated.json",
        feature_dim=4,
        max_nodes=6,
        max_edges=8,
        genes=genes,
        gene_to_idx=gene_to_idx,
        config={"seed": 42},
    )

    rng = np.random.default_rng(42)
    samples: list[GraphSample] = []
    for case_idx in range(num_samples):
        n_nodes = 3 + (case_idx % 3)
        n_edges = 2 + (case_idx % 3)
        x = rng.normal(size=(n_nodes, 4)).astype(np.float32)

        src = rng.integers(0, n_nodes, size=n_edges, dtype=np.int32)
        dst = rng.integers(0, n_nodes, size=n_edges, dtype=np.int32)
        edge_index = np.stack([src, dst], axis=0).astype(np.int32)

        gene_idx = case_idx % len(genes)
        samples.append(
            GraphSample(
                x=x,
                edge_index=edge_index,
                y=gene_idx,
                gene_idx=gene_idx,
                case_idx=case_idx,
            )
        )

    for start in range(0, len(samples), 4):
        writer.write_samples(samples[start : start + 4])

    writer.finalize()
    return dataset_root


def test_mrr_metric_matches_reference() -> None:
    logits = np.asarray(
        [
            [0.1, 0.9, 0.0],
            [0.4, 0.2, 0.3],
            [0.2, 0.1, 0.7],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([1, 2, 0], dtype=np.int32)

    mrr, top_k = compute_ranking_metrics(logits, targets, k_vals=(1, 2, 3))

    # Ranks: [1, 2, 2] -> MRR = (1 + 1/2 + 1/2) / 3
    assert np.isclose(mrr, (1.0 + 0.5 + 0.5) / 3.0)
    assert np.isclose(top_k[1], 1.0 / 3.0)
    assert np.isclose(top_k[2], 1.0)
    assert np.isclose(top_k[3], 1.0)


def test_split_resolver_uses_precomputed_indices(tmp_path: Path) -> None:
    dataset_root = _create_tiny_npz_dataset(tmp_path / "dataset")
    split_dir = dataset_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    train = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32)
    val = np.asarray([6, 7, 8], dtype=np.int32)
    test = np.asarray([9, 10, 11], dtype=np.int32)

    np.save(split_dir / "train_indices.npy", train)
    np.save(split_dir / "val_indices.npy", val)
    np.save(split_dir / "test_indices.npy", test)

    resolved = resolve_split_indices(
        dataset_root=dataset_root,
        precomputed_split_dir="splits",
        seed=123,
    )

    assert resolved.source == "precomputed"
    assert np.array_equal(resolved.train_indices, train)
    assert np.array_equal(resolved.val_indices, val)
    assert np.array_equal(resolved.test_indices, test)


def test_split_resolver_fallback_is_deterministic(tmp_path: Path) -> None:
    dataset_root = _create_tiny_npz_dataset(tmp_path / "dataset")

    first = resolve_split_indices(dataset_root=dataset_root, seed=77)
    second = resolve_split_indices(dataset_root=dataset_root, seed=77)

    assert first.source == "deterministic_random"
    assert second.source == "deterministic_random"
    assert np.array_equal(first.train_indices, second.train_indices)
    assert np.array_equal(first.val_indices, second.val_indices)
    assert np.array_equal(first.test_indices, second.test_indices)


def test_masked_layers_handle_padding_semantics() -> None:
    from training.models.keras_layers import MaskedAttentionalPooling, MaskedGCNLayer

    x = np.asarray(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [99.0, 99.0, 99.0],
                [77.0, 77.0, 77.0],
            ]
        ],
        dtype=np.float32,
    )
    node_mask = np.asarray([[True, True, False, False]], dtype=bool)
    edge_index = np.asarray([[[0, 1, -1, -1], [1, 0, -1, -1]]], dtype=np.int32)
    edge_mask = np.asarray([[True, True, False, False]], dtype=bool)

    gcn = MaskedGCNLayer(units=5)
    gcn_out = np.asarray(
        gcn(
            {
                "x": x,
                "node_mask": node_mask,
                "edge_index": edge_index,
                "edge_mask": edge_mask,
            }
        )
    )
    assert gcn_out.shape == (1, 4, 5)
    assert np.allclose(gcn_out[0, 2:, :], 0.0, atol=1e-6)

    pool = MaskedAttentionalPooling()
    pooled_a = np.asarray(pool({"x": gcn_out, "node_mask": node_mask}))

    gcn_out_variant = gcn_out.copy()
    gcn_out_variant[0, 2:, :] += 1e6
    pooled_b = np.asarray(pool({"x": gcn_out_variant, "node_mask": node_mask}))

    assert np.allclose(pooled_a, pooled_b, atol=1e-5)


def test_integration_one_epoch_training_and_artifacts(tmp_path: Path) -> None:
    dataset_root = _create_tiny_npz_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        "-m",
        "training.training",
        "--dataset_root",
        str(dataset_root),
        "--output_dir",
        str(output_dir),
        "--model_version",
        "2.0",
        "--epochs",
        "1",
        "--batch_size",
        "4",
        "--precomputed_split_dir",
        "splits",
        "--write_random_splits",
        "--log_level",
        "INFO",
        "--verbose",
        "0",
    ]
    env = os.environ.copy()
    env["KERAS_BACKEND"] = "jax"

    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"

    expected_files = [
        "best_model.keras",
        "last_model.keras",
        "last_model.state.json",
        "history.json",
        "test_results.json",
    ]
    for name in expected_files:
        assert (output_dir / name).exists(), f"Missing expected artifact: {name}"

    results_payload = json.loads((output_dir / "test_results.json").read_text())
    assert "test_metrics" in results_payload
    assert "loss" in results_payload["test_metrics"]
    assert "mrr" in results_payload["test_metrics"]
    assert "top_k" in results_payload["test_metrics"]


def test_integration_resume_from_last_model(tmp_path: Path) -> None:
    dataset_root = _create_tiny_npz_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "out"

    env = os.environ.copy()
    env["KERAS_BACKEND"] = "jax"
    cwd = Path(__file__).resolve().parents[1]

    first_cmd = [
        sys.executable,
        "-m",
        "training.training",
        "--dataset_root",
        str(dataset_root),
        "--output_dir",
        str(output_dir),
        "--epochs",
        "1",
        "--batch_size",
        "4",
        "--write_random_splits",
        "--verbose",
        "0",
    ]
    first = subprocess.run(first_cmd, cwd=cwd, capture_output=True, text=True, env=env)
    assert first.returncode == 0, f"stdout:\n{first.stdout}\n\nstderr:\n{first.stderr}"

    resume_cmd = [
        sys.executable,
        "-m",
        "training.training",
        "--dataset_root",
        str(dataset_root),
        "--output_dir",
        str(output_dir),
        "--resume_from_checkpoint",
        str(output_dir / "last_model.keras"),
        "--epochs",
        "2",
        "--batch_size",
        "4",
        "--verbose",
        "0",
    ]
    resumed = subprocess.run(resume_cmd, cwd=cwd, capture_output=True, text=True, env=env)
    assert resumed.returncode == 0, f"stdout:\n{resumed.stdout}\n\nstderr:\n{resumed.stderr}"

    state_payload = json.loads((output_dir / "last_model.state.json").read_text())
    assert int(state_payload["completed_epoch"]) >= 2

    test_results = json.loads((output_dir / "test_results.json").read_text())
    assert int(test_results["start_epoch"]) >= 2


def test_invalid_manifest_fails_fast(tmp_path: Path) -> None:
    dataset_root = tmp_path / "bad_dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "manifest.json").write_text('{"schema_version": "1.0.0"}', encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "training.training",
        "--dataset_root",
        str(dataset_root),
        "--output_dir",
        str(tmp_path / "out"),
        "--epochs",
        "1",
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode != 0
