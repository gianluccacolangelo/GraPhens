"""Keras Sequence adapter and split resolution for NPZ graph datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import keras

from src.simulation.phenotype_simulation.jax_dataset_schema import load_and_validate_manifest
from training.datasets.jax_npz_graph_dataset import JAXNPZGraphDataset


def _ensure_split_coverage(
    *,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    num_samples: int,
) -> None:
    for name, indices in {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }.items():
        if indices.ndim != 1:
            raise ValueError(f"{name} indices must be rank-1, got shape {indices.shape}.")
        if indices.size and (np.min(indices) < 0 or np.max(indices) >= num_samples):
            raise ValueError(
                f"{name} indices contain out-of-bounds values for dataset size {num_samples}."
            )

    train_set = set(train_indices.tolist())
    val_set = set(val_indices.tolist())
    test_set = set(test_indices.tolist())

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Train/val/test index sets must be disjoint.")

    all_indices = train_set | val_set | test_set
    if len(all_indices) != num_samples:
        raise ValueError(
            "Split indices must cover all samples exactly once. "
            f"Covered={len(all_indices)}, expected={num_samples}."
        )


@dataclass
class SplitResolution:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    source: str
    split_dir: Path


def resolve_split_indices(
    *,
    dataset_root: str | Path,
    precomputed_split_dir: str = "splits",
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    write_random_splits: bool = False,
) -> SplitResolution:
    """Resolve train/val/test indices with precomputed-first behavior."""
    if not (0.0 <= val_split < 1.0 and 0.0 <= test_split < 1.0):
        raise ValueError("val_split and test_split must be in [0, 1).")
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0.")

    root = Path(dataset_root)
    manifest = load_and_validate_manifest(root / "manifest.json")
    num_samples = int(manifest["num_samples"])
    split_dir = root / precomputed_split_dir

    train_path = split_dir / "train_indices.npy"
    val_path = split_dir / "val_indices.npy"
    test_path = split_dir / "test_indices.npy"

    has_precomputed = train_path.exists() and val_path.exists() and test_path.exists()
    if has_precomputed:
        train_indices = np.load(train_path).astype(np.int32, copy=False)
        val_indices = np.load(val_path).astype(np.int32, copy=False)
        test_indices = np.load(test_path).astype(np.int32, copy=False)

        _ensure_split_coverage(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            num_samples=num_samples,
        )
        return SplitResolution(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            source="precomputed",
            split_dir=split_dir,
        )

    indices = np.arange(num_samples, dtype=np.int32)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_test = int(test_split * num_samples)
    n_val = int(val_split * num_samples)
    n_train = num_samples - n_val - n_test

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    _ensure_split_coverage(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        num_samples=num_samples,
    )

    if write_random_splits:
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(train_path, train_indices)
        np.save(val_path, val_indices)
        np.save(test_path, test_indices)

    return SplitResolution(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        source="deterministic_random",
        split_dir=split_dir,
    )


class KerasNPZSequence(keras.utils.Sequence):
    """Keras-compatible sequence backed by NPZ graph shards."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_remainder: bool = False,
        validate_shards: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_remainder = bool(drop_remainder)

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        if self.indices.ndim != 1:
            raise ValueError(f"indices must be rank-1, got shape {self.indices.shape}.")

        self._dataset = JAXNPZGraphDataset(
            dataset_root=self.dataset_root,
            split="all",
            batch_size=self.batch_size,
            shuffle=False,
            seed=self.seed,
            include_trace_fields=False,
            drop_remainder=False,
            validate_shards=validate_shards,
        )

        num_samples = int(self._dataset.manifest["num_samples"])
        if self.indices.size and (np.min(self.indices) < 0 or np.max(self.indices) >= num_samples):
            raise ValueError(
                "indices contain out-of-bounds values for dataset size "
                f"{num_samples}."
            )

        self._epoch = 0
        self._epoch_indices = self.indices.copy()
        self.on_epoch_end()

    def __len__(self) -> int:
        if self.drop_remainder:
            return self._epoch_indices.shape[0] // self.batch_size
        return int(np.ceil(self._epoch_indices.shape[0] / self.batch_size))

    def __getitem__(self, batch_idx: int):
        if batch_idx < 0 or batch_idx >= len(self):
            raise IndexError(f"batch_idx out of range: {batch_idx}")

        start = batch_idx * self.batch_size
        stop = start + self.batch_size
        selected = self._epoch_indices[start:stop]

        if self.drop_remainder and selected.shape[0] < self.batch_size:
            raise IndexError(f"batch_idx {batch_idx} is dropped due to drop_remainder=True")

        batch = self._dataset.materialize_batch(selected)
        inputs = {
            "x": batch["x"],
            "node_mask": batch["node_mask"],
            "edge_index": batch["edge_index"],
            "edge_mask": batch["edge_mask"],
        }
        return inputs, batch["y"]

    def on_epoch_end(self) -> None:
        self._epoch_indices = self.indices.copy()
        if self.shuffle and self._epoch_indices.size > 0:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(self._epoch_indices)
        self._epoch += 1

    @property
    def num_samples(self) -> int:
        return int(self.indices.shape[0])

    @property
    def manifest(self) -> Dict[str, object]:
        return self._dataset.manifest


def build_sequences(
    *,
    dataset_root: str | Path,
    batch_size: int,
    seed: int,
    precomputed_split_dir: str = "splits",
    val_split: float = 0.15,
    test_split: float = 0.15,
    write_random_splits: bool = False,
    validate_shards: bool = False,
) -> tuple[KerasNPZSequence, KerasNPZSequence, KerasNPZSequence, SplitResolution]:
    """Build train/val/test Keras sequences using precomputed-first split policy."""
    split_resolution = resolve_split_indices(
        dataset_root=dataset_root,
        precomputed_split_dir=precomputed_split_dir,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        write_random_splits=write_random_splits,
    )

    train_seq = KerasNPZSequence(
        dataset_root=dataset_root,
        indices=split_resolution.train_indices,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        drop_remainder=False,
        validate_shards=validate_shards,
    )
    val_seq = KerasNPZSequence(
        dataset_root=dataset_root,
        indices=split_resolution.val_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        drop_remainder=False,
        validate_shards=False,
    )
    test_seq = KerasNPZSequence(
        dataset_root=dataset_root,
        indices=split_resolution.test_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        drop_remainder=False,
        validate_shards=False,
    )
    return train_seq, val_seq, test_seq, split_resolution
