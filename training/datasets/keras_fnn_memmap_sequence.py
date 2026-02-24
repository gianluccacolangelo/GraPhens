"""Keras Sequence adapter for FNN memmap datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import keras
import numpy as np


@dataclass
class FNNSplitResolution:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    source: str
    split_dir: Path


def _load_metadata(dataset_root: str | Path) -> Dict[str, object]:
    root = Path(dataset_root)
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must be a JSON object.")
    return metadata


def _resolve_data_path(dataset_root: Path, raw_path: str | None, default_name: str) -> Path:
    if raw_path:
        p = Path(raw_path)
        if p.exists():
            return p
        rel = dataset_root / raw_path
        if rel.exists():
            return rel
    return dataset_root / default_name


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


def resolve_fnn_split_indices(
    *,
    dataset_root: str | Path,
    precomputed_split_dir: str = "splits",
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    write_random_splits: bool = False,
) -> FNNSplitResolution:
    if not (0.0 <= val_split < 1.0 and 0.0 <= test_split < 1.0):
        raise ValueError("val_split and test_split must be in [0, 1).")
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0.")

    root = Path(dataset_root)
    metadata = _load_metadata(root)
    num_samples = int(metadata["num_samples"])
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
        return FNNSplitResolution(
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

    return FNNSplitResolution(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        source="deterministic_random",
        split_dir=split_dir,
    )


class KerasFNNMemmapSequence(keras.utils.Sequence):
    """Keras-compatible sequence backed by FNN memmap files."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_remainder: bool = False,
        workers: int = 1,
        use_multiprocessing: bool = False,
        max_queue_size: int = 10,
    ) -> None:
        if workers <= 0:
            raise ValueError(f"workers must be > 0, got {workers}.")
        if max_queue_size <= 0:
            raise ValueError(f"max_queue_size must be > 0, got {max_queue_size}.")
        super().__init__(
            workers=int(workers),
            use_multiprocessing=bool(use_multiprocessing),
            max_queue_size=int(max_queue_size),
        )

        self.dataset_root = Path(dataset_root)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_remainder = bool(drop_remainder)
        self.workers = int(workers)
        self.use_multiprocessing = bool(use_multiprocessing)
        self.max_queue_size = int(max_queue_size)

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        if self.indices.ndim != 1:
            raise ValueError(f"indices must be rank-1, got shape {self.indices.shape}.")

        self.metadata = _load_metadata(self.dataset_root)
        self.num_samples_total = int(self.metadata["num_samples"])
        self.feature_dim = int(self.metadata["feature_dim"])
        self.num_classes = int(self.metadata.get("num_genes", 0))
        if self.num_classes <= 0:
            raise ValueError("metadata.json must contain a positive 'num_genes'.")

        x_path = _resolve_data_path(
            self.dataset_root,
            self.metadata.get("x_memmap_path"),
            "X.float32.memmap",
        )
        y_path = _resolve_data_path(
            self.dataset_root,
            self.metadata.get("y_memmap_path"),
            "y.int32.memmap",
        )
        if not x_path.exists():
            raise FileNotFoundError(f"Missing X memmap file: {x_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"Missing y memmap file: {y_path}")

        self._x = np.memmap(
            x_path,
            mode="r",
            dtype=np.float32,
            shape=(self.num_samples_total, self.feature_dim),
        )
        self._y = np.memmap(
            y_path,
            mode="r",
            dtype=np.int32,
            shape=(self.num_samples_total,),
        )

        if self.indices.size and (
            np.min(self.indices) < 0 or np.max(self.indices) >= self.num_samples_total
        ):
            raise ValueError(
                "indices contain out-of-bounds values for dataset size "
                f"{self.num_samples_total}."
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

        x_batch = np.asarray(self._x[selected], dtype=np.float32)
        y_batch = np.asarray(self._y[selected], dtype=np.int32)
        return x_batch, y_batch

    def on_epoch_end(self) -> None:
        self._epoch_indices = self.indices.copy()
        if self.shuffle and self._epoch_indices.size > 0:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(self._epoch_indices)
        self._epoch += 1

    @property
    def num_samples(self) -> int:
        return int(self.indices.shape[0])


def build_fnn_sequences(
    *,
    dataset_root: str | Path,
    batch_size: int,
    seed: int,
    precomputed_split_dir: str = "splits",
    val_split: float = 0.15,
    test_split: float = 0.15,
    write_random_splits: bool = False,
    train_workers: int = 1,
    eval_workers: int = 1,
    use_multiprocessing: bool = False,
    max_queue_size: int = 10,
) -> tuple[KerasFNNMemmapSequence, KerasFNNMemmapSequence, KerasFNNMemmapSequence, FNNSplitResolution]:
    split_resolution = resolve_fnn_split_indices(
        dataset_root=dataset_root,
        precomputed_split_dir=precomputed_split_dir,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        write_random_splits=write_random_splits,
    )

    train_seq = KerasFNNMemmapSequence(
        dataset_root=dataset_root,
        indices=split_resolution.train_indices,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        drop_remainder=False,
        workers=train_workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size,
    )
    val_seq = KerasFNNMemmapSequence(
        dataset_root=dataset_root,
        indices=split_resolution.val_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        drop_remainder=False,
        workers=eval_workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size,
    )
    test_seq = KerasFNNMemmapSequence(
        dataset_root=dataset_root,
        indices=split_resolution.test_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        drop_remainder=False,
        workers=eval_workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size,
    )
    return train_seq, val_seq, test_seq, split_resolution
