"""NPZ-backed graph dataset loader for Keras 3 + JAX pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Literal, Optional

import numpy as np

from src.simulation.phenotype_simulation.jax_dataset_schema import (
    load_and_validate_manifest,
    validate_npz_shard_against_manifest,
)

SplitName = Literal["train", "val", "test", "all"]


class JAXNPZGraphDataset:
    """Reads NPZ graph shards and yields fixed-shape NumPy batches.

    Batches contain:
    - ``x``: ``[B, Nmax, D]`` float32
    - ``node_mask``: ``[B, Nmax]`` bool
    - ``edge_index``: ``[B, 2, Emax]`` int32
    - ``edge_mask``: ``[B, Emax]`` bool
    - ``y``: ``[B]`` int32

    Trace fields ``gene_idx`` and ``case_idx`` are included when
    ``include_trace_fields=True``.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: Optional[SplitName] = "all",
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int = 42,
        include_trace_fields: bool = False,
        drop_remainder: bool = False,
        validate_shards: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.include_trace_fields = bool(include_trace_fields)
        self.drop_remainder = bool(drop_remainder)
        self._epoch = 0

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")

        self.manifest = load_and_validate_manifest(self.dataset_root / "manifest.json")
        self._shards = list(self.manifest["shards"])
        if not self._shards:
            raise ValueError("Manifest has no shards.")

        self._shard_starts = np.asarray(
            [int(shard["index_start"]) for shard in self._shards], dtype=np.int64
        )
        self._shard_ends = np.asarray(
            [int(shard["index_end"]) for shard in self._shards], dtype=np.int64
        )

        if validate_shards:
            for shard_entry in self._shards:
                validate_npz_shard_against_manifest(
                    shard_path=self.dataset_root / shard_entry["file"],
                    shard_entry=shard_entry,
                    manifest=self.manifest,
                )

        self.indices = self._load_split_indices(split=split)
        self._split = "all" if split is None else split
        self._cached_shard_idx: Optional[int] = None
        self._cached_shard_payload: Optional[Dict[str, np.ndarray]] = None

    @property
    def split(self) -> str:
        return self._split

    @property
    def num_samples(self) -> int:
        return int(self.indices.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.manifest["feature_dim"])

    @property
    def max_nodes(self) -> int:
        return int(self.manifest["max_nodes"])

    @property
    def max_edges(self) -> int:
        return int(self.manifest["max_edges"])

    @property
    def num_classes(self) -> int:
        return len(self.manifest["genes"])

    def __len__(self) -> int:
        return self.num_samples

    def num_batches(self) -> int:
        if self.drop_remainder:
            return self.num_samples // self.batch_size
        return int(np.ceil(self.num_samples / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        """Set epoch used to derive deterministic shuffle order."""
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        return self.iter_batches()

    def iter_batches(self) -> Iterator[Dict[str, np.ndarray]]:
        batch_indices = self.indices.copy()
        if self.shuffle and batch_indices.size > 0:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(batch_indices)

        if self.drop_remainder:
            usable = (batch_indices.shape[0] // self.batch_size) * self.batch_size
            batch_indices = batch_indices[:usable]

        for start in range(0, batch_indices.shape[0], self.batch_size):
            chunk = batch_indices[start : start + self.batch_size]
            if chunk.shape[0] == 0:
                continue
            yield self._materialize_batch(chunk)

        if self.shuffle:
            self._epoch += 1

    def _load_split_indices(self, split: Optional[SplitName]) -> np.ndarray:
        if split is None or split == "all":
            return np.arange(int(self.manifest["num_samples"]), dtype=np.int32)

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Use train/val/test/all.")

        split_path = self.dataset_root / "splits" / f"{split}_indices.npy"
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}. Generate splits before using split='{split}'."
            )

        indices = np.load(split_path).astype(np.int32, copy=False)
        if indices.ndim != 1:
            raise ValueError(f"Split indices must be rank-1, got shape {indices.shape}.")
        if indices.size and (np.min(indices) < 0 or np.max(indices) >= int(self.manifest["num_samples"])):
            raise ValueError(
                f"Split '{split}' contains out-of-bounds sample indices for dataset size "
                f"{self.manifest['num_samples']}."
            )
        return indices

    def _materialize_batch(self, global_indices: np.ndarray) -> Dict[str, np.ndarray]:
        x_rows = []
        node_mask_rows = []
        edge_index_rows = []
        edge_mask_rows = []
        y_rows = []
        gene_idx_rows = []
        case_idx_rows = []

        for idx in global_indices:
            payload, local_idx = self._lookup_sample(int(idx))
            x_rows.append(payload["x"][local_idx])
            node_mask_rows.append(payload["node_mask"][local_idx])
            edge_index_rows.append(payload["edge_index"][local_idx])
            edge_mask_rows.append(payload["edge_mask"][local_idx])
            y_rows.append(payload["y"][local_idx])
            gene_idx_rows.append(payload["gene_idx"][local_idx])
            case_idx_rows.append(payload["case_idx"][local_idx])

        batch = {
            "x": np.stack(x_rows, axis=0).astype(np.float32, copy=False),
            "node_mask": np.stack(node_mask_rows, axis=0).astype(np.bool_, copy=False),
            "edge_index": np.stack(edge_index_rows, axis=0).astype(np.int32, copy=False),
            "edge_mask": np.stack(edge_mask_rows, axis=0).astype(np.bool_, copy=False),
            "y": np.asarray(y_rows, dtype=np.int32),
        }
        if self.include_trace_fields:
            batch["gene_idx"] = np.asarray(gene_idx_rows, dtype=np.int32)
            batch["case_idx"] = np.asarray(case_idx_rows, dtype=np.int32)
        return batch

    def _lookup_sample(self, global_idx: int) -> tuple[Dict[str, np.ndarray], int]:
        shard_pos = int(np.searchsorted(self._shard_ends, global_idx, side="left"))
        if shard_pos >= self._shard_ends.shape[0]:
            raise IndexError(f"Global index {global_idx} is out of shard bounds.")

        shard_payload = self._load_shard(shard_pos)
        local_idx = global_idx - int(self._shard_starts[shard_pos])
        if local_idx < 0 or local_idx >= int(self._shards[shard_pos]["num_samples"]):
            raise IndexError(
                f"Resolved local index {local_idx} is invalid for shard {self._shards[shard_pos]['file']}."
            )
        return shard_payload, local_idx

    def _load_shard(self, shard_pos: int) -> Dict[str, np.ndarray]:
        if self._cached_shard_idx == shard_pos and self._cached_shard_payload is not None:
            return self._cached_shard_payload

        shard_entry = self._shards[shard_pos]
        shard_path = self.dataset_root / shard_entry["file"]
        with np.load(shard_path) as data:
            payload = {
                "x": np.asarray(data["x"], dtype=np.float32),
                "node_mask": np.asarray(data["node_mask"], dtype=np.bool_),
                "edge_index": np.asarray(data["edge_index"], dtype=np.int32),
                "edge_mask": np.asarray(data["edge_mask"], dtype=np.bool_),
                "y": np.asarray(data["y"], dtype=np.int32),
                "gene_idx": np.asarray(data["gene_idx"], dtype=np.int32),
                "case_idx": np.asarray(data["case_idx"], dtype=np.int32),
            }

        self._cached_shard_idx = shard_pos
        self._cached_shard_payload = payload
        return payload
