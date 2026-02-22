"""Utilities to emit Keras/JAX migration delta reports."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict

REPORT_SCHEMA_VERSION = "1.0.0"


def build_migration_report(
    *,
    args: Dict[str, Any],
    split_source: str,
) -> Dict[str, Any]:
    """Build a structured migration delta report for each training run."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entrypoint": "training/training.py",
        "phase": "keras_jax_training",
        "summary": {
            "training_stack": "Keras 3 + JAX + NPZ shards",
            "split_policy": "precomputed-first with deterministic random fallback",
            "split_source_for_run": split_source,
        },
        "artifacts": {
            "dataset": "NPZ shards + manifest.json",
            "splits": ["train_indices.npy", "val_indices.npy", "test_indices.npy"],
            "checkpoints": ["last_model.keras", "best_model.keras"],
            "run_artifacts": [
                "history.json",
                "test_results.json",
                "migration_report.json",
            ],
        },
        "model_support": {
            "supported": ["2.0"],
            "removed_cli_args": ["num_workers", "device"],
        },
        "run_config": args,
    }


def write_migration_report(*, output_dir: str | Path, report: Dict[str, Any]) -> Path:
    """Write migration report to output directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "migration_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
    return report_path
