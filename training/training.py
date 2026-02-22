"""Keras + JAX training entrypoint for NPZ graph datasets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Dict, Iterable

import keras
import numpy as np

from src.simulation.phenotype_simulation.jax_dataset_schema import load_and_validate_manifest
from training.datasets.keras_npz_sequence import build_sequences
from training.migration_report import build_migration_report, write_migration_report

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None


def parse_top_k(top_k: str) -> tuple[int, ...]:
    values = []
    for token in top_k.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid top-k value '{token}'.") from exc
    if not values:
        raise ValueError("At least one top-k value must be provided.")
    if any(k <= 0 for k in values):
        raise ValueError("All top-k values must be > 0.")
    return tuple(sorted(set(values)))


def compute_ranking_metrics(
    logits: np.ndarray,
    target_y: np.ndarray,
    *,
    k_vals: Iterable[int],
) -> tuple[float, Dict[int, float]]:
    """Compute MRR and top-k accuracy from logits and integer class labels."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank-2, got shape {logits.shape}.")
    if target_y.ndim != 1:
        raise ValueError(f"target_y must be rank-1, got shape {target_y.shape}.")
    if logits.shape[0] != target_y.shape[0]:
        raise ValueError(
            f"Batch dimension mismatch: logits {logits.shape[0]} vs labels {target_y.shape[0]}."
        )

    if logits.shape[0] == 0:
        return 0.0, {k: 0.0 for k in k_vals}

    sorted_indices = np.argsort(-logits, axis=1)
    target_expanded = target_y[:, None]
    matches = sorted_indices == target_expanded

    # Every row should contain exactly one match when target labels are in bounds.
    if not np.all(np.any(matches, axis=1)):
        missing = np.where(~np.any(matches, axis=1))[0]
        raise ValueError(f"Missing target labels in ranking outputs for rows: {missing.tolist()}")

    ranks = np.argmax(matches, axis=1) + 1
    mrr = float(np.mean(1.0 / ranks))
    top_k = {int(k): float(np.mean(ranks <= int(k))) for k in k_vals}
    return mrr, top_k


def evaluate_sequence(
    *,
    model,
    sequence,
    loss_fn,
    k_vals: tuple[int, ...],
) -> Dict[str, object]:
    """Evaluate model on a Keras sequence and return loss + ranking metrics."""
    total_loss = 0.0
    total_samples = 0
    sum_reciprocal_ranks = 0.0
    top_k_correct = {k: 0.0 for k in k_vals}

    for batch_idx in range(len(sequence)):
        inputs, targets = sequence[batch_idx]
        logits = np.asarray(model(inputs, training=False))
        targets = np.asarray(targets, dtype=np.int32)

        per_sample_loss = np.asarray(loss_fn(targets, logits), dtype=np.float64)
        if per_sample_loss.ndim == 0:
            per_sample_loss = np.repeat(per_sample_loss, targets.shape[0])

        batch_size = int(targets.shape[0])
        batch_mrr, batch_top_k = compute_ranking_metrics(logits, targets, k_vals=k_vals)

        total_loss += float(np.sum(per_sample_loss))
        total_samples += batch_size
        sum_reciprocal_ranks += batch_mrr * batch_size
        for k in k_vals:
            top_k_correct[k] += batch_top_k[k] * batch_size

    if total_samples == 0:
        return {
            "loss": 0.0,
            "mrr": 0.0,
            "top_k": {k: 0.0 for k in k_vals},
            "num_samples": 0,
        }

    return {
        "loss": float(total_loss / total_samples),
        "mrr": float(sum_reciprocal_ranks / total_samples),
        "top_k": {k: float(top_k_correct[k] / total_samples) for k in k_vals},
        "num_samples": int(total_samples),
    }


def _load_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def _checkpoint_state_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".state.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GenePhenAI with Keras 3 + JAX from NPZ graph shards")

    # Data / split args
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to NPZ dataset root (manifest.json + shards/)")
    parser.add_argument("--output_dir", type=str, default="training_output", help="Directory for checkpoints/logs/results")
    parser.add_argument("--precomputed_split_dir", type=str, default="splits", help="Split directory under dataset_root (expects .npy files)")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation ratio for deterministic fallback split")
    parser.add_argument("--test_split", type=float, default=0.15, help="Test ratio for deterministic fallback split")
    parser.add_argument("--write_random_splits", action="store_true", help="Persist fallback deterministic split indices as .npy")
    parser.add_argument("--validate_shards", action="store_true", help="Validate all shards at training startup")

    # Model args
    parser.add_argument("--model_version", type=str, default="2.0", choices=["2.0"], help="Model version")
    parser.add_argument("--hidden_channels", type=int, default=128, help="Hidden dimension for GCN blocks")
    parser.add_argument("--dropout_rate", type=float, default=0.25, help="Dropout probability")

    # Training args
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate (defaults to 1e-3; optional override when resuming)")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="AdamW weight decay")
    parser.add_argument("--eval_epoch_interval", type=int, default=1, help="Compute ranking metrics every N epochs")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to .keras checkpoint to resume from")
    parser.add_argument("--jit_compile", action="store_true", help="Enable Keras jit_compile for model.compile")

    # Misc args
    parser.add_argument("--top_k", type=str, default="1,5,10,20", help="Comma-separated top-k values")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2], help="Keras fit verbosity")

    # W&B args
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="genphenia", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="gcolangelo", help="W&B entity")

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    top_k_vals = parse_top_k(args.top_k)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    args_payload = vars(args).copy()
    args_payload["top_k"] = list(top_k_vals)
    _save_json(output_dir / "args.json", args_payload)

    backend = keras.backend.backend()
    if backend != "jax":
        logger.error(
            "Keras backend must be 'jax' for this trainer, found '%s'. Set KERAS_BACKEND=jax.",
            backend,
        )
        return 2

    from training.models.keras_models import create_keras_model, get_custom_objects

    keras.utils.set_random_seed(args.seed)

    if args.use_wandb:
        if wandb is None:
            logger.warning("W&B requested but package is unavailable; disabling W&B logging.")
            args.use_wandb = False
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=args_payload,
                name=f"keras_jax_v{args.model_version}_{int(datetime.now(timezone.utc).timestamp())}",
                reinit=True,
            )

    manifest = load_and_validate_manifest(Path(args.dataset_root) / "manifest.json")
    feature_dim = int(manifest["feature_dim"])
    max_nodes = int(manifest["max_nodes"])
    max_edges = int(manifest["max_edges"])
    num_classes = len(manifest["genes"])

    train_seq, val_seq, test_seq, split_resolution = build_sequences(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        seed=args.seed,
        precomputed_split_dir=args.precomputed_split_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        write_random_splits=args.write_random_splits,
        validate_shards=args.validate_shards,
    )

    logger.info(
        "Dataset loaded from %s | samples train=%d val=%d test=%d | split_source=%s",
        args.dataset_root,
        train_seq.num_samples,
        val_seq.num_samples,
        test_seq.num_samples,
        split_resolution.source,
    )

    best_model_path = output_dir / "best_model.keras"
    last_model_path = output_dir / "last_model.keras"
    last_state_path = _checkpoint_state_path(last_model_path)
    history_path = output_dir / "history.json"

    previous_history = _load_json_if_exists(history_path)
    previous_metrics = previous_history.get("metrics", {}) if isinstance(previous_history, dict) else {}
    for key, values in list(previous_metrics.items()):
        if not isinstance(values, list):
            previous_metrics[key] = []

    start_epoch = 1
    custom_objects = get_custom_objects()

    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.resume_from_checkpoint)
        if not checkpoint_path.exists():
            logger.error("Resume checkpoint not found: %s", checkpoint_path)
            return 2

        model = keras.models.load_model(
            checkpoint_path,
            custom_objects=custom_objects,
            compile=True,
        )

        resume_state_path = _checkpoint_state_path(checkpoint_path)
        resume_state = _load_json_if_exists(resume_state_path)
        if resume_state:
            completed_epoch = int(resume_state.get("completed_epoch", 0))
            start_epoch = completed_epoch + 1

        if args.learning_rate is not None and getattr(model, "optimizer", None) is not None:
            try:
                model.optimizer.learning_rate.assign(args.learning_rate)
            except Exception:
                model.optimizer.learning_rate = args.learning_rate

        if getattr(model, "optimizer", None) is None:
            optimizer_lr = args.learning_rate if args.learning_rate is not None else 1e-3
            optimizer = keras.optimizers.AdamW(
                learning_rate=optimizer_lr,
                weight_decay=args.weight_decay,
            )
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            model.compile(optimizer=optimizer, loss=loss, jit_compile=args.jit_compile)

        logger.info("Resumed model from %s at start_epoch=%d", checkpoint_path, start_epoch)
    else:
        model = create_keras_model(
            model_version=args.model_version,
            feature_dim=feature_dim,
            max_nodes=max_nodes,
            max_edges=max_edges,
            hidden_channels=args.hidden_channels,
            num_classes=num_classes,
            dropout_rate=args.dropout_rate,
        )
        optimizer_lr = args.learning_rate if args.learning_rate is not None else 1e-3
        optimizer = keras.optimizers.AdamW(
            learning_rate=optimizer_lr,
            weight_decay=args.weight_decay,
        )
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss, jit_compile=args.jit_compile)

    eval_loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none",
    )

    class LastModelCheckpoint(keras.callbacks.Callback):
        def __init__(self, *, model_path: Path, state_path: Path) -> None:
            super().__init__()
            self.model_path = model_path
            self.state_path = state_path

        def on_epoch_end(self, epoch, logs=None):
            self.model.save(self.model_path)
            state_payload = {
                "completed_epoch": int(epoch) + 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "logs": {k: float(v) for k, v in (logs or {}).items() if v is not None},
            }
            _save_json(self.state_path, state_payload)

    class RankingEvalCallback(keras.callbacks.Callback):
        def __init__(
            self,
            *,
            val_sequence,
            loss_fn,
            k_vals: tuple[int, ...],
            eval_interval: int,
        ) -> None:
            super().__init__()
            self.val_sequence = val_sequence
            self.loss_fn = loss_fn
            self.k_vals = k_vals
            self.eval_interval = max(1, int(eval_interval))
            self.latest_metrics: Dict[str, object] = {}

        def on_epoch_end(self, epoch, logs=None):
            epoch_num = int(epoch) + 1
            if epoch_num % self.eval_interval != 0:
                return

            metrics = evaluate_sequence(
                model=self.model,
                sequence=self.val_sequence,
                loss_fn=self.loss_fn,
                k_vals=self.k_vals,
            )
            self.latest_metrics = metrics

            if logs is not None:
                logs["val_mrr"] = metrics["mrr"]
                for k, acc in metrics["top_k"].items():
                    logs[f"val_top{k}_acc"] = acc

            top_k_log = ", ".join([f"Top-{k}: {acc:.4f}" for k, acc in metrics["top_k"].items()])
            logger.info(
                "Epoch %d | Validation ranking metrics | MRR: %.4f | %s",
                epoch_num,
                metrics["mrr"],
                top_k_log,
            )

            if args.use_wandb and wandb is not None:
                payload = {
                    "epoch": epoch_num,
                    "val_mrr": metrics["mrr"],
                    "val_loss_eval": metrics["loss"],
                }
                for k, acc in metrics["top_k"].items():
                    payload[f"val_top{k}_acc"] = acc
                wandb.log(payload)

    ranking_callback = RankingEvalCallback(
        val_sequence=val_seq,
        loss_fn=eval_loss_fn,
        k_vals=top_k_vals,
        eval_interval=args.eval_epoch_interval,
    )

    callbacks = [
        ranking_callback,
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        LastModelCheckpoint(model_path=last_model_path, state_path=last_state_path),
        keras.callbacks.TerminateOnNaN(),
    ]

    if start_epoch <= args.epochs:
        logger.info(
            "Starting fit: start_epoch=%d, epochs=%d, batch_size=%d",
            start_epoch,
            args.epochs,
            args.batch_size,
        )
        fit_history = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=args.epochs,
            initial_epoch=start_epoch - 1,
            callbacks=callbacks,
            verbose=args.verbose,
        )
        new_history = fit_history.history
    else:
        logger.warning(
            "start_epoch (%d) is greater than epochs (%d); skipping training loop.",
            start_epoch,
            args.epochs,
        )
        new_history = {}

    merged_metrics = {k: list(v) for k, v in previous_metrics.items()}
    for key, values in new_history.items():
        merged_metrics.setdefault(key, [])
        merged_metrics[key].extend([float(v) for v in values])

    history_payload = {
        "schema_version": "1.0.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "start_epoch": start_epoch,
        "requested_epochs": args.epochs,
        "metrics": merged_metrics,
    }
    _save_json(history_path, history_payload)

    if not last_model_path.exists():
        model.save(last_model_path)
        _save_json(
            last_state_path,
            {
                "completed_epoch": max(start_epoch - 1, 0),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "logs": {},
            },
        )

    checkpoint_for_test = best_model_path if best_model_path.exists() else last_model_path
    test_model = keras.models.load_model(
        checkpoint_for_test,
        custom_objects=custom_objects,
        compile=True,
    )

    val_metrics = evaluate_sequence(
        model=test_model,
        sequence=val_seq,
        loss_fn=eval_loss_fn,
        k_vals=top_k_vals,
    )
    test_metrics = evaluate_sequence(
        model=test_model,
        sequence=test_seq,
        loss_fn=eval_loss_fn,
        k_vals=top_k_vals,
    )

    test_results = {
        "schema_version": "1.0.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": args.model_version,
        "checkpoint_used_for_test": str(checkpoint_for_test),
        "start_epoch": start_epoch,
        "requested_epochs": args.epochs,
        "split_source": split_resolution.source,
        "num_samples": {
            "train": train_seq.num_samples,
            "val": val_seq.num_samples,
            "test": test_seq.num_samples,
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    _save_json(output_dir / "test_results.json", test_results)

    migration_report = build_migration_report(
        args=args_payload,
        split_source=split_resolution.source,
    )
    report_path = write_migration_report(output_dir=output_dir, report=migration_report)

    logger.info("Saved history to %s", history_path)
    logger.info("Saved test results to %s", output_dir / "test_results.json")
    logger.info("Saved migration report to %s", report_path)
    logger.info(
        "Final Test Metrics | Loss: %.4f | MRR: %.4f | %s",
        test_metrics["loss"],
        test_metrics["mrr"],
        ", ".join([f"Top-{k}: {v:.4f}" for k, v in test_metrics["top_k"].items()]),
    )

    if args.use_wandb and wandb is not None:
        wandb_payload = {
            "test_loss": test_metrics["loss"],
            "test_mrr": test_metrics["mrr"],
            "val_loss_eval": val_metrics["loss"],
            "val_mrr_eval": val_metrics["mrr"],
        }
        for k, acc in test_metrics["top_k"].items():
            wandb_payload[f"test_top{k}_acc"] = acc
        for k, acc in val_metrics["top_k"].items():
            wandb_payload[f"val_top{k}_acc_eval"] = acc
        wandb.log(wandb_payload)
        wandb.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
