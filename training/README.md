# Training (Keras + JAX)

`training/training.py` is the training entrypoint and expects NPZ graph datasets (`manifest.json` + `shards/*.npz`).

PyTorch/PyG training is not supported in this repository.

## Dataset Contract

Expected dataset root layout:

```text
<dataset_root>/
  manifest.json
  shards/
    shard_00000.npz
    ...
  splits/                 # optional
    train_indices.npy
    val_indices.npy
    test_indices.npy
```

Split policy in the new trainer:

- Precomputed-first: uses `.npy` split files when present.
- Deterministic fallback: creates split indices in memory using `--seed`.
- Optional persistence of fallback splits: `--write_random_splits`.

## Usage

```bash
KERAS_BACKEND=jax python -m training.training \
  --dataset_root data/simulation/output_jax_npz \
  --output_dir training_output_jax \
  --model_version 2.0 \
  --epochs 20 \
  --batch_size 64 \
  --jit_compile
```

Resume training:

```bash
KERAS_BACKEND=jax python -m training.training \
  --dataset_root data/simulation/output_jax_npz \
  --output_dir training_output_jax \
  --resume_from_checkpoint training_output_jax/last_model.keras \
  --epochs 30
```

## Output Artifacts

- `best_model.keras`
- `last_model.keras`
- `last_model.state.json`
- `history.json`
- `test_results.json`
- `training.log`

## Model Support

- `model_version=2.0`
- Other model versions are not available in this trainer.

See `docs/training_keras_jax.md` for implementation details.
