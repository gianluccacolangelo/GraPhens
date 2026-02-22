#!/usr/bin/env python3
"""Validate local Keras 3 + JAX runtime configuration."""

from __future__ import annotations

import os


def main() -> int:
    print(f"KERAS_BACKEND env: {os.environ.get('KERAS_BACKEND', '<unset>')}")

    try:
        import keras
    except Exception as exc:  # pragma: no cover - runtime probe
        print(f"Failed to import keras: {exc}")
        return 1

    try:
        import jax
        import jaxlib
    except Exception as exc:  # pragma: no cover - runtime probe
        print(f"Failed to import jax/jaxlib: {exc}")
        return 1

    backend_name = keras.backend.backend()
    print(f"Keras version: {keras.__version__}")
    print(f"JAX version: {jax.__version__}")
    print(f"JAXLIB version: {jaxlib.__version__}")
    print(f"Keras backend: {backend_name}")

    if backend_name != "jax":
        print("ERROR: Keras is not using the JAX backend. Set KERAS_BACKEND=jax.")
        return 2

    devices = jax.devices()
    print(f"Detected devices ({len(devices)}):")
    for idx, device in enumerate(devices):
        print(f"  [{idx}] {device}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
