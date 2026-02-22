"""Keras/JAX model factory for Graphens training migration."""

from __future__ import annotations

from typing import Any, Dict

import keras

from training.models.keras_layers import MaskedAttentionalPooling, MaskedGCNLayer


SUPPORTED_MODEL_VERSIONS = ("2.0",)


def build_genephenai_v2_0_jax(
    *,
    feature_dim: int,
    max_nodes: int,
    max_edges: int,
    hidden_channels: int,
    num_classes: int,
    dropout_rate: float,
) -> keras.Model:
    """Build the Keras/JAX equivalent of GenePhenAI v2.0."""
    x_in = keras.Input(shape=(max_nodes, feature_dim), dtype="float32", name="x")
    node_mask_in = keras.Input(shape=(max_nodes,), dtype="bool", name="node_mask")
    edge_index_in = keras.Input(shape=(2, max_edges), dtype="int32", name="edge_index")
    edge_mask_in = keras.Input(shape=(max_edges,), dtype="bool", name="edge_mask")

    x = x_in
    for idx in range(4):
        block_id = idx + 1
        x = MaskedGCNLayer(hidden_channels, name=f"gcn_{block_id}")(
            {
                "x": x,
                "node_mask": node_mask_in,
                "edge_index": edge_index_in,
                "edge_mask": edge_mask_in,
            }
        )
        x = keras.layers.LayerNormalization(name=f"norm_{block_id}")(x)
        x = keras.layers.Activation("relu", name=f"relu_{block_id}")(x)
        x = keras.layers.Dropout(dropout_rate, name=f"dropout_{block_id}")(x)

    pooled = MaskedAttentionalPooling(name="attn_pool")(
        {"x": x, "node_mask": node_mask_in}
    )
    pooled = keras.layers.Dropout(dropout_rate, name="dropout_pool")(pooled)
    logits = keras.layers.Dense(num_classes, name="classifier")(pooled)

    model = keras.Model(
        inputs={
            "x": x_in,
            "node_mask": node_mask_in,
            "edge_index": edge_index_in,
            "edge_mask": edge_mask_in,
        },
        outputs=logits,
        name="GenePhenAIv2_0JAX",
    )
    return model


def create_keras_model(
    *,
    model_version: str,
    feature_dim: int,
    max_nodes: int,
    max_edges: int,
    hidden_channels: int,
    num_classes: int,
    dropout_rate: float,
) -> keras.Model:
    if model_version != "2.0":
        raise NotImplementedError(
            f"Model version '{model_version}' is not migrated yet. "
            "Supported in this phase: 2.0"
        )

    return build_genephenai_v2_0_jax(
        feature_dim=feature_dim,
        max_nodes=max_nodes,
        max_edges=max_edges,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )


def get_custom_objects() -> Dict[str, Any]:
    """Custom objects required for deserializing saved Keras checkpoints."""
    return {
        "MaskedGCNLayer": MaskedGCNLayer,
        "MaskedAttentionalPooling": MaskedAttentionalPooling,
    }
