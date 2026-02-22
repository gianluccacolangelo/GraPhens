"""Custom Keras layers for padded graph training with JAX backend."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="graphens")
class MaskedGCNLayer(keras.layers.Layer):
    """GCN-like message passing over padded edge-index graphs.

    Expected inputs:
    - x: [B, N, Din]
    - node_mask: [B, N] bool
    - edge_index: [B, 2, E] int32 (padded with -1)
    - edge_mask: [B, E] bool
    """

    def __init__(self, units: int, use_bias: bool = True, epsilon: float = 1e-6, **kwargs: Any):
        super().__init__(**kwargs)
        self.units = int(units)
        self.use_bias = bool(use_bias)
        self.epsilon = float(epsilon)

    def build(self, input_shape: Any) -> None:
        x_shape = self._extract_shape(input_shape, key="x", position=0)
        in_dim = int(x_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(in_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs: Dict[str, Any] | Tuple[Any, ...], training: bool = False):
        del training
        x, node_mask, edge_index, edge_mask = self._extract_inputs(inputs)

        if x.shape[1] is None:
            raise ValueError("MaskedGCNLayer requires statically known max_nodes dimension.")
        num_nodes = int(x.shape[1])

        node_mask_f = ops.cast(node_mask, x.dtype)
        edge_mask_f = ops.cast(edge_mask, x.dtype)

        src = edge_index[:, 0, :]
        dst = edge_index[:, 1, :]
        src_safe = ops.clip(src, 0, num_nodes - 1)
        dst_safe = ops.clip(dst, 0, num_nodes - 1)

        src_oh = ops.one_hot(src_safe, num_nodes)
        dst_oh = ops.one_hot(dst_safe, num_nodes)
        edge_w = ops.expand_dims(edge_mask_f, axis=-1)
        src_oh = ops.cast(src_oh, x.dtype) * edge_w
        dst_oh = ops.cast(dst_oh, x.dtype) * edge_w

        # A[target, source] for message aggregation at target.
        adjacency = ops.einsum("ben,bem->bnm", dst_oh, src_oh)

        # Add self-loops for valid nodes only.
        eye = ops.eye(num_nodes, dtype=x.dtype)
        adjacency = adjacency + ops.expand_dims(node_mask_f, axis=-1) * ops.expand_dims(eye, axis=0)

        degree = ops.sum(adjacency, axis=-1)
        inv_sqrt_degree = ops.rsqrt(ops.maximum(degree, self.epsilon))
        norm_adj = (
            adjacency
            * ops.expand_dims(inv_sqrt_degree, axis=-1)
            * ops.expand_dims(inv_sqrt_degree, axis=-2)
        )

        support = ops.matmul(x, self.kernel)
        out = ops.matmul(norm_adj, support)

        if self.bias is not None:
            out = out + self.bias

        out = out * ops.expand_dims(node_mask_f, axis=-1)
        return out

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "use_bias": self.use_bias,
                "epsilon": self.epsilon,
            }
        )
        return config

    @staticmethod
    def _extract_shape(input_shape: Any, *, key: str, position: int):
        if isinstance(input_shape, dict):
            return input_shape[key]
        if isinstance(input_shape, (list, tuple)):
            return input_shape[position]
        raise TypeError("Unsupported input shape type for MaskedGCNLayer")

    @staticmethod
    def _extract_inputs(inputs: Dict[str, Any] | Tuple[Any, ...]):
        if isinstance(inputs, dict):
            return (
                inputs["x"],
                inputs["node_mask"],
                inputs["edge_index"],
                inputs["edge_mask"],
            )
        if isinstance(inputs, (list, tuple)) and len(inputs) == 4:
            return inputs
        raise TypeError(
            "MaskedGCNLayer expects a dict with x/node_mask/edge_index/edge_mask "
            "or a 4-item tuple/list."
        )


@keras.saving.register_keras_serializable(package="graphens")
class MaskedAttentionalPooling(keras.layers.Layer):
    """Attention pooling that ignores padded nodes via node_mask."""

    def __init__(self, gate_hidden_units: int | None = None, epsilon: float = 1e-8, **kwargs: Any):
        super().__init__(**kwargs)
        self.gate_hidden_units = gate_hidden_units
        self.epsilon = float(epsilon)

    def build(self, input_shape: Any) -> None:
        x_shape = self._extract_shape(input_shape, key="x", position=0)
        hidden_dim = int(x_shape[-1])
        gate_hidden = self.gate_hidden_units or max(1, hidden_dim // 2)

        self.gate_kernel_1 = self.add_weight(
            name="gate_kernel_1",
            shape=(hidden_dim, gate_hidden),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.gate_bias_1 = self.add_weight(
            name="gate_bias_1",
            shape=(gate_hidden,),
            initializer="zeros",
            trainable=True,
        )
        self.gate_kernel_2 = self.add_weight(
            name="gate_kernel_2",
            shape=(gate_hidden, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.gate_bias_2 = self.add_weight(
            name="gate_bias_2",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: Dict[str, Any] | Tuple[Any, ...], training: bool = False):
        del training
        x, node_mask = self._extract_inputs(inputs)

        node_mask_bool = ops.cast(node_mask, "bool")
        node_mask_f = ops.cast(node_mask_bool, x.dtype)

        hidden = ops.matmul(x, self.gate_kernel_1) + self.gate_bias_1
        hidden = ops.relu(hidden)
        logits = ops.squeeze(ops.matmul(hidden, self.gate_kernel_2) + self.gate_bias_2, axis=-1)

        very_negative = ops.cast(-1e9, logits.dtype)
        masked_logits = ops.where(node_mask_bool, logits, very_negative)
        weights = ops.softmax(masked_logits, axis=-1)
        weights = weights * node_mask_f

        denom = ops.sum(weights, axis=-1, keepdims=True) + ops.cast(self.epsilon, weights.dtype)
        weights = weights / denom

        pooled = ops.sum(x * ops.expand_dims(weights, axis=-1), axis=1)
        return pooled

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "gate_hidden_units": self.gate_hidden_units,
                "epsilon": self.epsilon,
            }
        )
        return config

    @staticmethod
    def _extract_shape(input_shape: Any, *, key: str, position: int):
        if isinstance(input_shape, dict):
            return input_shape[key]
        if isinstance(input_shape, (list, tuple)):
            return input_shape[position]
        raise TypeError("Unsupported input shape type for MaskedAttentionalPooling")

    @staticmethod
    def _extract_inputs(inputs: Dict[str, Any] | Tuple[Any, ...]):
        if isinstance(inputs, dict):
            return inputs["x"], inputs["node_mask"]
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return inputs
        raise TypeError(
            "MaskedAttentionalPooling expects a dict with x/node_mask or a 2-item tuple/list."
        )
