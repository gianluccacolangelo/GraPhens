import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, AttentionalAggregation
from torch_geometric.nn import BatchNorm
from torch.nn import Linear, ReLU, Sequential, Dropout
from typing import Dict, Any, Optional, Tuple
from torch_geometric.utils import to_dense_batch


class GenePhenAIv2_0(nn.Module):
    """
    Graph Neural Network for phenotype-based gene classification.

    Consists of 3 GCN message passing layers followed by BatchNorm, ReLU, Dropout,
    an attention-based pooling layer, and a final linear classifier.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout_prob: float = 0.25):
        """
        Args:
            in_channels: Dimensionality of input node features.
            hidden_channels: Dimensionality of hidden layers.
            out_channels: Number of output classes (genes).
            dropout_prob: Probability for dropout layers.
        """
        super().__init__()
        torch.manual_seed(42) # For reproducibility
        self.dropout_prob = dropout_prob

        # --- GCNConv Explanation ---
        # GCNConv implements the graph convolutional layer from Kipf & Welling (ICLR 2017).
        # Formula: X' = D_hat^(-1/2) A_hat D_hat^(-1/2) X Theta
        # - X: Input node features [num_nodes, in_channels]
        # - A_hat: Adjacency matrix with added self-loops (A + I)
        # - D_hat: Diagonal degree matrix of A_hat
        # - Theta: Learnable weight matrix [in_channels, out_channels]
        # - D_hat^(-1/2) A_hat D_hat^(-1/2): Symmetrically normalized adjacency matrix
        # The `edge_index` provides A. PyG handles A_hat and D_hat calculation internally.
        # The layer performs: 1. Add self-loops to edge_index. 2. Compute node degrees.
        # 3. Calculate normalization factors. 4. Aggregate neighbor features (including self)
        #    using the normalized adjacency (message passing). 5. Apply linear transformation (Theta).
        # It does NOT use Q, K, V attention. Neighbor importance is determined by graph structure (degree normalization).
        # --------------------------
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.bn4 = BatchNorm(hidden_channels)
        # ADDED Dropout Layer
        self.dropout = Dropout(p=self.dropout_prob)

        # --- GlobalAttention Explanation ---
        # GlobalAttention performs pooling across all nodes in a graph using a learnable attention mechanism.
        # 1. Gate Network (`gate_nn`): Computes an unnormalized attention score `s_i` for each node `i` based on its features `x_i`.
        #    `s_i = gate_nn(x_i)` (Our `gate_nn` is a small MLP: Linear -> ReLU -> Linear -> scalar score)
        # 2. Softmax Normalization: Scores `s_i` are normalized per graph using softmax to get attention weights `a_i`.
        #    `a_i = softmax(s_i)` (The `batch` vector tells the layer which nodes belong to which graph).
        # 3. Weighted Aggregation: The graph embedding `h_G` is the weighted sum of node features.
        #    `h_G = sum(a_i * x_i)` for all nodes `i` in the graph.
        # 4. Optional NN (`nn` argument, None here): An optional MLP can transform `x_i` before the weighted sum.
        # Learnable parameters are solely within `gate_nn` (and `nn` if used). It does NOT use Q, K, V self-attention.
        # --------------------------------
        gate_nn = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            ReLU(),
            Linear(hidden_channels // 2, 1)
        )
        # It can also take an optional MLP to apply to the pooled features,
        # but we'll use our separate classifier layer for that.
        self.attention_pool = AttentionalAggregation(gate_nn=gate_nn, nn=None)

        # Classifier MLP (a single Linear layer in this case)
        self.classifier = Linear(hidden_channels, out_channels) # Input is pooled features

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Defines the computation performed at every call.

        Args:
            x: Node features matrix with shape [num_nodes, in_channels].
            edge_index: Graph connectivity in COO format with shape [2, num_edges].
            batch: Batch vector assigning each node to a graph, shape [num_nodes].
                   Required for pooling layers.

        Returns:
            Tensor of shape [batch_size, out_channels] representing logits.
        """
        # Apply message passing layers with BatchNorm, ReLU activation, and Dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Apply attention-based graph pooling
        # The batch vector is crucial for pooling layers to identify nodes per graph
        if batch is None:
             # If operating on a single graph, create a dummy batch vector
             batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        # Pass batch vector as a positional argument (index)
        x_pooled = self.attention_pool(x, batch) # [batch_size, hidden_channels]

        # Optional: Dropout before classifier
        x_pooled = self.dropout(x_pooled)

        # Apply the final classifier
        out = self.classifier(x_pooled) # [batch_size, out_channels]

        return out


class GenePhenAIv2_5(nn.Module):
    """
    Graph Neural Network using GCN layers followed by global Multi-Head Self-Attention pooling.

    Consists of 3 GCN message passing layers followed by BatchNorm, ReLU, Dropout,
    a MultiheadAttention pooling layer, and a final linear classifier.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_heads: int = 4, dropout_prob: float = 0.25):
        """
        Args:
            in_channels: Dimensionality of input node features.
            hidden_channels: Dimensionality of hidden layers.
            out_channels: Number of output classes (genes).
            num_heads: Number of attention heads for the MHA pooling layer.
            dropout_prob: Probability for dropout layers.
        """
        super().__init__()
        torch.manual_seed(42) # For reproducibility
        self.dropout_prob = dropout_prob
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

        # GCN Layers (same as v2.0)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)

        # Dropout Layer
        self.dropout = Dropout(p=self.dropout_prob)

        # Multi-Head Self-Attention Pooling Layer
        # We use hidden_channels as embed_dim. batch_first=False is default and expected by permutation below.
        self.mha_pool = nn.MultiheadAttention(embed_dim=self.hidden_channels, num_heads=self.num_heads, dropout=dropout_prob, batch_first=False)

        # Classifier MLP
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Defines the computation performed at every call.

        Args:
            x: Node features matrix with shape [num_nodes, in_channels].
            edge_index: Graph connectivity in COO format with shape [2, num_edges].
            batch: Batch vector assigning each node to a graph, shape [num_nodes]. Required.

        Returns:
            Tensor of shape [batch_size, out_channels] representing logits.
        """
        # Apply message passing layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x) # Final node embeddings [num_nodes, hidden_channels]
        # Note: Dropout is applied *within* MHA and again before classifier

        # Global Self-Attention Pooling
        if batch is None:
            # Self-attention pooling requires batch information
            # If only one graph, create a dummy batch vector
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            # log.debug("Created dummy batch vector for single graph MHA pooling.") # Example Debug log

        # Convert to dense batch format for MHA: [batch_size, max_nodes, hidden_channels]
        # Also get the mask indicating valid nodes (True for real nodes, False for padding)
        x_dense, mask = to_dense_batch(x, batch) # mask shape: [batch_size, max_nodes]
        # log.debug(f"Dense batch shape: {x_dense.shape}, Mask shape: {mask.shape}") # Example Debug log

        # MHA expects [seq_len, batch_size, embed_dim] by default (batch_first=False)
        x_dense = x_dense.permute(1, 0, 2) # Shape: [max_nodes, batch_size, hidden_channels]

        # Apply MultiheadAttention. Q, K, V are all the same for self-attention.
        # key_padding_mask needs shape [batch_size, max_nodes]. It should be True for padded values.
        # Our 'mask' is True for valid nodes, so we invert it (~mask).
        attn_output, _ = self.mha_pool(x_dense, x_dense, x_dense, key_padding_mask=~mask)
        # Output shape: [max_nodes, batch_size, hidden_channels]

        # Permute back: [batch_size, max_nodes, hidden_channels]
        attn_output = attn_output.permute(1, 0, 2)

        # Aggregate node embeddings after attention (masked mean pooling)
        # Zero out padded positions using the mask
        attn_output = attn_output * mask.unsqueeze(-1) # [batch_size, max_nodes, hidden_channels]
        # Sum over the node dimension
        pooled = attn_output.sum(dim=1) # [batch_size, hidden_channels]
        # Get count of non-padded nodes per graph
        num_nodes_per_graph = mask.sum(dim=1).unsqueeze(-1) # [batch_size, 1]
        # Divide by count (add epsilon for safety)
        pooled = pooled / (num_nodes_per_graph + 1e-6) # [batch_size, hidden_channels]
        # log.debug(f"Pooled output shape: {pooled.shape}") # Example Debug log

        # Optional: Dropout before classifier
        pooled = self.dropout(pooled)

        # Apply the final classifier
        out = self.classifier(pooled) # [batch_size, out_channels]

        return out