import torch
from training.models.models import GenePhenAIv2_0 # Assuming models.py is in the same directory or adjust path

# Define dummy parameters
in_channels = 756 # Example input feature dimension (e.g., LLM embedding size)
hidden_channels = 64 # Hidden dimension for GNN layers
out_channels = 5000  # Example number of gene classes
num_nodes = 50       # Total number of nodes in the dummy batch
num_edges = 100      # Number of edges in the dummy batch
batch_size = 4       # Number of graphs in the dummy batch

# Instantiate the model
model = GenePhenAIv2_0(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)
print("Model Instantiated:")
print(model)
print("\n")

# Generate dummy input data
# Node features (random)
x = torch.randn(num_nodes, in_channels)

# Edge index (random pairs of nodes within num_nodes range)
# Ensure indices are within [0, num_nodes - 1]
edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

# Batch vector (assigning each node to one of the graphs in the batch)
# Example: divide nodes roughly equally among graphs
batch = torch.sort(torch.randint(0, batch_size, (num_nodes,), dtype=torch.long))[0]
# Ensure batch covers all graph indices from 0 to batch_size-1
# (This is a simple way; real data loaders handle this more robustly)
for i in range(batch_size):
    if i not in batch:
         # If a batch index is missing, force one node to belong to it
         # This prevents potential errors in pooling layers if a batch index isn't present
         batch[torch.randint(0, num_nodes, (1,))] = i
batch = torch.sort(batch)[0] # Re-sort after potential modification

print(f"Dummy Input Shapes:")
print(f"  Node Features (x): {x.shape}")
print(f"  Edge Index (edge_index): {edge_index.shape}")
print(f"  Batch Vector (batch): {batch.shape}, Unique values: {torch.unique(batch).tolist()}")
print("\n")

# Perform a forward pass
try:
    # Set model to evaluation mode (good practice, disables dropout etc. if they were present)
    model.eval()
    # No gradient calculation needed for simple forward pass test
    with torch.no_grad():
        output = model(x=x, edge_index=edge_index, batch=batch)

    print("Forward Pass Successful!")
    print(f"  Output Shape: {output.shape} (Expected: [{batch_size}, {out_channels}])")

    # Optional: Print some output values
    print(f"  Example Output (first graph):\n{output[0, :5]}...") # Print first 5 logits for the first graph

except Exception as e:
    print(f"Forward Pass FAILED: {e}") 