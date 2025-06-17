import argparse
import logging
import torch
import json
from pathlib import Path
import sys

# Add project root to sys.path to allow relative imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from training.models.models import GenePhenAIv2_0, GenePhenAIv2_5
    from src.graphens import GraPhens
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure the script is run from the project root or the GraPhens package is installed correctly.")
    sys.exit(1)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Perform inference using a trained GenePhenAIv2 model.")

# Input Data Args
parser.add_argument('--hpo_ids', type=str, required=True, nargs='+', help='List of HPO IDs for inference (e.g., HP:0001250 HP:0001251).')
parser.add_argument('--dataset_root', type=str, default='/home/gcolangelo/GraPhens/data/simulation/output/', help='Path to the root directory of the LMDB dataset used for training (for metadata).')

# Model & Checkpoint Args
parser.add_argument('--checkpoint_path', type=str, default='training_output/checkpoint_epoch_1.pt', help='Path to the model checkpoint file (.pt).')
parser.add_argument('--embedding_path', type=str, default='data/embeddings/hpo_embeddings_gsarti_biobert-nli_20250317_162424.pkl', help='Path to the HPO lookup embeddings file (.pkl).')
parser.add_argument('--hidden_channels', type=int, help='Number of hidden units in GNN layers. Loaded from checkpoint args.json if available.')
parser.add_argument('--model_version', type=str, choices=['2.0', '2.5'], help='Model version to use. Loaded from checkpoint args.json if available.')
parser.add_argument('--num_heads', type=int, help='Number of attention heads for v2.5. Loaded from checkpoint args.json if available.')
parser.add_argument('--embedding_dim', type=int, default=768, help='Dimension of the node embeddings (must match training and embedding model).')

# Inference Args
parser.add_argument('--top_k', type=int, default=60, help='Number of top gene predictions to display.')
parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu).')
parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level.')

args = parser.parse_args()

# --- Setup ---
# Logging
log_level = getattr(logging, args.log_level.upper())
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Log to console
)
logger = logging.getLogger(__name__)

# Device
device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- Load Training Arguments and Override ---
training_args = {}
checkpoint_dir = Path(args.checkpoint_path).parent
args_json_path = checkpoint_dir / 'args.json'

if args_json_path.is_file():
    logger.info(f"Found training arguments file: {args_json_path}")
    with open(args_json_path, 'r') as f:
        training_args = json.load(f)

    # Override args with values from training_args
    args.model_version = training_args.get('model_version', args.model_version)
    args.hidden_channels = training_args.get('hidden_channels', args.hidden_channels)
    args.num_heads = training_args.get('num_heads', args.num_heads)
    
    logger.info(f"Loaded from args.json: model_version='{args.model_version}', hidden_channels={args.hidden_channels}, num_heads={args.num_heads}")

else:
    logger.warning(f"Training arguments file not found at {args_json_path}. Will rely on command-line arguments for model hyperparameters.")


# --- Load Metadata (Number of Classes and Gene Mapping) ---
logger.info(f"Loading metadata from dataset root: {args.dataset_root}")
try:
    metadata_path = Path(args.dataset_root) / 'metadata.json'
    if not metadata_path.exists():
         raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    gene_list = metadata.get('genes', [])
    if not gene_list:
        logger.error("Could not retrieve gene list from dataset metadata. Ensure 'genes' key exists in metadata.json.")
        sys.exit(1)

    num_classes = len(gene_list)

    # Create index-to-gene mapping
    idx_to_gene = {i: gene for i, gene in enumerate(gene_list)}
    logger.info(f"Successfully loaded metadata: Found {num_classes} classes (genes).")

except FileNotFoundError as e:
    logger.error(f"{e}. Please provide the correct path to the directory containing 'metadata.json'.")
    sys.exit(1)
except json.JSONDecodeError:
    logger.error(f"Error decoding JSON from {metadata_path}. The file may be corrupt.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error loading dataset metadata: {e}")
    sys.exit(1)


# --- Instantiate Model ---
logger.info("Instantiating model...")

# Check if necessary model parameters are available
if not args.model_version:
    logger.error("Model version not specified. Provide --model_version or a checkpoint with args.json.")
    sys.exit(1)
if not args.hidden_channels:
    logger.error("Hidden channels not specified. Provide --hidden_channels or a checkpoint with args.json.")
    sys.exit(1)

model_params = {
    'in_channels': args.embedding_dim,
    'hidden_channels': args.hidden_channels,
    'out_channels': num_classes
}

if args.model_version == '2.5':
    if not args.num_heads:
        logger.error("Number of heads (--num_heads) is required for model_version 2.5 and was not found in args.json.")
        sys.exit(1)
    model_class = GenePhenAIv2_5
    model_params['num_heads'] = args.num_heads
    logger.info(f"Instantiating GenePhenAIv2_5 model with params: {model_params}")
else: # Default to 2.0
    model_class = GenePhenAIv2_0
    logger.info(f"Instantiating GenePhenAIv2_0 model with params: {model_params}")

try:
    model = model_class(**model_params).to(device)
except Exception as e:
    logger.error(f"Error instantiating model: {e}")
    sys.exit(1)


# --- Load Checkpoint ---
checkpoint_path = Path(args.checkpoint_path)
if not checkpoint_path.exists():
    logger.error(f"Checkpoint file not found: {checkpoint_path}")
    sys.exit(1)

logger.info(f"Loading checkpoint from: {checkpoint_path}")
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set model to evaluation mode
    logger.info("Checkpoint loaded successfully and model set to evaluation mode.")
    # Log checkpoint metadata if available
    if 'epoch' in checkpoint: logger.info(f"  Checkpoint Epoch: {checkpoint['epoch']}")
    if checkpoint.get('val_loss') is not None: logger.info(f"  Checkpoint Val Loss: {checkpoint['val_loss']:.4f}")
    if checkpoint.get('val_mrr') is not None: logger.info(f"  Checkpoint Val MRR: {checkpoint['val_mrr']:.4f}")
    if checkpoint.get('val_top_k_acc'):
        # Filter out potential None values in the dictionary
        valid_accs = {k: acc for k, acc in checkpoint['val_top_k_acc'].items() if acc is not None}
        if valid_accs:
            top_k_str = ", ".join([f"Top-{k}: {acc:.4f}" for k, acc in valid_accs.items()])
            logger.info(f"  Checkpoint Val Top-K: {top_k_str}")

except Exception as e:
    logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
    sys.exit(1)

# --- Initialize GraPhens for Preprocessing ---
logger.info("Initializing GraPhens with settings matching training preprocessing...")
try:
    graphens = (
        GraPhens()
        .with_lookup_embeddings(args.embedding_path)
        .with_augmentation(include_ancestors=True)
        .with_adjacency_settings(include_reverse_edges=False)
    )
    logger.info("GraPhens initialized with lookup embeddings, ancestor augmentation, and forward-only edges.")
except Exception as e:
    logger.error(f"Error initializing GraPhens: {e}")
    sys.exit(1)

# --- Preprocess Input HPO IDs ---
logger.info(f"Preprocessing input HPO IDs: {args.hpo_ids}")
try:
    # Validate HPO IDs (basic check)
    valid_hpo_ids = []
    for hpo_id in args.hpo_ids:
        if hpo_id.startswith("HP:") and hpo_id[3:].isdigit():
            valid_hpo_ids.append(hpo_id)
        else:
            logger.warning(f"Skipping invalid HPO ID format: {hpo_id}. Should be 'HP:#######'.")
    if not valid_hpo_ids:
        logger.error("No valid HPO IDs provided.")
        sys.exit(1)

    # Create graph using GraPhens
    internal_graph = graphens.create_graph_from_phenotypes(valid_hpo_ids)
    logger.info(f"Internal graph created with {len(internal_graph.node_mapping)} nodes and {internal_graph.edge_index.shape[1]} edges.")

    # Export graph to PyTorch Geometric format
    pyg_data = graphens.export_graph(internal_graph, format="pytorch")
    pyg_data = pyg_data.to(device) # Move graph data to the target device
    logger.info("Graph exported to PyTorch Geometric format and moved to device.")

    # Sanity check dimensions
    if pyg_data.x.shape[1] != args.embedding_dim:
         logger.error(f"Input graph node feature dimension ({pyg_data.x.shape[1]}) does not match expected embedding dimension ({args.embedding_dim}). Check the '--embedding_dim' argument and the loaded embeddings.")
         sys.exit(1)

except ValueError as e:
     logger.error(f"Error during HPO ID validation or graph creation: {e}")
     sys.exit(1)
except Exception as e:
    logger.error(f"Error during graph preprocessing: {e}")
    sys.exit(1)

# --- Perform Inference ---
logger.info("Performing inference...")
try:
    with torch.no_grad():
        output_logits = model(pyg_data.x, pyg_data.edge_index, pyg_data.batch)
    logger.info("Inference complete.")

    # Ensure output shape is as expected [1, num_classes] for a single graph
    if output_logits.shape[0] != 1 or output_logits.shape[1] != num_classes:
        logger.warning(f"Unexpected output shape from model: {output_logits.shape}. Expected: [1, {num_classes}]. Results might be inaccurate.")
        # Attempt to use the first row if batch size > 1 unexpectedly
        if output_logits.shape[0] > 1 and output_logits.shape[1] == num_classes:
             output_logits = output_logits[0].unsqueeze(0)
             logger.warning("Using the first row of the output.")


except Exception as e:
    logger.error(f"Error during model inference: {e}")
    sys.exit(1)

# --- Interpret and Display Results ---
logger.info(f"Calculating top {args.top_k} predictions...")
try:
    # Apply Softmax to get probabilities
    probabilities = torch.softmax(output_logits, dim=1)

    # Get top K probabilities and their indices
    top_p, top_idx = torch.topk(probabilities, args.top_k, dim=1)

    # Detach tensors and move to CPU for numpy conversion/printing
    top_p = top_p.squeeze().cpu().numpy()
    top_idx = top_idx.squeeze().cpu().numpy()

    # Print results
    print("\n--- Top Gene Predictions ---")
    print(f"Input HPO Terms: {', '.join(valid_hpo_ids)}")
    print("-" * 30)
    for i in range(args.top_k):
        gene_index = top_idx[i]
        probability = top_p[i]
        gene_name = idx_to_gene.get(gene_index, f"Unknown Index {gene_index}") # Handle potential index issues
        print(f"{i+1}. Gene: {gene_name:<15} Probability: {probability:.4f}")
    print("-" * 30)

except Exception as e:
    logger.error(f"Error interpreting or displaying results: {e}")
    sys.exit(1)

logger.info("Script finished successfully.")
