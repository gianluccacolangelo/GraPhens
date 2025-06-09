import argparse
import logging
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.optim import Adam, AdamW
from pathlib import Path
import time
import math
import numpy as np

# Assuming the script is run from the workspace root
from training.datasets.lmdb_hpo_dataset import LMDBHPOGraphDataset
from training.models.models import GenePhenAIv2_0, GenePhenAIv2_5

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train Phenotype GNN Model using LMDB Dataset")

# Data Args
parser.add_argument('--dataset_root', type=str, required=True, help='Path to the root directory of the LMDB dataset.')
parser.add_argument('--output_dir', type=str, default='training_output', help='Directory to save models, logs, and results.')
parser.add_argument('--split_mode', type=str, default='random', choices=['random', 'precomputed'], help='Dataset split mode.') # Add 'gene_stratified' later
parser.add_argument('--val_split', type=float, default=0.15, help='Fraction of data for validation (if split_mode=random).')
parser.add_argument('--test_split', type=float, default=0.15, help='Fraction of data for testing (if split_mode=random).')
parser.add_argument('--precomputed_split_dir', type=str, default=None, help='Directory containing precomputed split indices (train_indices.pt, val_indices.pt, test_indices.pt) relative to dataset_root.')

# Model Args
parser.add_argument('--model_version', type=str, default='2.0', choices=['2.0', '2.5'], help='Version of the GenePhenAI model to use (e.g., 2.0, 2.5).')
parser.add_argument('--hidden_channels', type=int, default=128, help='Number of hidden units in GNN layers.')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads for MultiHeadAttention pooling (used only in v2.5).')
# Add other model hyperparameters if needed (e.g., dropout, number of layers if model becomes flexible)

# Training Args
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation.')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate (e.g., 0.1 for 1/10 decay per epoch).')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--eval_batches', type=int, default=100, help='Number of batches to use for validation evaluation.')
parser.add_argument('--eval_frequency', type=int, default=5, help='Number of evaluations per training epoch.')

# System Args
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu).')
parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
# ADDED: Argument to resume training from a checkpoint
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint file to resume training from.')

args = parser.parse_args()

# --- Setup ---
output_path = Path(args.output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# Logging
log_file = output_path / 'training.log'
log_level = getattr(logging, args.log_level.upper())
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler() # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# --- Get the file handler to flush manually ---
file_handler = None
for handler in logger.handlers:
    if isinstance(handler, logging.FileHandler):
        file_handler = handler
        break
if file_handler is None:
    logger.warning("Could not find FileHandler to enable manual flushing.")
# ---------------------------------------------

# Log arguments
logger.info("Starting training script with arguments:")
for arg, value in sorted(vars(args).items()):
    logger.info(f"  {arg}: {value}")
if file_handler: file_handler.flush() # Flush after args
# Save args
with open(output_path / 'args.json', 'w') as f:
    json.dump(vars(args), f, indent=4)


# Device
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Seed
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) # if you are using multi-GPU.
# Note: CUDA algorithms can be non-deterministic. Add below for more determinism if needed, but might impact performance.
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# --- Data Loading and Splitting ---
logger.info(f"Loading LMDB dataset from: {args.dataset_root}")
try:
    dataset = LMDBHPOGraphDataset(root_dir=args.dataset_root, readonly=True)
    logger.info(f"Dataset loaded successfully with {len(dataset)} samples.")
except FileNotFoundError:
    logger.error(f"LMDB database not found at {args.dataset_root}/hpo_lmdb. Please run convert_to_lmdb.py first.")
    exit(1)
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    exit(1)

# Infer input and output dimensions
try:
    num_node_features = dataset[0].num_node_features
    num_classes = dataset.num_classes
    if num_classes is None:
         logger.warning("Could not infer num_classes from dataset metadata. Attempting fallback.")
         # Fallback: Infer from unique genes if gene mapping exists
         if dataset._gene_idx_map:
             num_classes = len(dataset._gene_idx_map)
             logger.info(f"Inferred num_classes={num_classes} from gene map keys.")
         else:
              logger.error("Cannot determine number of classes. Ensure metadata.json or __gene_idx_map__ is present in LMDB.")
              exit(1)
    logger.info(f"Inferred input features: {num_node_features}, output classes: {num_classes}")
except Exception as e:
    logger.error(f"Failed to infer model dimensions from dataset: {e}")
    exit(1)


# Splitting
if args.split_mode == 'random':
    n_total = len(dataset)
    if args.val_split + args.test_split >= 1.0:
        logger.error("Sum of validation and test splits must be less than 1.")
        exit(1)
    n_test = int(args.test_split * n_total)
    n_val = int(args.val_split * n_total)
    n_train = n_total - n_val - n_test
    logger.info(f"Performing random split: Train={n_train}, Val={n_val}, Test={n_test}")
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test])

elif args.split_mode == 'precomputed':
    split_dir_path = Path(args.dataset_root) / (args.precomputed_split_dir or "splits") # Default to 'splits' subdir
    logger.info(f"Loading precomputed splits from: {split_dir_path}")
    try:
        train_indices = torch.load(split_dir_path / "train_indices.pt")
        val_indices = torch.load(split_dir_path / "val_indices.pt")
        test_indices = torch.load(split_dir_path / "test_indices.pt")
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        n_train, n_val, n_test = len(train_dataset), len(val_dataset), len(test_dataset)
        logger.info(f"Loaded precomputed splits: Train={n_train}, Val={n_val}, Test={n_test}")
    except FileNotFoundError as e:
        logger.error(f"Precomputed split file not found: {e}. Ensure files exist in {split_dir_path}")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading precomputed splits: {e}")
        exit(1)
else:
    # Placeholder for future split modes like gene_stratified
    logger.error(f"Split mode '{args.split_mode}' is not yet implemented.")
    exit(1)

# DataLoaders
train_loader = PyGDataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True if device.type == 'cuda' else False,
    persistent_workers=True if args.num_workers > 0 else False # Avoid recreating workers

)
val_loader = PyGDataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True if device.type == 'cuda' else False,
    persistent_workers=False
)
test_loader = PyGDataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True if device.type == 'cuda' else False,
    persistent_workers=False
)

logger.info("DataLoaders created.")

# --- Model, Loss, Optimizer ---
logger.info(f"Instantiating model version: GenePhenAIv{args.model_version}")

try:
    common_args = {
        'in_channels': num_node_features,
        'hidden_channels': args.hidden_channels,
        'out_channels': num_classes
        # Add common dropout arg if needed
    }
    if args.model_version == '2.0':
        model = GenePhenAIv2_0(**common_args).to(device)
    elif args.model_version == '2.5':
        model = GenePhenAIv2_5(**common_args, num_heads=args.num_heads).to(device)
    else:
        # This should not happen due to choices in argparse, but good practice
        logger.error(f"Unsupported model version: {args.model_version}")
        exit(1)

    logger.info("Model instantiated successfully:")
    logger.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")
    if file_handler: file_handler.flush()

except Exception as e:
    logger.error(f"Failed to instantiate model version {args.model_version}: {e}")
    if file_handler: file_handler.flush()
    exit(1)

# Map gene names (from metadata) to integer class indices (0 to num_classes-1)
logger.info("Creating gene-to-index mapping (no class weights will be used as dataset is balanced)")
gene_to_class_idx = {gene: i for i, gene in enumerate(dataset.metadata.get('genes', []))}

if not gene_to_class_idx:
     logger.warning("Gene list not found in metadata. Cannot determine class indices.")
     exit(1)  # Cannot proceed without the gene mapping
else:
     logger.info(f"Created mapping for {len(gene_to_class_idx)} genes.")

# Using uniform weights (None) as dataset is balanced
class_weights = None

criterion = torch.nn.CrossEntropyLoss(weight=class_weights) # Use weights if available
optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

logger.info("Optimizer and Loss function defined.")

# --- Checkpoint Loading (ADDED) ---
start_epoch = 1
if args.resume_from_checkpoint:
    checkpoint_path = Path(args.resume_from_checkpoint)
    if checkpoint_path.exists():
        logger.info(f"Attempting to resume training from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Determine the epoch to start from. The checkpoint saves the epoch it *completed*.
            start_epoch = checkpoint['epoch'] + 1
            # You might want to also restore other things like LR scheduler state if you add one

            # MODIFIED: Log and potentially warn about model version mismatch
            checkpoint_model_version = checkpoint.get('model_version')
            if checkpoint_model_version:
                logger.info(f"Checkpoint was saved with model version: {checkpoint_model_version}")
                if checkpoint_model_version != args.model_version:
                    logger.warning(f"Checkpoint model version ({checkpoint_model_version}) differs from requested model version ({args.model_version}). Loading weights anyway.")
            else:
                logger.warning("Checkpoint does not contain model version information.")
            # ---------------------------------------------------------------

            logger.info(f"Successfully loaded checkpoint. Resuming training from epoch {start_epoch}.")
            # Log the state of the loaded optimizer, specifically the learning rate
            loaded_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Optimizer state loaded. Current learning rate: {loaded_lr:.1e}")
            # Adjust initial learning rate based on the resumed epoch?
            # The current loop already adjusts LR based on epoch number, so just starting
            # the loop at the correct epoch number should handle this.
            if file_handler: file_handler.flush()
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_path}: {e}. Starting training from epoch 1.")
            start_epoch = 1
            if file_handler: file_handler.flush()
            # Reset optimizer if loading failed? Or exit? For now, we continue from scratch.
            optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) # Reinitialize optimizer
    else:
        logger.warning(f"Checkpoint file not found: {checkpoint_path}. Starting training from epoch 1.")
        start_epoch = 1
        if file_handler: file_handler.flush()
else:
    logger.info("No checkpoint specified. Starting training from epoch 1.")
    if file_handler: file_handler.flush()
# ------------------------------------

# --- Helper function for calculating batch metrics ---
@torch.no_grad()
def calculate_batch_metrics(out: torch.Tensor, target_y: torch.Tensor, k_vals: list[int] = [1, 10]) -> tuple[float, dict[int, float]]:
    """Calculates MRR and Top-K accuracy for a single batch.

    Args:
        out: Model output logits/scores (shape: [batch_size, num_classes]).
        target_y: Target class indices (shape: [batch_size]).
        k_vals: List of K values for Top-K accuracy calculation.

    Returns:
        A tuple containing: (batch_mrr, batch_top_k_acc_dict).
        Returns (0.0, {k: 0.0 for k in k_vals}) if calculation fails or batch is empty.
    """
    batch_mrr = 0.0
    batch_top_k_acc = {k: 0.0 for k in k_vals}
    num_graphs = target_y.size(0)

    if num_graphs == 0:
        return batch_mrr, batch_top_k_acc

    try:
        _, sorted_indices = torch.sort(out, dim=1, descending=True)
        target_expanded = target_y.unsqueeze(1).expand_as(sorted_indices)
        target_match = (sorted_indices == target_expanded)
        indices = torch.nonzero(target_match, as_tuple=True)[1]

        if len(indices) == num_graphs:
            ranks = indices + 1
            reciprocal_ranks = 1.0 / ranks.float()
            batch_mrr = torch.mean(reciprocal_ranks).item()

            top_k_correct = {k: torch.sum(ranks <= k).item() for k in k_vals}
            batch_top_k_acc = {k: (correct / num_graphs) for k, correct in top_k_correct.items()}
        else:
            # Log internally or handle mismatch if needed, for now return 0s
            pass # logger.warning("Rank finding mismatch in calculate_batch_metrics")

    except Exception as e:
        # Log internally or handle error if needed, for now return 0s
        pass # logger.error(f"Error in calculate_batch_metrics: {e}")

    return batch_mrr, batch_top_k_acc
# ------------------------------------------------------

# --- Training & Evaluation Functions ---

# Ensure gene_to_class_idx map is available in the scope of these functions
# It's defined globally in the script before these functions are called.

# MODIFIED evaluate function to include Top-K and MRR
@torch.no_grad()
def evaluate(loader, max_batches=None, k_vals=[1, 5, 10, 20]):
    model.eval()
    total_loss = 0
    total_samples = 0
    processed_batches = 0
    # Initialize accumulators for new metrics
    sum_reciprocal_ranks = 0.0
    # Use the full k_vals list provided for evaluation
    top_k_correct = {k: 0 for k in k_vals}

    start_time = time.time()
    # Update log message to reflect which k_vals are used
    logger.info(f"Starting evaluation for max {max_batches or 'all'} batches. Calculating Top-K for k={k_vals} and MRR.")
    if file_handler: file_handler.flush()

    for i, batch in enumerate(loader):
        batch_start_time = time.time()
        batch = batch.to(device)

        # --- Target Creation ---
        if not hasattr(batch, 'gene'):
            logger.warning(f"Eval Batch {i} is missing 'gene' attribute. Skipping.")
            continue
        try:
            target_y = torch.tensor([gene_to_class_idx[g] for g in batch.gene], dtype=torch.long, device=device)
        except KeyError as e:
            logger.warning(f"Eval Batch {i} contains unknown gene: {e}. Skipping batch.")
            continue
        except Exception as e:
            logger.warning(f"Error mapping genes to indices in eval batch {i}: {e}. Skipping batch.")
            continue
        # --------------------

        if not hasattr(batch, 'batch') or batch.batch is None:
             logger.warning(f"Eval Batch {i} is missing 'batch' attribute required for pooling. Skipping.")
             continue

        try:
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, target_y)

            samples_in_batch = batch.num_graphs
            total_loss += loss.item() * samples_in_batch
            total_samples += samples_in_batch
            processed_batches += 1

            # --- Calculate Ranking Metrics using helper --- Use full k_vals here
            batch_mrr, batch_top_k_acc = calculate_batch_metrics(out, target_y, k_vals=k_vals)

            # Accumulate metrics correctly
            sum_reciprocal_ranks += batch_mrr * samples_in_batch # Multiply batch avg MRR by batch size
            for k in k_vals:
                top_k_correct[k] += batch_top_k_acc[k] * samples_in_batch # Multiply batch acc by batch size
            # --------------------------------

            # Log per-batch results (including Top-1 for quick check)
            batch_eval_time = time.time() - batch_start_time
            current_batch_top1 = batch_top_k_acc[1] # Get Top-1 count up to this batch
            current_batch_mrr = batch_mrr # Get batch MRR
            # These are cumulative, maybe log batch-specific instead? Let's log batch-specific loss/acc
            # For simplicity, let's keep the debug log as is (batch loss/acc) for now
            # Or add batch-specific ranks? Let's add avg rank for the batch
            avg_rank_batch = 1.0 / batch_mrr if batch_mrr > 0 else float('inf') # Estimate avg rank from MRR
            logger.debug(f"  Eval Batch {i+1}/{max_batches or len(loader)} | Loss: {loss.item():.4f} | Approx Avg Rank: {avg_rank_batch:.2f} | Time: {batch_eval_time:.3f}s")
            # No flush for debug

            # Check if max_batches limit is reached
            if max_batches is not None and processed_batches >= max_batches:
                logger.info(f"Reached evaluation batch limit ({max_batches}). Stopping evaluation.")
                if file_handler: file_handler.flush()
                break

        except Exception as e:
             logger.error(f"Error during evaluation forward pass or metric calculation in batch {i}: {e}")
             # Optionally re-raise or log more details
             continue # Skip problematic batches during evaluation

    # Calculate final metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    # Final MRR is sum of batch MRRs * batch_size / total_samples
    final_mrr = sum_reciprocal_ranks / total_samples if total_samples > 0 else 0
    # Final TopK is sum of batch Acc * batch_size / total_samples
    final_top_k_acc = {k: (correct / total_samples if total_samples > 0 else 0) for k, correct in top_k_correct.items()}
    eval_time = time.time() - start_time

    # Log summary including new metrics
    top_k_log_str = ", ".join([f"Top-{k}: {acc:.4f}" for k, acc in final_top_k_acc.items()])
    logger.info(f"Finished evaluation. Avg Loss: {avg_loss:.4f}, MRR: {final_mrr:.4f}, {top_k_log_str} over {processed_batches} batches.")
    if file_handler: file_handler.flush()

    # Return all metrics
    return avg_loss, final_mrr, final_top_k_acc, eval_time

# --- Training Loop Modification --- needed to handle new return values
def train_epoch(loader, val_loader, epoch):
    model.train()
    total_loss = 0
    total_samples = 0
    batches_processed_since_eval = 0
    start_time = time.time()
    epoch_start_time = start_time # For overall epoch timing

    # Calculate evaluation interval
    if args.eval_frequency <= 0:
        eval_interval = len(loader) + 1 # Evaluate only at the end if frequency is 0 or less
    else:
        eval_interval = max(1, len(loader) // args.eval_frequency) # Ensure interval is at least 1
    logger.info(f"Epoch {epoch}: Evaluating every {eval_interval} training batches.")
    if file_handler: file_handler.flush()


    for i, batch in enumerate(loader):
        batch_start_time = time.time()
        batch = batch.to(device)
        optimizer.zero_grad()

        # --- Target Creation ---
        # Check if batch has the 'gene' attribute (list of strings)
        if not hasattr(batch, 'gene'):
            logger.error(f"Batch {i} is missing 'gene' attribute. Cannot determine target. Skipping.")
            continue
        try:
            # Map gene strings to integer class indices using the precomputed map
            # Ensure all genes in the batch are known
            target_y = torch.tensor([gene_to_class_idx[g] for g in batch.gene], dtype=torch.long, device=device)
        except KeyError as e:
            logger.error(f"Batch {i} contains unknown gene: {e}. Ensure metadata 'genes' list is complete. Skipping batch.")
            continue
        except Exception as e:
             logger.error(f"Error mapping genes to indices in batch {i}: {e}. Skipping batch.")
             continue
        # --------------------

        # Ensure batch vector exists (DataLoader should add it, but double check)
        if not hasattr(batch, 'batch') or batch.batch is None:
            logger.error(f"Batch {i} is missing 'batch' attribute required for pooling. Skipping.")
            continue

        try:
            # --- Forward pass, loss, backward, step ---
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, target_y)
            loss.backward()
            optimizer.step()
            # ---------------------------------------------

            # --- Calculate metrics using helper function ---
            # Use a subset of k values relevant for training log
            batch_mrr, batch_top_k_acc = calculate_batch_metrics(out, target_y, k_vals=[1, 10])
            # ---------------------------------------------

            # --- Accumulate overall epoch stats ---
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
            batches_processed_since_eval += 1
            # ------------------------------------

            # --- Log training progress every 40 batches ---
            if (i + 1) % 40 == 0:
                batch_time = time.time() - batch_start_time
                # Calculate metrics for this specific batch just before logging
                # Assumes calculate_batch_metrics function exists
                log_mrr, log_top_k_acc = calculate_batch_metrics(out, target_y, k_vals=[1, 10])
                # Format and log
                top_k_log_str_batch = ", ".join([f"Top-{k}: {acc:.3f}" for k, acc in log_top_k_acc.items()])
                logger.info(f' Epoch {epoch} | Train Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f} | MRR: {log_mrr:.3f} | {top_k_log_str_batch} | Time: {batch_time:.3f}s')
                if file_handler: file_handler.flush()
            # -------------------------------------------

        except Exception as e: # Catch errors from main forward/backward/step
            logger.error(f"Error during training forward/backward/step in batch {i}: {e}")
            logger.error(f"Batch data keys: {batch.keys}")
            logger.error(f"Node features shape: {batch.x.shape if hasattr(batch, 'x') else 'N/A'}")
            logger.error(f"Edge index shape: {batch.edge_index.shape if hasattr(batch, 'edge_index') else 'N/A'}")
            logger.error(f"Target shape: {target_y.shape}, dtype: {target_y.dtype}") # Log created target
            logger.error(f"Batch vector shape: {batch.batch.shape if hasattr(batch, 'batch') else 'N/A'}")
            # Decide whether to continue or raise the exception
            # For now, log and continue to avoid stopping training for one bad batch
            continue # Or raise e

        # --- Intra-Epoch Evaluation and Checkpointing ---
        # Evaluate if interval reached OR it's the last batch
        if batches_processed_since_eval >= eval_interval or (i + 1) == len(loader):
            logger.info(f"--- Evaluating at Epoch {epoch}, Batch {i+1}/{len(loader)} ---")
            if file_handler: file_handler.flush()

            # Capture all returned metrics
            val_loss, val_mrr, val_top_k, val_time = evaluate(val_loader, max_batches=args.eval_batches)
            top_k_log_str = ", ".join([f"Top-{k}: {acc:.4f}" for k, acc in val_top_k.items()])
            logger.info(f"Epoch {epoch} | Batch {i+1} | Val Loss: {val_loss:.4f}, MRR: {val_mrr:.4f}, {top_k_log_str} | Eval Time: {val_time:.2f}s")
            if file_handler: file_handler.flush()

            # Checkpoint saving - include new metrics
            # ... (checkpoint name generation) ...
            part_num = math.ceil((i + 1) / eval_interval) # Calculate part number (1-based)
            if (i + 1) == len(loader) and batches_processed_since_eval < eval_interval:
                 part_num = args.eval_frequency # Ensure last part is correctly numbered if epoch size not divisible

            checkpoint_name = f'checkpoint_epoch_{epoch}_part_{part_num}.pt'
            checkpoint_path = output_path / checkpoint_name
            torch.save({
                'epoch': epoch,
                'batch': i + 1,
                'model_version': args.model_version, # ADDED model version
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mrr': val_mrr, # ADDED
                'val_top_k_acc': val_top_k # ADDED
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            if file_handler: file_handler.flush()

            batches_processed_since_eval = 0
            model.train()

    # --- End of Epoch ---
    avg_epoch_loss = total_loss / total_samples if total_samples > 0 else 0
    epoch_time = time.time() - epoch_start_time
    # Return avg loss for the whole epoch and time
    return avg_epoch_loss, epoch_time

# --- Training Loop ---
logger.info(f"Starting training loop from epoch {start_epoch}...") # MODIFIED Log
total_start_time = time.time()

# MODIFIED: Start loop from start_epoch
for epoch in range(start_epoch, args.epochs + 1):
    # --- Set Learning Rate for the Epoch ---
    # Start with args.learning_rate for epoch 1, divide by 10 for subsequent epochs
    # This calculation uses the *current* epoch number, which is correct even when resuming.
    current_lr = args.learning_rate / (10**(epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    logger.info(f"--- Epoch {epoch}/{args.epochs} | Learning Rate: {current_lr:.1e} ---")
    if file_handler: file_handler.flush()

    # Pass val_loader and epoch to train_epoch for intra-epoch evaluation
    train_loss, train_time = train_epoch(train_loader, val_loader, epoch)
    # train_epoch now handles evaluation and checkpointing, so no separate evaluate call here

    logger.info(f"Epoch {epoch} Completed | Avg Train Loss: {train_loss:.4f} | Total Epoch Time: {train_time:.2f}s")
    if file_handler: file_handler.flush()

# --- Final Testing Modification ---
# ... (loading checkpoint logic) ...
logger.info("Attempting to load the last saved checkpoint for final testing...")
if file_handler: file_handler.flush()
# Find the last checkpoint file based on epoch and part number
last_epoch = epoch # The last epoch completed
last_part = args.eval_frequency # Assuming the last part was saved
last_checkpoint_name = f'checkpoint_epoch_{last_epoch}_part_{last_part}.pt'
last_checkpoint_path = output_path / last_checkpoint_name

try:
    if last_checkpoint_path.exists():
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded last checkpoint: {last_checkpoint_name} (Epoch {checkpoint.get('epoch', 'N/A')}, Batch {checkpoint.get('batch', 'N/A')})")
    else:
        logger.warning(f"Last checkpoint {last_checkpoint_name} not found. Testing with the final model state from end of training.")

    if file_handler: file_handler.flush()
except Exception as e:
    logger.error(f"Error loading last checkpoint: {e}. Testing with final model state.")
    if file_handler: file_handler.flush()

# Evaluate on the full test set using the potentially loaded model
logger.info("Running final evaluation on the full Test set...")
if file_handler: file_handler.flush()
# Evaluate on ALL test batches and capture all metrics
test_loss, test_mrr, test_top_k, test_time = evaluate(test_loader, max_batches=None)
test_top_k_log_str = ", ".join([f"Top-{k}: {acc:.4f}" for k, acc in test_top_k.items()])
logger.info(f"--- Test Results (Model from {last_checkpoint_name if last_checkpoint_path.exists() else 'End of Training'}) ---")
if file_handler: file_handler.flush()
logger.info(f"Test Loss: {test_loss:.4f}")
logger.info(f"Test MRR: {test_mrr:.4f}") # ADDED
logger.info(f"Test Top-K Accuracy: {test_top_k_log_str}") # ADDED
if file_handler: file_handler.flush()
logger.info(f"Test Time: {test_time:.2f}s")
if file_handler: file_handler.flush()

# Save test results - include new metrics
results = {
    'last_checkpoint_loaded': str(last_checkpoint_path) if last_checkpoint_path.exists() else 'None (final state used)',
    'model_version': args.model_version, # ADDED model version
    'test_loss': test_loss,
    'test_mrr': test_mrr, # ADDED
    'test_top_k_accuracy': test_top_k, # ADDED
    'total_training_time_minutes': (time.time() - total_start_time)/60,
    'args': vars(args)
}
with open(output_path / 'test_results.json', 'w') as f:
    json.dump(results, f, indent=4)

logger.info(f"Test results saved to {output_path / 'test_results.json'}")
if file_handler: file_handler.flush()
logger.info("Script finished.")
if file_handler: file_handler.flush()
