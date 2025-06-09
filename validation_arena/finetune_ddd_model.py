import argparse
import logging
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset as TorchDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.optim import Adam, AdamW
from pathlib import Path
import time
import math
import numpy as np
import sys

# Ensure src and training directories are in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from training.datasets.lmdb_hpo_dataset import LMDBHPOGraphDataset
from training.models.models import GenePhenAIv2_0, GenePhenAIv2_5

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Fine-tune a pre-trained Phenotype GNN Model.")

# Data Args
parser.add_argument('--ddd_dataset_pt', type=str, required=True, help='Path to the DDD fine-tuning dataset (.pt file).')
parser.add_argument('--ddd_label_map_json', type=str, required=True, help='Path to the DDD gene-to-label mapping (.json file).')
parser.add_argument('--pretrained_checkpoint_path', type=str, required=True, help='Path to the pre-trained model checkpoint.')
parser.add_argument('--original_dataset_root', type=str, required=True, help='Path to the root of the original LMDB dataset (for original validation).')
parser.add_argument('--original_training_args_json', type=str, required=True, help='Path to args.json from the original training run.')
parser.add_argument('--output_dir', type=str, default='finetuning_output', help='Directory to save fine-tuned models, logs, and results.')

# Model Args specific to fine-tuning (most will be from original_training_args_json or checkpoint)
# hidden_channels and num_heads will be inferred.

# Fine-tuning Hyperparameters
parser.add_argument('--epochs', type=int, default=50, help='Number of fine-tuning epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for fine-tuning.')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for fine-tuning.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 penalty) for fine-tuning.')
parser.add_argument('--freeze_gcn_layers', action='store_true', default=True, help='Freeze GCN convolutional layers.')
parser.add_argument('--no_freeze_gcn_layers', action='store_false', dest='freeze_gcn_layers', help='Do not freeze GCN layers (fine-tune all).')
parser.add_argument('--freeze_mha_layer', action='store_true', default=True, help='Freeze MHA pooling layer (for v2.5 model, default: True).')
parser.add_argument('--no_freeze_mha_layer', action='store_false', dest='freeze_mha_layer', help='Do not freeze MHA pooling layer (fine-tune it).')


# Evaluation Args
parser.add_argument('--eval_original_val_every_n_epochs', type=int, default=1, help='Frequency (in epochs) to evaluate on the original validation set. 0 to disable.')
parser.add_argument('--original_eval_max_batches', type=int, default=50, help='Max batches to use from original validation set for periodic checks.')
parser.add_argument('--ddd_val_split', type=float, default=0.2, help='Fraction of DDD data for validation.')


# System Args
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader.')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu).')
parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level.')
parser.add_argument('--seed', type=int, default=43, help='Random seed for reproducibility.') # Different seed than original training

args = parser.parse_args()

# --- Setup ---
output_path = Path(args.output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# Logging
log_file = output_path / 'finetuning.log'
log_level_val = getattr(logging, args.log_level.upper())
logging.basicConfig(
    level=log_level_val,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

file_handler = next((h for h in logger.handlers if isinstance(h, logging.FileHandler)), None)

logger.info("Starting fine-tuning script with arguments:")
for arg_name, value in sorted(vars(args).items()):
    logger.info(f"  {arg_name}: {value}")
if file_handler: file_handler.flush()

with open(output_path / 'finetuning_args.json', 'w') as f:
    json.dump(vars(args), f, indent=4)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)
    torch_cuda_manual_seed_all = getattr(torch.cuda, 'manual_seed_all', None) # For older PyTorch
    if torch_cuda_manual_seed_all:
        torch_cuda_manual_seed_all(args.seed)

# --- Load Original Training Args ---
try:
    with open(args.original_training_args_json, 'r') as f:
        original_train_args = json.load(f)
    logger.info(f"Loaded original training arguments from: {args.original_training_args_json}")
except Exception as e:
    logger.error(f"Error loading original training arguments: {e}. Exiting.")
    sys.exit(1)

# --- Helper: Custom Dataset for list of PyG Data objects ---
class ListDataset(TorchDataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

# --- Data Loading ---
# 1. DDD Fine-tuning Data
logger.info("Loading DDD fine-tuning dataset...")
try:
    ddd_data_list = torch.load(args.ddd_dataset_pt)
    with open(args.ddd_label_map_json, 'r') as f:
        ddd_gene_to_label = json.load(f)
    ddd_num_classes = len(ddd_gene_to_label)
    logger.info(f"Loaded {len(ddd_data_list)} samples for DDD fine-tuning with {ddd_num_classes} classes.")

    # Use the full DDD dataset for fine-tuning (no train/val split for DDD)
    ddd_full_dataset = ListDataset(ddd_data_list)
    # ddd_n_total = len(ddd_data_list)
    # ddd_n_val = int(args.ddd_val_split * ddd_n_total)
    # ddd_n_train = ddd_n_total - ddd_n_val
    # ddd_train_list, ddd_val_list = random_split(ddd_data_list, [ddd_n_train, ddd_n_val])
    
    # ddd_train_dataset = ListDataset(list(ddd_train_list))
    # ddd_val_dataset = ListDataset(list(ddd_val_list))

    ddd_train_loader = PyGDataLoader(ddd_full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, exclude_keys=['node_mapping'])
    # ddd_val_loader will be the same as ddd_train_loader but with shuffle=False, for consistent evaluation
    ddd_eval_loader = PyGDataLoader(ddd_full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, exclude_keys=['node_mapping'])
    logger.info(f"DDD DataLoaders created: Training on {len(ddd_full_dataset)} samples, Evaluating on {len(ddd_full_dataset)} samples.")

except Exception as e:
    logger.error(f"Error loading DDD data: {e}")
    sys.exit(1)

# 2. Original Validation Data (from LMDB)
original_val_loader = None
original_gene_to_class_idx = None
original_num_classes = None

if args.eval_original_val_every_n_epochs > 0:
    logger.info("Loading original validation dataset from LMDB...")
    try:
        original_dataset = LMDBHPOGraphDataset(root_dir=args.original_dataset_root, readonly=True)
        logger.info(f"Original LMDB dataset loaded successfully with {len(original_dataset)} samples.")
        
        original_num_node_features = original_dataset[0].num_node_features # For model instantiation check
        original_num_classes = original_dataset.num_classes
        if original_num_classes is None: # Fallback
             original_num_classes = len(original_dataset._gene_idx_map)

        original_gene_to_class_idx = {gene: i for i, gene in enumerate(original_dataset.metadata.get('genes', []))}
        if not original_gene_to_class_idx:
            logger.error("Original dataset metadata does not contain 'genes' list for mapping. Cannot evaluate.")
            sys.exit(1)

        # Use split settings from original_train_args
        original_split_mode = original_train_args.get('split_mode', 'random')
        if original_split_mode == 'random':
            ot_n_total = len(original_dataset)
            ot_n_test = int(original_train_args.get('test_split', 0.15) * ot_n_total)
            ot_n_val = int(original_train_args.get('val_split', 0.15) * ot_n_total)
            ot_n_train = ot_n_total - ot_n_val - ot_n_test
            _, original_val_dataset, _ = random_split(original_dataset, [ot_n_train, ot_n_val, ot_n_test])
        elif original_split_mode == 'precomputed':
            split_dir = Path(args.original_dataset_root) / (original_train_args.get('precomputed_split_dir') or "splits")
            val_indices = torch.load(split_dir / "val_indices.pt")
            original_val_dataset = torch.utils.data.Subset(original_dataset, val_indices)
        else:
            raise ValueError(f"Unsupported original split mode: {original_split_mode}")

        original_val_loader = PyGDataLoader(original_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        logger.info(f"Original validation DataLoader created with {len(original_val_dataset)} samples.")

    except Exception as e:
        logger.error(f"Error loading original validation data: {e}")
        args.eval_original_val_every_n_epochs = 0 # Disable if loading fails
        if file_handler: file_handler.flush()

# --- Model Loading and Preparation ---
logger.info(f"Loading pre-trained model from checkpoint: {args.pretrained_checkpoint_path}")
if not Path(args.pretrained_checkpoint_path).exists():
    logger.error(f"Pretrained checkpoint not found: {args.pretrained_checkpoint_path}")
    sys.exit(1)

checkpoint = torch.load(args.pretrained_checkpoint_path, map_location=device)

# --- Robustly infer hidden_channels and model_version from checkpoint ---
try:
    hidden_channels_from_checkpoint = checkpoint['model_state_dict']['conv1.bias'].shape[0]
    logger.info(f"Inferred hidden_channels={hidden_channels_from_checkpoint} from checkpoint conv1.bias layer.")
    hidden_channels = hidden_channels_from_checkpoint # Override value from args file
except KeyError as e:
    logger.warning(f"Could not infer hidden_channels from checkpoint state_dict (KeyError: {e}). Falling back to original_train_args.")
    hidden_channels = original_train_args.get('hidden_channels', 128) # Default if not in args either

# Infer model version based on keys present in the state_dict
inferred_model_version = None
if 'attention_pool.gate_nn.0.weight' in checkpoint['model_state_dict'] and 'attention_pool.gate_nn.2.weight' in checkpoint['model_state_dict']:
    inferred_model_version = '2.0'
    logger.info("Inferred model version '2.0' based on 'attention_pool' keys in checkpoint.")
elif 'mha_pool.in_proj_weight' in checkpoint['model_state_dict'] and 'mha_pool.out_proj.weight' in checkpoint['model_state_dict']:
    inferred_model_version = '2.5'
    logger.info("Inferred model version '2.5' based on 'mha_pool' keys in checkpoint.")

if inferred_model_version:
    pretrained_model_version = inferred_model_version
else:
    # Fallback to previous logic if keys are not definitive
    pretrained_model_version = checkpoint.get('model_version', original_train_args.get('model_version', '2.0'))
    logger.warning(f"Could not definitively infer model version from checkpoint keys. Using fallback: {pretrained_model_version}")
# --- End of robust inference ---

# pretrained_model_version = checkpoint.get('model_version', original_train_args.get('model_version', '2.0')) # Fallback
# hidden_channels = original_train_args.get('hidden_channels', 128) # Get from original args
num_heads_original = original_train_args.get('num_heads', 4)       # Get from original args

# Determine original_num_classes for model instantiation
# This should be the number of classes the checkpoint model was trained on.
# Try to get it from original_train_args if it had num_classes directly, or infer from its dataset.
# The checkpoint itself doesn't store num_classes directly, but original_train_args should reflect the setup.
# If original_val_loader is setup, we already have original_num_classes.
if original_num_classes is None:
    # This case should ideally not be hit if original_dataset_root is processed correctly
    # Or if original_train_args has a 'num_classes' field from a prior run.
    logger.warning("Could not determine original_num_classes precisely before model init. Estimating from checkpoint output layer.")
    # A bit risky: try to infer from classifier output size if possible, otherwise default
    try:
        original_num_classes = checkpoint['model_state_dict']['classifier.bias'].shape[0]
        logger.info(f"Inferred original_num_classes={original_num_classes} from checkpoint classifier.")
    except KeyError:
        logger.error("Cannot infer original_num_classes from checkpoint. Please ensure original_training_args.json is accurate or provides it.")
        sys.exit(1)


logger.info(f"Instantiating model version GenePhenAIv{pretrained_model_version} with hidden_channels={hidden_channels} and original_num_classes={original_num_classes}")

model_args = {'in_channels': ddd_data_list[0].num_node_features, # Use DDD's input features
              'hidden_channels': hidden_channels,
              'out_channels': original_num_classes} # Instantiate with ORIGINAL out_channels first

if pretrained_model_version == '2.5':
    model_args['num_heads'] = num_heads_original

try:
    if pretrained_model_version == '2.0':
        model = GenePhenAIv2_0(**model_args).to(device)
    elif pretrained_model_version == '2.5':
        model = GenePhenAIv2_5(**model_args).to(device)
    else:
        raise ValueError(f"Unsupported model version in checkpoint: {pretrained_model_version}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Pre-trained model loaded successfully.")
    if file_handler: file_handler.flush()
except Exception as e:
    logger.error(f"Error loading model state_dict: {e}")
    sys.exit(1)

# Save original classifier weights and architecture
original_classifier_state_dict = {k: v.clone() for k, v in model.classifier.state_dict().items()}
original_classifier_in_features = model.classifier.in_features
original_classifier_out_features = model.classifier.out_features

# Log trainable parameters before custom freezing
trainable_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Trainable parameters before custom freezing: {trainable_params_before:,}")

# Freeze GCN layers if requested
if args.freeze_gcn_layers:
    logger.info("Freezing GCN layers (conv1, bn1, conv2, bn2, conv3, bn3).")
    for name, param in model.named_parameters():
        if name.startswith('conv') or name.startswith('bn'):
            param.requires_grad = False
            logger.debug(f"Froze GCN parameter: {name}") # DEBUG log for individual params

    # Explicitly set track_running_stats to False for GCN BatchNorm modules
    # This ensures their running_mean and running_var are not updated during fine-tuning
    # even when model is in .train() mode.
    if hasattr(model, 'bn1') and isinstance(model.bn1, torch.nn.Module):
        model.bn1.track_running_stats = False
        logger.info("Set model.bn1.track_running_stats = False")
    if hasattr(model, 'bn2') and isinstance(model.bn2, torch.nn.Module):
        model.bn2.track_running_stats = False
        logger.info("Set model.bn2.track_running_stats = False")
    if hasattr(model, 'bn3') and isinstance(model.bn3, torch.nn.Module):
        model.bn3.track_running_stats = False
        logger.info("Set model.bn3.track_running_stats = False")
else:
    logger.info("GCN layers will be fine-tuned (NOT frozen).")

# Freeze MHA layer if requested (for model v2.5)
if pretrained_model_version == '2.5':
    if args.freeze_mha_layer:
        logger.info("Freezing MHA pooling layer (mha_pool) for model v2.5.")
        for name, param in model.named_parameters():
            if name.startswith('mha_pool'):
                param.requires_grad = False
                logger.debug(f"Froze MHA parameter: {name}") # DEBUG log for individual params
    else:
        logger.info("MHA pooling layer (mha_pool) will be fine-tuned for model v2.5 (NOT frozen).")


# Replace classifier for DDD task
model.classifier = torch.nn.Linear(original_classifier_in_features, ddd_num_classes).to(device)
logger.info(f"Replaced classifier head for DDD task. New out_features: {ddd_num_classes}")
logger.info("Updated model architecture for fine-tuning:")
logger.info(model)

total_params_ft = sum(p.numel() for p in model.parameters())
trainable_params_ft = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Total parameters (fine-tuning model): {total_params_ft:,}")
logger.info(f"Trainable parameters (fine-tuning model, after freezing and classifier swap): {trainable_params_ft:,}")


# Optimizer for fine-tuning (only for trainable parameters)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=args.learning_rate,
                  weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss()
logger.info("Optimizer and Loss function defined for fine-tuning.")
if file_handler: file_handler.flush()

# --- Metrics Calculation (Copied from training.py, ensure k_vals match) ---
@torch.no_grad()
def calculate_batch_metrics(out: torch.Tensor, target_y: torch.Tensor, k_vals: list[int] = [1, 10]) -> tuple[float, dict[int, float]]:
    batch_mrr = 0.0
    batch_top_k_acc = {k: 0.0 for k in k_vals}
    num_graphs = target_y.size(0)

    if num_graphs == 0: return batch_mrr, batch_top_k_acc
    try:
        _, sorted_indices = torch.sort(out, dim=1, descending=True)
        target_expanded = target_y.unsqueeze(1).expand_as(sorted_indices)
        target_match = (sorted_indices == target_expanded)
        indices = torch.nonzero(target_match, as_tuple=True)[1]

        if len(indices) == num_graphs: # Ensure all targets were found
            ranks = indices + 1
            reciprocal_ranks = 1.0 / ranks.float()
            batch_mrr = torch.mean(reciprocal_ranks).item()
            top_k_correct = {k: torch.sum(ranks <= k).item() for k in k_vals}
            batch_top_k_acc = {k: (correct / num_graphs) for k, correct in top_k_correct.items()}
    except Exception as e:
        logger.debug(f"Error in calculate_batch_metrics: {e}. Out shape: {out.shape}, Target shape: {target_y.shape}")
    return batch_mrr, batch_top_k_acc

# --- Evaluation Function (Adaptable) ---
@torch.no_grad()
def evaluate(loader, current_model, current_criterion, current_gene_to_label_map, eval_task_name="Eval", k_vals_eval=[1, 5, 10, 20], max_batches=None):
    current_model.eval()
    total_loss = 0
    total_samples = 0
    sum_reciprocal_ranks = 0.0
    top_k_correct = {k: 0 for k in k_vals_eval}
    processed_batches = 0 # Initialize counter for max_batches logic
    
    start_time = time.time()
    logger.info(f"Starting {eval_task_name}...")
    if file_handler: file_handler.flush()

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        
        # Target Y creation specific to the dataset being evaluated
        target_y_list = []
        valid_batch = True
        # LMDB batches have 'gene' attribute (list of gene strings)
        # ListDataset batches have 'gene_symbol' attribute (now a list of gene symbols due to PyGDataLoader)
        if hasattr(batch, 'gene'): # Original LMDB validation
            if not batch.gene: # Handle empty gene list case
                 logger.warning(f"{eval_task_name} Batch {i} has empty 'gene' attribute. Skipping.")
                 valid_batch = False
            else:
                for g_idx, g in enumerate(batch.gene):
                    if g not in current_gene_to_label_map:
                        logger.warning(f"{eval_task_name} Batch {i}, item {g_idx}: Unknown gene '{g}'. Skipping item.")
                        # valid_batch = False; break # Option to skip whole batch
                    else:
                        target_y_list.append(current_gene_to_label_map[g])
        elif hasattr(batch, 'gene_symbol'): # DDD ListDataset after PyGDataLoader
            if not isinstance(batch.gene_symbol, list):
                logger.warning(f"{eval_task_name} Batch {i}: batch.gene_symbol is not a list as expected. Type: {type(batch.gene_symbol)}. Skipping.")
                valid_batch = False
            elif not batch.gene_symbol: # Empty list
                logger.warning(f"{eval_task_name} Batch {i}: batch.gene_symbol is an empty list. Skipping.")
                valid_batch = False
            else:
                for gs_idx, gs_item in enumerate(batch.gene_symbol):
                    if gs_item not in current_gene_to_label_map:
                        logger.warning(f"{eval_task_name} Batch {i}, item {gs_idx}: Unknown gene_symbol '{gs_item}'. Skipping item.")
                        # valid_batch = False; break # Option to skip whole batch
                    else:
                        target_y_list.append(current_gene_to_label_map[gs_item])
        else: # Fallback for individual Data objects (should not happen with PyGDataLoader)
            if isinstance(batch, list): # list of Data from ListDataset not collated by PyGDataLoader
                for data_item_idx, data_item in enumerate(batch):
                    gs = getattr(data_item, 'gene_symbol', None)
                    if gs and gs in current_gene_to_label_map:
                        target_y_list.append(current_gene_to_label_map[gs])
                    else:
                        logger.warning(f"{eval_task_name} Data item {data_item_idx} in batch {i} has missing/unknown gene_symbol. Skipping.")
                        valid_batch = False; break 
            else: # Single Data object
                gs = getattr(batch, 'gene_symbol', None)
                if gs and gs in current_gene_to_label_map:
                     target_y_list.append(current_gene_to_label_map[gs])
                else:
                    logger.warning(f"{eval_task_name} Batch {i} (single item) has missing/unknown gene_symbol. Skipping.")
                    valid_batch = False


        if not valid_batch or not target_y_list:
            logger.debug(f"{eval_task_name} Batch {i} resulted in no valid targets. Skipping.")
            continue
        
        target_y = torch.tensor(target_y_list, dtype=torch.long, device=device)

        if not hasattr(batch, 'batch') or batch.batch is None:
             # If single graph, create dummy batch. For pre-batched lists, this might be an issue.
             if not isinstance(batch, list) and hasattr(batch, 'x'): # single Data object
                 current_batch_vector = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
             else: # If it's a list from ListDataset that wasn't properly batched by PyG, this is harder.
                 logger.warning(f"{eval_task_name} Batch {i} is missing 'batch' attribute and is not a single Data obj. Skipping.")
                 continue
        else:
            current_batch_vector = batch.batch

        try:
            out = current_model(batch.x, batch.edge_index, current_batch_vector)
            loss = current_criterion(out, target_y)
            
            samples_in_batch = target_y.size(0) # Number of actual samples with targets
            total_loss += loss.item() * samples_in_batch
            total_samples += samples_in_batch
            
            batch_mrr, batch_top_k_acc = calculate_batch_metrics(out, target_y, k_vals=k_vals_eval)
            sum_reciprocal_ranks += batch_mrr * samples_in_batch
            for k_val in k_vals_eval:
                top_k_correct[k_val] += batch_top_k_acc[k_val] * samples_in_batch
        except Exception as e:
            logger.error(f"Error during {eval_task_name} forward pass or metric calculation in batch {i}: {e}", exc_info=True)
            continue

        processed_batches += 1 # Increment after successful processing of a batch
        if max_batches is not None and processed_batches >= max_batches:
            logger.info(f"{eval_task_name} reached evaluation batch limit ({max_batches}). Stopping evaluation for this run.")
            if file_handler: file_handler.flush()
            break

    avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
    final_mrr = sum_reciprocal_ranks / total_samples if total_samples > 0 else 0.0
    final_top_k_acc = {k: (correct / total_samples if total_samples > 0 else 0.0) for k, correct in top_k_correct.items()}
    eval_time = time.time() - start_time

    top_k_log_str = ", ".join([f"Top-{k}: {acc:.4f}" for k, acc in final_top_k_acc.items()])
    logger.info(f"Finished {eval_task_name}. Avg Loss: {avg_loss:.4f}, MRR: {final_mrr:.4f}, {top_k_log_str}. Time: {eval_time:.2f}s")
    if file_handler: file_handler.flush()
    return avg_loss, final_mrr, final_top_k_acc


# --- Initial Evaluation on Original Validation Set (Baseline) ---
if args.eval_original_val_every_n_epochs > 0 and original_val_loader and original_gene_to_class_idx:
    logger.info("--- Performing Initial Baseline Evaluation on Original Validation Set (Before Fine-tuning) ---")
    # Ensure model is in eval mode and correct classifier is set for original task
    model.eval() # Set to eval mode
    # Temporarily set to original classifier for this baseline evaluation
    # The model currently has the DDD classifier; we need to swap for the original one.
    # Save DDD classifier state if it was already changed (it was, right after loading checkpoint)
    current_ddd_classifier_state_dict_baseline = {k: v.clone() for k, v in model.classifier.state_dict().items()}

    model.classifier = torch.nn.Linear(original_classifier_in_features, original_classifier_out_features).to(device)
    model.classifier.load_state_dict(original_classifier_state_dict) # Load original classifier weights
    
    baseline_orig_loss, baseline_orig_mrr, baseline_orig_top_k = evaluate(
        original_val_loader, 
        model, 
        criterion, # Can use the same criterion, or define one for original if different (e.g. class weights)
        original_gene_to_class_idx, 
        eval_task_name="Baseline Original Val", 
        max_batches=args.original_eval_max_batches
    )
    logger.info(f"Baseline Original Validation: Avg Loss: {baseline_orig_loss:.4f}, MRR: {baseline_orig_mrr:.4f}, "
                f"Top-1: {baseline_orig_top_k.get(1,0):.4f}, Top-10: {baseline_orig_top_k.get(10,0):.4f}")

    # Restore DDD classifier before starting fine-tuning loop
    model.classifier = torch.nn.Linear(original_classifier_in_features, ddd_num_classes).to(device)
    model.classifier.load_state_dict(current_ddd_classifier_state_dict_baseline)
    logger.info("Restored DDD classifier for fine-tuning.")
    if file_handler: file_handler.flush()
# --- End of Initial Baseline Evaluation ---



# --- Fine-tuning Loop ---
logger.info(f"Starting fine-tuning loop for {args.epochs} epochs...")
best_ddd_val_mrr = 0.0

for epoch in range(1, args.epochs + 1):
    model.train() # Set model to training mode
    epoch_train_loss = 0
    epoch_total_samples = 0
    epoch_start_time = time.time()

    logger.info(f"--- Fine-tuning Epoch {epoch}/{args.epochs} ---")
    if file_handler: file_handler.flush()

    for i, batch_data in enumerate(ddd_train_loader):
        batch_data = batch_data.to(device)
        optimizer.zero_grad()

        # DDD target_y comes from batch_data.y (already prepared by create_ddd_finetuning_set.py)
        target_y = batch_data.y.squeeze(-1) # Ensure it's [batch_size]

        if not hasattr(batch_data, 'batch') or batch_data.batch is None:
            logger.error(f"DDD Train Batch {i} missing 'batch' attribute. Skipping.")
            continue
        
        try:
            out = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            loss = criterion(out, target_y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * target_y.size(0)
            epoch_total_samples += target_y.size(0)

            if (i + 1) % 20 == 0: # Log training progress
                batch_mrr, batch_top_k = calculate_batch_metrics(out, target_y)
                logger.info(f"  Epoch {epoch} | Train Batch {i+1}/{len(ddd_train_loader)} | Loss: {loss.item():.4f} | MRR: {batch_mrr:.3f} | Top-1: {batch_top_k.get(1,0):.3f}")
                if file_handler: file_handler.flush()
        except Exception as e:
            logger.error(f"Error during DDD training batch {i}: {e}", exc_info=True)
            continue
            
    avg_epoch_train_loss = epoch_train_loss / epoch_total_samples if epoch_total_samples > 0 else float('nan')
    logger.info(f"Epoch {epoch} DDD Training: Avg Loss: {avg_epoch_train_loss:.4f}, Time: {time.time() - epoch_start_time:.2f}s")

    # Evaluate on DDD Validation set (now full DDD set)
    logger.info(f"Evaluating on full DDD dataset for Epoch {epoch}...")
    ddd_val_loss, ddd_val_mrr, ddd_val_top_k = evaluate(ddd_eval_loader, model, criterion, ddd_gene_to_label, "DDD Eval")

    if ddd_val_mrr > best_ddd_val_mrr:
        best_ddd_val_mrr = ddd_val_mrr
        checkpoint_name = f'finetuned_ddd_best_model_epoch_{epoch}.pt'
        checkpoint_path_save = output_path / checkpoint_name
        torch.save({
            'epoch': epoch,
            'model_version': pretrained_model_version,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ddd_val_mrr': ddd_val_mrr,
            'ddd_val_top_k_accuracy': ddd_val_top_k,
            'original_training_args': original_train_args,
            'finetuning_args': vars(args)
        }, checkpoint_path_save)
        logger.info(f"Saved new best DDD fine-tuned model to {checkpoint_path_save} (MRR: {best_ddd_val_mrr:.4f})")

    # Evaluate on Original Validation set (if enabled and loader available)
    if args.eval_original_val_every_n_epochs > 0 and epoch % args.eval_original_val_every_n_epochs == 0 and original_val_loader:
        logger.info(f"Evaluating on Original validation set for Epoch {epoch} (Using original classifier)...")
        
        # 1. Save current DDD classifier state
        current_ddd_classifier_state_dict = {k: v.clone() for k, v in model.classifier.state_dict().items()}
        
        # 2. Reconfigure model.classifier for original task
        model.classifier = torch.nn.Linear(original_classifier_in_features, original_classifier_out_features).to(device)
        model.classifier.load_state_dict(original_classifier_state_dict)
        
        # 3. Evaluate
        _, orig_val_mrr, orig_val_top_k = evaluate(original_val_loader, model, criterion, original_gene_to_class_idx, "Original Val", max_batches=args.original_eval_max_batches)
        # We mostly care about MRR and Top-K here, loss might be on a different scale/meaning.
        
        # 4. Restore DDD classifier
        model.classifier = torch.nn.Linear(original_classifier_in_features, ddd_num_classes).to(device)
        model.classifier.load_state_dict(current_ddd_classifier_state_dict)
        logger.info(f"Epoch {epoch} Original Validation: MRR: {orig_val_mrr:.4f}, Top-1: {orig_val_top_k.get(1,0):.4f}, Top-10: {orig_val_top_k.get(10,0):.4f}")

    if file_handler: file_handler.flush()

logger.info("Fine-tuning finished.")
if file_handler: file_handler.flush() 