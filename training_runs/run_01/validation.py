import argparse
import logging
import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader
from pathlib import Path
import time
import numpy as np
import csv  # Added for CSV output

# Assuming the script is run from the workspace root
from src.simulation.phenotype_simulation.create_hpo_dataset import HPOGraphDataset
from training.datasets.lmdb_hpo_dataset import LMDBHPOGraphDataset # Needed to load training metadata for gene mapping
from training.models.models import GenePhenAIv2_0 # Make sure the class name matches exactly

# --- Helper function for calculating batch metrics (copied from training.py) ---
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
            # logger.warning("Rank finding mismatch in calculate_batch_metrics")
            pass

    except Exception as e:
        # Log internally or handle error if needed, for now return 0s
        # logger.error(f"Error in calculate_batch_metrics: {e}")
        pass

    return batch_mrr, batch_top_k_acc
# ------------------------------------------------------------------------------

# --- New helper function to get individual sample rankings ---
@torch.no_grad()
def get_sample_rankings(out: torch.Tensor, target_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Get ranks and scores for individual samples in a batch.
    
    Args:
        out: Model output logits/scores (shape: [batch_size, num_classes])
        target_y: Target class indices (shape: [batch_size])
        
    Returns:
        Tuple of (ranks, scores) for each sample in the batch
        ranks: Tensor of ranks for correct class (1-indexed)
        scores: Tensor of scores (logits) for the correct class
    """
    # Get scores for the target classes
    target_scores = out.gather(1, target_y.unsqueeze(1)).squeeze(1)
    
    # Sort outputs to get ranks
    sorted_values, sorted_indices = torch.sort(out, dim=1, descending=True)
    target_expanded = target_y.unsqueeze(1).expand_as(sorted_indices)
    target_match = (sorted_indices == target_expanded)
    
    # Get positions (0-indexed) where targets appear
    indices = torch.nonzero(target_match, as_tuple=True)[1]
    
    # Convert to ranks (1-indexed)
    ranks = indices + 1
    
    return ranks, target_scores
# ------------------------------------------------------------------------------

# --- Evaluation Function (adapted from training.py) ---
@torch.no_grad()
def evaluate(loader, model, criterion, gene_to_class_idx, device, k_vals=[1, 5, 10, 20], 
             csv_output_path=None, class_idx_to_gene=None):
    """Evaluates the model on the provided data loader."""
    model.eval()
    total_loss = 0
    total_samples = 0
    processed_batches = 0
    sum_reciprocal_ranks = 0.0
    top_k_correct = {k: 0 for k in k_vals}
    
    # For detailed CSV output
    all_results = []
    
    # Get mapping from class index to gene name for CSV output
    if class_idx_to_gene is None and csv_output_path is not None:
        class_idx_to_gene = {idx: gene for gene, idx in gene_to_class_idx.items()}

    start_time = time.time()
    logger.info(f"Starting evaluation... Calculating Top-K for k={k_vals} and MRR.")
    if file_handler: file_handler.flush()
    
    # Calculate total number of samples for progress reporting
    total_dataset_size = len(loader.dataset)
    samples_processed = 0
    
    for i, batch in enumerate(loader):
        batch_start_time = time.time()
        batch = batch.to(device)
        batch_size = batch.num_graphs

        # --- Target Creation ---
        # HPOGraphDataset collates 'gene' attribute into a list for the batch
        if not hasattr(batch, 'gene') or not isinstance(batch.gene, list):
            logger.warning(f"Eval Batch {i} is missing 'gene' list attribute or has wrong format. Skipping.")
            continue
        try:
            # Ensure all genes in the batch are known by the mapping loaded from training
            target_y = torch.tensor([gene_to_class_idx[g] for g in batch.gene], dtype=torch.long, device=device)
        except KeyError as e:
            logger.warning(f"Eval Batch {i} contains unknown gene: {e}. Gene was not in the training set's metadata. Skipping batch.")
            continue
        except Exception as e:
            logger.warning(f"Error mapping genes to indices in eval batch {i}: {e}. Skipping batch.")
            continue
        # --------------------

        # Ensure batch vector exists (DataLoader should add it)
        if not hasattr(batch, 'batch') or batch.batch is None:
             logger.warning(f"Eval Batch {i} is missing 'batch' attribute required for pooling. Skipping.")
             continue

        try:
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, target_y)

            samples_in_batch = batch.num_graphs
            total_loss += loss.item() * samples_in_batch
            total_samples += samples_in_batch
            samples_processed += samples_in_batch
            processed_batches += 1

            # --- Calculate Ranking Metrics using helper ---
            batch_mrr, batch_top_k_acc = calculate_batch_metrics(out, target_y, k_vals=k_vals)

            # Accumulate metrics
            sum_reciprocal_ranks += batch_mrr * samples_in_batch
            for k in k_vals:
                top_k_correct[k] += batch_top_k_acc[k] * samples_in_batch
            # --------------------------------
            
            # --- Get individual sample rankings for CSV output ---
            if csv_output_path is not None:
                ranks, scores = get_sample_rankings(out, target_y)
                
                # Collect detailed results for each sample
                for j in range(samples_in_batch):
                    gene_name = batch.gene[j]
                    phenotype_ids = batch.phenotype_ids[j] if hasattr(batch, 'phenotype_ids') else []
                    phenotype_str = ','.join(phenotype_ids) if phenotype_ids else ''
                    
                    all_results.append({
                        'GENE': gene_name,
                        'PHENOTYPES': phenotype_str,
                        'RANK': ranks[j].item(),
                        'SCORE': scores[j].item()
                    })
            # --------------------------------

            batch_eval_time = time.time() - batch_start_time
            avg_rank_batch = 1.0 / batch_mrr if batch_mrr > 0 else float('inf') # Estimate avg rank from MRR
            
            # More detailed progress reporting
            progress_percent = (samples_processed / total_dataset_size) * 100
            logger.info(f"Eval Progress: {samples_processed}/{total_dataset_size} samples ({progress_percent:.1f}%) | "
                        f"Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f} | "
                        f"Avg Rank: {avg_rank_batch:.2f} | Time: {batch_eval_time:.3f}s")
            if file_handler: file_handler.flush()

        except Exception as e:
             logger.error(f"Error during evaluation forward pass or metric calculation in batch {i}: {e}")
             continue # Skip problematic batches during evaluation

    # Calculate final metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    final_mrr = sum_reciprocal_ranks / total_samples if total_samples > 0 else 0
    final_top_k_acc = {k: (correct / total_samples if total_samples > 0 else 0) for k, correct in top_k_correct.items()}
    eval_time = time.time() - start_time

    # Log summary
    top_k_log_str = ", ".join([f"Top-{k}: {acc:.4f}" for k, acc in final_top_k_acc.items()])
    logger.info(f"Finished evaluation. Avg Loss: {avg_loss:.4f}, MRR: {final_mrr:.4f}, {top_k_log_str} over {processed_batches} batches.")
    if file_handler: file_handler.flush()
    
    # Write detailed CSV output if path provided
    if csv_output_path is not None and all_results:
        try:
            with open(csv_output_path, 'w', newline='') as csvfile:
                fieldnames = ['GENE', 'PHENOTYPES', 'RANK', 'SCORE']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in all_results:
                    writer.writerow(result)
            logger.info(f"Detailed ranking results written to {csv_output_path}")
        except Exception as e:
            logger.error(f"Error writing CSV output to {csv_output_path}: {e}")
    
    # Return all metrics
    return avg_loss, final_mrr, final_top_k_acc, eval_time
# ---------------------------------------------------

# --- Main Script Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a pre-trained Phenotype GNN Model on a .pt dataset")

    # --- Path Arguments ---
    parser.add_argument('--validation_data_root', type=str, required=True,
                        help='Path to the ROOT directory containing the processed validation dataset (e.g., "data/real_validation"). '
                             'It should contain a "processed" subdirectory with batch_*.pt and index.pt files.')
    parser.add_argument('--training_output_dir', type=str, required=True,
                        help='Path to the output directory of the *training* run (e.g., "training_output/run_xyz") '
                             'which contains args.json and the checkpoint file.')
    parser.add_argument('--training_lmdb_dir', type=str, required=True,
                        help='Path to the root directory of the *LMDB dataset used for training*. '
                             'Needed to load the gene list for consistent class mapping.')
    parser.add_argument('--checkpoint_filename', type=str, default='checkpoint_epoch_1_part_4.pt',
                        help='Filename of the checkpoint to load from the training_output_dir.')
    parser.add_argument('--csv_output', type=str, default=None,
                        help='Path to save detailed CSV output with gene rankings')

    # --- Evaluation Arguments ---
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for validation.')

    # --- System Arguments ---
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu).')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level.')
    parser.add_argument('--output_log_file', type=str, default='validation_log.txt', help='File to save validation logs.')

    args = parser.parse_args()

    # --- Setup Logging ---
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.output_log_file),
            logging.StreamHandler() # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)

    # Get file handler for flushing
    file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
            break

    logger.info("Starting validation script with arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    if file_handler: file_handler.flush()

    # --- Device Setup ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Training Config ---
    training_args_path = Path(args.training_output_dir) / 'args.json'
    try:
        with open(training_args_path, 'r') as f:
            training_args = json.load(f)
        logger.info(f"Loaded training arguments from {training_args_path}")
    except FileNotFoundError:
        logger.error(f"Training arguments file not found: {training_args_path}. Cannot determine model hyperparameters.")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading training arguments: {e}")
        exit(1)

    hidden_channels = training_args.get('hidden_channels')
    if hidden_channels is None:
        logger.error("Could not find 'hidden_channels' in loaded training arguments.")
        exit(1)

    # --- Load Validation Dataset (.pt format) ---
    logger.info(f"Loading validation dataset from: {args.validation_data_root}")
    try:
        # HPOGraphDataset expects the root directory containing 'processed/'
        validation_dataset = HPOGraphDataset(root_dir=args.validation_data_root)
        if len(validation_dataset) == 0:
             logger.error("Validation dataset loaded but contains 0 samples. Check index.pt and batch files.")
             exit(1)
        logger.info(f"Validation dataset loaded successfully with {len(validation_dataset)} samples.")
        num_node_features = validation_dataset.get(0).num_node_features
        logger.info(f"Inferred input node features: {num_node_features}")
    except FileNotFoundError:
        logger.error(f"Validation dataset index or batch files not found in {Path(args.validation_data_root) / 'processed'}.")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading validation dataset: {e}")
        exit(1)


    # --- Load Gene Mapping from Original Training Data ---
    logger.info(f"Loading gene list from training LMDB metadata: {args.training_lmdb_dir}")
    try:
        # Instantiate LMDB dataset just to access metadata
        training_dataset_meta = LMDBHPOGraphDataset(root_dir=args.training_lmdb_dir, readonly=True)
        training_genes = training_dataset_meta.metadata.get('genes')
        if not training_genes:
            logger.error(f"Could not find 'genes' list in metadata of training LMDB dataset at {args.training_lmdb_dir}.")
            exit(1)
        gene_to_class_idx = {gene: i for i, gene in enumerate(training_genes)}
        class_idx_to_gene = {i: gene for i, gene in enumerate(training_genes)}
        num_classes = len(gene_to_class_idx)
        logger.info(f"Created gene-to-index mapping for {num_classes} classes based on training data.")
    except FileNotFoundError:
         logger.error(f"Training LMDB dataset not found at {args.training_lmdb_dir}. Cannot get gene mapping.")
         exit(1)
    except Exception as e:
        logger.error(f"Error loading metadata or creating gene map from training LMDB: {e}")
        exit(1)


    # --- Instantiate Model ---
    model = GenePhenAIv2_0(
        in_channels=num_node_features,
        hidden_channels=hidden_channels, # From loaded training args
        out_channels=num_classes         # From training gene list
    ).to(device)
    logger.info("Model instantiated:")
    # logger.info(model) # Optional: Log model structure

    # --- Load Checkpoint ---
    checkpoint_path = Path(args.training_output_dir) / args.checkpoint_filename
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Successfully loaded model weights from checkpoint (Epoch {checkpoint.get('epoch', 'N/A')}, Batch {checkpoint.get('batch', 'N/A')})")
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        exit(1)

    # --- Create DataLoader ---
    val_loader = PyGDataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle for validation
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    logger.info("Validation DataLoader created.")

    # --- Run Evaluation ---
    criterion = torch.nn.CrossEntropyLoss() # Same criterion as training
    val_loss, val_mrr, val_top_k, val_time = evaluate(
        loader=val_loader,
        model=model,
        criterion=criterion,
        gene_to_class_idx=gene_to_class_idx,
        class_idx_to_gene=class_idx_to_gene,
        device=device,
        csv_output_path=args.csv_output
        # k_vals default to [1, 5, 10, 20] in evaluate function
    )

    # --- Log Final Results ---
    logger.info(f"--- Validation Results ({args.checkpoint_filename}) ---")
    logger.info(f"  Dataset: {args.validation_data_root}")
    logger.info(f"  Avg Loss: {val_loss:.4f}")
    logger.info(f"  MRR: {val_mrr:.4f}")
    for k, acc in val_top_k.items():
         logger.info(f"  Top-{k} Accuracy: {acc:.4f}")
    logger.info(f"  Evaluation Time: {val_time:.2f}s")
    if file_handler: file_handler.flush()

    logger.info(f"Validation finished. Logs saved to {args.output_log_file}")
    if file_handler: file_handler.flush()