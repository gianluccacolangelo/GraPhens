import argparse
import importlib
import json
import logging
import os
import time
from pathlib import Path
from collections import defaultdict
import math # Added for ceiling division in batching

import torch
from torch_geometric.data import Batch, Data # Added Data import for loading graphs

# --- Arena Imports ---
# Import base class and specific evaluators dynamically later
from validation_arena.evaluators.base_evaluator import BaseModelEvaluator
# Import utility function
from validation_arena.utils import calculate_batch_metrics

# --- Training Data Import ---
# Required for loading gene list from training metadata
# Adjust the import path as necessary for your project structure
try:
    from training.datasets.lmdb_hpo_dataset import LMDBHPOGraphDataset
except ImportError:
    logging.basicConfig(level=logging.ERROR) # Basic config for early error
    logging.error("CRITICAL: Could not import LMDBHPOGraphDataset from training.datasets. Adjust path if needed.")
    # Exit if this core dependency is missing
    exit(1)

# --- Global Logger ---
# Will be configured properly in main()
logger = logging.getLogger(__name__)

# --- Evaluator Registry (Simple Factory) ---
# Maps evaluator type strings (from config) to their classes
# Add new evaluators here as they are implemented
EVALUATOR_REGISTRY = {}

def register_evaluator(name):
    """Decorator to register evaluator classes."""
    def decorator(cls):
        # Check inheritance here or later? Later is safer for import cycles.
        # if not issubclass(cls, BaseModelEvaluator):
        #     raise ValueError(f"Class {cls.__name__} must inherit from BaseModelEvaluator")
        EVALUATOR_REGISTRY[name] = cls
        logger.debug(f"Registered evaluator: '{name}' -> {cls.__name__}")
        return cls
    return decorator

# --- Dynamically Import and Register Known Evaluators ---
# This approach allows adding new evaluators without modifying the registry dict directly
# It assumes evaluator files are in the 'evaluators' directory and classes
# are decorated with @register_evaluator('TypeName')

def load_evaluators():
    evaluator_dir = Path(__file__).parent / "evaluators"
    logger.debug(f"Searching for evaluators in: {evaluator_dir}")
    for filename in os.listdir(evaluator_dir):
        if filename.endswith(".py") and not filename.startswith(("__", "base_")):
            module_name = f"validation_arena.evaluators.{filename[:-3]}"
            try:
                imported_module = importlib.import_module(module_name)
                logger.debug(f"Successfully imported module: {module_name}")
                # Now explicitly check for BaseModelEvaluator inheritance after import
                for name, obj in inspect.getmembers(imported_module):
                    if inspect.isclass(obj) and issubclass(obj, BaseModelEvaluator) and obj is not BaseModelEvaluator:
                         # Optional: could re-trigger registration here if needed,
                         # but decorator pattern should handle it.
                         pass # Already registered by decorator
            except ImportError as e:
                logger.warning(f"Could not import evaluator module {module_name}: {e}")
            except Exception as e:
                 logger.warning(f"Error processing module {module_name}: {e}")

# Import inspect for checking inheritance after dynamic import
import inspect

# Call this early to populate the registry
load_evaluators()

# Ensure PyG is registered if dynamic loading had issues (optional fallback)
try:
    from validation_arena.evaluators.pyg_evaluator import PyGModelEvaluator
    if 'PyG' not in EVALUATOR_REGISTRY:
        EVALUATOR_REGISTRY['PyG'] = PyGModelEvaluator
        logger.debug(f"Manually registered evaluator: 'PyG' -> PyGModelEvaluator")
except ImportError as e:
    logger.warning(f"Could not explicitly import PyGModelEvaluator: {e}. It might not be usable.")


# --- Main Arena Function ---
def run_arena(args):
    """Runs the validation arena for a single model with the given arguments."""
    logger.info("--- Starting Validation Arena --- ")
    overall_start_time = time.time()

    # --- 1. Validate Evaluator Choice ---
    if args.evaluator_type not in EVALUATOR_REGISTRY:
        logger.error(f"Unknown evaluator type: '{args.evaluator_type}'. Available: {list(EVALUATOR_REGISTRY.keys())}")
        return
    EvaluatorClass = EVALUATOR_REGISTRY[args.evaluator_type]
    logger.info(f"Using evaluator type: {args.evaluator_type} ({EvaluatorClass.__name__})")

    # --- 2. Construct Evaluator Configuration from Args ---
    evaluator_config = {
        "device": args.device,
        # Add arguments specific to the evaluator type
    }
    # Specific args for PyG, adapt or make more generic if needed
    if args.evaluator_type == 'PyG':
        pyg_required_args = ['model_class_fqn', 'checkpoint_path', 'training_args_path', 'num_node_features']
        missing_args = [arg for arg in pyg_required_args if not getattr(args, arg)]
        if missing_args:
            logger.error(f"Missing required arguments for PyG evaluator: {missing_args}")
            return
        evaluator_config.update({
            "model_class_fqn": args.model_class_fqn,
            "checkpoint_path": args.checkpoint_path,
            "training_args_path": args.training_args_path,
            "num_node_features": args.num_node_features,
            # We still need training_lmdb_dir for PyG evaluator to get num_classes
            "training_lmdb_dir": args.training_lmdb_dir,
        })
    # TODO: Add config construction for other evaluator types if necessary

    evaluator_name = Path(args.checkpoint_path).stem if args.checkpoint_path else args.evaluator_type # Use checkpoint name for uniqueness

    # --- 3. Load Validation Data Index ---
    validation_root = Path(args.validation_data_root)
    processed_dir = validation_root / 'processed'
    index_path = processed_dir / 'index.pt'
    logger.info(f"Loading validation index from: {index_path}")
    try:
        # index_data should be a list of tuples: (gene, case_id, file_idx)
        index_data = torch.load(index_path)
        if not index_data:
             logger.error("Validation index file loaded but is empty.")
             return
        num_total_samples = len(index_data)
        logger.info(f"Validation index loaded successfully with {num_total_samples} samples.")
    except FileNotFoundError:
        logger.error(f"Validation index file not found: {index_path}.")
        return
    except Exception as e:
        logger.error(f"Error loading validation index {index_path}: {e}")
        return

    # --- 4. Load Gene Mapping (from Training LMDB) ---
    # Still needed to map gene names from index to class indices for metrics
    logger.info(f"Loading gene list from training LMDB metadata: {args.training_lmdb_dir}")
    try:
        training_dataset_meta = LMDBHPOGraphDataset(root_dir=args.training_lmdb_dir, readonly=True)
        training_genes = training_dataset_meta.metadata.get('genes')
        if not training_genes:
            logger.error(f"Could not find 'genes' list in metadata of training LMDB at {args.training_lmdb_dir}.")
            return
        gene_to_class_idx = {gene: i for i, gene in enumerate(training_genes)}
        num_classes = len(gene_to_class_idx)
        logger.info(f"Created gene-to-index mapping for {num_classes} classes based on training data.")
        del training_dataset_meta # Release memory
    except FileNotFoundError:
         logger.error(f"Training LMDB dataset not found at {args.training_lmdb_dir}. Cannot get gene mapping.")
         return
    except Exception as e:
        logger.error(f"Error loading metadata or creating gene map from training LMDB {args.training_lmdb_dir}: {e}")
        return

    # --- 5. Define K values for metrics ---
    try:
        k_vals = [int(k.strip()) for k in args.k_values.split(',')]
        k_vals = sorted(list(set(k_vals))) # Ensure unique and sorted
        if not k_vals:
            raise ValueError("K values list cannot be empty.")
        logger.info(f"Using K values for Top-K calculation: {k_vals}")
    except ValueError as e:
        logger.error(f"Invalid k_values format '{args.k_values}'. Should be comma-separated integers (e.g., '1,5,10'). Error: {e}")
        return

    # --- 6. Evaluation Loop ---
    logger.info(f"--- Starting Evaluation for Model: {evaluator_name} ---")

    all_batch_mrr = []
    all_batch_top_k = {k: [] for k in k_vals}
    processed_samples = 0
    total_batches = math.ceil(num_total_samples / args.batch_size)

    # Instantiate evaluator using context manager for auto load/unload
    try:
        with EvaluatorClass(config=evaluator_config, name=evaluator_name) as evaluator:
            logger.info(f"Evaluator '{evaluator.name}' loaded successfully.")

            current_batch_graphs = []
            current_batch_targets = []

            for i, (gene, case_id, file_idx) in enumerate(index_data):
                # Load individual graph data
                graph_file_path = processed_dir / f"data_{file_idx}.pt"
                try:
                    graph_data = torch.load(graph_file_path)
                    # Ensure it's a PyG Data object (or adapt if format differs)
                    if not isinstance(graph_data, Data):
                        logger.warning(f"Loaded object from {graph_file_path} is not a PyG Data object (type: {type(graph_data)}). Skipping sample.")
                        continue

                    target_idx = gene_to_class_idx.get(gene)
                    if target_idx is None:
                        logger.warning(f"Gene '{gene}' from index (case {case_id}, file {file_idx}) not found in training gene map. Skipping sample.")
                        continue

                    current_batch_graphs.append(graph_data)
                    current_batch_targets.append(target_idx)

                except FileNotFoundError:
                    logger.warning(f"Graph file not found: {graph_file_path} (referenced by index entry: gene={gene}, case={case_id}, idx={file_idx}). Skipping sample.")
                    continue
                except Exception as e:
                    logger.error(f"Error loading graph file {graph_file_path}: {e}. Skipping sample.")
                    continue

                # Process batch when full or on the last sample
                if len(current_batch_graphs) == args.batch_size or (i == num_total_samples - 1 and current_batch_graphs):
                    batch_num = (processed_samples // args.batch_size) + 1
                    logger.debug(f"Processing Batch {batch_num}/{total_batches} (Size: {len(current_batch_graphs)})")

                    try:
                        # Create PyG Batch object
                        pyg_batch = Batch.from_data_list(current_batch_graphs)
                        target_y = torch.tensor(current_batch_targets, dtype=torch.long)

                        # Run prediction
                        # Ensure data is on the correct device inside predict method
                        output_logits = evaluator.predict(pyg_batch) # Returns tensor on CPU

                        # Calculate metrics for the batch
                        batch_mrr, batch_top_k_acc = calculate_batch_metrics(output_logits, target_y, k_vals)

                        all_batch_mrr.append(batch_mrr)
                        for k in k_vals:
                            all_batch_top_k[k].append(batch_top_k_acc[k])

                        processed_samples += len(current_batch_graphs)
                        logger.debug(f" Batch {batch_num}: MRR={batch_mrr:.4f}, TopK={batch_top_k_acc}")

                    except Exception as e:
                        logger.error(f"Error processing batch {batch_num}: {e}", exc_info=True)
                        # Skip metrics for this batch if processing fails

                    # Clear batch lists for next iteration
                    current_batch_graphs = []
                    current_batch_targets = []

    except Exception as e:
        logger.error(f"Failed to initialize or run evaluator '{evaluator_name}': {e}", exc_info=True)
        return # Cannot proceed if evaluator fails

    # --- 7. Calculate and Report Final Metrics ---
    logger.info("--- Evaluation Complete --- ")

    if not all_batch_mrr:
        logger.warning("No batches were successfully processed. Cannot calculate final metrics.")
    else:
        final_mrr = sum(all_batch_mrr) / len(all_batch_mrr)
        final_top_k = {k: (sum(all_batch_top_k[k]) / len(all_batch_top_k[k])) for k in k_vals}

        logger.info(f"Evaluation Summary for Model: {evaluator_name}")
        logger.info(f" Total Samples Processed: {processed_samples} / {num_total_samples}")
        logger.info(f" Average MRR: {final_mrr:.4f}")
        for k in sorted(final_top_k.keys()):
            logger.info(f" Average Top-{k} Accuracy: {final_top_k[k]:.4f}")

        # Save results to a simple text file
        results_file_path = Path(args.output_dir) / f"results_{evaluator_name}.txt"
        try:
            with open(results_file_path, 'w') as f:
                f.write(f"Evaluation Summary for Model: {evaluator_name}\n")
                f.write(f"Evaluator Type: {args.evaluator_type}\n")
                if args.checkpoint_path: f.write(f"Checkpoint: {args.checkpoint_path}\n")
                f.write(f"Validation Data: {args.validation_data_root}\n")
                f.write(f"Training Gene Map: {args.training_lmdb_dir}\n")
                f.write(f"Total Samples Processed: {processed_samples} / {num_total_samples}\n")
                f.write(f"Batch Size: {args.batch_size}\n")
                f.write("="*30 + "\n")
                f.write(f"Average MRR: {final_mrr:.4f}\n")
                for k in sorted(final_top_k.keys()):
                    f.write(f"Average Top-{k} Accuracy: {final_top_k[k]:.4f}\n")
            logger.info(f"Results saved to: {results_file_path}")
        except Exception as e:
            logger.error(f"Failed to save results to {results_file_path}: {e}")


    overall_duration = time.time() - overall_start_time
    logger.info(f"Validation Arena finished in {overall_duration:.2f} seconds.")

# --- Argument Parser and Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation Arena: Benchmark a single model against a validation dataset.")

    # --- Core Paths ---
    parser.add_argument('--validation_data_root', type=str, required=True,
                        help='Path to the ROOT directory of the processed validation dataset (containing processed/index.pt, processed/data_*.pt, etc.).')
    parser.add_argument('--training_lmdb_dir', type=str, required=True,
                        help='Path to the root directory of the LMDB dataset used for training (needed for gene mapping).')
    parser.add_argument('--output_dir', type=str, default='arena_output',
                        help='Directory to save evaluation results and logs.')

    # --- Model Specification ---
    parser.add_argument('--evaluator_type', type=str, required=True,
                        choices=list(EVALUATOR_REGISTRY.keys()), # Dynamically list choices
                        help='Type of the evaluator to use.')
    parser.add_argument('--checkpoint_path', type=str, # Required for many evaluators, checked in run_arena
                        help='Path to the model checkpoint file (e.g., .pt file for PyG).')
    # --- PyG Specific Args (Add args for other evaluators as needed) ---
    parser.add_argument('--model_class_fqn', type=str,
                        help='Fully qualified name of the PyG model class (e.g., training.models.models.MyModel).')
    parser.add_argument('--training_args_path', type=str,
                        help='Path to the training args.json file for the model.')
    parser.add_argument('--num_node_features', type=int,
                        help='Number of input node features for the PyG model.')

    # --- Evaluation Parameters ---
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation inference.')
    parser.add_argument('--k_values', type=str, default='1,5,10,20',
                        help='Comma-separated list of K values for Top-K accuracy (e.g., "1,5,10").')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'], help="Device to run inference on.")

    # --- Logging & Output ---
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level.')
    # Removed leaderboard file, added generic output log file name
    parser.add_argument('--output_log_file', type=str, default='arena_run_log.txt',
                        help='Filename for the output log file within output_dir.')
    # Removed leaderboard_file argument

    args = parser.parse_args()

    # --- Setup Output Directory ---
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Setup Logging ---
    log_level_enum = getattr(logging, args.log_level.upper())
    log_file_path = output_path / args.output_log_file
    # Ensure handlers are cleared if running multiple times in same interpreter session (e.g., testing)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    logging.basicConfig(
        level=log_level_enum,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler() # Log to console as well
        ]
    )
    # Re-assign logger to the configured root logger after basicConfig
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Level: {args.log_level}, File: {log_file_path}")
    logger.info("Validation Arena started with arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

    # --- Run the Arena ---
    run_arena(args)

