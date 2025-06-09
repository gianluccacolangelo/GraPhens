import torch
import logging
from typing import Dict, List, Tuple

# Use standard Python logging
logger = logging.getLogger(__name__)

@torch.no_grad()
def calculate_batch_metrics(out: torch.Tensor, target_y: torch.Tensor, k_vals: List[int] = [1, 5, 10, 20]) -> Tuple[float, Dict[int, float]]:
    """Calculates MRR and Top-K accuracy for a single batch.

    Args:
        out: Model output logits/scores (shape: [batch_size, num_classes]). 
             Assumed to be on CPU.
        target_y: Target class indices (shape: [batch_size]). 
                  Assumed to be on CPU.
        k_vals: List of K values for Top-K accuracy calculation.

    Returns:
        A tuple containing: (batch_mrr, batch_top_k_acc_dict).
        Returns (0.0, {k: 0.0 for k in k_vals}) if calculation fails or batch is empty.
    """
    batch_mrr = 0.0
    # Ensure k_vals are sorted for consistent output ordering if needed
    k_vals = sorted(k_vals) 
    batch_top_k_acc = {k: 0.0 for k in k_vals}
    num_graphs = target_y.size(0)

    if num_graphs == 0:
        logger.debug("calculate_batch_metrics received empty batch, returning 0s.")
        return batch_mrr, batch_top_k_acc

    # Ensure tensors are on CPU
    if out.device.type != 'cpu':
        out = out.cpu()
    if target_y.device.type != 'cpu':
        target_y = target_y.cpu()

    try:
        # Get the predicted class indices by sorting scores
        _, sorted_indices = torch.sort(out, dim=1, descending=True) 
        
        # Expand target_y to match the shape of sorted_indices for comparison
        target_expanded = target_y.unsqueeze(1).expand_as(sorted_indices)
        
        # Find where the prediction matches the target
        target_match = (sorted_indices == target_expanded)
        
        # Get the rank (position) of the correct prediction for each sample
        # nonzero returns a tuple of tensors, one for each dimension. We need the column indices ([1])
        match_indices = torch.nonzero(target_match, as_tuple=True)
        
        # Check if a match was found for every sample in the batch
        if len(match_indices[0]) == num_graphs:
            # Ranks are 1-based (position + 1)
            ranks = match_indices[1] + 1 
            reciprocal_ranks = 1.0 / ranks.float()
            batch_mrr = torch.mean(reciprocal_ranks).item()

            # Calculate Top-K accuracy
            top_k_correct = {k: torch.sum(ranks <= k).item() for k in k_vals}
            batch_top_k_acc = {k: (correct / num_graphs) for k, correct in top_k_correct.items()}
        else:
            # This can happen if the target class index is out of bounds of the output shape,
            # or potentially if there are duplicate max scores (less likely with floats).
            logger.warning(f"Rank finding mismatch in calculate_batch_metrics. "
                           f"Found matches for {len(match_indices[0])} out of {num_graphs} samples. "
                           f"Returning MRR=0, TopK=0 for this batch.")
            # Keep MRR and TopK as 0.0

    except Exception as e:
        logger.error(f"Error in calculate_batch_metrics: {e}", exc_info=True) 
        # Return 0s in case of any unexpected error during calculation
        batch_mrr = 0.0
        batch_top_k_acc = {k: 0.0 for k in k_vals}

    return batch_mrr, batch_top_k_acc 