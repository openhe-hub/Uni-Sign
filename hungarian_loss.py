"""
Hungarian Loss for CSLR Task
Implements Hungarian matching algorithm for gloss sequence prediction
where the order of glosses may not strictly align with video frames.

Based on DETR's Hungarian matcher, adapted for CSLR task.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """
    Hungarian algorithm-based gloss sequence matcher.
    Finds optimal assignment between predictions and targets.

    Args:
        cost_class: Weight for classification cost in matching
    """

    def __init__(self, cost_class: float = 1.0):
        super().__init__()
        self.cost_class = cost_class

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs Hungarian matching between predictions and targets.

        Args:
            outputs: Model prediction logits, shape (B, L, V)
                    B=batch_size, L=sequence_length, V=vocab_size
            targets: Target gloss sequence token ids, shape (B, T)
                    T=target_length, -100 indicates padding

        Returns:
            List of tuples (src_idx, tgt_idx) for each batch sample
            src_idx: matched prediction indices
            tgt_idx: matched target indices
        """
        bs, num_queries = outputs.shape[:2]

        # Compute softmax probabilities for all predictions
        # Shape: (B*L, V)
        out_prob = outputs.flatten(0, 1).softmax(-1)

        indices = []

        for i in range(bs):
            # Get targets for current sample
            tgt = targets[i]  # (T,)

            # Remove padding tokens (-100)
            valid_tgt = tgt[tgt != -100]

            if len(valid_tgt) == 0:
                # No valid targets, return empty matching
                indices.append((torch.tensor([]), torch.tensor([])))
                continue

            # Compute cost matrix: (L, T)
            # For each prediction position and each target token,
            # compute negative log-likelihood as cost
            start_idx = i * num_queries
            end_idx = (i + 1) * num_queries

            # Gather probabilities for target tokens
            # Shape: (L, T)
            cost_class = -out_prob[start_idx:end_idx, valid_tgt]

            # Total cost
            C = self.cost_class * cost_class
            # Convert to float32 first (BFloat16 is not supported by numpy)
            C = C.float().cpu().numpy()

            # Solve optimal assignment using Hungarian algorithm
            src_idx, tgt_idx = linear_sum_assignment(C)

            # Convert to tensors
            src_idx = torch.as_tensor(src_idx, dtype=torch.int64)
            tgt_idx = torch.as_tensor(tgt_idx, dtype=torch.int64)

            indices.append((src_idx, tgt_idx))

        return indices


class HungarianLoss(nn.Module):
    """
    Hungarian matching-based loss for CSLR task.
    Combines Hungarian matching with cross-entropy loss.

    Args:
        matcher: HungarianMatcher instance
        vocab_size: Size of vocabulary
        label_smoothing: Label smoothing factor
    """

    def __init__(self, matcher, vocab_size, label_smoothing=0.0):
        super().__init__()
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing

    def loss_labels(self, outputs, targets, indices):
        """
        Compute classification loss based on Hungarian matching.

        Args:
            outputs: Prediction logits, shape (B, L, V)
            targets: Target token ids, shape (B, T)
            indices: Matching indices from Hungarian matcher

        Returns:
            Dictionary containing the loss
        """
        # Get batch and source indices for matched predictions
        idx = self._get_src_permutation_idx(indices)

        # Gather matched target tokens
        target_classes_list = []
        for i, (_, tgt_idx) in enumerate(indices):
            tgt = targets[i]
            valid_tgt = tgt[tgt != -100]

            if len(tgt_idx) > 0:
                target_classes_list.append(valid_tgt[tgt_idx])
            else:
                # No matches for this sample
                target_classes_list.append(
                    torch.tensor([], dtype=torch.long, device=tgt.device)
                )

        # Check if we have any valid matches
        valid_matches = [t for t in target_classes_list if len(t) > 0]

        if len(valid_matches) > 0:
            target_classes = torch.cat(target_classes_list)

            # Get matched prediction logits
            src_logits = outputs[idx]  # (num_matched, V)

            # Compute cross-entropy loss
            loss_ce = nn.functional.cross_entropy(
                src_logits,
                target_classes,
                label_smoothing=self.label_smoothing,
                reduction='mean'
            )
        else:
            # No valid matches in batch, return zero loss
            loss_ce = torch.tensor(0.0, device=outputs.device, requires_grad=True)

        return {'loss_ce_hungarian': loss_ce}

    def _get_src_permutation_idx(self, indices):
        """
        Convert matching indices to batch indices for tensor indexing.

        Args:
            indices: List of (src_idx, tgt_idx) tuples

        Returns:
            Tuple of (batch_idx, src_idx) for advanced indexing
        """
        batch_idx = torch.cat([
            torch.full_like(src, i)
            for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        """
        Compute Hungarian loss.

        Args:
            outputs: Prediction logits, shape (B, L, V)
            targets: Target token ids, shape (B, T)

        Returns:
            Scalar loss tensor
        """
        # Perform Hungarian matching
        indices = self.matcher(outputs, targets)

        # Compute loss based on matching
        losses = self.loss_labels(outputs, targets, indices)

        return losses['loss_ce_hungarian']


def compute_set_based_metrics(predictions, targets, ignore_index=-100):
    """
    Compute set-based metrics (ignoring order).
    Useful for evaluating gloss recognition without order constraints.

    Args:
        predictions: List of predicted token sequences
        targets: List of target token sequences
        ignore_index: Token to ignore (padding)

    Returns:
        Dictionary containing precision, recall, and F1 score
    """
    total_precision = 0.0
    total_recall = 0.0
    count = 0

    for pred, tgt in zip(predictions, targets):
        # Remove padding
        pred_set = set(pred[pred != ignore_index].tolist())
        tgt_set = set(tgt[tgt != ignore_index].tolist())

        if len(tgt_set) == 0:
            continue

        # Compute intersection
        intersection = pred_set & tgt_set

        # Precision: |intersection| / |predictions|
        precision = len(intersection) / len(pred_set) if len(pred_set) > 0 else 0.0

        # Recall: |intersection| / |targets|
        recall = len(intersection) / len(tgt_set)

        total_precision += precision
        total_recall += recall
        count += 1

    if count == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    avg_precision = total_precision / count
    avg_recall = total_recall / count
    f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)
          if (avg_precision + avg_recall) > 0 else 0.0)

    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': f1
    }
