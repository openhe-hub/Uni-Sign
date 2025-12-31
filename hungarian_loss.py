"""
Hungarian Loss for CSLR Task
Implements Hungarian matching algorithm for gloss sequence prediction
where the order of glosses may not strictly align with video frames.

Based on DETR's Hungarian matcher, adapted for CSLR task.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """
    Hungarian algorithm-based gloss sequence matcher.
    Finds optimal assignment between predictions and targets.

    Args:
        cost_class: Weight for classification cost in matching
        use_no_object: If True, unmatched predictions are assigned to "no object"
        allow_null_match: If True, add a dummy "no object" column so predictions
                          can explicitly match to empty targets
        no_object_cost: Cost for matching to the dummy column
    """

    def __init__(self, cost_class: float = 1.0, use_no_object: bool = False,
                 allow_null_match: bool = False, no_object_cost: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.use_no_object = use_no_object
        self.allow_null_match = allow_null_match
        self.no_object_cost = no_object_cost

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
            tgt_idx: matched target indices (-1 indicates "no object" if use_no_object=True)
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

            # Optional dummy "no object" column so predictions can choose to
            # match an empty target even when L == T
            if self.use_no_object and self.allow_null_match:
                dummy_cost = torch.full(
                    (num_queries, 1),
                    self.no_object_cost,
                    device=cost_class.device,
                    dtype=cost_class.dtype,
                )
                cost_class = torch.cat([cost_class, dummy_cost], dim=1)
                dummy_col_idx = cost_class.shape[1] - 1
            else:
                dummy_col_idx = None

            # Total cost
            C = self.cost_class * cost_class
            # Convert to float32 first (BFloat16 is not supported by numpy)
            C = C.float().cpu().numpy()

            # Solve optimal assignment using Hungarian algorithm
            src_idx, tgt_idx = linear_sum_assignment(C)

            # If a dummy column exists, mark those matches as "no object"
            if dummy_col_idx is not None:
                tgt_idx = np.where(tgt_idx == dummy_col_idx, -1, tgt_idx)

            # Handle unmatched predictions (if enabled and L > T)
            if self.use_no_object and num_queries > len(valid_tgt):
                # Find unmatched prediction indices
                all_src_idx = set(range(num_queries))
                matched_src_idx = set(src_idx.tolist())
                unmatched_src_idx = sorted(all_src_idx - matched_src_idx)

                if len(unmatched_src_idx) > 0:
                    # Mark unmatched predictions as "no object" (-1)
                    unmatched_src = list(unmatched_src_idx)
                    unmatched_tgt = [-1] * len(unmatched_src)

                    # Combine matched and unmatched results
                    src_idx = np.concatenate([src_idx, unmatched_src])
                    tgt_idx = np.concatenate([tgt_idx, unmatched_tgt])

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
        no_object_token_id: Token ID for "no object" (e.g., PAD or EOS)
        no_object_weight: Weight for no object loss (default: 0.1)
    """

    def __init__(self, matcher, vocab_size, label_smoothing=0.0,
                 no_object_token_id=None, no_object_weight=0.1):
        super().__init__()
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.no_object_token_id = no_object_token_id
        self.no_object_weight = no_object_weight

    def loss_labels(self, outputs, targets, indices):
        """
        Compute classification loss based on Hungarian matching.

        Args:
            outputs: Prediction logits, shape (B, L, V)
            targets: Target token ids, shape (B, T)
            indices: Matching indices from Hungarian matcher

        Returns:
            Dictionary containing the loss(es)
        """
        # Separate normal matches (tgt_idx >= 0) and no object matches (tgt_idx == -1)
        normal_batch_idx = []
        normal_src_idx = []
        normal_target_classes = []

        no_object_batch_idx = []
        no_object_src_idx = []

        for i, (src_idx, tgt_idx) in enumerate(indices):
            tgt = targets[i]
            valid_tgt = tgt[tgt != -100]

            # Separate matches by type
            normal_mask = tgt_idx >= 0
            no_object_mask = tgt_idx == -1

            # Process normal matches
            if normal_mask.any():
                matched_src = src_idx[normal_mask]
                matched_tgt = tgt_idx[normal_mask]

                normal_batch_idx.append(torch.full_like(matched_src, i))
                normal_src_idx.append(matched_src)
                normal_target_classes.append(valid_tgt[matched_tgt])

            # Process no object matches
            if no_object_mask.any():
                matched_src = src_idx[no_object_mask]
                no_object_batch_idx.append(torch.full_like(matched_src, i))
                no_object_src_idx.append(matched_src)

        # Compute normal classification loss
        if len(normal_batch_idx) > 0:
            batch_idx = torch.cat(normal_batch_idx)
            src_idx = torch.cat(normal_src_idx)
            target_classes = torch.cat(normal_target_classes)

            src_logits = outputs[batch_idx, src_idx]

            loss_ce = nn.functional.cross_entropy(
                src_logits,
                target_classes,
                label_smoothing=self.label_smoothing,
                reduction='mean'
            )
        else:
            loss_ce = torch.tensor(0.0, device=outputs.device, requires_grad=True)

        # Compute no object loss (if enabled and there are no object matches)
        if self.no_object_token_id is not None and len(no_object_batch_idx) > 0:
            batch_idx = torch.cat(no_object_batch_idx)
            src_idx = torch.cat(no_object_src_idx)

            src_logits = outputs[batch_idx, src_idx]

            # Target is no_object_token_id for all unmatched predictions
            no_object_targets = torch.full(
                (src_logits.shape[0],),
                self.no_object_token_id,
                dtype=torch.long,
                device=outputs.device
            )

            loss_no_object = nn.functional.cross_entropy(
                src_logits,
                no_object_targets,
                reduction='mean'
            )

            # Weighted combination
            total_loss = loss_ce + self.no_object_weight * loss_no_object

            return {
                'loss_ce_hungarian': total_loss,
                'loss_ce_normal': loss_ce,
                'loss_no_object': loss_no_object
            }
        else:
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
