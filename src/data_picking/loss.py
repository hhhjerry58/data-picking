import torch
from torch.nn import functional as F


def per_example_nll(logits, input_ids, response_mask, chunk_size=128):
    """Causal, response-only NLL; float32 CE, equal example weighting."""
    mask = response_mask[:, 1:]
    counts = mask.sum(1)
    if (counts == 0).any():
        raise ValueError("At least one example has zero scored response tokens")
    positions = mask.nonzero(as_tuple=False)
    sums = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
    for part in positions.split(chunk_size):
        batch, time = part[:, 0], part[:, 1]
        losses = F.cross_entropy(logits[batch, time].float(), input_ids[batch, time + 1], reduction="none")
        sums = sums.index_add(0, batch, losses)
    return sums, sums / counts, counts
