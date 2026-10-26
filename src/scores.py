import torch
import torch.nn.functional as F

def id_confidence(logits, method="msp", temperature=1.0):
    """Larger means more ID-like. Energy confidence is negative free energy."""
    if method == "msp":
        return F.softmax(logits, dim=1).max(dim=1).values
    if method == "energy":
        return temperature * torch.logsumexp(logits / temperature, dim=1)
    raise ValueError(f"Unknown score method: {method}")

