import torch
import torch.nn.functional as F

def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Computes the entropy of a probability distribution.
    """
    entropy = -torch.sum(
        probs * torch.log2(probs + 1e-12), dim=-1
    )
    return entropy

def entropy_regularization_loss(
    model: nn.Module, primary_loss: torch.Tensor
) -> torch.Tensor:
    """
    Computes the total loss including entropy regularization.
    """
    entropy = model.moe.entropy
    lambda_entropy = model.moe.lambda_entropy
    total_loss = primary_loss + lambda_entropy * entropy
    return total_loss
