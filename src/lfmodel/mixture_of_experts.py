import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptive_linear import AdaptiveLinear

class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) module that dynamically selects experts
    based on input.
    Operates after channel and token mixing.
    """

    def __init__(self, expert_dim: int, num_experts: int,
                 adapt_dim: int, lambda_entropy: float = 0.01):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([
            AdaptiveLinear(expert_dim, expert_dim, adapt_dim)
            for _ in range(num_experts)
        ])
        self.gating = nn.Linear(adapt_dim, num_experts)
        self.lambda_entropy = lambda_entropy  # Regularization strength

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor
                ) -> torch.Tensor:
        gate_logits = self.gating(adapt_input)
        gate_scores = F.softmax(gate_logits, dim=-1)
        # Compute entropy of gating scores
        entropy = -torch.sum(
            gate_scores * torch.log2(gate_scores + 1e-12), dim=-1
        )
        # Store entropy for use in loss calculation
        self.entropy = entropy.mean()
        # Compute output
        output = sum(
            gate_scores[:, i].unsqueeze(1) * expert(x, adapt_input)
            for i, expert in enumerate(self.experts)
        )
        return output
