import torch
import torch.nn as nn
from token_mixing import TokenMixing
from channel_mixing import ChannelMixing
from mixture_of_experts import MixtureOfExperts

class LFModel(nn.Module):
    """
    Custom LF Model architecture combining token mixing, channel mixing,
    and MoE with entropy regularization.
    Accepts 3D input tensor: [batch_size, sequence_length, embedding_dim].
    """

    def __init__(self, token_dim: int, channel_dim: int,
                 expert_dim: int, adapt_dim: int, num_experts: int,
                 lambda_entropy: float = 0.01):
        super(LFModel, self).__init__()
        self.token_dim = token_dim
        self.featurizer = nn.Linear(token_dim, adapt_dim)
        self.token_mixer = TokenMixing(token_dim, adapt_dim)
        self.channel_mixer = ChannelMixing(channel_dim, adapt_dim)
        self.moe = MixtureOfExperts(
            expert_dim, num_experts, adapt_dim, lambda_entropy
        )
        self.output_layer = nn.Linear(expert_dim, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Featurization stage
        adapt_input = self.featurizer(
            x.mean(dim=1)
        )  # Aggregate across sequence for adaptation

        # Token Mixing
        token_mixed = self.token_mixer(x, adapt_input)

        # Channel Mixing
        channel_mixed = self.channel_mixer(token_mixed, adapt_input)

        # Mixture of Experts
        expert_output = self.moe(channel_mixed, adapt_input)

        # Final Output
        output = self.output_layer(expert_output)
        return output
