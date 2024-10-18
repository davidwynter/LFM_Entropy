import torch
import torch.nn as nn
from adaptive_linear import AdaptiveLinear

class ChannelMixing(nn.Module):
    """
    Channel mixing layer that performs cross-channel interactions using
    adaptive linear layers.
    Operates across the embedding dimension (embedding_dim).
    """

    def __init__(self, channel_dim: int, adapt_dim: int):
        super(ChannelMixing, self).__init__()
        self.channel_mixing = AdaptiveLinear(
            channel_dim, channel_dim, adapt_dim
        )

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor
                ) -> torch.Tensor:
        # x: [batch_size, sequence_length, embedding_dim]
        return self.channel_mixing(x, adapt_input)
