import torch
import torch.nn as nn
from adaptive_linear import AdaptiveLinear

class TokenMixing(nn.Module):
    """
    Token mixing layer that performs token-wise interactions using
    adaptive linear layers.
    Operates across the sequence dimension (sequence_length).
    """

    def __init__(self, token_dim: int, adapt_dim: int):
        super(TokenMixing, self).__init__()
        self.token_mixing = AdaptiveLinear(
            token_dim, token_dim, adapt_dim
        )

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor
                ) -> torch.Tensor:
        # x: [batch_size, sequence_length, embedding_dim]
        batch_size, seq_length, embed_dim = x.shape
        x = x.view(
            batch_size * seq_length, embed_dim
        )  # Flatten sequence for linear transformation
        x_mixed = self.token_mixing(x, adapt_input)
        return x_mixed.view(batch_size, seq_length, embed_dim)
