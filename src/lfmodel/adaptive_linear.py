import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveLinear(nn.Module):
    """
    Adaptive Linear layer whose weight and bias adapt based on input.
    """

    def __init__(self, in_features: int, out_features: int, adapt_dim: int):
        super(AdaptiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features)
        )
        self.bias = nn.Parameter(torch.randn(out_features))

        # Linear transformation for adapting the weight based on input
        self.adapt = nn.Linear(adapt_dim, out_features * in_features)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor
                ) -> torch.Tensor:
        adapt_weight = self.adapt(adapt_input).view(
            self.out_features, self.in_features
        )
        weight = self.weight + adapt_weight
        return F.linear(x, weight, self.bias)
