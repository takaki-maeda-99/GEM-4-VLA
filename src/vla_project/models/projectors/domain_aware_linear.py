import torch
import torch.nn as nn


class DomainAwareLinear(nn.Module):
    """Per-domain linear: y = x @ W[domain_id] + b[domain_id].

    Weights and biases are stored as `nn.Embedding` rows so that lookup is
    a single embedding gather. Adapted from X-VLA's `DomainAwareLinear`
    (X-VLA/models/transformer.py).
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_domains = num_domains
        self.fc = nn.Embedding(num_domains, input_size * output_size)
        self.bias = nn.Embedding(num_domains, output_size)
        nn.init.normal_(self.fc.weight, std=(input_size ** -0.5))
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        squeeze_T = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_T = True
        B = domain_id.shape[0]
        assert x.shape[0] == B, f"batch mismatch: x={x.shape[0]} vs dom={B}"
        W = self.fc(domain_id).view(B, self.input_size, self.output_size)
        b = self.bias(domain_id).view(B, 1, self.output_size)
        y = torch.matmul(x, W) + b
        if squeeze_T:
            y = y.squeeze(1)
        return y
