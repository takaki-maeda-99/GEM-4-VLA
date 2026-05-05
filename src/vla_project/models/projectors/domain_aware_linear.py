import torch
import torch.nn as nn


class DomainAwareTwoLayerMLP(nn.Module):
    """Per-domain 2-layer MLP with GELU between, X-VLA convention extended.

    Each domain has its own (W1, b1, W2, b2). Stored as four nn.Embeddings.
    Use as a drop-in replacement for ``DomainAwareLinear`` when more capacity
    is wanted (mid-dim controlled by ``hidden_size``).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_domains: int,
        activation: nn.Module = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_domains = num_domains
        self.fc1 = DomainAwareLinear(input_size, hidden_size, num_domains)
        self.fc2 = DomainAwareLinear(hidden_size, output_size, num_domains)
        self.act = activation if activation is not None else nn.GELU()

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x, domain_id)), domain_id)


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
