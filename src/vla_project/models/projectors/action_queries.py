import torch
import torch.nn as nn


class ActionQueryHub(nn.Module):
    """Shared learnable action queries (NOT per-domain).

    Design choice: action queries are shared across domains. The class is
    kept named "Hub" for naming consistency with SoftPromptHub, but it does
    not index by domain_id — it broadcasts a single [Q, D] parameter to
    [B, Q, D] in forward.
    """

    def __init__(self, num_queries: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        # Match vla-gemma-4 73% baseline:
        # ``action_queries = nn.Embedding(NUM, dim); weight.data.zero_()``.
        # Earlier we used N(0, 0.02²) which gave per-position random offsets
        # while the reference starts every position from the same zero
        # vector and lets gradient differentiate them via RoPE-rotated
        # queries.
        self.queries = nn.Parameter(torch.zeros(num_queries, hidden_dim))

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.queries.unsqueeze(0).expand(
            batch_size, self.num_queries, self.hidden_dim
        )
