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
        self.queries = nn.Parameter(torch.zeros(num_queries, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.queries.unsqueeze(0).expand(
            batch_size, self.num_queries, self.hidden_dim
        )
