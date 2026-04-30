import torch
import torch.nn as nn


class ActionQueryHub(nn.Module):
    def __init__(self, num_domains: int, num_queries: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_domains, num_queries * hidden_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, domain_id: torch.Tensor) -> torch.Tensor:
        B = domain_id.shape[0]
        return self.embedding(domain_id).view(B, self.num_queries, self.hidden_dim)
