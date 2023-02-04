import torch
from torch import nn

from src.tinyGPT.backend import Backend


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = Backend.device

        self.token_embedding_table = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim
        )
        # language model head
        self.lm_head = nn.Linear(
            in_features=self.embedding_dim, out_features=self.vocab_size
        )

    def forward(self, index: torch.Tensor, targets: torch.Tensor = None) -> tuple:
        token_embeddings = self.token_embedding_table(index)  # (B, T, C)
        logits = self.lm_head(token_embeddings)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # reshape to fit cross entropy
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # scores
            targets = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self):
        pass
