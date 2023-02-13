import torch
from torch import nn

from src.tinyGPT.backend import Backend


class AttentionHead(nn.Module):
    def __init__(
        self, embedding_dim: int = 32, head_size: int = 16, block_size: int = 8
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.head_size = head_size
        self.block_size = block_size
        self.key = nn.Linear(
            in_features=embedding_dim, out_features=head_size, bias=False
        )
        self.query = nn.Linear(
            in_features=embedding_dim, out_features=head_size, bias=False
        )
        self.value = nn.Linear(
            in_features=embedding_dim, out_features=head_size, bias=False
        )
        # tril is not a variable, register as buffer
        self.register_buffer(
            name="tril", tensor=torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x) -> torch.Tensor:
        B, T, C = x.shape  # (B:4, T:8, C:32)
        k = self.key(x)  # (B, T, C)
        q = self.key(x)  # (B, T, C)

        # compute attention scores == affinities
        weights = q @ k.transpose(-2, -1)  # (B:4, T:8, 16) @ (B, 16, T) => (B, T, T)
        # normalize
        weights *= self.head_size**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = torch.softmax(weights, dim=-1)  # (B:4, T:8, T:8)

        # value
        v = self.value(x)  # (B:4, T:8, C:32)
        out = weights @ v  # (B:4, T:8, C:32)

        return out


class TinyGPT(nn.Module):
    def __init__(
        self, vocab_size: int, block_size: int = 8, embedding_dim: int = 32
    ) -> None:
        """

        Parameters:
             block_size: int
                the size of the text chunk
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.device = Backend.device()

        self.token_embedding_table = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim
        )
        # positional encoding of the block size
        self.position_embedding_table = nn.Embedding(
            num_embeddings=block_size, embedding_dim=self.embedding_dim
        )
        self.self_attention_head = AttentionHead(
            embedding_dim=self.embedding_dim, head_size=self.embedding_dim, block_size=self.block_size
        )
        # language model head
        self.lm_head = nn.Linear(
            in_features=self.embedding_dim, out_features=self.vocab_size
        )

    def forward(self, index: torch.Tensor, targets: torch.Tensor = None) -> tuple:
        B, T = index.shape
        token_embeddings = self.token_embedding_table(
            index
        )  # (B, T, C) => (Batch, Time = block_size, Channels = vocab_size)
        positional_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, C)
        x = token_embeddings + positional_embeddings  # (B:4, T:8, C:32)
        x = self.self_attention_head(x)
        logits = self.lm_head(x)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            # reshape to fit cross entropy
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # scores
            targets = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index: torch.Tensor, max_tokens: int):
        for _ in range(max_tokens):
            # guard clause to avoid out of index if index should be langer than block size
            context = index[:, -self.block_size:]
            logits, loss = self(context)
            # focus only on the last time step
            logits = logits[:, -1, :]  # => (B, C)
            probabilities = torch.softmax(logits, dim=-1)
            # sample from the distribution
            sample = torch.multinomial(probabilities, num_samples=1)  # => (B, 1)
            # concat index with new generated sample
            index = torch.cat((index, sample), dim=1)  # => (B, T+1)

        return index


