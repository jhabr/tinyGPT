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


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        no_heads: int = 4,
        embedding_dim: int = 32,
        head_size: int = 32,
        block_size: int = 8,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            modules=[
                AttentionHead(
                    embedding_dim=embedding_dim,
                    head_size=head_size,
                    block_size=block_size,
                )
                for _ in range(no_heads)
            ]
        )
        self.projection = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, x) -> torch.Tensor:
        out = torch.cat(
            [head(x) for head in self.heads], dim=-1
        )  # concatenate over the C dimension

        out = self.projection(out)  # residual connection
        return out


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int = 32) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            # according to paper, the inner layer dim is 4x bigger then the input and output
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim),  # residual connection as projection
        )

    def forward(self, x) -> torch.Tensor:
        return self.sequential(x)


class Block(nn.Module):
    def __init__(
        self,
        no_heads: int = 4,
        embedding_dim: int = 32,
        block_size: int = 8,
    ) -> None:
        """
        Transformer block: communication between tokens (multi head self attention),
        followed by a computation (feed forward layer)

        Parameters:
            no_heads: int
                the number of attention heads
            embedding_dim: int
                the dimension of the embedding
            block_size: int
                the size of the text chunk

        Returns:
            None
        """
        super().__init__()
        self.head_size = embedding_dim // no_heads
        self.multi_head_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            head_size=self.head_size,
            block_size=block_size,
        )
        self.feed_forward = FeedForward(embedding_dim=embedding_dim)
        # layer normalization
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embedding_dim)

    def forward(self, x) -> torch.Tensor:
        x = x + self.multi_head_attention(self.layer_norm_1(x))  # x + => residual connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


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

        self.blocks = nn.Sequential(
            Block(no_heads=4, block_size=self.block_size, embedding_dim=self.embedding_dim),
            Block(no_heads=4, block_size=self.block_size, embedding_dim=self.embedding_dim),
            Block(no_heads=4, block_size=self.block_size, embedding_dim=self.embedding_dim),
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
        x = self.blocks(x)
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
            context = index[:, -self.block_size :]
            logits, loss = self(context)
            # focus only on the last time step
            logits = logits[:, -1, :]  # => (B, C)
            probabilities = torch.softmax(logits, dim=-1)
            # sample from the distribution
            sample = torch.multinomial(probabilities, num_samples=1)  # => (B, 1)
            # concat index with new generated sample
            index = torch.cat((index, sample), dim=1)  # => (B, T+1)

        return index
