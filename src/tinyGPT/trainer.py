import torch.optim

from src.tinyGPT.backend import Backend
from src.tinyGPT.data_loader import DataLoader
from src.tinyGPT.gpt import TinyGPT
from src.tinyGPT.tokenizer import Tokenizer


class Trainer:
    def __init__(
        self, model: TinyGPT, data_loader: DataLoader, optimizer: torch.optim.Optimizer, tokenizer: Tokenizer
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.tokenizer = tokenizer

    def fit(self, epochs: int = 100):
        for epoch in range(epochs):
            x_batch, y_batch = self.data_loader.get_batch(split="train")

            # forward pass
            logits, loss = self.model(x_batch, targets=y_batch)
            print(f"[INFO] epoch: {epoch}, loss: {loss.item()}")

            # zero grad
            self.optimizer.zero_grad(set_to_none=True)

            # backward pass
            loss.backward()

            # update weights and biases
            self.optimizer.step()

    def test(self, max_tokens: int = 500) -> str:
        print("[INFO] Generating text...")

        context = torch.zeros((1, 1), dtype=torch.long, device=Backend.device())
        tokens = self.model.generate(index=context, max_tokens=max_tokens)[0].tolist()
        text = self.tokenizer.decode(tokens=tokens)

        print(f"[INFO] Generated text:\n{text}")

        return text
