import torch.optim

from src.tinyGPT.data_loader import DataLoader
from src.tinyGPT.gpt import TinyGPT


class Trainer:
    def __init__(
        self, model: TinyGPT, data_loader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader

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
