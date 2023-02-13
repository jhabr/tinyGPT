import torch.optim

from src.tinyGPT.backend import Backend
from src.tinyGPT.data_loader import DataLoader
from src.tinyGPT.gpt import TinyGPT
from src.tinyGPT.tokenizer import Tokenizer


class Trainer:
    def __init__(
        self,
        model: TinyGPT,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        tokenizer: Tokenizer,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.tokenizer = tokenizer

    def fit(self, iterations: int = 1_000, eval_interval: int = 500):
        for iteration in range(iterations):
            x_batch, y_batch = self.data_loader.get_batch(split="train")

            # forward pass
            logits, loss = self.model(x_batch, targets=y_batch)

            # zero grad
            self.optimizer.zero_grad(set_to_none=True)

            # backward pass
            loss.backward()

            # update weights and biases
            self.optimizer.step()

            # print loss every 500 iteration
            if iteration % eval_interval == 0 or iter == iterations - 1:
                loss = self._estimate_loss()
                print(
                    f"[INFO]: step {iteration} - train loss: {round(loss['train'], 4)}, val loss: {round(loss['valid'], 4)}"
                )

    def _estimate_loss(self, eval_iterations: int = 200) -> dict:
        out = {}

        self.model.eval()

        for split in ["train", "valid"]:
            losses = []
            for iteration in range(eval_iterations):
                x_batch, y_batch = self.data_loader.get_batch(split=split)
                logits, loss = self.model(x_batch, targets=y_batch)
                losses.append(loss.item())

            out[split] = sum(losses) / len(losses)

        self.model.train()

        return out

    def test(self, max_tokens: int = 500) -> str:
        print("[INFO] Generating text...")

        context = torch.zeros((1, 1), dtype=torch.long, device=Backend.device())
        tokens = self.model.generate(index=context, max_tokens=max_tokens)[0].tolist()
        text = self.tokenizer.decode(tokens=tokens)

        print(f"[INFO] Generated text:\n{text}")

        return text
