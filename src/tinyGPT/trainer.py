import logging

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
        """
        Trainer class to train the model.

        Parameters:
            model: TinyGPT
                the model to be trained
            data_loader: DataLoader
                the data loader to load the data
            optimizer: torch.optim.Optimizer
                the optimizer to update the weights and biases
            tokenizer: Tokenizer
                the tokenizer to tokenize the text

        Returns:
            None
        """
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.tokenizer = tokenizer

    def fit(self, iterations: int = 1_000, eval_interval: int = 500):
        """
        Fit the model.

        Parameters:
            iterations: int
                the number of iterations to train the model
            eval_interval: int
                the interval to evaluate the model

        Returns:
            None
        """
        logging.info(
            f"Trainable params: {round(sum([param.numel() for param in self.model.parameters()])/1e6, 1)}M"
        )
        logging.info("Fitting model...")

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
                logging.info(
                    f"step {iteration} - "
                    f"train loss: {round(loss['train'], 4)}, "
                    f"val loss: {round(loss['valid'], 4)}"
                )
        logging.info("Done.")

    def _estimate_loss(self, eval_iterations: int = 200) -> dict:
        """
        Estimate the loss on the train and validation set.

        Parameters:
            eval_iterations: int
                the number of iterations to estimate the loss

        Returns:
            out: dict
                the estimated loss on the train and validation set
        """

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
        """
        Test the model.

        Parameters:
            max_tokens: int
                the maximum number of tokens to generate

        Returns:
            text: str
                the generated text
        """
        logging.info("Generating text...")

        context = torch.zeros((1, 1), dtype=torch.long, device=Backend.device())
        tokens = self.model.generate(index=context, max_tokens=max_tokens)[0].tolist()
        text = self.tokenizer.decode(tokens=tokens)

        logging.info(f"Generated text:\n{text}")

        return text
