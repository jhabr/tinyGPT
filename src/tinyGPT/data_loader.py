from typing import Union

import torch

from src.tinyGPT.backend import Backend
from src.tinyGPT.tokenizer import Tokenizer


class DataLoader:
    torch.manual_seed(42)

    def __init__(self, file_path: str, tokenizer: Tokenizer) -> None:
        """
        A data loader for the tinyGPT model.

        Parameters:
             file_path: str
                The path to the file that will be used to train the model.
            tokenizer: Tokenizer
                The tokenizer that will be used to tokenize the text.

        Returns:
            None
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.text = None
        self.train_dataset = None
        self.valid_dataset = None
        self.device = Backend.device()

    def load_corpus(self) -> str:
        """
        Loads the corpus from the file path.

        Returns:
            text: str
                The text from the file.
        """
        with open(self.file_path, "r") as file:
            self.text = file.read()
            return self.text

    def create_datasets(
        self, train_split_size: float = 0.9
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Creates the training and validation datasets.

        Parameters:
            train_split_size: float
                The size of the training split. The validation split will be 1 - train_split_size.

        Returns:
            train_dataset: torch.Tensor
                The training dataset.
        """
        tokens = self.tokenizer.encode(self.text)
        data = torch.tensor(tokens, dtype=torch.long)

        no_tokens = int(train_split_size * len(data))
        self.train_dataset = data[:no_tokens].to(self.device)
        self.valid_dataset = data[no_tokens:].to(self.device)

        return self.train_dataset, self.valid_dataset

    def get_batch(
        self,
        split: str = Union["train", "valid"],
        block_size: int = 8,
        batch_size: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the next batch of data.

        Parameters:
            split: str
                The split of the data to use. Either "train" or "valid".
            block_size: int
                The size of the block of data to return - number of tokens to consider at once.
            batch_size: int
                The size of the batch to return.

        Returns:
            batch: torch.Tensor
                The batch of data.
        """
        data = self.train_dataset if split == "train" else self.valid_dataset
        indices = torch.randint(high=len(data) - block_size, size=(batch_size,))
        x = torch.stack([data[index : index + block_size] for index in indices]).to(
            self.device
        )
        y = torch.stack(
            [data[index + 1 : index + block_size + 1] for index in indices]
        ).to(
            self.device
        )  # offset by 1 == target is the next character in the sequence
        return x, y
