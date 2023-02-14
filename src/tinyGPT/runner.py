import os

import torch

from src.tinyGPT.backend import Backend
from src.tinyGPT.data_loader import DataLoader
from src.tinyGPT.gpt import TinyGPT
from src.tinyGPT.tokenizer import Tokenizer
from src.tinyGPT.trainer import Trainer


def main():
    file_path = os.path.join(os.getcwd(), "..", "..", "data", "tiny_shakespeare.txt")
    tokenizer = Tokenizer()
    data_loader = DataLoader(file_path=file_path, tokenizer=tokenizer)
    data_loader.load_corpus()
    data_loader.create_datasets()
    model = TinyGPT(
        vocab_size=len(tokenizer.vocabulary), block_size=256, embedding_dim=384
    ).to(Backend.device())
    trainer = Trainer(
        model=model,
        data_loader=data_loader,
        optimizer=torch.optim.AdamW(params=model.parameters(), lr=1e-3),
        tokenizer=tokenizer,
    )
    trainer.fit(iterations=5_000)
    trainer.test()


if __name__ == "__main__":
    main()
