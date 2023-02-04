import os
import unittest

from src.tinyGPT.data_loader import DataLoader
from src.tinyGPT.tokenizer import Tokenizer


class DataLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_loader = DataLoader(
            file_path=os.path.join(
                os.getcwd(), "..", "..", "data", "tiny_shakespeare.txt"
            ),
            tokenizer=Tokenizer(),
        )
        self.data_loader.load_corpus()

    def tearDown(self) -> None:
        self.data_loader = None

    def test_load(self) -> None:
        self.assertIsNotNone(self.data_loader.text)

    def test_create_dataset(self) -> None:
        train_dataset, valid_dataset = self.data_loader.create_datasets()

        self.assertIsNotNone(train_dataset)
        self.assertIsNotNone(valid_dataset)

    def test_get_batch(self) -> None:
        self.data_loader.create_datasets()
        x, y = self.data_loader.get_batch(split="train")

        # e.g.
        # tensor([[52,  1, 58, 56, 47, 59, 51, 54],
        #         [53, 58, 46, 47, 52, 45,  1, 40],
        #         [53,  1, 57, 43, 58,  1, 53, 52],
        #         [12,  1, 58, 53,  1, 58, 46, 43]])
        self.assertEqual((4, 8), x.shape)

        # e.g.
        # tensor([[ 1, 58, 56, 47, 59, 51, 54, 46],
        #         [58, 46, 47, 52, 45,  1, 40, 59],
        #         [ 1, 57, 43, 58,  1, 53, 52, 43],
        #
        # if input x is [52], output y = [1]
        # x: [52, 1] => y: 58
        # x: [52, 1, 58] => y: 56
        #
        # => 32 independent examples packed in one batch
        self.assertEqual((4, 8), y.shape)

    def test_device(self) -> None:
        self.assertEqual("mps", self.data_loader.device.type)
