import os
import unittest

from src.tinyGPT.data_loader import DataLoader


class DataLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_loader = DataLoader(
            file_path=os.path.join(
                os.getcwd(), "..", "..", "data", "tiny_shakespeare.txt"
            )
        )

    def tearDown(self) -> None:
        self.data_loader = None

    def test_load(self) -> None:
        text = self.data_loader.load_corpus()

        self.assertIsNotNone(text)
