import unittest

from src.tinyGPT.tokenizer import Tokenizer


class TokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = Tokenizer()

    def tearDown(self) -> None:
        self.tokenizer = None

    def test_vocabulary(self) -> None:
        self.assertEqual(65, len(self.tokenizer.vocabulary))

    def test_encode(self) -> None:
        text = "hello world"
        tokens = [46, 43, 50, 50, 53, 1, 61, 53, 56, 50, 42]

        self.assertEqual(tokens, self.tokenizer.encode(text=text))

    def test_decode(self) -> None:
        tokens = [46, 43, 50, 50, 53, 1, 61, 53, 56, 50, 42]

        self.assertEqual("hello world", self.tokenizer.decode(tokens=tokens))
