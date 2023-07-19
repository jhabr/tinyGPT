import unittest

import torch.backends.mps

from src.tinyGPT.backend import Backend


class BackendTests(unittest.TestCase):
    def test_device(self) -> None:
        if torch.backends.mps.is_available():
            self.assertEqual("mps", Backend.device().type)
        elif torch.cuda.is_available():
            self.assertEqual("cuda", Backend.device().type)
        else:
            self.assertEqual("cpu", Backend.device().type)
