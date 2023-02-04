import unittest

from src.tinyGPT.backend import Backend


class BackendTests(unittest.TestCase):
    def test_device(self) -> None:
        self.assertEqual("mps", Backend.device().type)
