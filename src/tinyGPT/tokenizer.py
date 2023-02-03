class Tokenizer:
    def __init__(self) -> None:
        """
        A tokenizer for the tinyGPT model.

        The vocabulary is a list of characters that the model will be trained on.
        The vocabulary was taken from the tiny_shakespeare.txt file using the following code:

        text = self.data_loader.load_corpus()
        tokens = sorted(list(set(text)))

        Returns:
            None
        """
        self.vocabulary = [
            "\n",
            " ",
            "!",
            "$",
            "&",
            "'",
            ",",
            "-",
            ".",
            "3",
            ":",
            ";",
            "?",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
        ]

    def encode(self, text: str) -> list[int]:
        return [self.vocabulary.index(token) for token in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join([self.vocabulary[token] for token in tokens])
