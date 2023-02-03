class DataLoader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load_corpus(self) -> str:
        with open(self.file_path, "r") as file:
            return file.read()
