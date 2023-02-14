import torch.backends.mps


class Backend:
    @staticmethod
    def device() -> torch.device:

        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
