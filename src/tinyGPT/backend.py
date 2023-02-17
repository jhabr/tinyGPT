import torch.backends.mps


class Backend:
    @staticmethod
    def device() -> torch.device:
        """
        Returns the device to be used for training.

        Returns:
            device: torch.device
                the device to be used for training
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
