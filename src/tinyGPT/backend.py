import platform

import torch.backends.mps


class Backend:
    @staticmethod
    def device() -> torch.device:
        device_type = "cpu"

        if Backend.is_mac_os() and torch.backends.mps.is_available():
            device_type = "mps"
        elif torch.cuda.is_available():
            device_type = "cuda"

        return torch.device(device_type)

    @staticmethod
    def is_mac_os() -> bool:
        """
        Check if the underlying os is macOS.

        Return:
            is_mac_os: bool
                True if is macOS
        """
        return "macOS" in platform.platform()
