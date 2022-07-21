import torch

class Design():
    @staticmethod
    def from_str(s: str):
        raise NotImplementedError

    @staticmethod
    def from_torch(param: torch.Tensor):
        raise NotImplementedError

    def to_xml(self, path: str) -> None:
        raise NotImplementedError

    def to_str(self) -> str:
        raise NotImplementedError

    def to_torch(self) -> torch.Tensor:
        raise NotImplementedError
