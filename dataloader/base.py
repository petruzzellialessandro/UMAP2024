from abc import *


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass