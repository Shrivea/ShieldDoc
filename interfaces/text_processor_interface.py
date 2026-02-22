from abc import ABC, abstractmethod
from pathlib import Path

class TextProcessorInterface(ABC):

    def __init__(self, input_path: Path):
        self.input_path = input_path

    @abstractmethod
    def process(self) -> str:
        pass