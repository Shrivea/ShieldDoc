from pathlib import Path
from interfaces.text_processor_interface import TextProcessorInterface

class TextProcessorTxt(TextProcessorInterface):
    
    def __init__(self, input_path: Path):
        self.input_path = input_path
    
    def process(self) -> str:
        return self.input_path.read_text()
    

