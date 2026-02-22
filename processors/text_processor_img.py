from pathlib import Path
from interfaces.text_processor_interface import TextProcessorInterface
import easyocr

class TextProcessorImg(TextProcessorInterface):

    def __init__(self, input_path: Path):
        self.input_path = input_path
        self.reader = easyocr.Reader(['en'])

    def process(self) -> str:
        results = self.reader.readtext(str(self.input_path))
        return "\n".join([text for (_, text, _) in results])