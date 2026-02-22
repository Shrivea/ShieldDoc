from pathlib import Path
from interfaces.text_processor_interface import TextProcessorInterface
import fitz
import easyocr
import numpy as np

class TextProcessorPdf(TextProcessorInterface):

    def __init__(self, input_path: Path):
        self.input_path = input_path
        self._ocr_reader = None  

    @property
    def ocr_reader(self):
        if self._ocr_reader is None:
            self._ocr_reader = easyocr.Reader(['en'])
        return self._ocr_reader

    def process(self) -> str:
        doc = fitz.open(self.input_path)
        text_data = ""

        for page in doc:
            text = page.get_text(sort=True).strip()
            if text:
                text_data += text + "\n"
            else:
                #fallback to OCR
                pix = page.get_pixmap(dpi=300)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                results = self.ocr_reader.readtext(img)
                text_data += "\n".join([t for (_, t, _) in results]) + "\n"

        doc.close()
        return text_data