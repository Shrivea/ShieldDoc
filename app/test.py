
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil, tempfile
from processors.text_processor_img import TextProcessorImg
from processors.text_processor_pdf import TextProcessorPdf
from processors.text_processor_txt import TextProcessorTxt
from app.pii_detector import RuleBasedPIIDetector

app = FastAPI(title="ShieldDoc PII API")
detector = RuleBasedPIIDetector("rules.yaml")


def run_detection(text: str):
    matches = detector.detect(text)
    return {
        "extracted_text": text,
        "pii_found": [
            {
                "entity_type": m.entity_type,
                "value": m.value,
                "start": m.start,
                "end": m.end,
                "confidence": m.confidence
            }
            for m in matches
        ],
        "redacted_text": detector.redact(text)
    }


@app.post("/scan/image")
async def scan_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    text = TextProcessorImg(tmp_path).process()
    tmp_path.unlink()
    return run_detection(text)


@app.post("/scan/pdf")
async def scan_pdf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    text = TextProcessorPdf(tmp_path).process()
    tmp_path.unlink()
    return run_detection(text)


@app.post("/scan/text")
async def scan_text(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    text = TextProcessorTxt(tmp_path).process()
    tmp_path.unlink()
    return run_detection(text)