from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil, tempfile, json
import torch
import numpy as np
from transformers import AutoModelForTokenClassification, BertTokenizerFast
from processors.text_processor_img import TextProcessorImg
from processors.text_processor_pdf import TextProcessorPdf
from processors.text_processor_txt import TextProcessorTxt
from app.pii_detector import RuleBasedPIIDetector

app = FastAPI(title="ShieldDoc PII API")

# ── Regex engine ─────────────────────────────────────────────────────────────
detector = RuleBasedPIIDetector("rules.yaml")

# ── BERT engine ──────────────────────────────────────────────────────────────
MODEL_PATH = "./models/bert-pii"

try:
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()
    with open(f"{MODEL_PATH}/label_config.json") as f:
        label_config = json.load(f)
    id2label = {int(k): v for k, v in label_config["id2label"].items()}
    BERT_AVAILABLE = True
except Exception as e:
    print(f"BERT model not loaded: {e}")
    BERT_AVAILABLE = False


def bert_detect(text: str) -> list:
    """Run BERT NER inference and return list of detected PII entities."""
    if not BERT_AVAILABLE:
        return []
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    entities = []
    current_entity = None

    for token, pred_id, offsets in zip(tokens, predictions, offset_mapping):
        label = id2label.get(pred_id, "O")

        # Skip special tokens
        if token in ("[CLS]", "[SEP]", "[PAD]") or offsets == (0, 0):
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "entity_type": label[2:],  # strip B-
                "value": text[offsets[0]:offsets[1]],
                "start": offsets[0],
                "end": offsets[1],
                "confidence": 0.95,
                "source": "bert"
            }
        elif label.startswith("I-") and current_entity:
            # Extend current entity
            current_entity["value"] = text[current_entity["start"]:offsets[1]]
            current_entity["end"] = offsets[1]
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities


def run_detection(text: str):
    # Regex detections
    regex_matches = detector.detect(text)
    regex_results = [
        {
            "entity_type": m.entity_type,
            "value": m.value,
            "start": m.start,
            "end": m.end,
            "confidence": m.confidence,
            "source": "regex"
        }
        for m in regex_matches
    ]

    # BERT detections
    bert_results = bert_detect(text)

    # Merge — deduplicate by character span
    all_results = regex_results + bert_results
    seen_spans = set()
    merged = []
    for r in all_results:
        span = (r["start"], r["end"])
        if span not in seen_spans:
            seen_spans.add(span)
            merged.append(r)

    merged.sort(key=lambda x: x["start"])

    return {
        "extracted_text": text,
        "pii_found": merged,
        "redacted_text": detector.redact(text),
        "bert_available": BERT_AVAILABLE
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