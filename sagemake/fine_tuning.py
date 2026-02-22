"""
 — PII Shield BERT Fine-tuning
=======================================
Runs on SageMaker. Do not run this on your local machine.

BEFORE RUNNING THIS:
  Run check_dataset.py first to confirm column names.
  The two variables TOKEN_COLUMN and LABEL_COLUMN below must match
  exactly what check_dataset.py prints.
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
# These are all the libraries this script needs.
# SageMaker installs them from requirements_sagemaker.txt before running this.

import os           # reads SageMaker environment variables
import ast          # converts string "['a','b']" → actual list ['a','b']
import json         # saves label config to a file at the end
import logging      # prints progress messages to CloudWatch
import numpy as np  # needed to convert model output probabilities → class numbers
from datasets import Dataset, DatasetDict
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,                   # BERT's tokenizer
    AutoModelForTokenClassification,     # BERT + classification head on top
    TrainingArguments,                   # training settings object
    Trainer,                             # the training loop
    DataCollatorForTokenClassification,  # batches + pads your data
)
import evaluate  # calculates F1 score

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CONFIG
# Everything you might want to change is up here.
# ─────────────────────────────────────────────────────────────────────────────

# !! IMPORTANT — run check_dataset.py first and update these two lines !!
# These are the column names from the ai4privacy HuggingFace dataset.
# Based on the CSV preview the likely names are shown below,
# but verify with check_dataset.py before training.
TOKEN_COLUMN = "Tokenised Filled Template"           # the column that has ['in','our','dr',...]
LABEL_COLUMN = "Tokens"    # the column that has ['O','O','B-NAME',...]

# Model — starting from a clean base with no PII knowledge
BASE_MODEL = "bert-base-uncased"

# Dataset
DATASET_NAME = "ai4privacy/pii-masking-43k"

# Training settings
MAX_LENGTH  = 256   # max tokens per example (BERT supports up to 512)
BATCH_SIZE  = 16    # examples processed at once (safe for 16GB GPU)
NUM_EPOCHS  = 4     # how many full passes through the training data
LEARNING_RATE = 2e-5  # how fast the model updates (0.00002 — standard for BERT)

# SageMaker paths — set automatically by SageMaker on the GPU instance.
# If you run this locally by accident, it falls back to ./output/
MODEL_DIR  = os.environ.get("SM_MODEL_DIR",         "./output/model")
OUTPUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR",   "./output/data")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DEFINE YOUR LABEL SYSTEM
#
# The dataset has ~60 fine-grained labels like FULLNAME, FIRSTNAME, LASTNAME.
# We collapse those into 9 broader categories so each category has more
# training examples and the model learns better.
# ─────────────────────────────────────────────────────────────────────────────

# This dictionary maps original dataset labels → your collapsed labels.
# Key   = what the dataset calls it
# Value = what you want to call it
COLLAPSE_MAP = {
    # All name variants → NAME
    "FULLNAME":     "NAME",
    "FIRSTNAME":    "NAME",
    "LASTNAME":     "NAME",
    "MIDDLENAME":   "NAME",
    "PREFIX":       "NAME",
    "SUFFIX":       "NAME",
    "NAME":         "NAME",

    # Contact info → CONTACT
    "EMAIL":        "CONTACT",
    "PHONE":        "CONTACT",
    "TEL":          "CONTACT",
    "FAX":          "CONTACT",

    # Location → ADDRESS
    "STREET":       "ADDRESS",
    "BUILDINGNUMBER":"ADDRESS",
    "CITY":         "ADDRESS",
    "STATE":        "ADDRESS",
    "ZIPCODE":      "ADDRESS",
    "COUNTRY":      "ADDRESS",
    "COUNTY":       "ADDRESS",
    "SECONDARYADDRESS": "ADDRESS",

    # Online identifiers → DIGITAL_ID
    "USERNAME":     "DIGITAL_ID",
    "URL":          "DIGITAL_ID",
    "IPV4":         "DIGITAL_ID",
    "IPV6":         "DIGITAL_ID",

    # Financial → FINANCIAL
    "CREDITCARDNUMBER": "FINANCIAL",
    "IBAN":         "FINANCIAL",
    "ACCOUNTNUMBER":"FINANCIAL",
    "SALARY":       "FINANCIAL",
    "AMOUNT":       "FINANCIAL",

    # Government IDs → GOV_ID
    "SSN":          "GOV_ID",
    "DRIVERLICENSE":"GOV_ID",
    "PASSPORT":     "GOV_ID",
    "TAXIDENTIFICATIONNUMBER": "GOV_ID",

    # Medical → MEDICAL
    "MEDICALRECORD":"MEDICAL",
    "DIAGNOSIS":    "MEDICAL",
    "MEDICATION":   "MEDICAL",

    # Dates → DATETIME
    "DATE":         "DATETIME",
    "DOB":          "DATETIME",
    "AGE":          "DATETIME",

    # Gender, job area, other contextual PII
    "GENDER":       "OTHER_PII",
    "JOBAREA":      "OTHER_PII",
    "JOBTITLE":     "OTHER_PII",
}

# Build the full label list with BIO prefixes.
# Result: ["O", "B-NAME", "I-NAME", "B-CONTACT", "I-CONTACT", ...]
BASE_CATEGORIES = [
    "NAME", "CONTACT", "ADDRESS", "DIGITAL_ID",
    "FINANCIAL", "GOV_ID", "MEDICAL", "DATETIME", "OTHER_PII"
]
LABEL_LIST = ["O"]
for cat in BASE_CATEGORIES:
    LABEL_LIST.append(f"B-{cat}")
    LABEL_LIST.append(f"I-{cat}")

# label2id: "B-NAME" → 1,  "I-NAME" → 2,  "O" → 0, etc.
# id2label: 1 → "B-NAME",  2 → "I-NAME",  0 → "O", etc.
label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label  = {i: label for label, i in label2id.items()}

# Set up logging so you can watch progress in CloudWatch
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"Label set ({len(LABEL_LIST)} classes): {LABEL_LIST}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — HELPER: COLLAPSE ONE LABEL
#
# Takes one raw label from the dataset like "B-FULLNAME"
# and returns your collapsed version "B-NAME"
# ─────────────────────────────────────────────────────────────────────────────

def collapse_label(raw_label: str) -> str:
    """
    Examples:
        "O"          → "O"
        "B-FULLNAME" → "B-NAME"
        "I-FIRSTNAME"→ "I-NAME"
        "B-EMAIL"    → "B-CONTACT"
        "B-FOOBAR"   → "O"   (unknown label, treat as not-PII)
    """
    if raw_label == "O":
        return "O"

    # Split "B-FULLNAME" into prefix="B" and entity="FULLNAME"
    if "-" not in raw_label:
        return "O"  # malformed — ignore it

    prefix, entity = raw_label.split("-", 1)  # split on first hyphen only

    collapsed = COLLAPSE_MAP.get(entity, None)

    if collapsed is None:
        return "O"  # entity type we don't recognize → treat as not-PII

    return f"{prefix}-{collapsed}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PROCESS ONE DATASET ROW
#
# The dataset stores tokens and labels as STRINGS that look like lists:
#   "['in', 'our', 'video', ...]"   ← this is a string, not a list
#   "['O', 'O', 'O', ...]"          ← this is also a string, not a list
#
# ast.literal_eval converts those strings into actual Python lists.
#
# Then we convert each token → its BERT vocabulary number (input_id)
# and each label → your label number (label_id)
# ─────────────────────────────────────────────────────────────────────────────

def process_row(row, tokenizer):
    """
    Takes one dataset row and returns a dict with:
        input_ids      — list of token numbers BERT understands
        attention_mask — 1 for real tokens, 0 for padding (added later)
        labels         — list of label numbers (-100 for special tokens)
    """

    # ── Parse the string columns into actual Python lists ───────────────────
    # If the column is already a list (HuggingFace sometimes auto-parses),
    # ast.literal_eval still works fine on a list — it just returns it as-is.
    raw_tokens = row[TOKEN_COLUMN]
    raw_labels = row[LABEL_COLUMN]

    if isinstance(raw_tokens, str):
        tokens = ast.literal_eval(raw_tokens)  # "['in','our']" → ['in','our']
    else:
        tokens = raw_tokens  # already a list

    if isinstance(raw_labels, str):
        labels = ast.literal_eval(raw_labels)  # "['O','B-NAME']" → ['O','B-NAME']
    else:
        labels = raw_labels  # already a list

    # ── Sanity check ─────────────────────────────────────────────────────────
    # Every token must have exactly one label 
    if len(tokens) != len(labels):
        return {"input_ids": [tokenizer.cls_token_id, tokenizer.sep_token_id],
            "attention_mask": [1, 1],
            "labels": [-100, -100]}

    # ── Truncate if too long (BERT has a 512 token limit) ────────────────────
    # Save 2 slots for [CLS] and [SEP] that we add below
    max_content = MAX_LENGTH - 2
    tokens = tokens[:max_content]
    labels = labels[:max_content]

    # ── Convert each token string → its BERT vocabulary ID ──────────────────
    # The dataset tokens are already WordPiece tokens (they have ## in them).
    # tokenizer.convert_tokens_to_ids() just looks each one up in BERT's vocab.
    # Unknown tokens get the [UNK] id.
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # ── Collapse and convert labels → label IDs ──────────────────────────────
    label_ids = [label2id.get(collapse_label(l), label2id["O"]) for l in labels]

    # ── Add [CLS] at the start and [SEP] at the end ──────────────────────────
    # BERT expects every sequence to start with [CLS] and end with [SEP].
    # We give these special tokens the label -100 so the loss function
    # ignores them — we don't want to train the model to predict labels
    # for [CLS] and [SEP], only for real words.
    cls_id = tokenizer.cls_token_id  # [CLS] = 101
    sep_id = tokenizer.sep_token_id  # [SEP] = 102

    input_ids  = [cls_id] + token_ids  + [sep_id]
    label_ids  = [-100]   + label_ids  + [-100]

    # ── Attention mask ────────────────────────────────────────────────────────
    # 1 means "this is a real token, pay attention to it"
    # 0 means "this is padding, ignore it" (we add padding later in batches)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         label_ids,
    }

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — METRICS
#
# After each epoch the Trainer calls this to score the model.
# seqeval is the standard library for NER evaluation.
# It gives precision, recall, and F1 per entity class.
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    seqeval = evaluate.load("seqeval")
    logits, labels = eval_pred

    # logits shape: (batch_size, seq_length, num_labels)
    # argmax along last axis gives the predicted class for each token
    predictions = np.argmax(logits, axis=2)

    # Convert IDs back to label strings, skipping -100 (special tokens)
    true_labels = [
        [LABEL_LIST[l] for l in label_row if l != -100]
        for label_row in labels
    ]
    true_preds = [
        [LABEL_LIST[p] for p, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]

    result = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": result["overall_precision"],
        "recall":    result["overall_recall"],
        "f1":        result["overall_f1"],
        "accuracy":  result["overall_accuracy"],
    }

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — MAIN
# Everything above was just defining functions.
# This is where the actual work happens, in order.
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 50)
    logger.info("PII Shield — Starting Fine-tuning")
    logger.info(f"Base model : {BASE_MODEL}")
    logger.info(f"Dataset    : {DATASET_NAME}")
    logger.info(f"Epochs     : {NUM_EPOCHS}")
    logger.info(f"Batch size : {BATCH_SIZE}")
    logger.info(f"Model save : {MODEL_DIR}")
    logger.info("=" * 50)

    # ── 6a. Load tokenizer ───────────────────────────────────────────────────
    # The tokenizer knows BERT's vocabulary — it's the lookup table that
    # converts token strings like "rolf" into numbers like 14912.
    logger.info("Loading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL)

    # ── 6b. Load dataset ─────────────────────────────────────────────────────
    logger.info("Loading dataset from HuggingFace...")
    import pandas as pd
    from datasets import Dataset, DatasetDict
    logger.info("Loading dataset with pandas (skipping bad lines)...")
    df = pd.read_csv(
    "https://huggingface.co/datasets/ai4privacy/pii-masking-43k/resolve/main/PII43k.csv",
    on_bad_lines="skip",
    engine="python"
)
# Split 90% train, 10% validation
    split = int(len(df) * 0.9)
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df   = df.iloc[split:].reset_index(drop=True)

    dataset = DatasetDict({
        "train":      Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
    })
    logger.info(f"Dataset loaded: {dataset}")
    logger.info(f"Dataset loaded: {dataset}")
    logger.info(f"Column names: {dataset['train'].column_names}")
    logger.info(f"First row keys: {list(dataset['train'][0].keys())}")

    # ── 6c. Process every row ────────────────────────────────────────────────
    # Apply process_row() to every example in the dataset.
    # batched=False means one row at a time (simpler to debug).
    # remove_columns drops the original string columns — you only keep
    # input_ids, attention_mask, and labels.
    logger.info("Processing dataset rows (tokenizing + label conversion)...")
    processed = dataset.map(
        lambda row: process_row(row, tokenizer),
        batched=False,
        remove_columns=dataset["train"].column_names,
        desc="Processing rows",
    )
    logger.info(f"Processed dataset: {processed}")

    # Check one processed example looks right
    sample = processed["train"][0]
    logger.info(f"Sample input_ids (first 10): {sample['input_ids'][:10]}")
    logger.info(f"Sample labels   (first 10): {sample['labels'][:10]}")
    logger.info(f"Sequence length: {len(sample['input_ids'])}")

    # ── 6d. Load base BERT model ─────────────────────────────────────────────
    # AutoModelForTokenClassification loads standard BERT and adds a small
    # linear layer on top. That linear layer takes BERT's output for each
    # token and predicts which of your 19 labels it is.
    # This new layer is initialized randomly and is what gets trained.
    logger.info("Loading base BERT model...")
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_LIST),     # tells it to make a 19-class output layer
        id2label=id2label,              # saves label mapping inside the model config
        label2id=label2id,
        ignore_mismatched_sizes=True,   # needed because we're replacing the output layer
    )

    # ── 6e. Training settings ────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,

        # Evaluate and save after every epoch
        evaluation_strategy="epoch",
        save_strategy="epoch",

        # At the end, load whichever epoch had the best F1
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # Logging
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=100,              # print loss every 100 batches

        save_total_limit=2,             # only keep last 2 checkpoints to save disk
        fp16=torch.cuda.is_available(), # use mixed precision if GPU is available
        report_to="none",               # disable wandb
    )

    # ── 6f. Data collator ────────────────────────────────────────────────────
    # Your processed rows all have different lengths.
    # The DataCollator groups them into batches and pads shorter sequences
    # to match the longest one in each batch.
    # It automatically adds 0s to input_ids and attention_mask for padding,
    # and adds -100 to labels for padding positions (so loss ignores them).
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )

    # ── 6g. Build trainer ────────────────────────────────────────────────────
    # You hand the Trainer everything it needs.
    # It handles the training loop, gradient updates, evaluation, saving.
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=processed["train"],
        eval_dataset=processed["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ── 6h. Train ─────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info(f"Training done: {train_result.metrics}")

    # ── 6i. Final evaluation ──────────────────────────────────────────────────
    logger.info("Running final evaluation on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Final F1    : {eval_results.get('eval_f1', 'N/A'):.4f}")
    logger.info(f"Precision   : {eval_results.get('eval_precision', 'N/A'):.4f}")
    logger.info(f"Recall      : {eval_results.get('eval_recall', 'N/A'):.4f}")

    # ── 6j. Save model ────────────────────────────────────────────────────────
    # SageMaker automatically uploads /opt/ml/model/ to S3 when done.
    logger.info(f"Saving model to {MODEL_DIR}...")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # Save label config alongside the model so your app can load it
    label_config = {"label_list": LABEL_LIST, "label2id": label2id, "id2label": id2label}
    with open(os.path.join(MODEL_DIR, "label_config.json"), "w") as f:
        json.dump(label_config, f, indent=2)

    # Save eval metrics for your presentation
    with open(os.path.join(MODEL_DIR, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    logger.info("All done. Model saved and ready to download from S3.")


if __name__ == "__main__":
    main()