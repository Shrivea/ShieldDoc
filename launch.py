"""
launch_job.py — Run this on YOUR LOCAL MACHINE to kick off SageMaker training.
This script does NOT train anything — it just tells SageMaker to spin up a GPU
instance and run train.py on it.

Before running:
  pip install sagemaker boto3
  aws configure   (enter your AWS Access Key ID + Secret + region)
"""

import sagemaker
from sagemaker.huggingface import HuggingFace
import boto3

# ── Your AWS Config — Fill These In ───────────────────────────────────────────
AWS_REGION = "us-east-1"                        # Your preferred AWS region
S3_BUCKET = "pii-shield-training"               # Your S3 bucket name
#ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerRole"  # From IAM setup
ROLE_ARN = 'arn:aws:iam::058264464409:role/SageMakerRole'

# ── Job Config ─────────────────────────────────────────────────────────────────
JOB_NAME = "pii-shield-bert-finetune"
INSTANCE_TYPE = "ml.g4dn.xlarge"    # 1x T4 GPU, 16GB VRAM, ~$0.70/hr
                                     # Upgrade to ml.g5.xlarge for faster runs

# ── Setup ──────────────────────────────────────────────────────────────────────
boto3.setup_default_session(region_name=AWS_REGION)
sess = sagemaker.Session()

# ── HuggingFace Estimator ──────────────────────────────────────────────────────
# This tells SageMaker:
#   - What script to run (entry_point)
#   - What Python/transformers versions to use
#   - What instance type to provision
#   - What dependencies to install
huggingface_estimator = HuggingFace(
    entry_point="fine_tuning.py",          # Your training script
    source_dir="./sagemake",        # Folder containing train.py
    role=ROLE_ARN,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    transformers_version="4.36",     # HuggingFace transformers version
    pytorch_version="2.1",           # PyTorch version
    py_version="py310",              # Python version
    hyperparameters={},              # Add any CLI args to train.py here
    requirements_txt="requirements.txt",
    base_job_name=JOB_NAME,
    # Saves model output to: s3://pii-shield-training/pii-shield-bert-finetune-.../output/
    output_path=f"s3://{S3_BUCKET}/model-output/",
)

# ── The dataset is loaded from HuggingFace Hub inside train.py ────────────────
# No need to upload data to S3. The GPU instance has internet access during
# training and will pull ai4privacy/pii-masking-43k directly from HuggingFace.
# SM_CHANNEL_TRAINING will NOT be set in this setup — the dataset loads via
# load_dataset() inside the script.

# ── Launch ─────────────────────────────────────────────────────────────────────
print(f"Launching SageMaker training job: {JOB_NAME}")
print(f"Instance: {INSTANCE_TYPE}")
print(f"Model output will be saved to: s3://{S3_BUCKET}/model-output/")
print("Watch logs in AWS Console → SageMaker → Training Jobs")
print("Or watch live: aws logs tail /aws/sagemaker/TrainingJobs --follow")

huggingface_estimator.fit(wait=False)  # wait=False returns immediately
                                        # change to wait=True to block until done

print(f"\nJob launched. Job name: {huggingface_estimator.latest_training_job.name}")
print("Download model when done:")
print(f"  aws s3 cp s3://{S3_BUCKET}/model-output/<JOB_NAME>/output/model.tar.gz .")
print("  tar -xzf model.tar.gz -C ./models/bert-pii/")
