"""
Uploads the fine-tuned COBS Legal-BERT model to Hugging Face Hub.
Run once: python upload_model_to_hub.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from huggingface_hub import HfApi, whoami

MODEL_DIR = os.path.join('models', 'transformers', 'legal-bert')
REPO_NAME = 'cobs-legal-bert-fca'

api = HfApi()

# Get logged-in username automatically
user = whoami()
username = user['name']
repo_id  = f"{username}/{REPO_NAME}"

print(f"Logged in as: {username}")
print(f"Uploading to: https://huggingface.co/{repo_id}")

# Create repo (private by default — change to private=False to make public)
api.create_repo(repo_id=repo_id, repo_type='model', private=True, exist_ok=True)
print("Repository created/confirmed.")

# Upload all model files
print(f"Uploading files from {MODEL_DIR} ...")
api.upload_folder(
    folder_path=MODEL_DIR,
    repo_id=repo_id,
    repo_type='model',
    commit_message='Upload fine-tuned COBS Legal-BERT (FCA COBS classification)'
)

print(f"\nDone! Model is at: https://huggingface.co/{repo_id}")
print(f"\nAdd this to .env or update api/app.py:")
print(f"  HF_MODEL_REPO = '{repo_id}'")
