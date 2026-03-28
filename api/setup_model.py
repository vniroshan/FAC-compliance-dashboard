"""
Run ONCE to set up the local Legal-BERT model directory.

The fine-tuned weights live on Hugging Face Hub:
  https://huggingface.co/vniroshan/cobs-legal-bert-fca

This script downloads them locally so the API can run fully offline.
In production, app.py loads directly from the Hub when no local weights exist.
"""

import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR  = os.path.join('models', 'transformers', 'legal-bert')
BASE_MODEL = 'nlpaueb/legal-bert-base-uncased'
LABEL_MAP  = json.load(open(os.path.join('data', 'processed', 'label_map.json')))

# Only 3 labels actually present in the dataset
ID2LABEL = {'0': 'R', '1': 'G', '2': 'E'}
LABEL2ID = {'R': 0, 'G': 1, 'E': 2}
NUM_LABELS = 3

print(f'Downloading config + tokenizer from {BASE_MODEL} …')

# 1. Build config with our fine-tuning settings
config = AutoConfig.from_pretrained(
    BASE_MODEL,
    num_labels   = NUM_LABELS,
    id2label     = ID2LABEL,
    label2id     = LABEL2ID,
    finetuning_task = 'text-classification',
)
config.save_pretrained(MODEL_DIR)
print(f'  config saved → {MODEL_DIR}/config.json')

# 2. Download and save tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(MODEL_DIR)
print(f'  tokenizer saved → {MODEL_DIR}/')

# 3. Quick sanity-check: load the full model with our weights
print('Loading fine-tuned model from weights …')
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
print(f'  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters')
print(f'  Output labels: {model.config.id2label}')

# 4. Test forward pass
tokenizer_out = tokenizer(
    'A firm must ensure best execution.',
    return_tensors='pt', truncation=True, max_length=512
)
with torch.no_grad():
    logits = model(**tokenizer_out).logits
    probs  = torch.softmax(logits, dim=-1)[0]
    pred   = logits.argmax(dim=-1).item()

print(f'\nSanity check — "A firm must ensure best execution."')
for i, (lbl, p) in enumerate(zip(ID2LABEL.values(), probs.tolist())):
    marker = ' ←' if i == pred else ''
    print(f'  {lbl}: {p*100:.1f}%{marker}')

print('\nModel directory is ready. You can now run: python api/app.py')
