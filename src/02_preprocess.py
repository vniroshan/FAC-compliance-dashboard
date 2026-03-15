import pandas as pd
import numpy as np
import re
import json
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('../data/raw/cobs_dataset.csv')

# Label encoding
LABEL2ID = {'R': 0, 'G': 1, 'E': 2, 'D': 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
df['label'] = df['type_code'].map(LABEL2ID)
df = df.dropna(subset=['label', 'clean_text'])
df['label'] = df['label'].astype(int)

# Text cleaning
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)           # collapse whitespace
    text = re.sub(r'\[deleted\]', '', text)    # remove [deleted] stubs
    text = re.sub(r'http\S+', '', text)         # remove raw URLs
    return text.strip()

df['text'] = df['clean_text'].apply(clean_text)
df = df[df['text'].str.len() > 20]    # drop near-empty stubs

# Stratified split: 70% train / 15% val / 15% test
# Use provision_ref as the key — not doc_id — to avoid data leakage
train_val, test = train_test_split(
    df, test_size=0.15, stratify=df['label'], random_state=42)
train, val = train_test_split(
    train_val, test_size=0.15/0.85, stratify=train_val['label'], random_state=42)

print(f'Train: {len(train)} | Val: {len(val)} | Test: {len(test)}')
for split_name, split_df in [('train',train),('val',val),('test',test)]:
    dist = split_df['type_code'].value_counts().to_dict()
    print(f'  {split_name}: {dist}')

# Save splits
train.to_csv('../data/processed/train.csv', index=False)
val.to_csv('../data/processed/val.csv',   index=False)
test.to_csv('../data/processed/test.csv',  index=False)

# Save label map for later use
with open('../data/processed/label_map.json', 'w') as f:
    json.dump({'label2id': LABEL2ID, 'id2label': ID2LABEL}, f)

print('Splits saved to ../data/processed/')
