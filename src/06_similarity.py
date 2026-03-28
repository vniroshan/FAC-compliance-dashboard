import pandas as pd, numpy as np, json
from sentence_transformers import SentenceTransformer, util
import torch

#Load corpus
df = pd.read_csv('data/raw/cobs_dataset.csv')
df = df.dropna(subset=['clean_text'])
df['text'] = df['clean_text'].str.strip()

# Choose embedding model
EMBED_MODEL = 'all-MiniLM-L6-v2'

embedder = SentenceTransformer(EMBED_MODEL)

#Encode all provisions
print(f'Encoding {len(df)} provisions...')
embeddings = embedder.encode(
    df['text'].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=True    # for cosine similarity
)

#Save embeddings + metadata
np.save('data/processed/cobs_embeddings.npy', embeddings.cpu().numpy())

meta = df[['doc_id','provision_ref','type','type_code','clean_text',
           'word_count','url','provision_date']].copy()
meta.to_json('data/processed/cobs_metadata.json', orient='records', indent=2)

print(f'Embeddings shape: {embeddings.shape}')
print('Saved: data/processed/cobs_embeddings.npy')
print('Saved: data/processed/cobs_metadata.json')
