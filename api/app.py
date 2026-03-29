"""
FastAPI server that serves the trained COBS NLP models:
  - Legal-BERT  (nlpaueb/legal-bert-base-uncased, fine-tuned)  → /classify
  - SentenceTransformer (all-MiniLM-L6-v2, precomputed)       → /similarity
  - SVM + TF-IDF pipeline (sklearn, trained)                   → /classify-svm

Run:
    python api/app.py
    # -> http://localhost:8000
    # -> docs at http://localhost:8000/docs
"""

import os, sys, json, time, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

#Paths
BASE          = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BERT_DIR      = os.path.join(BASE, 'models', 'transformers', 'legal-bert')  # local fallback
HF_MODEL_REPO    = os.environ.get('HF_MODEL_REPO',    'vniroshan/cobs-legal-bert-fca')
HF_ROBERTA_REPO  = os.environ.get('HF_ROBERTA_REPO', 'vniroshan/cobs-roberta-fca')
ROBERTA_DIR      = os.path.join(BASE, 'models', 'transformers', 'roberta')
SVM_PATH      = os.path.join(BASE, 'models', 'baselines', 'svm_tfidf.joblib')
EMB_PATH      = os.path.join(BASE, 'data', 'processed', 'cobs_embeddings.npy')
BERT_EMB_PATH = os.path.join(BASE, 'data', 'processed', 'cobs_bert_embeddings.npy')
META_PATH     = os.path.join(BASE, 'data', 'processed', 'cobs_metadata.json')

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

#App
app = FastAPI(
    title       = 'COBS NLP Model API',
    description = 'Serves Legal-BERT classifier & SentenceTransformer similarity search for FCA COBS provisions.',
    version     = '1.0.0',
)
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ['*'],   # dashboard opens from file:// or localhost
    allow_methods  = ['*'],
    allow_headers  = ['*'],
)

#Global model state (loaded once at startup)
state: dict = {
    'bert_model'      : None,
    'bert_tokenizer'  : None,
    'roberta_model'   : None,
    'roberta_tokenizer': None,
    'svm_pipeline'    : None,
    'st_model'       : None,
    'embeddings'     : None,   # (802, 384) float32  — SentenceTransformer
    'bert_embeddings': None,   # (802, 768) float32  — Legal-BERT [CLS]
    'metadata'       : None,   # list of provision dicts
    'id2label'       : {0: 'R', 1: 'G', 2: 'E'},
    'label_names'    : {0: 'Rule', 1: 'Guidance', 2: 'Evidential Provision'},
    'ready'          : False,
    'errors'         : [],
}

@app.on_event('startup')
async def load_models():
    log.info('Loading models …')

    # 1. Legal-BERT — try HF Hub first, fall back to local directory
    bert_source = HF_MODEL_REPO if not os.path.isfile(os.path.join(BERT_DIR, 'model.safetensors')) else BERT_DIR
    try:
        state['bert_tokenizer'] = AutoTokenizer.from_pretrained(bert_source)
        state['bert_model']     = AutoModelForSequenceClassification.from_pretrained(bert_source)
        state['bert_model'].eval()
        log.info('Legal-BERT loaded  (%s)', bert_source)
    except Exception as e:
        msg = f'Legal-BERT load failed from {bert_source}: {e}'
        state['errors'].append(msg)
        log.error(' %s', msg)

    # 2. RoBERTa — try local first, fall back to HF Hub
    roberta_source = ROBERTA_DIR if os.path.isfile(os.path.join(ROBERTA_DIR, 'model.safetensors')) else HF_ROBERTA_REPO
    try:
        state['roberta_tokenizer'] = AutoTokenizer.from_pretrained(roberta_source)
        state['roberta_model']     = AutoModelForSequenceClassification.from_pretrained(roberta_source)
        state['roberta_model'].eval()
        log.info('RoBERTa loaded  (%s)', roberta_source)
    except Exception as e:
        msg = f'RoBERTa load failed: {e}'
        state['errors'].append(msg)
        log.warning(' %s', msg)

    # 3. SVM baseline
    try:
        state['svm_pipeline'] = joblib.load(SVM_PATH)
        log.info(' SVM + TF-IDF loaded')
    except Exception as e:
        state['errors'].append(f'SVM load failed: {e}')
        log.warning('SVM load failed: %s', e)

    # 4. SentenceTransformer for query encoding
    try:
        state['st_model']  = SentenceTransformer('all-MiniLM-L6-v2')
        state['embeddings']= np.load(EMB_PATH).astype(np.float32)
        state['metadata']  = json.load(open(META_PATH, encoding='utf-8'))
        log.info('SentenceTransformer + %d provision embeddings loaded', len(state['metadata']))
    except Exception as e:
        state['errors'].append(f'SentenceTransformer load failed: {e}')
        log.error('%s', e)

    # 5. Legal-BERT precomputed corpus embeddings (for similarity search)
    try:
        state['bert_embeddings'] = np.load(BERT_EMB_PATH).astype(np.float32)
        log.info(' Legal-BERT corpus embeddings loaded  shape=%s', state['bert_embeddings'].shape)
    except Exception as e:
        log.warning('Legal-BERT corpus embeddings not found (%s) — run scripts/precompute_bert_embeddings.py', e)

    state['ready'] = (state['bert_model'] is not None)
    log.info('Startup complete  ready=%s', state['ready'])


# Schemas
class ClassifyRequest(BaseModel):
    text   : str
    model  : Optional[str] = 'legal-bert'   # 'legal-bert' | 'roberta' | 'svm'

class ClassifyResponse(BaseModel):
    label       : str           # 'R' | 'G' | 'E'
    label_name  : str           # 'Rule' | 'Guidance' | 'Evidential Provision'
    confidence  : float         # 0–1
    probabilities: dict         # {R: float, G: float, E: float}
    citations   : List[str]
    model_used  : str
    inference_ms: float

class SimilarityRequest(BaseModel):
    query  : str
    top_k  : Optional[int] = 5
    model  : Optional[str] = 'sentence-transformer'   # 'legal-bert' | 'sentence-transformer'

class ProvisionResult(BaseModel):
    rank           : int
    provision_ref  : str
    type_code      : str
    score          : float
    text_snippet   : str

class SimilarityResponse(BaseModel):
    query      : str
    results    : List[ProvisionResult]
    model_used : str
    inference_ms: float


# Helpers
import re as _re
_CIT_RE = _re.compile(r'COBS\s+[\d]+\.[\dA-Z.]+')

def extract_citations(text: str) -> List[str]:
    return list(dict.fromkeys(_CIT_RE.findall(text)))   # deduplicated, order-preserved

def bert_classify(text: str) -> ClassifyResponse:
    tok   = state['bert_tokenizer']
    model = state['bert_model']
    inputs = tok(text, return_tensors='pt', truncation=True,
                  max_length=512, padding=True)
    t0 = time.monotonic()
    with torch.no_grad():
        logits = model(**inputs).logits
    ms  = (time.monotonic() - t0) * 1000
    probs     = torch.softmax(logits, dim=-1)[0].tolist()
    pred_idx  = int(torch.argmax(logits, dim=-1).item())
    id2label  = state['id2label']
    return ClassifyResponse(
        label        = id2label[pred_idx],
        label_name   = state['label_names'][pred_idx],
        confidence   = round(probs[pred_idx], 4),
        probabilities= {id2label[i]: round(p, 4) for i, p in enumerate(probs)},
        citations    = extract_citations(text),
        model_used   = 'Legal-BERT (nlpaueb/legal-bert-base-uncased, fine-tuned)',
        inference_ms = round(ms, 2),
    )

def roberta_classify(text: str) -> ClassifyResponse:
    tok   = state['roberta_tokenizer']
    model = state['roberta_model']
    inputs = tok(text, return_tensors='pt', truncation=True,
                  max_length=512, padding=True)
    t0 = time.monotonic()
    with torch.no_grad():
        logits = model(**inputs).logits
    ms  = (time.monotonic() - t0) * 1000
    probs     = torch.softmax(logits, dim=-1)[0].tolist()
    pred_idx  = int(torch.argmax(logits, dim=-1).item())
    id2label  = state['id2label']
    return ClassifyResponse(
        label        = id2label[pred_idx],
        label_name   = state['label_names'][pred_idx],
        confidence   = round(probs[pred_idx], 4),
        probabilities= {id2label[i]: round(p, 4) for i, p in enumerate(probs)},
        citations    = extract_citations(text),
        model_used   = 'RoBERTa-base (fine-tuned)',
        inference_ms = round(ms, 2),
    )

def svm_classify(text: str) -> ClassifyResponse:
    pipeline  = state['svm_pipeline']
    t0   = time.monotonic()
    pred = pipeline.predict([text])[0]
    # SVM decision function for confidence proxy
    try:
        dec   = pipeline.decision_function([text])[0]
        probs = torch.softmax(torch.tensor(dec, dtype=torch.float), dim=-1).tolist()
    except Exception:
        # Fallback if decision_function not available
        probs = [0.33, 0.33, 0.33]
    ms   = (time.monotonic() - t0) * 1000
    id2label  = state['id2label']
    labels    = list(id2label.values())   # ['R','G','E']
    pred_idx  = labels.index(pred) if pred in labels else 0
    return ClassifyResponse(
        label        = pred,
        label_name   = state['label_names'].get(pred_idx, pred),
        confidence   = round(max(probs), 4),
        probabilities= {labels[i]: round(p, 4) for i, p in enumerate(probs[:len(labels)])},
        citations    = extract_citations(text),
        model_used   = 'SVM + TF-IDF (sklearn, trained)',
        inference_ms = round(ms, 2),
    )


# Endpoints
@app.get('/health')
def health():
    return {
        'status'          : 'ready' if state['ready'] else 'degraded',
        'bert_loaded'     : state['bert_model']      is not None,
        'roberta_loaded'  : state['roberta_model']   is not None,
        'svm_loaded'      : state['svm_pipeline']    is not None,
        'st_loaded'       : state['st_model']        is not None,
        'corpus_size'     : len(state['metadata']) if state['metadata'] else 0,
        'errors'          : state['errors'],
    }


@app.post('/classify', response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    """
    Classify a COBS provision text using Legal-BERT or SVM.
    Returns label (R/G/E), confidence, softmax probabilities, and extracted citations.
    """
    if not req.text.strip():
        raise HTTPException(400, 'text field is empty')

    if req.model == 'svm':
        if state['svm_pipeline'] is None:
            raise HTTPException(503, 'SVM model not loaded')
        return svm_classify(req.text)
    elif req.model == 'roberta':
        if state['roberta_model'] is None:
            raise HTTPException(503, 'RoBERTa model not loaded')
        return roberta_classify(req.text)
    else:
        if state['bert_model'] is None:
            raise HTTPException(503, 'Legal-BERT model not loaded')
        return bert_classify(req.text)


@app.post('/similarity', response_model=SimilarityResponse)
def similarity(req: SimilarityRequest):
    """
    Find the top-K most semantically similar COBS provisions.
    model='legal-bert'         -> Legal-BERT [CLS] encoder (768-dim, domain-specific)
    model='sentence-transformer' -> all-MiniLM-L6-v2 (384-dim, fast general-purpose)
    """
    if not req.query.strip():
        raise HTTPException(400, 'query is empty')

    use_bert = (req.model or 'sentence-transformer').lower() == 'legal-bert'
    t0 = time.monotonic()

    if use_bert:
        # Legal-BERT [CLS] similarity
        if state['bert_model'] is None:
            raise HTTPException(503, 'Legal-BERT model not loaded')
        if state['bert_embeddings'] is None:
            raise HTTPException(503, 'Legal-BERT corpus embeddings not found — run scripts/precompute_bert_embeddings.py')

        enc = state['bert_tokenizer'](
            req.query, max_length=512, truncation=True,
            padding=True, return_tensors='pt'
        )
        import torch
        with torch.no_grad():
            # Extract base encoder hidden states (ignore the classification head)
            out = state['bert_model'].bert(**enc) if hasattr(state['bert_model'], 'bert') \
                  else state['bert_model'].base_model(**enc)
            q_vec = out.last_hidden_state[:, 0, :].squeeze(0).float().numpy()  # (768,)

        emb   = state['bert_embeddings']                                        # (802, 768)
        model_label = 'Legal-BERT [CLS] encoder (fine-tuned, 768-dim) - cosine similarity - 802 provisions'
    else:
        # SentenceTransformer similarity
        if state['st_model'] is None or state['embeddings'] is None:
            raise HTTPException(503, 'SentenceTransformer / embeddings not loaded')

        q_vec = state['st_model'].encode(req.query, normalize_embeddings=True)  # (384,)
        emb   = state['embeddings']                                              # (802, 384)
        model_label = 'SentenceTransformer (all-MiniLM-L6-v2) - cosine similarity - 802 provisions'

    # Cosine similarity (works for both 384-dim and 768-dim)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    emb_norm = emb / norms
    q_norm   = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    scores   = (emb_norm @ q_norm).tolist()                                    # (802,)

    top_k  = min(req.top_k or 5, len(scores))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    ms   = (time.monotonic() - t0) * 1000
    meta = state['metadata']

    results = []
    for rank, (idx, score) in enumerate(ranked, 1):
        m = meta[idx] if idx < len(meta) else {}
        results.append(ProvisionResult(
            rank          = rank,
            provision_ref = str(m.get('provision_ref', f'[{idx}]')),
            type_code     = str(m.get('type_code', '?')),
            score         = round(float(score), 4),
            text_snippet  = str(m.get('clean_text', ''))[:300],
        ))

    return SimilarityResponse(
        query       = req.query,
        results     = results,
        model_used  = model_label,
        inference_ms= round(ms, 2),
    )


# Run
if __name__ == '__main__':
    import uvicorn
    log.info('Starting COBS NLP API  →  http://localhost:8000')
    log.info('Swagger UI             →  http://localhost:8000/docs')
    uvicorn.run('api.app:app', host='0.0.0.0', port=8000, reload=False, workers=1)
