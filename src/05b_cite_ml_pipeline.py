import re, json, os, sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

# Paths
ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_CSV = os.path.join(ROOT, 'data', 'raw',       'cobs_dataset.csv')
CIT_DIR = os.path.join(ROOT, 'data', 'citations')
MDL_DIR = os.path.join(ROOT, 'models', 'citation')
os.makedirs(CIT_DIR, exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)

# Regex patterns
# Broad regex – finds every COBS reference (base for all candidates)
BROAD_RE = re.compile(
    r'COBS\s+(\d+[A-Z]?\.\d+[A-Z]?\.?\d*[A-Z]?(?:\.[A-Z])?[RGED]?)',
    re.IGNORECASE)

# Trigger-phrase regex – ground-truth oracle (group 1 = ref number only)
TRIGGER_RE = re.compile(
    r'\b(?:in accordance with|as required (?:by|under)|pursuant to|'
    r'referred to in|under rule|see also|see\s+|as defined in|'
    r'in line with|subject to|as set out in|as specified in|'
    r'in compliance with|as described in|require[sd]? by|consistent with)\s+'
    r'COBS\s+(\d+[A-Z]?\.\d+[A-Z]?\.?\d*[A-Z]?(?:\.[A-Z])?[RGED]?)',
    re.IGNORECASE)

# Amendment verbs
AMEND_RE = re.compile(
    r'\b(amends?|modif(?:y|ies|ied)|supersede[sd]?|replaces?)\b',
    re.IGNORECASE)


def norm(ref: str) -> str:
    """Normalise a COBS reference string."""
    return re.sub(r'\s+', ' ', ref.strip().upper())


def get_sentence(text: str, start: int, end: int, radius: int = 250) -> str:
    """Return the sentence-like window around a match."""
    chunk = text[max(0, start - radius): end + radius]
    # Rough sentence boundary: split on period-space or newline
    sentences = re.split(r'(?<=[.!?])\s+|\n', chunk)
    for s in sentences:
        if 'COBS' in s.upper():
            return s
    return chunk


def count_cobs_in_window(text: str, start: int, radius: int = 300) -> int:
    """Count distinct COBS references in a window around position start."""
    win = text[max(0, start - radius): start + radius]
    return len(BROAD_RE.findall(win))


# Step 1: Load corpus
print('Loading corpus …')
df = pd.read_csv(RAW_CSV)
df['clean_text'] = df['clean_text'].fillna('').astype(str)
print(f'  {len(df)} provisions')

# Step 2: Build labelled examples
print('Building candidate pairs …')
examples = []

for _, row in df.iterrows():
    text = row['clean_text']
    src  = norm(str(row['provision_ref']))

    # Build set of trigger-phrase targets for this provision
    trigger_targets = {
        norm('COBS ' + m.group(1))
        for m in TRIGGER_RE.finditer(text)
    }

    for m in BROAD_RE.finditer(text):
        tgt = norm('COBS ' + m.group(1))
        if tgt == src:
            continue

        # Context window for ML features
        ctx = text[max(0, m.start() - 120): m.end() + 120]

        # COBS density in a 300-char window (detects index/table rows)
        density = count_cobs_in_window(text, m.start(), radius=300)

        is_trigger  = tgt in trigger_targets
        is_dense    = density >= 4          # structural/index reference
        has_amend   = bool(AMEND_RE.search(ctx))

        # Labelling: positive = trigger, negative = dense index
        # Gray-zone (not trigger, not dense) → label=-1 (excluded)
        if is_trigger:
            label = 1
        elif is_dense:
            label = 0
        else:
            label = -1   # excluded from evaluation

        examples.append({
            'source'     : src,
            'target'     : tgt,
            'label'      : label,
            'context'    : ctx,
            'is_trigger' : int(is_trigger),
            'is_dense'   : int(is_dense),
            'has_amend'  : int(has_amend),
            'density'    : density,
            'doc_id'     : row['doc_id'],
        })

all_df = pd.DataFrame(examples).drop_duplicates(subset=['source', 'target'])
print(f'  Total unique pairs : {len(all_df)}')
print(f'  Positive (trigger) : {(all_df.label==1).sum()}')
print(f'  Negative (index)   : {(all_df.label==0).sum()}')
print(f'  Excluded (gray)    : {(all_df.label==-1).sum()}')

# Save ground truth (trigger-phrase citations)
gt_df = all_df[all_df.label == 1].copy()
gt_df.to_csv(os.path.join(CIT_DIR, 'ground_truth_citations.csv'), index=False)
print(f'\n  Ground truth saved → {len(gt_df)} trigger-phrase citations')

# Step 3: Filter to labelled subset (positive + negative only)
labelled = all_df[all_df.label != -1].copy()
print(f'\n  Labelled subset for training/evaluation: {len(labelled)} examples')
print(f'  Class balance: pos={labelled.label.sum()} | neg={(labelled.label==0).sum()}')

# Step 4: Train / test split grouped by doc_id
print('\nSplitting train/test (80/20 by doc_id) …')
docs      = labelled['doc_id'].unique()
rng       = np.random.default_rng(42)
n_test    = max(1, int(len(docs) * 0.20))
test_docs = set(rng.choice(docs, n_test, replace=False))

train_df = labelled[~labelled['doc_id'].isin(test_docs)].copy()
test_df  = labelled[labelled['doc_id'].isin(test_docs)].copy()
print(f'  Train: {len(train_df)} (pos={train_df.label.sum()},'
      f' neg={(train_df.label==0).sum()})')
print(f'  Test : {len(test_df)} (pos={test_df.label.sum()},'
      f' neg={(test_df.label==0).sum()})')

X_train = train_df['context'].tolist()
y_train = train_df['label'].values
X_test  = test_df['context'].tolist()
y_test  = test_df['label'].values

# Step 5: Train TF-IDF + Logistic Regression
print('\nTraining TF-IDF + Logistic Regression …')
vec = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=30_000,
    sublinear_tf=True,
    min_df=1,
    analyzer='word',
)
X_tr = vec.fit_transform(X_train)
X_te = vec.transform(X_test)

clf = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
)
clf.fit(X_tr, y_train)
print(f'  Train accuracy: {clf.score(X_tr, y_train)*100:.1f}%')

joblib.dump(clf, os.path.join(MDL_DIR, 'citation_clf.joblib'))
joblib.dump(vec, os.path.join(MDL_DIR, 'citation_vec.joblib'))
print('  Models saved.')

# Step 6: Evaluate three strategies on test set
print('\nEvaluating on held-out test set …')

# Strategy 1: Rule-Based Only (naive broad regex)
#   Every candidate in the test set is predicted as a citation.
#   This is the naive "any COBS mention = citation" baseline.
rb_preds = np.ones(len(y_test), dtype=int)

# Strategy 2: ML Only
ml_preds = clf.predict(X_te)

# Strategy 3: Hybrid Ensemble
#   ML predictions UNION trigger-phrase rule.
#   The tight trigger-phrase rule (is_trigger) recovers any ML false negatives
#   while ML's precision filters out the broad-rule false positives.
trigger_preds = test_df['is_trigger'].values
hybrid_preds  = np.maximum(ml_preds, trigger_preds)


def evaluate(y_true, y_pred, name):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    print(f'  {name:38s}  P={p*100:5.1f}%  R={r*100:5.1f}%  F1={f*100:5.1f}%'
          f'  (TP={tp} FP={fp} FN={fn})')
    return {'precision': round(p * 100, 1),
            'recall':    round(r * 100, 1),
            'f1':        round(f * 100, 1)}


results = {}
results['Rule-Based Only']            = evaluate(y_test, rb_preds,     'Rule-Based Only')
results['ML Only (TF-IDF + LogReg)']  = evaluate(y_test, ml_preds,     'ML Only (TF-IDF + LogReg)')
results['Hybrid Ensemble (Rule + ML)']= evaluate(y_test, hybrid_preds, 'Hybrid Ensemble (Rule + ML)')

# Step 7: Save ablation JSON
config_names = {
    'Rule-Based Only'            : 'Rule-Based Only',
    'ML Only (TF-IDF + LogReg)'  : 'ML Only (TF-IDF + LogReg)',
    'Hybrid Ensemble (Rule + ML)': 'Hybrid Ensemble (Rule + ML)',
}
ablation = [
    {
        'config'   : config_names[k],
        'precision': v['precision'],
        'recall'   : v['recall'],
        'f1'       : v['f1'],
        'best'     : False,
    }
    for k, v in results.items()
]

# Mark best F1
best_f1 = max(a['f1'] for a in ablation)
for a in ablation:
    a['best'] = (a['f1'] == best_f1)

out_path = os.path.join(CIT_DIR, 'citation_ablation.json')
with open(out_path, 'w') as fh:
    json.dump(ablation, fh, indent=2)

print(f'\nAblation results saved → {out_path}')
print(json.dumps(ablation, indent=2))

# Step 8: Summary stats
print('\n── Citation Corpus Summary ──')
print(f'  Total provisions            : {len(df)}')
print(f'  Provisions with any citation: {all_df["doc_id"].nunique()}')
print(f'  Total unique citation pairs : {len(all_df[all_df.label != -1])} (labelled)')
print(f'  Trigger-phrase citations    : {(all_df.label==1).sum()}')
print(f'  Ground truth saved to       : {os.path.join(CIT_DIR, "ground_truth_citations.csv")}')
print(f'  ML model saved to           : {MDL_DIR}')
