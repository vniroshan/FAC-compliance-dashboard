# src/03_classify_baseline.py
import pandas as pd, numpy as np, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
import joblib

# ── Load splits ───────────────────────────────────────────────
train = pd.read_csv('data/processed/train.csv')
val   = pd.read_csv('data/processed/val.csv')
test  = pd.read_csv('data/processed/test.csv')

with open('data/processed/label_map.json') as f:
    lm = json.load(f)
ID2LABEL = {int(k): v for k, v in lm['id2label'].items()}

X_tr, y_tr = train['text'], train['label']
X_val, y_val = val['text'], val['label']
X_te, y_te = test['text'], test['label']

# ── SVM + TF-IDF ──────────────────────────────────────────────
svm_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,3),
                              sublinear_tf=True, min_df=2)),
    ('clf',   LinearSVC(max_iter=2000, dual='auto'))
])

# 5-fold CV on training data
cv_svm = cross_val_score(svm_pipe, X_tr, y_tr, cv=5, scoring='f1_macro')
print(f'SVM CV F1: {cv_svm.mean():.4f} ± {cv_svm.std():.4f}')

# Hyperparameter search
param_grid = {'clf__C': [0.1, 0.5, 1.0, 5.0, 10.0]}
gs_svm = GridSearchCV(svm_pipe, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
gs_svm.fit(X_tr, y_tr)
print(f'Best C: {gs_svm.best_params_}')

svm_best = gs_svm.best_estimator_
svm_preds = svm_best.predict(X_te)

# Only use labels that actually appear in the data
all_labels = sorted(set(y_tr) | set(y_te))
label_names = [ID2LABEL[i] for i in all_labels]

print('\n── SVM Test Results ──')
print(classification_report(y_te, svm_preds,
      labels=all_labels, target_names=label_names))

joblib.dump(svm_best, 'models/baselines/svm_tfidf.joblib')

# ── Random Forest ─────────────────────────────────────────────
rf_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=30000, ngram_range=(1,2), sublinear_tf=True)),
    ('clf',   RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])
rf_pipe.fit(X_tr, y_tr)
rf_preds = rf_pipe.predict(X_te)
print('\n── Random Forest Test Results ──')
print(classification_report(y_te, rf_preds,
      labels=all_labels, target_names=label_names))

joblib.dump(rf_pipe, 'models/baselines/random_forest.joblib')

# ── Confusion matrix (save for dissertation Figure 6) ─────────
for name, preds in [('SVM', svm_preds), ('RandomForest', rf_preds)]:
    cm = confusion_matrix(y_te, preds, labels=all_labels)
    labels = label_names
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'{name} Confusion Matrix'); ax.set_ylabel('True'); ax.set_xlabel('Pred')
    plt.tight_layout()
    plt.savefig(f'data/processed/cm_{name.lower()}.png', dpi=150)
    print(f'Saved: data/processed/cm_{name.lower()}.png')
