import json, pandas as pd, numpy as np
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load test data
test_df = pd.read_csv('../data/processed/test.csv')
X_te = test_df['text']
y_te = test_df['label'].values

with open('../data/processed/label_map.json') as f:
    lm = json.load(f)
ID2LABEL = {int(k): v for k, v in lm['id2label'].items()}

# Only use labels present in the data
all_labels = sorted(set(y_te))
label_names = [ID2LABEL[i] for i in all_labels]

#Baseline predictions
print('Loading baseline models...')
svm_model = joblib.load('../models/baselines/svm_tfidf.joblib')
rf_model  = joblib.load('../models/baselines/random_forest.joblib')

svm_preds = svm_model.predict(X_te)
rf_preds  = rf_model.predict(X_te)

# Legal-BERT results (from Colab)
bert_report = json.load(open('../data/processed/legal_bert_report.json'))

# Build comparison table
results = []

# SVM
svm_report = classification_report(y_te, svm_preds, labels=all_labels,
                                   target_names=label_names, output_dict=True)
results.append({
    'Model': 'SVM + TF-IDF',
    'Precision': f"{svm_report['macro avg']['precision']*100:.1f}%",
    'Recall':    f"{svm_report['macro avg']['recall']*100:.1f}%",
    'F1 (macro)':f"{svm_report['macro avg']['f1-score']*100:.1f}%",
    'Accuracy':  f"{accuracy_score(y_te, svm_preds)*100:.1f}%"
})

# Random Forest
rf_report = classification_report(y_te, rf_preds, labels=all_labels,
                                  target_names=label_names, output_dict=True)
results.append({
    'Model': 'Random Forest',
    'Precision': f"{rf_report['macro avg']['precision']*100:.1f}%",
    'Recall':    f"{rf_report['macro avg']['recall']*100:.1f}%",
    'F1 (macro)':f"{rf_report['macro avg']['f1-score']*100:.1f}%",
    'Accuracy':  f"{accuracy_score(y_te, rf_preds)*100:.1f}%"
})

# Legal-BERT
results.append({
    'Model': 'Legal-BERT',
    'Precision': f"{bert_report['macro avg']['precision']*100:.1f}%",
    'Recall':    f"{bert_report['macro avg']['recall']*100:.1f}%",
    'F1 (macro)':f"{bert_report['macro avg']['f1-score']*100:.1f}%",
    'Accuracy':  f"{bert_report['accuracy']*100:.1f}%"
})

# Print & save
results_df = pd.DataFrame(results)
print('\nTable 2: Model Comparison')
print(results_df.to_string(index=False))
results_df.to_csv('../data/processed/table2_model_comparison.csv', index=False)
print('\nSaved -> ../data/processed/table2_model_comparison.csv')

#Per-class detail
print('\nSVM Classification Report')
print(classification_report(y_te, svm_preds, labels=all_labels, target_names=label_names))

print('Random Forest Classification Report')
print(classification_report(y_te, rf_preds, labels=all_labels, target_names=label_names))

print('Legal-BERT Classification Report')
for cls in label_names:
    if cls in bert_report:
        r = bert_report[cls]
        print(f"  {cls:25s}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  support={int(r['support'])}")

