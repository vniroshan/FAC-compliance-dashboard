import pandas as pd, numpy as np, json, os
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback)
from datasets import Dataset
import evaluate
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
import wandb

#Config
#MODEL_NAME = 'nlpaueb/legal-bert-base-uncased'  # Legal-BERT
MODEL_NAME = 'roberta-base'                   # swap for RoBERTa

OUTPUT_DIR  = '../models/transformers/legal-bert'
MAX_LEN     = 512
BATCH_SIZE  = 8    
EPOCHS      = 4
LR          = 2e-5
WARMUP_RATIO= 0.1
SEED        = 42

#Load data
train_df = pd.read_csv('../data/processed/train.csv')
val_df   = pd.read_csv('../data/processed/val.csv')
test_df  = pd.read_csv('../data/processed/test.csv')

with open('../data/processed/label_map.json') as f:
    lm = json.load(f)
ID2LABEL_FULL = {int(k): v for k,v in lm['id2label'].items()}

# Only keep labels that actually appear in the data
present_labels = sorted(int(x) for x in set(train_df['label'].unique()) | set(val_df['label'].unique()) | set(test_df['label'].unique()))
NUM_LABELS = len(present_labels)
ID2LABEL = {i: ID2LABEL_FULL[i] for i in present_labels}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

print(f'Labels found: {ID2LABEL} (NUM_LABELS={NUM_LABELS})')

#Tokenise
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenise(batch):
    return tokenizer(batch['text'], truncation=True, max_length=MAX_LEN, padding=False)

train_ds = Dataset.from_pandas(train_df[['text','label']]).map(tokenise, batched=True)
val_ds   = Dataset.from_pandas(val_df[['text','label']]).map(tokenise,   batched=True)
test_ds  = Dataset.from_pandas(test_df[['text','label']]).map(tokenise,  batched=True)

#Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

#Metrics
f1_metric = evaluate.load('f1')
acc_metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1  = f1_metric.compute(predictions=preds, references=labels, average='macro')['f1']
    acc = acc_metric.compute(predictions=preds, references=labels)['accuracy']
    return {'f1_macro': f1, 'accuracy': acc}

#Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True,
    seed=SEED,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# Final evaluation on test set
print('\nTest set evaluation ')
preds_out = trainer.predict(test_ds)
y_pred = np.argmax(preds_out.predictions, axis=-1)
y_true = test_df['label'].values

all_labels = sorted(set(y_true) | set(y_pred))
label_names = [ID2LABEL[i] for i in all_labels]
report = classification_report(y_true, y_pred, labels=all_labels, target_names=label_names, output_dict=True)
print(classification_report(y_true, y_pred, labels=all_labels, target_names=label_names))

# Save report for dashboard export
import json
with open('data/processed/legal_bert_report.json', 'w') as f:
    json.dump(report, f, indent=2)

#Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=all_labels)
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names, ax=ax)
ax.set_title('Legal-BERT Confusion Matrix (Test Set)')
ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('../data/processed/cm_legal_bert.png', dpi=150)

#Save model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f'Model saved → {OUTPUT_DIR}')
