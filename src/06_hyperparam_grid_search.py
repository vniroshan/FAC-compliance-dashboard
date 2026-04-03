
import os, sys, json, shutil
import pandas as pd
import numpy as np
import torch

# Resolve paths relative to this script
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA_DIR = os.path.join(ROOT, 'data', 'processed')
OUT_JSON = os.path.join(DATA_DIR, 'hyperparam_results.json')
TMP_BASE = os.path.join(ROOT, 'tmp_grid_search')

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset
import evaluate

# Config
MODEL_NAME  = 'nlpaueb/legal-bert-base-uncased'
MAX_LEN     = 128      # fast CPU training; captures ~60% of text sequences fully
NUM_EPOCHS  = 2
SEED        = 42
LEARNING_RATES = [1e-5, 2e-5, 5e-5]
BATCH_SIZES    = [8, 16]

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
val_df   = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))

with open(os.path.join(DATA_DIR, 'label_map.json')) as f:
    lm = json.load(f)
ID2LABEL_FULL = {int(k): v for k, v in lm['id2label'].items()}

present_labels = sorted(
    int(x) for x in set(train_df['label'].unique()) | set(val_df['label'].unique())
)
NUM_LABELS = len(present_labels)
ID2LABEL   = {i: ID2LABEL_FULL[i] for i in present_labels}
LABEL2ID   = {v: k for k, v in ID2LABEL.items()}

print(f"Labels: {ID2LABEL}")
print(f"Train: {len(train_df)} | Val: {len(val_df)}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"Grid: {len(LEARNING_RATES)} LRs × {len(BATCH_SIZES)} BSs = "
      f"{len(LEARNING_RATES)*len(BATCH_SIZES)} runs\n")

# Tokeniser (shared across runs)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenise(batch):
    return tokenizer(batch['text'], truncation=True, max_length=MAX_LEN, padding=False)

train_ds = Dataset.from_pandas(train_df[['text', 'label']]).map(tokenise, batched=True)
val_ds   = Dataset.from_pandas(val_df[['text', 'label']]).map(tokenise, batched=True)

#  Metric 
f1_metric = evaluate.load('f1')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_metric.compute(predictions=preds, references=labels, average='macro')['f1']
    return {'f1_macro': f1}

#  Grid search 
results = []
total   = len(LEARNING_RATES) * len(BATCH_SIZES)
run_idx = 0

for lr in LEARNING_RATES:
    for bs in BATCH_SIZES:
        run_idx += 1
        lr_str  = f"{lr:.0e}".replace('e-0', 'e-')   # e.g. "2e-5"
        run_tag = f"lr{lr_str}_bs{bs}"
        out_dir = os.path.join(TMP_BASE, run_tag)

        print(f"[{run_idx}/{total}] LR={lr:.0e}  BS={bs}  -> {run_tag}")

        # Fresh model for every run
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        training_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=32,
            learning_rate=lr,
            warmup_ratio=0.1,
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
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Best val F1 seen during training
        best_f1 = max(
            log['eval_f1_macro']
            for log in trainer.state.log_history
            if 'eval_f1_macro' in log
        )
        best_f1 = round(best_f1 * 100, 2)   # store as percentage

        results.append({'lr': lr, 'bs': bs, 'val_f1_macro': best_f1})
        print(f"    -> best val macro-F1 = {best_f1:.2f}%\n")

        # Free GPU/CPU memory
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clean up checkpoint files to save disk space
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)

# Save results
os.makedirs(DATA_DIR, exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print("=" * 60)
print(f"Grid search complete. Results saved -> {OUT_JSON}")
print()
print(f"{'LR':>10}  {'BS':>4}  {'Val macro-F1 (%)':>17}")
print("-" * 36)
for r in results:
    print(f"{r['lr']:>10.0e}  {r['bs']:>4}  {r['val_f1_macro']:>17.2f}")

# Clean up temp directory
if os.path.isdir(TMP_BASE):
    shutil.rmtree(TMP_BASE, ignore_errors=True)
