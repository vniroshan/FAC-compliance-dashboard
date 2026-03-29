# src/08_export_dashboard.py
# Generates dashboard_data.json from your real results

import pandas as pd, numpy as np, json, re

# ── Load your real data ───────────────────────────────────────
df      = pd.read_csv('../data/raw/cobs_dataset.csv')
report  = json.load(open('../data/processed/legal_bert_report.json'))
cit_df  = pd.read_csv('../data/citations/rule_based_citations.csv')
meta    = pd.read_json('../data/processed/cobs_metadata.json')

# ── 1. METRICS (for the 4 top cards) ─────────────────────────
metrics = {
    'classification_accuracy': round(report['accuracy'] * 100, 1),
    'citation_precision':       88.3,  # replace with your ablation result
    'citation_recall':          79.1,  # replace with your ablation result
    'corpus_size':              len(df)
}

# ── 2. CORPUS DISTRIBUTION (donut chart) ──────────────────────
dist = df['type_code'].value_counts().to_dict()
corpus_dist = {
    'R': {'count': dist.get('R',0), 'label': 'Rule'},
    'G': {'count': dist.get('G',0), 'label': 'Guidance'},
    'E': {'count': dist.get('E',0), 'label': 'Evidential Provision'},
    'D': {'count': dist.get('D',0), 'label': 'Direction'},
    'total': len(df)
}

# ── 3. PER-CLASS F1 (bar chart) ───────────────────────────────
per_class = {}
for code_val, label in [('R','Rule'),('G','Guidance'),('E','Evidential Provision'),('D','Direction')]:
    # report keys are the short codes (R, G, E), not the full names
    key = code_val
    if key in report:
        per_class[code_val] = {
            'f1':        round(report[key]['f1-score'] * 100, 1),
            'precision': round(report[key]['precision'] * 100, 1),
            'recall':    round(report[key]['recall'] * 100, 1),
            'support':   int(report[key]['support'])
        }

# ── 4. CORPUS for SIMILARITY SEARCH (corpus array) ───────────
# Extract citations from each provision
CITE_REGEX = re.compile(r'COBS\s+[\d]+\.[\d.]+[RGED]?')

def get_citations_for(provision_ref, cit_df_local):
    rels = cit_df_local[cit_df_local['source'] == provision_ref]
    cits = rels['target'].tolist()
    rel_list = rels[['source','target','relation']].rename(
        columns={'source':'s','target':'t','relation':'r'}).to_dict('records')
    return cits, rel_list

corpus_js = []
for _, row in meta.iterrows():
    cits, rels = get_citations_for(row['provision_ref'], cit_df)
    # Derive keywords from text (simple approach: nouns/key phrases)
    words = re.findall(r'\b[a-z]{4,}\b', str(row['clean_text']).lower())
    stop = {'that','with','this','from','have','been','will','into','when',
            'where','which','their','than','each','also','must','firm','shall'}
    kw = list(dict.fromkeys([w for w in words if w not in stop]))[:12]
    corpus_js.append({
        'ref':   row['provision_ref'],
        'type':  row['type_code'],
        'ch':    row['provision_ref'].split()[1].split('.')[0] if ' ' in str(row['provision_ref']) else '?',
        'title': str(row['provision_ref']),
        'text':  str(row['clean_text'])[:600],
        'cits':  cits[:8],
        'rels':  rels[:6],
        'kw':    kw
    })

# ── 5. RECENT EXTRACTIONS (extraction feed) ───────────────────
recent_ext = cit_df.head(10)[['source','target','relation']].rename(
    columns={'source':'s','target':'t','relation':'r'}).to_dict('records')
for r in recent_ext: r['c'] = 0.92  # placeholder confidence

# ── 6. NETWORK NODES (citation network) ──────────────────────
# Pick top-cited provisions for the network visualisation
top_cited = cit_df['target'].value_counts().head(6).index.tolist()
all_refs = set(top_cited) | set(cit_df['source'].value_counts().head(4).index)
ref_list = list(all_refs)[:6]

# ── Bundle and save ───────────────────────────────────────────
dashboard_data = {
    'metrics':       metrics,
    'corpus_dist':   corpus_dist,
    'per_class':     per_class,
    'corpus':        corpus_js,
    'recent_ext':    recent_ext,
    'network_refs':  ref_list
}

with open('../dashboard/dashboard_data.json', 'w') as f:
    json.dump(dashboard_data, f, indent=2)
print(f'Dashboard data exported → ../dashboard/dashboard_data.json')
print(f'  {len(corpus_js)} provisions, {len(recent_ext)} extractions')
