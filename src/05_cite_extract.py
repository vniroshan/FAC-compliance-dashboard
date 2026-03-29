# src/05_cite_extract.py  ── Part A: Rule-based
import pandas as pd, re, json
from collections import defaultdict

df = pd.read_csv('../data/raw/cobs_dataset.csv')

#Pattern library
CITE_PATTERN = re.compile(
    r'COBS\s+(?P<ref>\d+[A-Z]?\.\d+[A-Z]?\.?\d*[A-Z]?'
    r'(?:\.[A-Z])?[RGED]?)',
    re.IGNORECASE
)
AMEND_VERBS = re.compile(
    r'\b(amends?|modif(?:y|ies|ied)|supersede[sd]?|replaces?)\b',
    re.IGNORECASE
)

def extract_citations(row):
    text = str(row['clean_text'])
    src  = str(row['provision_ref'])
    matches = CITE_PATTERN.finditer(text)
    relations = []
    # Check for amendment verbs in surrounding window
    for m in matches:
        tgt = 'COBS ' + m.group('ref').strip()
        if tgt.replace(' ','') == src.replace(' ',''):
            continue
        window = text[max(0,m.start()-60):m.end()+60]
        rel = 'AMENDS' if AMEND_VERBS.search(window) else 'CITES'
        relations.append({'source': src, 'target': tgt, 'relation': rel,
                          'method': 'rule', 'confidence': 0.97,
                          'doc_id': row['doc_id']})
    return relations

all_relations = []
for _, row in df.iterrows():
    all_relations.extend(extract_citations(row))

rel_df = pd.DataFrame(all_relations).drop_duplicates(subset=['source','target','relation'])
print(f'Extracted {len(rel_df)} unique relations')
print(rel_df['relation'].value_counts())

rel_df.to_csv('../data/citations/rule_based_citations.csv', index=False)
print('Saved → ../data/citations/rule_based_citations.csv')
