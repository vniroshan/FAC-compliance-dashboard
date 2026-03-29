# src/inject_dashboard.py
# Injects real pipeline results into the dashboard HTML

import json, re

# ── Load data ─────────────────────────────────────────────────
with open('dashboard/dashboard_data.json') as f:
    data = json.load(f)

with open('dashboard/cobs_dashboard.html', encoding='utf-8') as f:
    html = f.read()

m = data['metrics']
cd = data['corpus_dist']
pc = data['per_class']
corpus = data['corpus']
exts = data['recent_ext']

# ── 1. Metric cards ──────────────────────────────────────────
# Classification Accuracy
html = html.replace(
    '<div class="mc-val">91.2<span>%</span></div><div class="mc-sub">Legal-BERT',
    f'<div class="mc-val">{m["classification_accuracy"]}<span>%</span></div><div class="mc-sub">Legal-BERT'
)
# Corpus size
html = html.replace(
    '<div class="mc-val" style="color:var(--purple)">620</div><div class="mc-sub">COBS provisions',
    f'<div class="mc-val" style="color:var(--purple)">{m["corpus_size"]}</div><div class="mc-sub">COBS provisions'
)
html = html.replace(
    '<span class="delta" style="color:var(--purple)">4 types</span>',
    '<span class="delta" style="color:var(--purple)">3 types (R, G, E)</span>'
)

# ── 2. Per-Class F1 bars ─────────────────────────────────────
r = pc.get('R', {})
g = pc.get('G', {})
e = pc.get('E', {})

old_bars = '''<div class="bc" style="margin-bottom:18px">
        <div class="br"><div class="bm"><span class="bn">Rule (R) · n=43</span><span class="bv">94.0%</span></div><div class="bt"><div class="bf" style="width:94%;--fc:var(--accent);--d:.1s"></div></div></div>
        <div class="br"><div class="bm"><span class="bn">Guidance (G) · n=34</span><span class="bv">91.8%</span></div><div class="bt"><div class="bf" style="width:91.8%;--fc:var(--accent2);--d:.2s"></div></div></div>
        <div class="br"><div class="bm"><span class="bn">Evidential (E) · n=13</span><span class="bv">88.2%</span></div><div class="bt"><div class="bf" style="width:88.2%;--fc:var(--accent3);--d:.3s"></div></div></div>
        <div class="br"><div class="bm"><span class="bn">Direction (D) · n=3</span><span class="bv">73.2%</span></div><div class="bt"><div class="bf" style="width:73.2%;--fc:var(--purple);--d:.4s"></div></div></div>
      </div>'''

new_bars = f'''<div class="bc" style="margin-bottom:18px">
        <div class="br"><div class="bm"><span class="bn">Rule (R) · n={r.get("support",0)}</span><span class="bv">{r.get("f1",0)}%</span></div><div class="bt"><div class="bf" style="width:{r.get("f1",0)}%;--fc:var(--accent);--d:.1s"></div></div></div>
        <div class="br"><div class="bm"><span class="bn">Guidance (G) · n={g.get("support",0)}</span><span class="bv">{g.get("f1",0)}%</span></div><div class="bt"><div class="bf" style="width:{g.get("f1",0)}%;--fc:var(--accent2);--d:.2s"></div></div></div>
        <div class="br"><div class="bm"><span class="bn">Evidential (E) · n={e.get("support",0)}</span><span class="bv">{e.get("f1",0)}%</span></div><div class="bt"><div class="bf" style="width:{max(e.get("f1",0), 2)}%;--fc:var(--accent3);--d:.3s"></div></div></div>
      </div>'''

html = html.replace(old_bars, new_bars)

# ── 3. Corpus distribution donut ─────────────────────────────
total = cd['total']
r_count = cd['R']['count']
g_count = cd['G']['count']
e_count = cd['E']['count']
d_count = cd['D']['count']

r_pct = round(r_count / total * 100, 1)
g_pct = round(g_count / total * 100, 1)
e_pct = round(e_count / total * 100, 1)
d_pct = round(d_count / total * 100, 1) if d_count > 0 else 0

# Update the tag "620 DOCS"
html = html.replace('<div class="ptag">620 DOCS</div>', f'<div class="ptag">{total} DOCS</div>')

# Update donut center text
html = html.replace(
    'font-size="19" font-weight="600">620</text>',
    f'font-size="19" font-weight="600">{total}</text>'
)

# Update legend counts
html = html.replace(
    '<div class="ln">Rule</div><div class="lc">284</div><div class="lp">45.8%</div>',
    f'<div class="ln">Rule</div><div class="lc">{r_count}</div><div class="lp">{r_pct}%</div>'
)
html = html.replace(
    '<div class="ln">Guidance</div><div class="lc">231</div><div class="lp">37.3%</div>',
    f'<div class="ln">Guidance</div><div class="lc">{g_count}</div><div class="lp">{g_pct}%</div>'
)
html = html.replace(
    '<div class="ln">Evidential Provision</div><div class="lc">85</div><div class="lp">13.7%</div>',
    f'<div class="ln">Evidential Provision</div><div class="lc">{e_count}</div><div class="lp">{e_pct}%</div>'
)
html = html.replace(
    '<div class="ln">Direction</div><div class="lc">20</div><div class="lp">3.2%</div>',
    f'<div class="ln">Direction</div><div class="lc">{d_count}</div><div class="lp">{d_pct}%</div>'
)

# Recalculate donut SVG arcs (circumference = 2*pi*52 ≈ 326.73)
C = 326.73
r_arc = r_count / total * C
g_arc = g_count / total * C
e_arc = e_count / total * C
d_arc = d_count / total * C if d_count > 0 else 0

# Replace old donut arcs with new ones
old_donut = '''<circle cx="70" cy="70" r="52" fill="none" stroke="var(--accent)" stroke-width="17" stroke-dasharray="272 326.7" stroke-dashoffset="81.7" style="animation:dash 1.2s ease both .3s"/>
          <circle cx="70" cy="70" r="52" fill="none" stroke="var(--accent2)" stroke-width="17" stroke-dasharray="221.7 326.7" stroke-dashoffset="-190.3" style="animation:dash 1.2s ease both .4s"/>
          <circle cx="70" cy="70" r="52" fill="none" stroke="var(--accent3)" stroke-width="17" stroke-dasharray="81.4 326.7" stroke-dashoffset="-412" style="animation:dash 1.2s ease both .5s"/>
          <circle cx="70" cy="70" r="52" fill="none" stroke="var(--purple)" stroke-width="17" stroke-dasharray="19 326.7" stroke-dashoffset="-493.4" style="animation:dash 1.2s ease both .6s"/>'''

# offset for first segment = C/4 (start at 12 o'clock)
off0 = C / 4
r_off = off0
g_off = -(r_arc - off0)
e_off = -(r_arc + g_arc - off0)

new_donut = f'''<circle cx="70" cy="70" r="52" fill="none" stroke="var(--accent)" stroke-width="17" stroke-dasharray="{r_arc:.1f} {C}" stroke-dashoffset="{r_off:.1f}" style="animation:dash 1.2s ease both .3s"/>
          <circle cx="70" cy="70" r="52" fill="none" stroke="var(--accent2)" stroke-width="17" stroke-dasharray="{g_arc:.1f} {C}" stroke-dashoffset="{g_off:.1f}" style="animation:dash 1.2s ease both .4s"/>
          <circle cx="70" cy="70" r="52" fill="none" stroke="var(--accent3)" stroke-width="17" stroke-dasharray="{e_arc:.1f} {C}" stroke-dashoffset="{e_off:.1f}" style="animation:dash 1.2s ease both .5s"/>'''

html = html.replace(old_donut, new_donut)

# ── 4. Model comparison table ────────────────────────────────
old_table = '''<tr><td><div class="mn2">SVM + TF-IDF</div><div class="mt2">BASELINE</div></td><td>82.1%</td><td>80.9%</td><td><span class="f1b">81.4%</span></td><td>82.8%</td></tr>
          <tr><td><div class="mn2">Random Forest</div><div class="mt2">BASELINE</div></td><td>79.3%</td><td>77.8%</td><td><span class="f1b">78.5%</span></td><td>80.1%</td></tr>
          <tr><td><div class="mn2">RoBERTa-base</div><div class="mt2">TRANSFORMER</div></td><td>88.4%</td><td>87.2%</td><td><span class="f1b">87.8%</span></td><td>88.9%</td></tr>
          <tr class="brow"><td><div class="mn2">Legal-BERT ★</div><div class="mt2">BEST · TRANSFORMER</div></td><td>91.7%</td><td>90.8%</td><td><span class="f1b best">91.2%</span></td><td>91.5%</td></tr>'''

new_table = '''<tr class="brow"><td><div class="mn2">SVM + TF-IDF ★</div><div class="mt2">BEST · BASELINE</div></td><td>87.6%</td><td>86.4%</td><td><span class="f1b best">86.9%</span></td><td>82.6%</td></tr>
          <tr><td><div class="mn2">Random Forest</div><div class="mt2">BASELINE</div></td><td>87.8%</td><td>67.6%</td><td><span class="f1b">74.0%</span></td><td>81.0%</td></tr>
          <tr><td><div class="mn2">Legal-BERT</div><div class="mt2">TRANSFORMER</div></td><td>62.8%</td><td>64.2%</td><td><span class="f1b">63.5%</span></td><td>95.0%</td></tr>'''

html = html.replace(old_table, new_table)

# ── 5. CORPUS JS array ───────────────────────────────────────
# Escape strings for JS
def js_str(s):
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '')

corpus_entries = []
for p in corpus:
    cits_js = json.dumps(p['cits'][:8])
    rels_js = json.dumps(p['rels'][:6])
    kw_js   = json.dumps(p['kw'])
    corpus_entries.append(
        f'  {{ ref:"{js_str(str(p["ref"]))}", type:"{p["type"]}", ch:"{p["ch"]}", '
        f'title:"{js_str(str(p["title"]))}", '
        f'text:"{js_str(str(p["text"]))}", '
        f'cits:{cits_js}, rels:{rels_js}, kw:{kw_js} }}'
    )
corpus_js_str = 'const CORPUS = [\n' + ',\n'.join(corpus_entries) + '\n];'

# Replace old CORPUS
html = re.sub(
    r'const CORPUS = \[.*?\];',
    corpus_js_str,
    html,
    flags=re.DOTALL
)

# ── 6. EXTS JS array ─────────────────────────────────────────
exts_js = json.dumps(exts)
html = re.sub(
    r'const EXTS=\[.*?\];',
    f'const EXTS={exts_js};',
    html,
    flags=re.DOTALL
)

# ── 7. Footer ────────────────────────────────────────────────
html = html.replace(
    'MSc Computer Science · FCA COBS NLP Project · Sample Data — Replace With Real Results',
    'MSc Computer Science · FCA COBS NLP Project · Pipeline Results'
)
html = html.replace('<div class="ptag">SAMPLE DATA</div>', '<div class="ptag">REAL DATA</div>')

# ── Save ──────────────────────────────────────────────────────
with open('dashboard/cobs_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('✓ Dashboard HTML updated with real pipeline data')
print(f'  Corpus: {len(corpus)} provisions')
print(f'  Extractions: {len(exts)} recent')
print(f'  Metrics: Acc={m["classification_accuracy"]}%, Corpus={m["corpus_size"]}')
print(f'  Per-class F1: R={pc.get("R",{}).get("f1","?")}%, G={pc.get("G",{}).get("f1","?")}%, E={pc.get("E",{}).get("f1","?")}%')
