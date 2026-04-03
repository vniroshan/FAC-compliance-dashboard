"""
Research charts for COBS NLP dashboard.

Charts generated:
  1. class_distribution.png   — pie chart, N=802
  2. model_metrics_bar.png    — grouped bar chart, classification metrics by model
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(__file__))), "..", "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)


# Chart 2 – Classification Metrics by Model (%)
def plot_model_metrics():
    models   = ["SVM + TF-IDF\n(Baseline)", "Random Forest\n(Baseline)", "Legal-BERT\n(Transformer)"]
    metrics  = ["Precision", "Recall", "F1-Score", "Accuracy"]
    data = np.array([
        [90.1, 71.5, 77.7, 85.1],   # SVM + TF-IDF
        [55.0, 36.4, 31.9, 66.1],   # Random Forest
        [62.8, 64.2, 63.5, 95.0],   # Legal-BERT
    ])

    bar_colors  = ["#64789D", "#A9B8D0", "#3B4573"]   # lighter→darker for 3 models
    n_models    = len(models)
    n_metrics   = len(metrics)
    x           = np.arange(n_metrics)
    bar_w       = 0.22
    offsets     = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor("#F5F3EF")
    ax.set_facecolor("#F5F3EF")

    for i, (model, vals, col, off) in enumerate(zip(models, data, bar_colors, offsets)):
        bars = ax.bar(x + off, vals, bar_w, color=col, label=model,
                      zorder=3, linewidth=0.6, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=7.8, color="#1E1E24",
            )

    # Accent line at y=95 for Legal-BERT accuracy
    ax.axhline(95.0, color="#76BA70", lw=1.1, ls="--", alpha=0.7, zorder=2)
    ax.text(n_metrics - 0.08, 96.2, "Legal-BERT 95.0%", color="#76BA70",
            fontsize=8, ha="right", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_title("Classification Metrics by Model (%)", fontsize=14,
                 fontweight="bold", color="#1E1E24", pad=14)

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9.5,
        handlelength=1.2,
        handletextpad=0.6,
    )

    out = os.path.join(OUT_DIR, "model_metrics_bar.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {os.path.abspath(out)}")
    plt.close(fig)


plot_model_metrics()


# Chart 1 – Class Distribution pie (N = 802)
def plot_class_distribution():
    labels  = ["R – Rule", "G – Guidance", "E – Evidential\nProvision", "D – Direction"]
    counts  = [505, 285, 12, 0]
    colors  = ["#3B4573", "#64789D", "#76BA70", "#FFC107"]
    explode = [0.03, 0.03, 0.06, 0.0]
    total   = sum(counts)  # 802

    nonzero = [(l, c, col, ex) for l, c, col, ex in zip(labels, counts, colors, explode) if c > 0]
    nz_labels, nz_counts, nz_colors, nz_explode = zip(*nonzero)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    fig.patch.set_facecolor("#F5F3EF")
    ax.set_facecolor("#F5F3EF")

    wedges, _, autotexts = ax.pie(
        nz_counts,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct * total / 100))})" if pct > 3 else "",
        colors=nz_colors,
        explode=nz_explode,
        startangle=140,
        pctdistance=0.68,
        wedgeprops=dict(linewidth=1.4, edgecolor="white"),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color("white")

    # Arrow callout for tiny E slice
    e_wedge = wedges[2]
    ang = (e_wedge.theta1 + e_wedge.theta2) / 2
    rad = np.radians(ang)
    ax.annotate(
        "1.5%\n(12)",
        xy=(0.55 * np.cos(rad), 0.55 * np.sin(rad)),
        xytext=(1.25 * np.cos(rad + 0.3), 1.25 * np.sin(rad + 0.3)),
        arrowprops=dict(arrowstyle="->", color="#555", lw=1.0),
        fontsize=9, fontweight="bold", color="#333", ha="center",
    )

    patches = [mpatches.Patch(color=col, label=f"{lbl}  (n={cnt})")
               for lbl, cnt, col in zip(labels, counts, colors)]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False, fontsize=9.5, handlelength=1.4)
    ax.set_title(f"Class Distribution  (N = {total})", fontsize=14,
                 fontweight="bold", color="#1E1E24", pad=16)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {os.path.abspath(out)}")
    plt.close(fig)


plot_class_distribution()

# Chart 3 – Ablation Study: Effect of Each Pipeline Component
def plot_ablation_study():
    configs   = ["Full Hybrid", "Without ML\n(Rule only)", "Without\nConfidence Override"]
    precision = [88, 96, 85]
    recall    = [79, 58, 79]

    x      = np.arange(len(configs))
    bar_w  = 0.32
    p_color = "#2ABFBF"   # teal  – matches screenshot
    r_color = "#FFC107"   # amber – matches screenshot

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    fig.patch.set_facecolor("#F5F3EF")
    ax.set_facecolor("#F5F3EF")

    bars_p = ax.bar(x - bar_w / 2, precision, bar_w, color=p_color,
                    label="Precision", zorder=3, edgecolor="white", linewidth=0.6)
    bars_r = ax.bar(x + bar_w / 2, recall,    bar_w, color=r_color,
                    label="Recall",    zorder=3, edgecolor="white", linewidth=0.6)

    # Value labels above bars
    for bars in (bars_p, bars_r):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                str(int(bar.get_height())),
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#1E1E24",
            )

    # Highlight the trade-off for "Without ML": recall drops sharply
    ax.annotate(
        "Recall drops\n−21pp",
        xy=(1 + bar_w / 2, 58),
        xytext=(1.55, 65),
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1),
        fontsize=8.5, color="#c0392b", ha="center",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(50, 110)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_title("Ablation Study – Effect of Each Pipeline Component\nPrecision & Recall Ablation (%)",
                 fontsize=13, fontweight="bold", color="#1E1E24", pad=14)

    ax.legend(frameon=False, fontsize=10, handlelength=1.2, loc="upper right")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ablation_study_bar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {os.path.abspath(out)}")
    plt.close(fig)


plot_ablation_study()


# Chart 4 – Precision-Recall Curve: Citation Extraction (Rule-Based vs ML vs Hybrid)
def plot_citation_pr_curve():
    """
    Builds a Precision-Recall curve for the citation extraction ablation.

    - ML Only  : real PR curve via predicted probabilities from the saved
                 Logistic Regression model on the held-out test set.
    - Rule-Based Only / Hybrid Ensemble : single operating points from
                 the ablation study (no probability output available).
    """
    import joblib
    import pandas as pd
    import re
    from sklearn.metrics import precision_recall_curve, average_precision_score

    # ── Paths ──────────────────────────────────────────────────────────────
    root     = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    raw_csv  = os.path.join(root, "data", "raw",       "cobs_dataset.csv")
    clf_path = os.path.join(root, "models", "citation", "citation_clf.joblib")
    vec_path = os.path.join(root, "models", "citation", "citation_vec.joblib")

    if not os.path.isfile(clf_path):
        print("citation_clf.joblib not found – skipping PR curve chart.")
        return

    clf = joblib.load(clf_path)
    vec = joblib.load(vec_path)

    # ── Rebuild dataset (same logic as 05b_cite_ml_pipeline.py) ────────────
    BROAD_RE   = re.compile(r'COBS\s+(\d+[A-Z]?\.\d+[A-Z]?\.?\d*[A-Z]?(?:\.[A-Z])?[RGED]?)', re.I)
    TRIGGER_RE = re.compile(
        r'\b(?:in accordance with|as required (?:by|under)|pursuant to|'
        r'referred to in|under rule|see also|see\s+|as defined in|'
        r'in line with|subject to|as set out in|as specified in|'
        r'in compliance with|as described in|require[sd]? by|consistent with)\s+'
        r'COBS\s+(\d+[A-Z]?\.\d+[A-Z]?\.?\d*[A-Z]?(?:\.[A-Z])?[RGED]?)', re.I)

    df = pd.read_csv(raw_csv)
    df['clean_text'] = df['clean_text'].fillna('').astype(str)

    def norm(r): return re.sub(r'\s+', ' ', r.strip().upper())
    def cobs_density(text, start, radius=300):
        return len(BROAD_RE.findall(text[max(0, start - radius): start + radius]))

    rows = []
    for _, row in df.iterrows():
        text = row['clean_text']
        src  = norm(str(row['provision_ref']))
        trigger_tgts = {norm('COBS ' + m.group(1)) for m in TRIGGER_RE.finditer(text)}
        for m in BROAD_RE.finditer(text):
            tgt = norm('COBS ' + m.group(1))
            if tgt == src:
                continue
            ctx     = text[max(0, m.start() - 120): m.end() + 120]
            density = cobs_density(text, m.start())
            is_trigger = tgt in trigger_tgts
            is_dense   = density >= 4
            label = 1 if is_trigger else (0 if is_dense else -1)
            rows.append({'label': label, 'context': ctx,
                         'is_trigger': int(is_trigger), 'doc_id': row['doc_id']})

    all_df   = pd.DataFrame(rows).drop_duplicates(subset=['context'])
    labelled = all_df[all_df.label != -1].copy()

    # Reproduce the same 80/20 doc-stratified split
    docs      = labelled['doc_id'].unique()
    rng       = np.random.default_rng(42)
    n_test    = max(1, int(len(docs) * 0.20))
    test_docs = set(rng.choice(docs, n_test, replace=False))
    test_df   = labelled[labelled['doc_id'].isin(test_docs)].copy()

    X_test = test_df['context'].tolist()
    y_test = test_df['label'].values

    # ── ML probabilities → real PR curve ───────────────────────────────────
    X_te     = vec.transform(X_test)
    y_scores = clf.predict_proba(X_te)[:, 1]   # P(positive)
    ap       = average_precision_score(y_test, y_scores)
    prec_ml, rec_ml, _ = precision_recall_curve(y_test, y_scores)

    # ── Single operating points (from report) ──────────────────────────────
    rb_point  = (0.584, 0.962)   # Rule-Based Only  (recall, precision)
    hyb_point = (0.791, 0.883)   # Hybrid Ensemble

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    fig.patch.set_facecolor("#F5F3EF")
    ax.set_facecolor("#F5F3EF")

    # ML Only curve
    ax.plot(rec_ml, prec_ml,
            color="#3B4573", lw=2.2, zorder=3,
            label=f"ML Only – TF-IDF + LogReg  (AP = {ap:.2f})")

    # Random baseline
    positive_rate = float(y_test.mean())
    ax.axhline(positive_rate, color="#aaa", lw=1.2, ls=":", zorder=1,
               label=f"Random baseline  (P = {positive_rate:.2f})")

    # Rule-Based point
    ax.scatter(*rb_point, s=130, zorder=5,
               color="#FFC107", edgecolors="#1E1E24", linewidths=0.8,
               label=f"Rule-Based Only  (P={rb_point[1]:.2f}, R={rb_point[0]:.2f})")
    ax.annotate("Rule-Based\nOnly",
                xy=rb_point,
                xytext=(rb_point[0] - 0.18, rb_point[1] - 0.09),
                arrowprops=dict(arrowstyle="->", color="#777", lw=0.9),
                fontsize=8.5, color="#1E1E24", ha="center")

    # Hybrid point
    ax.scatter(*hyb_point, s=160, marker="D", zorder=5,
               color="#76BA70", edgecolors="#1E1E24", linewidths=0.8,
               label=f"Hybrid Ensemble  (P={hyb_point[1]:.2f}, R={hyb_point[0]:.2f})")
    ax.annotate("Hybrid\nEnsemble",
                xy=hyb_point,
                xytext=(hyb_point[0] + 0.10, hyb_point[1] - 0.09),
                arrowprops=dict(arrowstyle="->", color="#777", lw=0.9),
                fontsize=8.5, color="#1E1E24", ha="center")

    # Axes formatting
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.0,   1.08)
    ax.set_xlabel("Recall",    fontsize=12, color="#1E1E24")
    ax.set_ylabel("Precision", fontsize=12, color="#1E1E24")
    ax.set_title("Precision-Recall Curve – Citation Extraction\n"
                 "(Rule-Based vs ML-Only vs Hybrid Ensemble)",
                 fontsize=13, fontweight="bold", color="#1E1E24", pad=14)

    ax.xaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(frameon=True, fontsize=9, handlelength=1.4,
              facecolor="#F5F3EF", edgecolor="#ccc", loc="lower left")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "citation_pr_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {os.path.abspath(out)}")
    plt.close(fig)


plot_citation_pr_curve()


# Chart 5 – Hyperparameter Sensitivity Heatmap (Legal-BERT)
def plot_hyperparam_heatmap():
    """
    3×2 heatmap of validation macro-F1 (%) across:
      rows: Learning Rate  ∈ {1e-5, 2e-5, 5e-5}
      cols: Batch Size     ∈ {8, 16}
    Data loaded from data/processed/hyperparam_results.json
    """
    import json as _json

    results_path = os.path.join(OUT_DIR, "hyperparam_results.json")
    if not os.path.isfile(results_path):
        print("hyperparam_results.json not found – skipping heatmap chart.")
        return

    with open(results_path) as f:
        results = _json.load(f)

    lr_vals = sorted(set(r['lr'] for r in results))
    bs_vals = sorted(set(r['bs'] for r in results))

    # Build matrix: rows = LR, cols = BS
    matrix = np.zeros((len(lr_vals), len(bs_vals)))
    for r in results:
        i = lr_vals.index(r['lr'])
        j = bs_vals.index(r['bs'])
        matrix[i, j] = r['val_f1_macro']

    lr_labels = [f"{lr:.0e}" for lr in lr_vals]   # e.g. "1e-05"
    bs_labels = [f"BS = {bs}" for bs in bs_vals]

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
    fig.patch.set_facecolor("#F5F3EF")
    ax.set_facecolor("#F5F3EF")

    vmin = matrix.min() - 2
    vmax = matrix.max() + 2

    im = ax.imshow(matrix, cmap="Blues", aspect="auto",
                   vmin=vmin, vmax=vmax, zorder=2)

    # Annotate each cell
    best_val = matrix.max()
    for i in range(len(lr_vals)):
        for j in range(len(bs_vals)):
            val = matrix[i, j]
            is_best = (val == best_val)
            text_color = "white" if val > (vmin + vmax) / 2 else "#1E1E24"
            ax.text(j, i, f"{val:.2f}%",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold" if is_best else "normal",
                    color=text_color, zorder=4)
            if is_best:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=2.5, edgecolor="#76BA70",
                    facecolor="none", zorder=5
                )
                ax.add_patch(rect)

    ax.set_xticks(range(len(bs_vals)))
    ax.set_xticklabels(bs_labels, fontsize=11)
    ax.set_yticks(range(len(lr_vals)))
    ax.set_yticklabels(lr_labels, fontsize=11)
    ax.set_xlabel("Batch Size", fontsize=12, color="#1E1E24", labelpad=8)
    ax.set_ylabel("Learning Rate", fontsize=12, color="#1E1E24", labelpad=8)
    ax.set_title("Legal-BERT Hyperparameter Sensitivity\nValidation Macro-F1 (%)",
                 fontsize=13, fontweight="bold", color="#1E1E24", pad=14)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Macro-F1 (%)", fontsize=10, color="#1E1E24")
    cbar.ax.yaxis.set_tick_params(labelcolor="#1E1E24")
    cbar.ax.set_facecolor("#F5F3EF")

    # Remove all spines for clean look
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False, left=False)

    # Best-config annotation below chart
    best_i, best_j = np.unravel_index(matrix.argmax(), matrix.shape)
    best_lr = lr_labels[best_i]
    best_bs = bs_vals[best_j]
    fig.text(0.5, 0.01,
             f"Best config: LR = {best_lr}, BS = {best_bs}  →  F1 = {best_val:.2f}%  "
             f"(green border)",
             ha="center", fontsize=8.5, color="#555", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(OUT_DIR, "hyperparam_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {os.path.abspath(out)}")
    plt.close(fig)


plot_hyperparam_heatmap()