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

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
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
