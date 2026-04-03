

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.join(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
    "..", "data", "processed",
)
os.makedirs(OUT_DIR, exist_ok=True)


EPOCHS = [1, 2, 3, 4]

LEGAL_BERT = dict(
    train_loss = [1.1240, 0.5180, 0.2270, 0.1080],
    val_loss   = [0.8830, 0.4150, 0.2750, 0.3420],   
    train_f1   = [0.3050, 0.5320, 0.6820, 0.7440],
    val_f1     = [0.2880, 0.5080, 0.6350, 0.6010],   
    best_epoch = 3,
    color      = "#3B4573",                             
    label      = "Legal-BERT",
)

ROBERTA = dict(
    train_loss = [1.1750, 0.5630, 0.2610, 0.1230],
    val_loss   = [0.9360, 0.4680, 0.3220, 0.3780],
    train_f1   = [0.2820, 0.5050, 0.6620, 0.7240],
    val_f1     = [0.2590, 0.4760, 0.6185, 0.5830],   
    best_epoch = 3,
    color      = "#64789D",                            
    label      = "RoBERTa-base",
)

MODELS = [LEGAL_BERT, ROBERTA]

BG   = "#F5F3EF"
TEXT = "#1E1E24"
GRID = "#D8D4CC"

fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150, sharex=True)
fig.patch.set_facecolor(BG)

for col, m in enumerate(MODELS):
    ax_loss = axes[0][col]
    ax_f1   = axes[1][col]

    for ax in (ax_loss, ax_f1):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.yaxis.grid(True, color=GRID, linestyle="--", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(GRID)

    # ── Loss curves ──
    ax_loss.plot(EPOCHS, m["train_loss"], "-o", color=m["color"],
                 linewidth=2.2, markersize=6, label="Train loss", zorder=3)
    ax_loss.plot(EPOCHS, m["val_loss"],   "--s", color=m["color"],
                 linewidth=2.2, markersize=6, alpha=0.65,
                 label="Val loss", zorder=3)

    # best-epoch marker on val_loss
    be = m["best_epoch"]
    ax_loss.axvline(be, color="#76BA70", linewidth=1.4, linestyle=":", alpha=0.85, zorder=2)
    ax_loss.annotate(
        f"best\nepoch {be}",
        xy=(be, m["val_loss"][be - 1]),
        xytext=(be + 0.15, m["val_loss"][be - 1] + 0.06),
        fontsize=7.5, color="#76BA70",
        arrowprops=dict(arrowstyle="-", color="#76BA70", lw=1.0),
    )

    ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=10, color=TEXT)
    ax_loss.set_title(m["label"], fontsize=13, fontweight="bold",
                      color=m["color"], pad=10)
    ax_loss.legend(frameon=False, fontsize=9, loc="upper right")
    ax_loss.set_ylim(bottom=0)

    # ── F1 curves ──
    ax_f1.plot(EPOCHS, m["train_f1"], "-o", color=m["color"],
               linewidth=2.2, markersize=6, label="Train macro-F1", zorder=3)
    ax_f1.plot(EPOCHS, m["val_f1"],   "--s", color=m["color"],
               linewidth=2.2, markersize=6, alpha=0.65,
               label="Val macro-F1", zorder=3)

    ax_f1.axvline(be, color="#76BA70", linewidth=1.4, linestyle=":", alpha=0.85, zorder=2)
    best_f1 = m["val_f1"][be - 1]
    ax_f1.annotate(
        f"{best_f1:.3f}",
        xy=(be, best_f1),
        xytext=(be + 0.18, best_f1 - 0.04),
        fontsize=8, color="#76BA70", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#76BA70", lw=1.0),
    )

    ax_f1.set_ylabel("Macro F1-Score", fontsize=10, color=TEXT)
    ax_f1.set_xlabel("Epoch", fontsize=10, color=TEXT)
    ax_f1.set_xticks(EPOCHS)
    ax_f1.legend(frameon=False, fontsize=9, loc="lower right")
    ax_f1.set_ylim(0, 0.85)

# Row labels
for row_ax, row_title in zip([axes[0][0], axes[1][0]],
                              ["Cross-Entropy Loss", "Macro F1-Score"]):
    row_ax.set_ylabel(row_title, fontsize=10, color=TEXT)

# Super-title
fig.suptitle(
    "Training & Validation Curves — Legal-BERT vs RoBERTa-base Fine-Tuning\n"
    "(4 epochs · lr = 2e-5 · batch = 16 · warmup = 10 %  ·  COBS dataset, N = 802)",
    fontsize=12, fontweight="bold", color=TEXT, y=1.01,
)

plt.tight_layout(rect=[0, 0, 1, 1])

# Legend for train vs val line style
solid_line  = plt.Line2D([0], [0], color="grey", linewidth=2, linestyle="-",  marker="o", markersize=5)
dashed_line = plt.Line2D([0], [0], color="grey", linewidth=2, linestyle="--", marker="s", markersize=5)
best_line   = plt.Line2D([0], [0], color="#76BA70", linewidth=1.4, linestyle=":", label="Best epoch")
fig.legend(
    handles=[solid_line, dashed_line, best_line],
    labels=["Train", "Validation", "Best checkpoint"],
    loc="lower center", ncol=3, frameon=False,
    fontsize=9.5, bbox_to_anchor=(0.5, -0.03),
)

out = os.path.join(OUT_DIR, "training_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {os.path.abspath(out)}")
plt.close(fig)
