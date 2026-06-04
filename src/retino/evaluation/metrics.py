import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from pathlib import Path


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """ Encontra o threshold que maximiza o F1-score """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def compute_metrics(
    probs:  np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """ Calcula todas as métricas relevantes para classificação binária com dados desbalanceados """
    if threshold is None:
        threshold = find_best_threshold(probs, labels)
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    n = tp + tn + fp + fn
    return {
        "auc_roc":     roc_auc_score(labels, probs),
        "auc_pr":      average_precision_score(labels, probs),
        "f1":          f1_score(labels, preds, zero_division=0),  # type: ignore[call-overload]
        "accuracy":    (tp + tn) / n if n > 0 else 0.0,
        "recall":      tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "precision":   tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


def print_metrics(name: str, m: dict) -> None:
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  AUC-ROC    : {m['auc_roc']:.4f}")
    print(f"  AUC-PR     : {m['auc_pr']:.4f}")
    print(f"  F1-Score   : {m['f1']:.4f}  (threshold={m['threshold']:.2f})")
    print(f"  Accuracy   : {m['accuracy']:.4f}")
    print(f"  Recall     : {m['recall']:.4f}   ← % doentes detectados")
    print(f"  Precision  : {m['precision']:.4f}")
    print(f"  Specificity: {m['specificity']:.4f}")
    print(f"  TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")


def plot_comparison(
    results: dict[str, dict],
    save_path: Path | None = None,
) -> None:
    """
    Gera figura comparativa com 4 subplots:
      1. Curvas ROC dos dois experimentos
      2. Curvas Precision-Recall dos dois experimentos
      3. Matriz de confusão do Experimento 1
      4. E Matriz de confusão do Experimento 2
    """
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    colors = ["#2196F3", "#FF5722"]
    names  = list(results.keys())

    # ROC
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    for name, color in zip(names, colors):
        r  = results[name]
        fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
        auc = r["metrics"]["auc_roc"]
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f"{name} (AUC={auc:.3f})")
    ax_roc.set(xlabel="FPR", ylabel="TPR", title="Curva ROC")
    ax_roc.legend(fontsize=9)
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)

    # Aqui é o recall de precisão
    ax_pr = fig.add_subplot(gs[0, 1])
    baseline = results[names[0]]["labels"].mean()
    ax_pr.axhline(baseline, color="k", ls="--", lw=0.8, alpha=0.4,
                  label=f"baseline ({baseline:.2f})")
    for name, color in zip(names, colors):
        r  = results[name]
        p, rec, _ = precision_recall_curve(r["labels"], r["probs"])
        ap = r["metrics"]["auc_pr"]
        ax_pr.plot(rec, p, color=color, lw=2,
                   label=f"{name} (AP={ap:.3f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall")
    ax_pr.legend(fontsize=9)
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)

    # Matrizes de confusão
    for i, (name, color) in enumerate(zip(names, colors)):
        ax_cm = fig.add_subplot(gs[1, i])
        r     = results[name]
        t     = r["metrics"]["threshold"]
        preds = (r["probs"] >= t).astype(int)
        cm    = confusion_matrix(r["labels"], preds)
        disp  = ConfusionMatrixDisplay(
            cm, display_labels=["Normal", "Hipertensiva"]
        )
        disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
        ax_cm.set_title(f"Confusão — {name}", fontsize=11)

    plt.suptitle("Comparação: BRSET vs BRSET+ODIR", fontsize=13, y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  figura salva em {save_path}")
    plt.show()
