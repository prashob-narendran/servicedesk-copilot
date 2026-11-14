"""Shared evaluation helpers so every model is scored the same way."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

MET = Path("reports/metrics")
FIG = Path("reports/figures")


def score(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), 4),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def save_json(obj: dict, name: str) -> Path:
    MET.mkdir(parents=True, exist_ok=True)
    path = MET / name
    path.write_text(json.dumps(obj, indent=2))
    return path


def plot_confusion(y_true, y_pred, labels, title: str, filename: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(5, 0.6 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > 0.5 else "#1F2937")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share of true class")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / filename
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def top_confusions(y_true, y_pred, texts, k: int = 3, examples: int = 3) -> list[dict]:
    pairs: dict[tuple[str, str], list[str]] = {}
    for t, p, x in zip(y_true, y_pred, texts):
        if t != p:
            pairs.setdefault((t, p), []).append(x)
    ranked = sorted(pairs.items(), key=lambda kv: len(kv[1]), reverse=True)[:k]
    return [{"true": t, "predicted": p, "count": len(xs),
             "examples": [" ".join(x.split())[:240] for x in xs[:examples]]} for (t, p), xs in ranked]
