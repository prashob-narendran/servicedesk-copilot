"""Part II: fine-tune a transformer to classify support tickets (queue or priority).

    python -m classifier.train --label queue                       # -> models/queue-classifier
    python -m classifier.train --label priority                    # -> models/priority-classifier
    python -m classifier.train --label queue --push-to-hub <user>/servicedesk-queue-classifier

Recipe: microsoft/deberta-v3-base + classification head, weighted cross-entropy for class imbalance,
learning rate 5e-5 with linear warm-up (10%), 8 epochs, batch 32, 128 tokens, early stopping on validation
macro-F1, best checkpoint kept. (A first run with lr 2e-5 / 4 epochs / 256 tokens under-fitted.) Every run is logged
to MLflow (parameters, per-epoch metrics, final test metrics, plots).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from classifier.metrics import FIG, plot_confusion, save_json, score, top_confusions

DATA = "data/processed"


def plot_loss(log_history: list[dict], label: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = [(h["epoch"], h["loss"]) for h in log_history if "loss" in h]
    ev = [(h["epoch"], h["eval_loss"]) for h in log_history if "eval_loss" in h]
    f1 = [(h["epoch"], h["eval_macro_f1"]) for h in log_history if "eval_macro_f1" in h]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if tr:
        ax.plot(*zip(*tr), label="train loss", color="#2F5597", alpha=0.8)
    if ev:
        ax.plot(*zip(*ev), "o-", label="validation loss", color="#C0392B")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.spines[["top", "right"]].set_visible(False)
    if f1:
        ax2 = ax.twinx()
        ax2.plot(*zip(*f1), "s--", label="validation macro-F1", color="#27AE60")
        ax2.set_ylabel("Macro-F1")
        ax2.set_ylim(0, 1)
        ax2.spines[["top"]].set_visible(False)
        lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        names = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
        ax.legend(lines, names, frameon=False, fontsize=8, loc="center right")
    else:
        ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Fine-tuning curves ({label} classifier)", loc="left", fontsize=11)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"training_curves_{label}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def model_card(args, labels, test_metrics, stats) -> str:
    rows = "\n".join(f"| {lab} | {test_metrics['per_class'][lab]['precision']:.3f} | "
                     f"{test_metrics['per_class'][lab]['recall']:.3f} | {test_metrics['per_class'][lab]['f1-score']:.3f} | "
                     f"{int(test_metrics['per_class'][lab]['support'])} |" for lab in labels)
    return f"""---
license: cc-by-nc-4.0
base_model: {args.model}
datasets: [Tobi-Bueck/customer-support-tickets]
language: [en]
pipeline_tag: text-classification
---
# Service-desk {args.label} classifier (Group 60, AIMLZG549)

Fine-tuned `{args.model}` that assigns a customer-support ticket to a **{args.label}**.
Input format: `subject + "\\n\\n" + body`, max {args.max_length} tokens.

## Data
English tickets from Tobi-Bueck/customer-support-tickets (CC BY-NC 4.0), deduplicated and split 80/10/10
stratified by queue (seed 42). Train {stats['train']:,} · validation {stats['val']:,} · test {stats['test']:,}.

## Training
learning rate {args.lr} · batch size {args.batch_size} · up to {args.epochs} epochs · warm-up 10% · weight decay 0.01 ·
class-weighted cross-entropy · early stopping (patience 2) on validation macro-F1 · fp16={args.fp16} ·
seed {args.seed} · training time {stats['train_minutes']} min.

## Test results
Accuracy **{test_metrics['accuracy']:.3f}** · macro-F1 **{test_metrics['macro_f1']:.3f}** · weighted-F1 {test_metrics['weighted_f1']:.3f}

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
{rows}

## Intended use and limits
Suggests a {args.label} to a human agent; predictions below 0.5 confidence are flagged for manual routing.
Trained on synthetic-style tickets in English; accuracy on a real organisation's tickets must be re-measured.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="queue", choices=["queue", "priority"])
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--output", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--no-class-weights", dest="class_weights", action="store_false")
    ap.add_argument("--push-to-hub", default=None, help="Hub repo id, e.g. <user>/servicedesk-queue-classifier")
    ap.add_argument("--mlflow-experiment", default="servicedesk-copilot")
    args = ap.parse_args()
    args.output = args.output or f"models/{args.label}-classifier"

    import mlflow
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

    set_seed(args.seed)
    args.fp16 = args.fp16 and torch.cuda.is_available()
    train_df = pd.read_csv(f"{DATA}/train.csv")
    val_df = pd.read_csv(f"{DATA}/val.csv")
    test_df = pd.read_csv(f"{DATA}/test.csv")
    sample_df = pd.read_csv(f"{DATA}/test_sample300.csv")
    labels = sorted(train_df[args.label].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    model = model.float()  # some checkpoints load in fp16; mixed-precision training needs fp32 master weights

    def to_ds(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_dict({"text": df["text"].tolist(), "labels": [label2id[x] for x in df[args.label]]})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_length),
                      batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)

    counts = train_df[args.label].value_counts().reindex(labels).values.astype(float)
    weights = torch.tensor(counts.sum() / (len(labels) * counts), dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            y = inputs.pop("labels")
            outputs = model(**inputs)
            w = weights.to(outputs.logits.device) if args.class_weights else None
            loss = torch.nn.functional.cross_entropy(outputs.logits, y, weight=w)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        pred = np.argmax(logits, axis=-1)
        s = score([id2label[i] for i in y], [id2label[i] for i in pred], labels)
        return {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"], "weighted_f1": s["weighted_f1"]}

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    targs = TrainingArguments(
        output_dir=f"checkpoints/{args.label}",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=max(10, steps_per_epoch // 10),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        warmup_steps=int(0.1 * steps_per_epoch * args.epochs),
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=args.fp16,
        seed=args.seed,
        report_to=["mlflow"],
        run_name=f"{args.label}-{args.model.split('/')[-1]}",
    )
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")  # view with: mlflow ui --backend-store-uri sqlite:///mlflow.db
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", args.mlflow_experiment)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name=targs.run_name) as run:
        mlflow.log_params({"label": args.label, "base_model": args.model, "class_weights": args.class_weights,
                           "max_length": args.max_length, "train_rows": len(train_ds), "num_labels": len(labels)})
        trainer = WeightedTrainer(
            model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
            processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        t0 = time.time()
        trainer.train()
        train_minutes = round((time.time() - t0) / 60, 1)

        # ---- final evaluation on the untouched test set ----
        pred = trainer.predict(test_ds)
        y_pred = [id2label[i] for i in np.argmax(pred.predictions, axis=-1)]
        y_true = test_df[args.label].tolist()
        test_metrics = score(y_true, y_pred, labels)
        pos = {tid: i for i, tid in enumerate(test_df["ticket_id"])}
        idx = [pos[t] for t in sample_df["ticket_id"] if t in pos]
        sample_metrics = score([y_true[i] for i in idx], [y_pred[i] for i in idx], labels)

        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)
        Path(args.output, "label_map.json").write_text(json.dumps({"labels": labels}, indent=2))
        stats = {"train": len(train_df), "val": len(val_df), "test": len(test_df), "train_minutes": train_minutes}
        Path(args.output, "README.md").write_text(model_card(args, labels, test_metrics, stats))

        cm_path = plot_confusion(y_true, y_pred, labels, f"Confusion matrix — fine-tuned {args.label} classifier",
                                 f"confusion_matrix_{args.label}.png")
        curve_path = plot_loss(trainer.state.log_history, args.label)
        errors = top_confusions(y_true, y_pred, test_df["text"].tolist())
        result = {"model": f"fine-tuned {args.model}", "label": args.label, "output_dir": args.output,
                  "mlflow_run_id": run.info.run_id, "train_minutes": train_minutes,
                  "best_checkpoint": trainer.state.best_model_checkpoint,
                  "epochs_run": round(trainer.state.epoch or 0, 2),
                  "test": test_metrics, "test_sample300": sample_metrics, "top_confusions": errors}
        metrics_path = save_json(result, f"finetuned_{args.label}.json")
        Path(args.output, "metrics.json").write_text(json.dumps(result, indent=2))

        mlflow.log_metrics({"test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"],
                            "test_weighted_f1": test_metrics["weighted_f1"], "train_minutes": train_minutes})
        for p in (cm_path, curve_path, metrics_path):
            mlflow.log_artifact(str(p))

    print(f"\nTest accuracy {test_metrics['accuracy']:.3f} · macro-F1 {test_metrics['macro_f1']:.3f} "
          f"· saved to {args.output} · MLflow run {run.info.run_id}")
    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, exist_ok=True)
        api.upload_folder(folder_path=args.output, repo_id=args.push_to_hub,
                          commit_message=f"{args.label} classifier, macro-F1 {test_metrics['macro_f1']:.3f}")
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
