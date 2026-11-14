"""Download, clean and split the customer-support ticket dataset; draw the EDA charts.

    python -m data.prepare                     # downloads Tobi-Bueck/customer-support-tickets from the Hub
    python -m data.prepare --csv my_copy.csv   # or use a local CSV with the same columns

Outputs
  data/processed/{train,val,test}.csv      stratified 80/10/10 split by queue, seed 42
  data/processed/test_sample300.csv        fixed 300-ticket test sample shared by all models (fair comparison)
  data/processed/kb_tickets.csv            resolved tickets from the TRAIN split only, for the RAG index
  reports/metrics/data_stats.json          row counts after each cleaning step, class distribution
  reports/figures/*.png                    queue, priority and length distributions
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET = "Tobi-Bueck/customer-support-tickets"
OUT = Path("data/processed")
FIG = Path("reports/figures")
MET = Path("reports/metrics")
SEED = 42


def load(csv: str | None, hub_file: str | None) -> pd.DataFrame:
    if csv:
        return pd.read_csv(csv)
    from datasets import load_dataset

    try:
        return load_dataset(DATASET, data_files=hub_file, split="train").to_pandas()
    except Exception as e:  # e.g. the repo holds several CSV versions with different columns
        from huggingface_hub import HfApi

        files = [f for f in HfApi().list_repo_files(DATASET, repo_type="dataset") if f.endswith(".csv")]
        raise SystemExit(f"Could not load {DATASET}: {e}\nCSV files in the repo: {files}\n"
                         "Re-run with --hub-file <one of these>, ideally the largest English/German release.")


def charts(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    blue = "#2F5597"

    counts = df["queue"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(counts.index, counts.values, color=blue)
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    ax.set_title("Tickets per queue (English subset)", loc="left", fontsize=11)
    ax.set_xlabel("Tickets")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "queue_distribution.png", dpi=200)
    plt.close(fig)

    pr = df["priority"].value_counts().reindex(["high", "medium", "low"]).fillna(0)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(pr.index, pr.values, color=["#C0392B", "#E67E22", "#27AE60"])
    for i, v in enumerate(pr.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Tickets per priority", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "priority_distribution.png", dpi=200)
    plt.close(fig)

    words = df["text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(words.clip(upper=400), bins=40, color=blue)
    ax.axvline(words.median(), color="#C0392B", linestyle="--", label=f"median {int(words.median())} words")
    ax.set_title("Ticket length (subject + body), clipped at 400 words", loc="left", fontsize=11)
    ax.set_xlabel("Words")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "ticket_length.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="local CSV instead of downloading")
    ap.add_argument("--hub-file", default=None, help="specific CSV file inside the Hub dataset repo")
    ap.add_argument("--kb-size", type=int, default=2000, help="resolved tickets to index for RAG")
    args = ap.parse_args()

    df = load(args.csv, args.hub_file)
    stats = {"source": args.csv or f"{DATASET}/{args.hub_file or ''}".rstrip("/"), "raw_rows": len(df)}
    df = df[df["language"].str.lower().eq("en")].copy()
    stats["english_rows"] = len(df)
    df["subject"] = df["subject"].fillna("").astype(str).str.strip()
    df["body"] = df["body"].fillna("").astype(str).str.strip()
    df = df[df["body"].str.len() > 0]
    stats["non_empty_body"] = len(df)
    df = df.drop_duplicates(subset=["subject", "body"])
    stats["after_dedup"] = len(df)
    df = df.dropna(subset=["queue", "priority"])
    df["priority"] = df["priority"].str.lower().str.strip()
    df["text"] = (df["subject"] + "\n\n" + df["body"]).str.strip()
    df = df.reset_index(drop=True)
    df["ticket_id"] = [f"{i:05d}" for i in range(len(df))]

    # drop queues too small to stratify (fewer than 10 tickets)
    small = df["queue"].value_counts()[lambda s: s < 10].index.tolist()
    if small:
        df = df[~df["queue"].isin(small)]
    stats["dropped_small_queues"] = small
    stats["final_rows"] = len(df)

    train, rest = train_test_split(df, test_size=0.2, stratify=df["queue"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["queue"], random_state=SEED)
    if len(test) > 300:
        sample, _ = train_test_split(test, train_size=300, stratify=test["queue"], random_state=SEED)
    else:
        sample = test

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["ticket_id", "subject", "body", "text", "queue", "priority", "type", "answer"]
    cols = [c for c in cols if c in df.columns]
    for name, part in [("train", train), ("val", val), ("test", test), ("test_sample300", sample)]:
        part[cols].to_csv(OUT / f"{name}.csv", index=False)
    kb = train[train["answer"].fillna("").str.len() > 20]
    kb = kb.sample(min(args.kb_size, len(kb)), random_state=SEED)
    kb[["ticket_id", "subject", "body", "answer", "queue"]].to_csv(OUT / "kb_tickets.csv", index=False)

    stats.update({
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test), "test_sample300": len(sample)},
        "kb_tickets": len(kb),
        "queues": df["queue"].value_counts().to_dict(),
        "priorities": df["priority"].value_counts().to_dict(),
        "median_words": int(df["text"].str.split().str.len().median()),
    })
    MET.mkdir(parents=True, exist_ok=True)
    (MET / "data_stats.json").write_text(json.dumps(stats, indent=2))
    charts(df)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
