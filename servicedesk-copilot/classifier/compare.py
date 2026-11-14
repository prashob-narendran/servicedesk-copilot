"""Build the before/after comparison table for the report from the saved metric files.

    python -m classifier.compare --label queue
Writes reports/metrics/comparison_<label>.md and comparison_<label>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MET = Path("reports/metrics")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="queue")
    args = ap.parse_args()
    files = [f"baseline_tfidf_{args.label}.json", f"baseline_zeroshot_{args.label}.json",
             f"finetuned_{args.label}.json"]
    rows = []
    for fname in files:
        path = MET / fname
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        full, sample = d.get("test"), d.get("test_sample300")
        rows.append({
            "model": d["model"],
            "test_accuracy": full["accuracy"] if full else None,
            "test_macro_f1": full["macro_f1"] if full else None,
            "sample300_accuracy": sample["accuracy"] if sample else None,
            "sample300_macro_f1": sample["macro_f1"] if sample else None,
        })
    if not rows:
        raise SystemExit("No metric files found. Run the baselines and training first.")

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else "—"

    md = [f"## {args.label.capitalize()} classification: baselines vs fine-tuned model", "",
          "| Model | Test accuracy | Test macro-F1 | 300-sample accuracy | 300-sample macro-F1 |",
          "|---|---|---|---|---|"]
    md += [f"| {r['model']} | {fmt(r['test_accuracy'])} | {fmt(r['test_macro_f1'])} | "
           f"{fmt(r['sample300_accuracy'])} | {fmt(r['sample300_macro_f1'])} |" for r in rows]
    md += ["", "The zero-shot LLM is scored only on the fixed 300-ticket sample to limit API cost; "
               "all models are compared on that same sample in the last two columns."]
    (MET / f"comparison_{args.label}.md").write_text("\n".join(md))
    (MET / f"comparison_{args.label}.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
