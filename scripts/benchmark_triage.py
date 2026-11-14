"""Send the sample tickets through the running API and record per-step latency for the report.

    python scripts/benchmark_triage.py --runs 2          # API on http://localhost:8000
Writes reports/metrics/latency.csv and reports/metrics/latency.md (median and p95 per step).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def pct(values: list[int], q: float) -> int:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.getenv("API_URL", "http://localhost:8000/api/v1"))
    ap.add_argument("--runs", type=int, default=2, help="passes over the sample tickets (first pass warms up)")
    args = ap.parse_args()
    headers = {"X-API-Key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}
    tickets = json.loads((ROOT / "ui" / "sample_tickets.json").read_text(encoding="utf-8"))

    rows = []
    for run in range(args.runs):
        for t in tickets:
            r = requests.post(f"{args.api}/triage", json={"subject": t["subject"], "body": t["body"]},
                              headers=headers, timeout=180)
            r.raise_for_status()
            d = r.json()
            row = {"run": run + 1, "ticket": t["name"], "queue": (d["queue"] or {}).get("label"),
                   "priority": (d["priority"] or {}).get("label"), "grounded_reply": (d["draft_reply"] or {}).get("grounded"),
                   "total_ms": d["meta"]["latency_ms"], **d["step_latency_ms"]}
            rows.append(row)
            print(f"run {run + 1} · {t['name'][:40]:40s} · {row['total_ms']} ms · {row['queue']} · {row['priority']}")

    out = ROOT / "reports" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "latency.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    measured = [r for r in rows if r["run"] > 1] or rows  # ignore the warm-up pass when possible
    steps = ["redact", "classify", "priority", "sentiment", "summarise", "draft_reply", "total_ms"]
    md = ["| Step | Median (ms) | p95 (ms) |", "|---|---|---|"]
    for s in steps:
        vals = [r[s] for r in measured if s in r]
        if vals:
            md.append(f"| {s.replace('_ms', '')} | {int(statistics.median(vals))} | {pct(vals, 0.95)} |")
    md.append(f"\n{len(measured)} requests measured after warm-up; CPU inference, LLM calls over the OpenAI API.")
    (out / "latency.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
