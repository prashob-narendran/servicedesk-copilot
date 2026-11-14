"""Two baselines the fine-tuned model must beat, on the same test data.

    python -m classifier.baselines --label queue               # TF-IDF + logistic regression (free, ~1 min)
    python -m classifier.baselines --label queue --zero-shot   # + zero-shot LLM on the 300-ticket sample
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from classifier.metrics import save_json, score

DATA = "data/processed"


def tfidf_baseline(label: str) -> dict:
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    sample = pd.read_csv(f"{DATA}/test_sample300.csv")
    labels = sorted(train[label].unique())
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0),
    )
    t0 = time.time()
    model.fit(train["text"], train[label])
    res = {"model": "TF-IDF (1-2 grams) + logistic regression", "label": label,
           "train_seconds": round(time.time() - t0, 1),
           "test": score(test[label].tolist(), model.predict(test["text"]).tolist(), labels),
           "test_sample300": score(sample[label].tolist(), model.predict(sample["text"]).tolist(), labels)}
    save_json(res, f"baseline_tfidf_{label}.json")
    return res


def zero_shot_baseline(label: str) -> dict:
    from app.config import get_settings
    from app.llm import LLMClient
    from app.services.redact import Redactor

    train = pd.read_csv(f"{DATA}/train.csv")
    sample = pd.read_csv(f"{DATA}/test_sample300.csv")
    labels = sorted(train[label].unique())
    llm, redactor = LLMClient(get_settings()), Redactor(None)
    system = (f"Classify the support ticket into exactly one {label} from this list: {labels}. "
              'Reply as JSON: {"label": "<one item from the list>"}')
    preds = []
    for text in sample["text"]:
        out = llm.complete_json(system, redactor.redact(text)["redacted_text"][:3000], max_tokens=30)
        pred = str(out.get("label", "")).strip()
        preds.append(pred if pred in labels else "INVALID")
    res = {"model": f"zero-shot {llm.model}", "label": label,
           "invalid_outputs": preds.count("INVALID"),
           "test_sample300": score(sample[label].tolist(), preds, labels)}
    save_json(res, f"baseline_zeroshot_{label}.json")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="queue", choices=["queue", "priority"])
    ap.add_argument("--zero-shot", action="store_true", help="also run the LLM zero-shot baseline (needs OPENAI_API_KEY)")
    args = ap.parse_args()
    r = tfidf_baseline(args.label)
    print(f"TF-IDF + LR   test macro-F1 {r['test']['macro_f1']:.3f}  accuracy {r['test']['accuracy']:.3f}")
    if args.zero_shot:
        z = zero_shot_baseline(args.label)
        print(f"Zero-shot LLM sample300 macro-F1 {z['test_sample300']['macro_f1']:.3f}  "
              f"accuracy {z['test_sample300']['accuracy']:.3f}  invalid {z['invalid_outputs']}")


if __name__ == "__main__":
    main()
