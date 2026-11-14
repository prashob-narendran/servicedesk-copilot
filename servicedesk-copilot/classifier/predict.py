"""Inference wrapper for the fine-tuned ticket classifiers (queue and priority).

Hand-over interface from the API contract:

    clf = QueueClassifier("models/queue-classifier")      # local folder or Hub repo id
    clf.predict(["subject\\n\\nbody"], top_k=3)
    -> [[{"label": "Technical Support", "score": 0.81}, ...]]
"""
from __future__ import annotations


class TextClassifier:
    def __init__(self, model_id: str, device: str = "cpu", max_length: int = 256, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @property
    def labels(self) -> list[str]:
        return [self.id2label[i] for i in sorted(self.id2label)]

    def predict(self, texts: list[str], top_k: int = 3) -> list[list[dict]]:
        import torch

        top_k = max(1, min(top_k, len(self.id2label)))
        results: list[list[dict]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(batch, truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1)
            scores, idx = probs.topk(top_k, dim=-1)
            for s_row, i_row in zip(scores.tolist(), idx.tolist()):
                results.append([{"label": self.id2label[j], "score": round(s, 4)} for s, j in zip(s_row, i_row)])
        return results


# Name used in the API contract
QueueClassifier = TextClassifier


if __name__ == "__main__":  # quick smoke test: python -m classifier.predict models/queue-classifier
    import sys

    clf = TextClassifier(sys.argv[1] if len(sys.argv) > 1 else "models/queue-classifier")
    samples = [
        "VPN not connecting\n\nSince the update the VPN drops every 5 minutes for the whole finance team.",
        "Refund request\n\nI was charged twice for my annual subscription. Please refund the duplicate payment.",
        "Server outage\n\nOur production database is unreachable since 02:00. All customers are affected.",
        "Pricing question\n\nCould you send a quote for 50 additional licences?",
        "Laptop replacement\n\nMy laptop screen is broken and I need a replacement device.",
    ]
    for text, preds in zip(samples, clf.predict(samples, top_k=2)):
        print(text.split("\n")[0], "->", preds)
