## Priority classification: baselines vs fine-tuned model

| Model | Test accuracy | Test macro-F1 | 300-sample accuracy | 300-sample macro-F1 |
|---|---|---|---|---|
| TF-IDF (1-2 grams) + logistic regression | 0.667 | 0.659 | 0.680 | 0.670 |
| fine-tuned microsoft/deberta-v3-base | 0.607 | 0.611 | 0.630 | 0.636 |

The zero-shot LLM is scored only on the fixed 300-ticket sample to limit API cost; all models are compared on that same sample in the last two columns.