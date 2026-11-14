## Queue classification: baselines vs fine-tuned model

| Model | Test accuracy | Test macro-F1 | 300-sample accuracy | 300-sample macro-F1 |
|---|---|---|---|---|
| TF-IDF (1-2 grams) + logistic regression | 0.602 | 0.602 | 0.557 | 0.566 |
| zero-shot LLM (openai/gpt-oss-20b) | — | — | 0.327 | 0.253 |
| fine-tuned microsoft/deberta-v3-base | 0.434 | 0.494 | 0.417 | 0.491 |

The zero-shot LLM is scored only on the fixed 300-ticket sample to limit API cost; all models are compared on that same sample in the last two columns.