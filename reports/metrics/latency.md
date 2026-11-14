| Step | Median (ms) | p95 (ms) |
|---|---|---|
| redact | 585 | 630 |
| classify | 367 | 394 |
| priority | 334 | 372 |
| sentiment | 136 | 193 |
| summarise | 465 | 803 |
| draft_reply | 2353 | 5937 |
| total | 4190 | 8211 |

6 requests measured after warm-up; CPU inference, LLM calls over the provider API (Groq or OpenAI).