"""Application settings, read from environment variables or a local .env file."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Service ---
    app_name: str = "AI Service Desk Copilot"
    version: str = "1.0.0"
    api_key: str | None = None            # when set, every request needs a matching X-API-Key header
    preload_models: bool = False          # load all models at start-up instead of on first use
    device: str = "cpu"                   # "cpu" or "cuda"

    # --- LLM: "groq" (free tier, no card needed) or "openai" ---
    llm_provider: str = "groq"
    groq_api_key: str | None = None
    openai_api_key: str | None = None
    llm_model: str | None = None          # default: openai/gpt-oss-20b (groq) / gpt-4o-mini (openai)
    llm_base_url: str | None = None       # optional: any other OpenAI-compatible endpoint
    llm_judge_model: str | None = None    # evaluation judge; default openai/gpt-oss-120b (groq) / same model (openai)
    llm_temperature: float | None = 0.2   # set empty for models that do not accept temperature
    llm_timeout_s: float = 30.0

    # --- Local models ---
    classifier_model_id: str = "models/queue-classifier"       # local folder or Hugging Face Hub repo
    priority_model_id: str | None = "models/priority-classifier"  # optional; LLM fallback when missing
    sentiment_model_id: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ner_enabled: bool = True              # GLiNER entity model; regex PII detection always runs
    ner_model_id: str = "urchade/gliner_multi_pii-v1"
    ner_threshold: float = 0.5
    review_threshold: float = 0.5         # queue confidence below this is flagged for human review

    # --- RAG ---
    embedding_model_id: str = "BAAI/bge-small-en-v1.5"
    rag_index_dir: str = "rag/index"
    rag_collection: str = "techserv_kb"
    rag_score_threshold: float = 0.5      # cosine similarity; calibrate with rag/eval/run_eval.py
    rag_top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
