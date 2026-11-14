"""Builds and holds every model the API needs. Models load lazily on first use (or at start-up when
PRELOAD_MODELS=true), and load failures are kept so /health can report them."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.config import Settings
from app.errors import ModelNotReady
from app.llm import LLMClient

log = logging.getLogger("copilot")


class Services:
    COMPONENTS = ("ner", "classifier", "priority", "summariser", "rag")

    def __init__(self, settings: Settings, **overrides):
        """overrides (redactor=, classifier=, priority=, summariser=, rag=, llm=) let tests inject fakes."""
        self.settings = settings
        self.llm = overrides.pop("llm", None) or LLMClient(settings)
        self._objs: dict[str, object] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()
        name_map = {"redactor": "ner", "classifier": "classifier", "priority": "priority",
                    "summariser": "summariser", "rag": "rag"}
        for key, comp in name_map.items():
            if key in overrides:
                self._objs[comp] = overrides[key]

    # ---- builders ----
    def _build(self, comp: str):
        s = self.settings
        if comp == "ner":
            from app.services.redact import Redactor
            return Redactor.from_settings(s)
        if comp == "classifier":
            from classifier.predict import TextClassifier
            mid = s.classifier_model_id
            if mid.startswith(("models/", "./", "/")) and not Path(mid).exists():
                raise FileNotFoundError(f"{mid} not found. Train it (notebooks/colab_finetune.ipynb) or set "
                                        "CLASSIFIER_MODEL_ID to your Hugging Face Hub repo")
            return TextClassifier(mid, device=s.device)
        if comp == "priority":
            from app.services.priority import PriorityService
            return PriorityService.from_settings(s, self.llm)
        if comp == "summariser":
            from app.services.summarise import Summariser
            return Summariser(self.llm)
        if comp == "rag":
            from rag.pipeline import RAGPipeline
            return RAGPipeline(index_dir=s.rag_index_dir, score_threshold=s.rag_score_threshold, llm=self.llm,
                               collection_name=s.rag_collection, embedding_model_id=s.embedding_model_id,
                               device=s.device, default_top_k=s.rag_top_k)
        raise KeyError(comp)

    def get(self, comp: str):
        if comp in self._objs:
            return self._objs[comp]
        with self._lock:
            if comp not in self._objs:
                try:
                    log.info("loading %s", comp)
                    self._objs[comp] = self._build(comp)
                    self._errors.pop(comp, None)
                except Exception as e:  # keep the reason for /health and the 503 message
                    self._errors[comp] = f"{type(e).__name__}: {e}"
                    log.exception("failed to load %s", comp)
                    raise ModelNotReady(comp, str(e)) from e
        return self._objs[comp]

    def preload(self) -> None:
        for comp in self.COMPONENTS:
            try:
                self.get(comp)
            except ModelNotReady:
                pass

    # ---- convenience accessors ----
    @property
    def redactor(self):
        return self.get("ner")

    @property
    def classifier(self):
        return self.get("classifier")

    @property
    def priority(self):
        return self.get("priority")

    @property
    def summariser(self):
        return self.get("summariser")

    @property
    def rag(self):
        return self.get("rag")

    def status(self) -> dict[str, str]:
        out = {}
        for comp in self.COMPONENTS:
            if comp in self._objs:
                out[comp] = "loaded"
                if comp == "rag":
                    try:
                        out["vector_index"] = f"loaded ({self._objs['rag'].count():,} chunks)"
                    except Exception as e:  # pragma: no cover
                        out["vector_index"] = f"error: {e}"
            elif comp in self._errors:
                out[comp] = f"error: {self._errors[comp][:200]}"
            else:
                out[comp] = "not loaded yet (loads on first use)"
        out["llm"] = f"configured ({self.settings.llm_model})" if self.llm.configured else "missing OPENAI_API_KEY"
        return out
