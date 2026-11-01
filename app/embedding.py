"""Turns a thought into a meaning-fingerprint (vector), fully offline on CPU.

The first call downloads the model (~120 MB) once; afterwards it is cached in
/opt/not-alone/models and no network is used.
"""
import os
import threading

os.environ.setdefault("HF_HOME", "/opt/not-alone/models")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer

from .config import settings

_model: SentenceTransformer | None = None
_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = SentenceTransformer(settings.embedding_model, device="cpu")
    return _model


def embed(text: str) -> list[float]:
    vec = get_model().encode(
        text.strip(),
        normalize_embeddings=True,  # unit vectors -> cosine distance is meaningful
        convert_to_numpy=True,
    )
    return vec.tolist()
