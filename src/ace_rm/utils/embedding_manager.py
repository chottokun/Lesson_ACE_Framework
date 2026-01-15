"""
Shared embedding model manager.
Provides a singleton-like access to the SentenceTransformer model
to avoid redundant model loading across components.
"""
import threading
from typing import Optional
from sentence_transformers import SentenceTransformer

from ace_rm.config import EMBEDDING_MODEL_NAME

_model: Optional[SentenceTransformer] = None
_lock = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    """
    Returns the shared SentenceTransformer model instance.
    Thread-safe initialization ensures the model is loaded only once.
    """
    global _model
    if _model is None:
        with _lock:
            # Double-check locking pattern
            if _model is None:
                _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model
