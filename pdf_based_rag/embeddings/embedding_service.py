from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from config import Config
from utils import get_logger


logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(query, show_progress_bar=False)
        return np.asarray(vector, dtype=np.float32).tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(list(texts), show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32).tolist()
