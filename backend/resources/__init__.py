"""
resources/__init__.py

Equivalent of the reference pattern's `models/__init__.py` (engine, Base,
get_db) -- except our "database" is a local embedding model and a Qdrant
client. Both are loaded ONCE via the FastAPI lifespan in main.py, not
per-request, since the embedding model is ~1GB and reloading it per
request would be prohibitively slow.
"""

import logging

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from configs import BaseConfig

logger = logging.getLogger(__name__)

_embedding_model: SentenceTransformer | None = None
_qdrant_client: QdrantClient | None = None


def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model once. Safe to call multiple times --
    subsequent calls return the already-loaded instance."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model '%s'...", BaseConfig.EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(BaseConfig.EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _embedding_model


def get_embedding_model() -> SentenceTransformer:
    """Dependency-style accessor for use inside request handlers.
    Raises clearly if called before lifespan startup ran."""
    if _embedding_model is None:
        raise RuntimeError(
            "Embedding model not loaded -- load_embedding_model() must be "
            "called during app startup (see main.py lifespan)."
        )
    return _embedding_model


def connect_qdrant() -> QdrantClient:
    """Connect to Qdrant once. Safe to call multiple times."""
    global _qdrant_client
    if _qdrant_client is None:
        logger.info("Connecting to Qdrant at %s:%s...", BaseConfig.QDRANT_HOST, BaseConfig.QDRANT_PORT)
        _qdrant_client = QdrantClient(
            host=BaseConfig.QDRANT_HOST,
            port=BaseConfig.QDRANT_PORT,
            check_compatibility=False,
        )
        logger.info("Connected to Qdrant.")
    return _qdrant_client


def get_qdrant_client() -> QdrantClient:
    """Dependency-style accessor for use inside request handlers."""
    if _qdrant_client is None:
        raise RuntimeError(
            "Qdrant client not connected -- connect_qdrant() must be "
            "called during app startup (see main.py lifespan)."
        )
    return _qdrant_client
