"""
configs/base_config.py

Centralizes every environment-dependent value so nothing is hardcoded
in business logic. Loaded once at import time via python-dotenv.
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class BaseConfig:
    # Qdrant -- set QDRANT_URL (+ QDRANT_API_KEY) for a hosted cluster such as
    # Qdrant Cloud; otherwise falls back to a local host/port.
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "thirukural")

    # Embedding model -- same model validated end-to-end in Phase 6/7
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-mpnet-base-v2"
    )

    # Retrieval tuning -- carried over from eval/run_eval.py's validated values
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    SEARCH_LIMIT_RAW: int = int(os.getenv("SEARCH_LIMIT_RAW", "20"))

    # Explanation generation (Gemini)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    EXPLAIN_MODEL_NAME: str = os.getenv("EXPLAIN_MODEL_NAME", "gemini-3.5-flash-lite")

    # PostgreSQL Database for Chat History & Sessions
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:2003@localhost:5432/pgdb"
    )

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
