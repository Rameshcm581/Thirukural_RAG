"""
main.py

FastAPI entrypoint for ThirukuralService.

Follows the reference project's main.py pattern: structured logging is
configured before any other import that logs, resources (embedding
model + Qdrant client) are loaded once via `lifespan` rather than
per-request, and routes are registered as a single aggregated router.
"""

import logging
import os
import pathlib
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Ensure backend directory is in sys.path so imports work regardless of CWD
backend_dir = str(pathlib.Path(__file__).parent.resolve())
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv(pathlib.Path(__file__).parent / ".env")

# ── Logging setup (must run before any other import that logs) ─────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from resources import load_embedding_model, connect_qdrant, check_health  # noqa: E402
from resources.db import init_db, check_db_health  # noqa: E402
from routes import router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model, connect to Qdrant, and initialize PostgreSQL at startup."""
    logger.info("ThirukuralService starting -- loading embedding model, connecting to Qdrant, and initializing PostgreSQL")
    load_embedding_model()
    connect_qdrant()
    init_db()
    logger.info("ThirukuralService startup complete")
    yield
    logger.info("ThirukuralService shutting down")


app = FastAPI(
    title="ThirukuralService",
    description="RAG service retrieving Thirukural couplets based on modern life scenarios",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root():
    """Service info endpoint."""
    return {
        "service": "ThirukuralService",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint checking embedding model, Qdrant, and PostgreSQL readiness."""
    health = check_health()
    db_status = check_db_health()
    health["database"] = db_status

    is_healthy = (
        health["status"] == "healthy"
        and db_status == "connected"
    )
    health["status"] = "healthy" if is_healthy else "degraded"
    status_code = 200 if is_healthy else 503

    return JSONResponse(
        content={"service": "ThirukuralService", **health},
        status_code=status_code,
    )


app.include_router(router)
