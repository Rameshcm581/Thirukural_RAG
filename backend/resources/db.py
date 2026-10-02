"""
resources/db.py

PostgreSQL persistence layer for multi-turn chat sessions and messages.
Uses psycopg v3 with JSONB support.
"""

import json
import logging
from contextlib import contextmanager
from typing import Optional

import psycopg
from psycopg.rows import dict_row

from configs import BaseConfig

logger = logging.getLogger(__name__)

INIT_SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    title VARCHAR(255) NOT NULL DEFAULT 'New Conversation',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    kural_results JSONB,
    mode VARCHAR(30) DEFAULT 'answered',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated ON chat_sessions(updated_at DESC);
"""


@contextmanager
def get_db():
    conn = psycopg.connect(BaseConfig.DATABASE_URL, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initializes tables in PostgreSQL if they don't already exist."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(INIT_SQL)
        logger.info("PostgreSQL chat database initialized successfully.")
    except Exception as exc:
        logger.error("Failed to initialize PostgreSQL database: %s", exc, exc_info=True)


def check_db_health() -> str:
    """Verifies PostgreSQL connectivity."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                return "connected"
    except Exception as exc:
        return f"unreachable: {exc}"


def ensure_session(session_id: str, title: Optional[str] = None):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_sessions (session_id, title, updated_at)
                VALUES (%s, COALESCE(%s, 'New Conversation'), CURRENT_TIMESTAMP)
                ON CONFLICT (session_id) DO UPDATE
                SET updated_at = CURRENT_TIMESTAMP,
                    title = CASE WHEN EXCLUDED.title IS NOT NULL AND EXCLUDED.title != 'New Conversation'
                                 THEN EXCLUDED.title ELSE chat_sessions.title END;
                """,
                (session_id, title),
            )


def save_message(
    session_id: str,
    role: str,
    content: str,
    kural_results: Optional[list] = None,
    mode: str = "answered",
) -> int:
    ensure_session(session_id)
    kural_json = json.dumps(kural_results) if kural_results else None

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, kural_results, mode)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (session_id, role, content, kural_json, mode),
            )
            msg_id = cur.fetchone()["id"]

            # Update session's updated_at timestamp
            cur.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = %s;",
                (session_id,),
            )
            return msg_id


def get_session_messages(session_id: str) -> list[dict]:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, role, content, kural_results, mode, created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC;
                """,
                (session_id,),
            )
            rows = cur.fetchall()
            messages = []
            for r in rows:
                k_res = r["kural_results"]
                if isinstance(k_res, str):
                    try:
                        k_res = json.loads(k_res)
                    except Exception:
                        pass
                messages.append({
                    "id": r["id"],
                    "role": r["role"],
                    "content": r["content"],
                    "kural_results": k_res,
                    "mode": r["mode"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                })
            return messages


def list_sessions(limit: int = 50) -> list[dict]:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.updated_at,
                       COUNT(m.id) as message_count
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON s.session_id = m.session_id
                GROUP BY s.session_id, s.title, s.created_at, s.updated_at
                ORDER BY s.updated_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return [
                {
                    "session_id": r["session_id"],
                    "title": r["title"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                    "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                    "message_count": r["message_count"],
                }
                for r in rows
            ]


def delete_session(session_id: str) -> bool:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_sessions WHERE session_id = %s;", (session_id,))
            return cur.rowcount > 0
