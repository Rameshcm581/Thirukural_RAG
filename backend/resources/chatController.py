"""
resources/chatController.py

FastAPI router for conversational Thirukural counseling and session management.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from resources.chat import process_chat_message
from resources.db import get_session_messages, list_sessions, delete_session
from schemas import ChatRequest, ChatResponse, KuralResult

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat_endpoint(request: ChatRequest):
    """
    Multi-turn chat endpoint:
    Processes user message within session, retrieves kurals, stores to PostgreSQL,
    and returns assistant reply with grounded kural citations.
    """
    try:
        session_id = request.session_id.strip()
        message = request.message.strip()

        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id cannot be empty",
            )
        if not message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="message cannot be empty",
            )

        reply, raw_kurals, mode = process_chat_message(session_id, message)

        kural_results = [KuralResult(**k) for k in raw_kurals] if raw_kurals else None

        logger.info("chat | session=%s | mode=%s | kurals=%d",
                    session_id, mode, len(raw_kurals) if raw_kurals else 0)

        return ChatResponse(
            session_id=session_id,
            reply=reply,
            kural_results=kural_results,
            mode=mode,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("chat error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get("/chat/sessions", status_code=status.HTTP_200_OK)
def get_sessions():
    """Returns list of recent chat sessions stored in PostgreSQL."""
    try:
        return {"sessions": list_sessions()}
    except Exception as exc:
        logger.error("get_sessions error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get("/chat/sessions/{session_id}", status_code=status.HTTP_200_OK)
def get_history(session_id: str):
    """Returns full message history for a given session from PostgreSQL."""
    try:
        messages = get_session_messages(session_id)
        return {"session_id": session_id, "messages": messages}
    except Exception as exc:
        logger.error("get_history error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.delete("/chat/sessions/{session_id}", status_code=status.HTTP_200_OK)
def delete_chat_session(session_id: str):
    """Deletes a chat session and all its messages from PostgreSQL."""
    try:
        deleted = delete_session(session_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )
        return {"session_id": session_id, "deleted": True}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("delete_session error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
