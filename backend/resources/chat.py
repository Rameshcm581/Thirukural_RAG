"""
resources/chat.py

Multi-turn conversational counseling engine using Thirukural wisdom.
Retrieves relevant kurals per turn, incorporates conversation history,
prompts Gemini for grounded counseling, and stores state in PostgreSQL.
"""

import json
import logging
import re
from typing import Literal

from google import genai
from google.genai import types

from configs import BaseConfig
from resources.db import get_session_messages, save_message, ensure_session
from resources.explain import _get_client
from resources.retrieval import retrieve_kurals
from schemas import KuralResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a compassionate, deeply perceptive mentor and counselor whose guidance is rooted in the 2,000-year-old ethical wisdom of the Thirukural.

When someone shares their personal or ethical dilemma:
1. Speak warmly and directly to them as a thoughtful mentor.
2. Ground your counsel in the provided Thirukural couplets. Mention the Kural number and interpret its wisdom specifically in the context of their modern situation.
3. Choose the most appropriate conversational approach:
   - "answered": Offer a clear, actionable perspective and ethical conclusion.
   - "clarifying_question": If the situation is ambiguous, ask an insightful question to help them reflect deeper.
   - "multiple_options": If there are competing virtues (e.g. speaking uncomfortable truth vs showing kindness), present both perspectives for them to balance.
4. Output your response followed by a final line indicating your approach mode:
   MODE: answered OR MODE: clarifying_question OR MODE: multiple_options
"""


def _parse_reply_and_mode(raw_text: str) -> tuple[str, str]:
    mode: Literal["answered", "clarifying_question", "multiple_options"] = "answered"
    text = raw_text.strip()

    # Check for MODE: tag at the end
    match = re.search(r"MODE:\s*(answered|clarifying_question|multiple_options)", text, re.IGNORECASE)
    if match:
        mode_val = match.group(1).lower()
        if mode_val in ("answered", "clarifying_question", "multiple_options"):
            mode = mode_val
        text = re.sub(r"MODE:\s*(answered|clarifying_question|multiple_options)", "", text, flags=re.IGNORECASE).strip()

    return text, mode


def process_chat_message(session_id: str, user_message: str) -> tuple[str, list[dict], str]:
    """
    Executes a multi-turn chat interaction:
    1. Saves user message to PostgreSQL.
    2. Retrieves top kurals for context.
    3. Queries Gemini with conversation history + retrieved kurals.
    4. Saves assistant response to PostgreSQL.
    Returns (reply_text, kural_results, mode).
    """
    ensure_session(session_id, title=user_message[:50])

    # Save user message
    save_message(session_id, role="user", content=user_message)

    # Fetch recent history (up to last 6 messages prior to current turn)
    history = get_session_messages(session_id)
    # The last one is the user message we just inserted
    past_turns = history[:-1][-6:] if len(history) > 1 else []

    # Retrieve relevant Kurals based on current user message
    try:
        raw_kurals = retrieve_kurals(user_message, top_k=3)
    except Exception as exc:
        logger.warning("Failed to retrieve kurals for chat: %s", exc)
        raw_kurals = []

    kural_context_parts = []
    for k in raw_kurals:
        kural_context_parts.append(
            f"- Kural #{k['kural_number']} ({k['book']} / {k['adikaram']}):\n"
            f"  Couplet: \"{k['english_couplet']}\"\n"
            f"  Meaning: {k['english_explanation']}\n"
        )
    kural_context_str = "\n".join(kural_context_parts) if kural_context_parts else "No specific kurals matched."

    # Build prompt
    history_str = ""
    for msg in past_turns:
        role_label = "User" if msg["role"] == "user" else "Counselor"
        history_str += f"{role_label}: {msg['content']}\n"

    prompt = f"""CONVERSATION HISTORY:
{history_str or '(New conversation)'}

USER'S CURRENT MESSAGE:
{user_message}

RELEVANT THIRUKURAL WISDOM RETRIEVED:
{kural_context_str}

Please respond to the user based on the Thirukural wisdom. Include the MODE indicator at the very end."""

    client = _get_client()
    reply_text = ""
    mode = "answered"

    if client:
        try:
            response = client.models.generate_content(
                model=BaseConfig.EXPLAIN_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.6,
                    max_output_tokens=600,
                ),
            )
            if response.text and response.text.strip():
                reply_text, mode = _parse_reply_and_mode(response.text)
        except Exception as exc:
            logger.error("Chat generation failed: %s", exc, exc_info=True)

    # Fallback if Gemini not available or failed
    if not reply_text:
        if raw_kurals:
            top_k = raw_kurals[0]
            reply_text = (
                f"Regarding your thoughts, Thirukural #{top_k['kural_number']} in the chapter on "
                f"'{top_k['adikaram']}' offers timeless counsel:\n\n"
                f"\"{top_k['english_couplet']}\"\n\n"
                f"{top_k['english_explanation']}.\n\n"
                f"Reflect on how this perspective applies to what you are currently navigating."
            )
        else:
            reply_text = (
                "I hear you. The Thirukural teaches us that patience, self-restraint, and clear discernment "
                "guide us through moments of confusion. Could you share a bit more about the specific situation?"
            )
            mode = "clarifying_question"

    # Save assistant message to PostgreSQL
    save_message(
        session_id=session_id,
        role="assistant",
        content=reply_text,
        kural_results=raw_kurals,
        mode=mode,
    )

    return reply_text, raw_kurals, mode
