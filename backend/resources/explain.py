"""
resources/explain.py

Grounded generation: given the user's scenario and a retrieved kural,
generate a short explanation of WHY it fits -- explicitly instructed to
reference only the kural's actual couplet/explanation, not invent advice
beyond what the text says.
"""

import logging
import time

from google import genai
from google.genai import types

from configs import BaseConfig

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=BaseConfig.GEMINI_API_KEY)
    return _client


SYSTEM_PROMPT = """You explain why a classical Thirukural couplet applies to someone's modern situation.

Rules:
- Reference ONLY the couplet and explanation text given to you. Do not invent advice, facts, or interpretations beyond what the kural actually says.
- Keep it to 2-3 sentences.
- Speak directly to the person ("you"), connecting their specific situation to the kural's actual wisdom.
- Do not repeat the couplet verbatim -- paraphrase its meaning in the context of their situation.
"""

USER_PROMPT_TEMPLATE = """User's situation: {scenario}

Kural couplet (English): {english_couplet}
Kural explanation: {english_explanation}

Explain why this kural's wisdom applies to the user's situation."""


def generate_explanation(scenario: str, kural: dict, max_retries: int = 2) -> str:
    """Returns a short grounded explanation, or a safe fallback string
    if generation fails after retries (never raises -- an explanation
    failure shouldn't break the whole /search response)."""
    client = _get_client()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        scenario=scenario,
        english_couplet=kural["english_couplet"],
        english_explanation=kural["english_explanation"],
    )

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=BaseConfig.EXPLAIN_MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.5,
                    max_output_tokens=200,
                ),
            )
            text = response.text
            if text and text.strip():
                return text.strip()
            raise ValueError("empty explanation response")

        except Exception as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning("generate_explanation | retry %d/%d | kural=%s | error=%s",
                               attempt + 1, max_retries, kural["kural_number"], exc)
                time.sleep(wait)
                continue
            logger.error("generate_explanation | failed after retries | kural=%s | error=%s",
                         kural["kural_number"], exc, exc_info=True)
            return (f"This kural (#{kural['kural_number']}) speaks to your situation "
                    f"through its teaching: {kural['english_explanation']}")
