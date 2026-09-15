"""
generate_tags_full.py

Phase 4 of the Thirukural RAG project: full scenario tag generation
across all 1330 kurals.

Switched from Groq to Gemini (gemini-2.5-flash-lite) after Groq's free
tier daily token cap (200K TPD, org-wide across models) proved too small
to finish 1330 kurals in a reasonable number of sessions. Gemini's free
tier has no published daily *token* cap for Flash-Lite -- just 1,500
RPD and 250K TPM -- which comfortably covers a single-sitting run of
1330 requests.

Same validated prompt and resume/incremental-save design as the Groq
version:
  - Processes chapter-by-chapter (grouped by `adikaram`).
  - RESUMABLE: skips any kural that already has non-empty scenario_tags
    in data/processed/kurals_enriched.json.
  - Saves after every single kural, not just per chapter.

Requires: pip install google-genai
Set GEMINI_API_KEY in your environment (free key from
https://aistudio.google.com/apikey).
"""

import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import ClientError

INPUT_PATH = Path("data/processed/kurals_flat.json")
OUTPUT_PATH = Path("data/processed/kurals_enriched.json")
MODEL = "gemini-3.5-flash-lite"  # gemini-2.5-flash-lite is no longer
                                    # available to new users as of this key
MAX_OUTPUT_TOKENS = 900
BASE_DELAY_SECONDS = 4.5  # free tier is ~10-15 RPM; 4.5s keeps us comfortably under

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def extract_suggested_wait(error_message: str) -> float | None:
    match = re.search(r"retry.*?(\d+(?:\.\d+)?)\s*s", str(error_message), re.IGNORECASE)
    return float(match.group(1)) if match else None


def is_daily_quota_error(error_message: str) -> bool:
    """Gemini's 429 messages mention 'PerDay' for daily quota exhaustion,
    vs 'PerMinute' for transient per-minute limits."""
    msg = str(error_message)
    return "PerDay" in msg or "per day" in msg.lower()


class DailyQuotaExhausted(Exception):
    pass


SYSTEM_PROMPT = """You are helping build a semantic search index for the Thirukural, a classical Tamil text of ethical couplets.
For the given kural, generate realistic modern-day scenarios where a person seeking guidance would benefit from this specific teaching.

Rules:
- Write each scenario as a first-person situation a real person might type into a search bar or ask a friend -- natural, specific, not abstract.
- Do NOT reuse words from the kural's English translation or explanation. The goal is to bridge vocabulary gaps, not repeat them.
- Cover DIFFERENT life domains where relevant (work, family, friendship, money, relationships, self-discipline, leadership, health) -- don't generate several variations of the same domain. IMPORTANT: "different domains" means different areas of LIFE, not different job titles within the same area. Four scenarios that are all "as a [project lead/manager/executive/committee member]..." are NOT diverse even though the job titles differ -- they are all still "workplace" and count as one domain. Unless the kural's concept is genuinely restricted to one domain (e.g. it is specifically about parenting, or specifically about a monarch's court), spread scenarios across multiple domains -- do not settle for only workplace scenarios when personal, family, or relationship scenarios would fit just as well.
- If the kural involves gratitude or receiving a benefit, vary WHO or WHAT the benefit/gratitude is directed at (a person, an institution, a circumstance, a stranger) rather than defaulting to "a person" every time.
- Keep each scenario to one sentence, specific enough to be realistic (not generic like "having a problem").
- IMPORTANT: no matter how many tags you generate, ALWAYS include at least ONE tag that is a direct, literal, first-person paraphrase of the kural's central concept -- using the plainest possible modern words for the core idea (e.g. if the kural is about losing one's appetite from heartbreak, include a tag that literally says something like "I've lost my appetite since we broke up," not only creative indirect angles like grief over other losses). This guarantees the tag set always represents the kural's most obvious, most literal modern equivalent, even while other tags explore more creative or indirect angles. Do not let every tag be a clever indirect scenario -- the single most literal, obvious paraphrase must always be present.
- Decide the right NUMBER of scenarios based on how broadly the kural applies to modern life -- but count is not the goal, faithfulness to the kural's specific concept is:
    - Narrow/archaic kurals (kingship mechanics, war strategy, caste-era social structure): generate as few or as many as genuinely fit -- could be 2, could be 7. Never pad with a weak or drifted scenario just to hit a target number, and never force it down to an artificially low count if you've found several genuinely distinct, structurally-faithful domains.
    - Moderately specific kurals (specific relationship dynamics, specific vices/virtues): 5-8 scenarios.
    - Broadly universal virtues (anger, honesty, gratitude, diligence, friendship): 8-12 scenarios.
- CRITICAL for narrow/archaic kurals: do not satisfy the "different domains" rule by templating -- i.e. taking one sentence structure and swapping a role noun into it (e.g. "As a CEO/mayor/founder/coach managing X, Y, Z resources..." repeated with only the role and nouns changed). That produces scenarios that look different but are structurally identical and near-duplicate in meaning. If you cannot vary the actual underlying situation -- not just the job title -- generate fewer tags instead of reusing the same template.
- Equally important: do not chase a higher tag count by drifting away from the kural's actual specific concept into generic "I have a lot going on" scenarios that merely sound thematically adjacent. A kural about a king needing SIX SPECIFIC, DISTINCT, COMPLEMENTARY assets simultaneously (army, subjects, wealth, ministers, allies, a fortress) is NOT the same concept as generic busyness, marathon training, or personal relationship juggling -- those scenarios don't actually involve the same structure (several distinct types of support held at once) and would hurt retrieval by attaching this kural to queries where it isn't truly relevant. If a scenario doesn't genuinely mirror the kural's specific structure, leave it out even if it would raise the count.
- IMPORTANT CALIBRATION: the "narrow/archaic" tier is reserved specifically for administrative, military, or governance mechanics (e.g. a king's ministers, army composition, envoys, fortification, specific court protocol). It is NOT for emotional or relational content. Chapters about love, longing, separation, reunion, jealousy, sulking between partners, grief, or personal virtues (anger, patience, truthfulness, compassion, gratitude, self-restraint) describe near-universal human experiences and should almost always land in the moderate (5-8) or broad (8-12) tier, never the narrow (2-4) tier, even when the kural's literal imagery is about lovers or courtship rather than modern dating -- the underlying emotion transfers to countless modern scenarios. If you find yourself generating only 2-4 tags for a kural about love, longing, or a personal virtue, that is a signal to look harder for more scenarios, not a sign the kural is narrow.

Example of correct calibration for a narrow/archaic kural (six essential, distinct assets a king must hold simultaneously -- army, subjects, wealth, ministers, allies, fortress):
CORRECT (2 honest, structurally-faithful tags):
  - "I'm building a startup and thinking through what core assets I actually need to survive at once: a team, funding, advisors, and a safety net"
  - "As a new manager, I'm realizing leadership requires several different kinds of support in place simultaneously, not just one strength"
INCORRECT (padding with loosely-related filler to hit a higher count):
  - "I'm training for a marathon while juggling a demanding job and family duties" (this is about personal discipline, not holding multiple distinct assets at once -- do not include this kind of tag)
  - "My partner and I are merging finances to buy a house" (this is a two-person financial decision, not the multi-asset-simultaneously structure -- do not include this kind of tag)

Output strictly as JSON, no other text: {"scenario_tags": ["...", "...", ...]}"""

USER_PROMPT_TEMPLATE = """KURAL:
English couplet: {english_couplet}
English explanation: {english_explanation}
Chapter (adikaram): {adikaram}
Book: {book}"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "scenario_tags": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        }
    },
    "required": ["scenario_tags"],
}


def generate_tags_for_kural(record: dict, max_retries: int = 2) -> list[str]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        english_couplet=record["english_couplet"],
        english_explanation=record["english_explanation"],
        adikaram=record["adikaram"],
        book=record["book"],
    )

    last_error = None
    rate_limit_hits = 0
    max_rate_limit_waits = 5

    attempt = 0
    while attempt <= max_retries + max_rate_limit_waits:
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                ),
            )
            content = response.text
            if not content or not content.strip():
                raise ValueError("empty response content")
            parsed = json.loads(content)
            tags = parsed.get("scenario_tags", [])
            if tags:
                return tags
            raise ValueError("parsed JSON but scenario_tags was empty")

        except ClientError as e:
            is_rate_limit = getattr(e, "code", None) == 429 or "429" in str(e)
            if is_rate_limit:
                if is_daily_quota_error(str(e)):
                    raise DailyQuotaExhausted(str(e)) from e

                rate_limit_hits += 1
                if rate_limit_hits > max_rate_limit_waits:
                    last_error = e
                    break
                wait = extract_suggested_wait(str(e)) or 15
                wait += 1
                print(f"    rate limited (per-minute); waiting {wait:.1f}s "
                      f"(hit {rate_limit_hits}/{max_rate_limit_waits})...")
                time.sleep(wait)
                attempt += 1
                continue

            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"    retry {attempt + 1}/{max_retries} after error ({e}); waiting {wait}s...")
                time.sleep(wait)
            attempt += 1
            continue

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"    retry {attempt + 1}/{max_retries} after error ({e}); waiting {wait}s...")
                time.sleep(wait)
            attempt += 1
            continue

    print(f"    FAILED for kural {record['kural_number']} after retries: {last_error}")
    return []


def load_existing_progress() -> dict[int, dict]:
    if not OUTPUT_PATH.exists():
        return {}
    with OUTPUT_PATH.open(encoding="utf-8") as f:
        existing = json.load(f)
    return {r["kural_number"]: r for r in existing}


def save_progress(records_by_number: dict[int, dict]) -> None:
    ordered = [records_by_number[k] for k in sorted(records_by_number)]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


def group_by_chapter(records: list[dict]) -> list[tuple[str, list[dict]]]:
    chapters: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in records:
        key = r["adikaram"]
        if key not in chapters:
            chapters[key] = []
            order.append(key)
        chapters[key].append(r)
    return [(k, chapters[k]) for k in order]


def main():
    with INPUT_PATH.open(encoding="utf-8") as f:
        all_records = json.load(f)

    progress = load_existing_progress()
    already_done = {num for num, r in progress.items() if r.get("scenario_tags")}
    if already_done:
        print(f"Resuming: {len(already_done)}/{len(all_records)} kurals already tagged, skipping those.\n")

    chapters = group_by_chapter(all_records)
    total = len(all_records)
    processed_count = len(already_done)
    failed_numbers = []

    for chapter_name, chapter_records in chapters:
        pending = [r for r in chapter_records if r["kural_number"] not in already_done]
        if not pending:
            continue

        print(f"=== Chapter: {chapter_name} ({len(pending)}/{len(chapter_records)} pending) ===")

        for record in pending:
            processed_count += 1
            print(f"[{processed_count}/{total}] Kural {record['kural_number']}...")

            try:
                tags = generate_tags_for_kural(record)
            except DailyQuotaExhausted:
                print(f"\nDaily quota exhausted after {processed_count - 1} kurals this run.")
                print(f"Progress saved: {len([r for r in progress.values() if r.get('scenario_tags')])}/{total} tagged so far.")
                print("Gemini's free-tier daily quota resets ~24h after your first call today.")
                print("Just re-run this same script tomorrow -- it will automatically skip "
                      "everything already tagged and resume from here.")
                return

            record_copy = dict(record)
            record_copy["scenario_tags"] = tags
            progress[record["kural_number"]] = record_copy

            if tags:
                print(f"    -> {len(tags)} tags")
            else:
                failed_numbers.append(record["kural_number"])

            save_progress(progress)
            time.sleep(BASE_DELAY_SECONDS)

        print()

    print(f"\nDone. {len(progress)} total records in {OUTPUT_PATH}")
    if failed_numbers:
        print(f"WARNING: {len(failed_numbers)} kurals ended up with 0 tags: {failed_numbers}")
        print("Re-run this script to retry just these (they'll be picked up automatically "
              "since they have no tags yet).")


if __name__ == "__main__":
    main()
