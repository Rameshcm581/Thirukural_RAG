"""
generate_tags.py

Phase 3 of the Thirukural RAG project: small-batch scenario tag generation.

This script:
  1. Loads data/processed/kurals_flat.json
  2. Selects a deliberate sample of kurals spanning all three books
     (Aram/Virtue, Porul/Wealth, Inbam/Love), including at least one
     narrow/archaic chapter, so the sample stress-tests variety
  3. Calls Groq's API (openai/gpt-oss-120b -- the current recommended
     free-tier model; llama-3.3-70b-versatile was deprecated by Groq
     on 2026-08-16) with the tiered scenario-count prompt
  4. Writes results to data/processed/kurals_tagged_sample.json
     (kept separate from the real dataset until the prompt is validated)

Requires: pip install groq
Set GROQ_API_KEY in your environment (get a free key at console.groq.com).
"""

import json
import os
import sys
import time
from pathlib import Path

from groq import Groq

INPUT_PATH = Path("data/processed/kurals_flat.json")
OUTPUT_PATH = Path("data/processed/kurals_tagged_sample.json")
MODEL = "openai/gpt-oss-120b"  # current Groq free-tier recommendation

client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """You are helping build a semantic search index for the Thirukural, a classical Tamil text of ethical couplets.
For the given kural, generate realistic modern-day scenarios where a person seeking guidance would benefit from this specific teaching.

Rules:
- Write each scenario as a first-person situation a real person might type into a search bar or ask a friend -- natural, specific, not abstract.
- Do NOT reuse words from the kural's English translation or explanation. The goal is to bridge vocabulary gaps, not repeat them.
- Cover DIFFERENT life domains where relevant (work, family, friendship, money, relationships, self-discipline, leadership, health) -- don't generate several variations of the same domain.
- If the kural involves gratitude or receiving a benefit, vary WHO or WHAT the benefit/gratitude is directed at (a person, an institution, a circumstance, a stranger) rather than defaulting to "a person" every time.
- Keep each scenario to one sentence, specific enough to be realistic (not generic like "having a problem").
- Decide the right NUMBER of scenarios based on how broadly the kural applies to modern life -- but count is not the goal, faithfulness to the kural's specific concept is:
    - Narrow/archaic kurals (kingship mechanics, war strategy, caste-era social structure): generate as few or as many as genuinely fit -- could be 2, could be 7. Never pad with a weak or drifted scenario just to hit a target number, and never force it down to an artificially low count if you've found several genuinely distinct, structurally-faithful domains.
    - Moderately specific kurals (specific relationship dynamics, specific vices/virtues): 5-8 scenarios.
    - Broadly universal virtues (anger, honesty, gratitude, diligence, friendship): 8-12 scenarios.
- CRITICAL for narrow/archaic kurals: do not satisfy the "different domains" rule by templating -- i.e. taking one sentence structure and swapping a role noun into it (e.g. "As a CEO/mayor/founder/coach managing X, Y, Z resources..." repeated with only the role and nouns changed). That produces scenarios that look different but are structurally identical and near-duplicate in meaning. If you cannot vary the actual underlying situation -- not just the job title -- generate fewer tags instead of reusing the same template.
- Equally important: do not chase a higher tag count by drifting away from the kural's actual specific concept into generic "I have a lot going on" scenarios that merely sound thematically adjacent. A kural about a king needing SIX SPECIFIC, DISTINCT, COMPLEMENTARY assets simultaneously (army, subjects, wealth, ministers, allies, a fortress) is NOT the same concept as generic busyness, marathon training, or personal relationship juggling -- those scenarios don't actually involve the same structure (several distinct types of support held at once) and would hurt retrieval by attaching this kural to queries where it isn't truly relevant. If a scenario doesn't genuinely mirror the kural's specific structure, leave it out even if it would raise the count.

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


def select_sample(records: list[dict]) -> list[dict]:
    """Deliberately pick ~25-30 kurals spanning all three books, including
    at least one narrow/archaic chapter, rather than a random sample."""
    by_book: dict[str, list[dict]] = {}
    for r in records:
        by_book.setdefault(r["book"], []).append(r)

    sample = []
    # Take a spread from each book, sampling across different chapters
    # within each book rather than consecutive kurals.
    for book, book_records in by_book.items():
        book_records_sorted = sorted(book_records, key=lambda r: r["kural_number"])
        step = max(1, len(book_records_sorted) // 10)
        picks = book_records_sorted[::step][:10]
        sample.extend(picks)

    return sample


def generate_tags_for_kural(record: dict, max_retries: int = 2) -> list[str]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        english_couplet=record["english_couplet"],
        english_explanation=record["english_explanation"],
        adikaram=record["adikaram"],
        book=record["book"],
    )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("empty response content")
            parsed = json.loads(content)
            tags = parsed.get("scenario_tags", [])
            if tags:
                return tags
            raise ValueError("parsed JSON but scenario_tags was empty")

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  retry {attempt + 1}/{max_retries} after error ({e}); waiting {wait}s...")
                time.sleep(wait)
            continue

    print(f"  FAILED after {max_retries + 1} attempts for kural "
          f"{record['kural_number']}: {last_error}")
    return []


def main():
    with INPUT_PATH.open(encoding="utf-8") as f:
        all_records = json.load(f)

    if len(sys.argv) > 1:
        # Targeted re-test mode: python3 generate_tags.py 381 731 941
        requested = {int(n) for n in sys.argv[1:]}
        sample = [r for r in all_records if r["kural_number"] in requested]
        output_path = Path("data/processed/kurals_tagged_retest.json")
        print(f"Targeted re-test on {len(sample)} kurals: "
              f"{[r['kural_number'] for r in sample]}")
    else:
        sample = select_sample(all_records)
        output_path = OUTPUT_PATH
        print(f"Selected {len(sample)} kurals for the small-batch test, "
              f"spanning books: {sorted(set(r['book'] for r in sample))}")

    results = []
    failed_numbers = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, record in enumerate(sample, 1):
        print(f"[{i}/{len(sample)}] Kural {record['kural_number']} "
              f"({record['book']} / {record['adikaram']})...")
        tags = generate_tags_for_kural(record)
        record_copy = dict(record)
        record_copy["scenario_tags"] = tags
        results.append(record_copy)

        if tags:
            print(f"  -> {len(tags)} tags generated")
        else:
            failed_numbers.append(record["kural_number"])

        # Save after every kural, not just at the end -- a failure partway
        # through shouldn't cost the tags already generated.
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(1)  # be polite to the free-tier rate limit

    print(f"\nWrote {len(results)} tagged sample records to {output_path}")
    if failed_numbers:
        print(f"WARNING: {len(failed_numbers)} kurals ended up with 0 tags "
              f"after retries: {failed_numbers}")


if __name__ == "__main__":
    main()
