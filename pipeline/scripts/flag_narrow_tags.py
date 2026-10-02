"""
flag_narrow_tags.py

Identifies kurals in kurals_enriched.json whose scenario_tags look
suspiciously narrow (low count, or heavy reuse of the same domain
keyword like "manager"/"project"/"team" across every tag) and clears
their scenario_tags so generate_tags_full.py will pick them back up
and regenerate them under the strengthened domain-diversity prompt.

This is a heuristic flag, not a guarantee -- it prints what it's about
to clear so you can sanity-check before committing.

Usage:
    python3 scripts/flag_narrow_tags.py           # dry run, just reports
    python3 scripts/flag_narrow_tags.py --apply   # actually clears them
"""

import json
import sys
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data/processed/kurals_enriched.json"

WORKPLACE_WORDS = ["manager", "project", "team", "company", "office",
                    "career", "colleague", "boss", "department", "meeting",
                    "client", "workplace", "coworker", "employee"]

# Chapters where a genuinely low tag count (2-4) is expected and correct --
# real administrative/military/governance mechanics, not emotional content.
# Low counts OUTSIDE this list (especially in the Love book, or personal
# virtue chapters) are suspicious, not legitimate narrowness.
LEGITIMATELY_NARROW_ADIKARAMS = {
    "The Greatness of a King", "The Office of Minister of state",
    "The Envoy", "Conduct in the Presence of the King", "The Right Sceptre",
    "The Cruel Sceptre", "The Excellence of an Army", "Military Spirit",
    "The Fortification", "Detectives", "Absence of Terrorism",
    "Selection and Confidence", "Selection and Employment",
    "Not to dread the Council", "The Knowledge of the Council Chamber",
    # Added after these kept getting re-flagged even post-regeneration --
    # confirmed genuinely narrow statecraft/administrative/geographic
    # content, not under-generation:
    "The Land", "Way of Accumulating Wealth", "Knowing the Place",
    "Seeking the Aid of Great Men", "The Knowledge of Indications",
    "Modes of Action", "Power in Action", "Purity in Action",
}


def looks_single_domain(tags: list[str]) -> bool:
    if len(tags) < 3:
        return False  # too few to judge domain spread meaningfully
    hits = sum(1 for t in tags if any(w in t.lower() for w in WORKPLACE_WORDS))
    return hits == len(tags)  # every single tag is workplace-flavored


def main():
    apply = "--apply" in sys.argv

    with OUTPUT_PATH.open(encoding="utf-8") as f:
        records = json.load(f)

    flagged = []
    for r in records:
        tags = r.get("scenario_tags", [])
        if not tags:
            continue
        adikaram = r["adikaram"]
        is_legit_narrow_chapter = adikaram in LEGITIMATELY_NARROW_ADIKARAMS
        low_count = len(tags) <= 4 and not is_legit_narrow_chapter
        single_domain = looks_single_domain(tags)
        if low_count or single_domain:
            flagged.append((r["kural_number"], adikaram, len(tags), single_domain))

    print(f"Flagged {len(flagged)} kurals out of {len([r for r in records if r.get('scenario_tags')])} tagged:")
    for num, adikaram, count, single_domain in flagged:
        reason = []
        if count <= 4:
            reason.append(f"low count ({count})")
        if single_domain:
            reason.append("single-domain")
        print(f"  Kural {num} ({adikaram}): {', '.join(reason)}")

    if not apply:
        print("\nDry run -- nothing changed. Re-run with --apply to clear these "
              "and let generate_tags_full.py regenerate them.")
        return

    flagged_numbers = {f[0] for f in flagged}
    for r in records:
        if r["kural_number"] in flagged_numbers:
            r["scenario_tags"] = []

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nCleared scenario_tags for {len(flagged)} kurals. "
          f"Re-run generate_tags_full.py to regenerate them.")


if __name__ == "__main__":
    main()
