"""
flatten_dataset.py

Phase 2 of the Thirukural RAG project.

Merges two source files from vijayanandrp/Thirukkural-Tamil-Dataset into
one clean, flat record per kural:

  - thirukkural.json                 (primary: clean schema, English
                                       translation/explanation, book/iyal/
                                       adikaram hierarchy, short commentaries)
  - all_thirukkural_information.json (secondary: full-length Parimelazhagar
                                       and Manikudavar commentaries, plus
                                       a couple of others)

Output: data/processed/kurals_flat.json
        one record per kural (target ~1330), matching the schema:

{
  "kural_number": int,
  "book": str,              # English, e.g. "Virtue"
  "book_tamil": str,
  "iyal": str,               # English, e.g. "Prologue"
  "iyal_tamil": str,
  "adikaram": str,           # English chapter name, e.g. "The Praise of God"
  "adikaram_tamil": str,
  "tamil_couplet": [line1, line2],
  "transliteration": str,    # transliteration1 (line-by-line romanization)
  "english_couplet": str,    # the poetic "couplet" field
  "english_explanation": str,
  "commentaries": {
      "mu_varatharasanar": str | None,   # short, from primary file (mv)
      "salaman_pappaiyah": str | None,   # short, from primary file (sp)
      "mu_karunanidhi": str | None,      # short, from primary file (mk)
      "parimelazhagar": str | None,      # full, from secondary file
      "manikudavar": str | None,         # full, from secondary file
      "v_munusami": str | None           # full, from secondary file, if present
  },
  "scenario_tags": []        # populated later in Phase 3/4
}

Any kural where the secondary-file merge found nothing is logged at the
end so you know exactly which records are missing the fuller commentaries,
rather than silently having gaps.
"""

import json
from pathlib import Path

PRIMARY_PATH = Path("data/raw/thirukkural-source/Thirukkural-Tamil-Dataset/data/thirukkural.json")
SECONDARY_PATH = Path("data/raw/thirukkural-source/Thirukkural-Tamil-Dataset/data/all_thirukkural_information.json")
OUTPUT_PATH = Path("data/processed/kurals_flat.json")

# secondary-file key -> our commentary field name
SECONDARY_COMMENTARY_KEYS = {
    "6_parimela": "parimelazhagar",
    "6_manikudavar": "manikudavar",
    "6_v_munusami": "v_munusami",
    # salaman/mu_varatha/mu_karu also exist in the secondary file but we
    # already get short versions of these from the primary file (sp/mv/mk).
    # Uncomment below if you want the secondary file's versions instead/too:
    # "6_salaman": "salaman_pappaiyah_full",
    # "6_mu_varatha": "mu_varatharasanar_full",
    # "6_mu_karu": "mu_karunanidhi_full",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_secondary_lookup(secondary_raw: dict) -> dict:
    """Key the secondary file by its 0_number field, not by its dict key,
    since the two are not guaranteed to align 1:1."""
    lookup = {}
    for _, record in secondary_raw.items():
        number = record.get("0_number")
        if number is None:
            continue
        lookup[number] = record
    return lookup


def extract_commentary_body(record: dict, key: str) -> str | None:
    """Secondary-file commentary fields are [title, ...body_paragraphs].
    Some entries have a single body paragraph, others have multiple
    (e.g. a main explanation plus a separate விளக்கம் elaboration) —
    join every element after the title rather than assuming exactly 2."""
    value = record.get(key)
    if isinstance(value, list) and len(value) >= 2:
        paragraphs = [p.strip() for p in value[1:] if isinstance(p, str) and p.strip()]
        return "\n\n".join(paragraphs) if paragraphs else None
    return None


def flatten(primary_raw: dict, secondary_lookup: dict) -> tuple[list[dict], list[int], dict]:
    flat_records = []
    missing_secondary = []
    extraction_gaps: dict[str, list[int]] = {}

    for entry in primary_raw["kural"]:
        number = entry["Number"]
        secondary = secondary_lookup.get(number)

        commentaries = {
            "mu_varatharasanar": entry.get("mv"),
            "salaman_pappaiyah": entry.get("sp"),
            "mu_karunanidhi": entry.get("mk"),
            "parimelazhagar": None,
            "manikudavar": None,
            "v_munusami": None,
        }

        if secondary:
            for sec_key, field_name in SECONDARY_COMMENTARY_KEYS.items():
                body = extract_commentary_body(secondary, sec_key)
                if body:
                    commentaries[field_name] = body
                elif sec_key in secondary:
                    # secondary record exists and has this key, but extraction
                    # still yielded nothing — flag it, don't fail silently
                    extraction_gaps.setdefault(field_name, []).append(number)
        else:
            missing_secondary.append(number)

        t1 = (entry.get("transliteration1") or "").strip()
        t2 = (entry.get("transliteration2") or "").strip()
        transliteration = f"{t1}\n{t2}" if t2 else t1

        flat_records.append({
            "kural_number": number,
            "book": entry.get("paul_translation"),
            "book_tamil": entry.get("paul_name"),
            "iyal": entry.get("iyal_translation"),
            "iyal_tamil": entry.get("iyal_name"),
            "adikaram": entry.get("adikaram_translation"),
            "adikaram_tamil": entry.get("adikaram_name"),
            "tamil_couplet": [entry.get("Line1"), entry.get("Line2")],
            "transliteration": transliteration,
            "english_couplet": entry.get("couplet"),
            "english_explanation": entry.get("explanation"),
            "commentaries": commentaries,
            "scenario_tags": [],
        })

    return flat_records, missing_secondary, extraction_gaps


def validate(records: list[dict]) -> None:
    numbers = [r["kural_number"] for r in records]
    assert len(numbers) == len(set(numbers)), "Duplicate kural numbers found!"
    print(f"Total records: {len(records)}")
    if len(records) != 1330:
        print(f"WARNING: expected 1330 kurals, got {len(records)}")

    required_fields = ["book", "iyal", "adikaram", "tamil_couplet", "english_couplet"]
    incomplete = [
        r["kural_number"] for r in records
        if any(r.get(f) in (None, "", [None, None]) for f in required_fields)
    ]
    if incomplete:
        print(f"WARNING: {len(incomplete)} records missing required fields: {incomplete[:20]}"
              f"{' ...' if len(incomplete) > 20 else ''}")
    else:
        print("All records have required fields populated.")


def main():
    primary_raw = load_json(PRIMARY_PATH)
    secondary_raw = load_json(SECONDARY_PATH)
    secondary_lookup = build_secondary_lookup(secondary_raw)

    flat_records, missing_secondary, extraction_gaps = flatten(primary_raw, secondary_lookup)
    validate(flat_records)

    if missing_secondary:
        print(f"\n{len(missing_secondary)} kurals had no secondary-file match "
              f"(missing full Parimelazhagar/Manikudavar commentary): {missing_secondary}")

    if extraction_gaps:
        print("\nWARNING: secondary record existed but extraction yielded nothing for:")
        for field_name, numbers in extraction_gaps.items():
            print(f"  {field_name}: {len(numbers)} kurals -> {numbers[:20]}"
                  f"{' ...' if len(numbers) > 20 else ''}")
    else:
        print("\nNo silent extraction gaps detected.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(flat_records, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(flat_records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
