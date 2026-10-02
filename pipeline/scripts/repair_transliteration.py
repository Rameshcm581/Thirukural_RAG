"""
repair_transliteration.py

Patches missing transliteration line 2 into kurals_flat.json and
kurals_enriched.json directly from the raw upstream thirukkural.json,
preserving all generated scenario tags.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data/raw/thirukkural-source/Thirukkural-Tamil-Dataset/data/thirukkural.json"
FLAT_PATH = BASE_DIR / "data/processed/kurals_flat.json"
ENRICHED_PATH = BASE_DIR / "data/processed/kurals_enriched.json"


def main():
    if not RAW_PATH.exists():
        print(f"Error: Raw dataset not found at {RAW_PATH}")
        return

    print("Loading raw dataset...")
    with RAW_PATH.open(encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_lookup = {}
    for entry in raw_data["kural"]:
        number = entry["Number"]
        t1 = (entry.get("transliteration1") or "").strip()
        t2 = (entry.get("transliteration2") or "").strip()
        translit = f"{t1}\n{t2}" if t2 else t1
        raw_lookup[number] = translit

    for path, label in [(FLAT_PATH, "kurals_flat.json"), (ENRICHED_PATH, "kurals_enriched.json")]:
        if not path.exists():
            print(f"Skipping {label} (does not exist)")
            continue
        print(f"Patching {label}...")
        with path.open(encoding="utf-8") as f:
            records = json.load(f)

        patched_count = 0
        for r in records:
            num = r["kural_number"]
            if num in raw_lookup and raw_lookup[num] != r.get("transliteration"):
                r["transliteration"] = raw_lookup[num]
                patched_count += 1

        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"Updated {patched_count} kurals in {label}.")


if __name__ == "__main__":
    main()
