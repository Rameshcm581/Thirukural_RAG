"""
clear_specific_kurals.py

Clears scenario_tags for a specific, hand-picked list of kural numbers
so generate_tags_full.py will regenerate just those under an improved
prompt. Used here for the 9 kurals diagnosed as genuine gaps or
embedding-noise misses in Phase 7 eval.

Usage:
    python3 scripts/clear_specific_kurals.py 106 221 131 181 951 961 1231 611 111
"""

import json
import sys
from pathlib import Path

OUTPUT_PATH = Path("data/processed/kurals_enriched.json")


def main():
    numbers = {int(n) for n in sys.argv[1:]}
    if not numbers:
        print("Usage: python3 scripts/clear_specific_kurals.py <kural_number> [<kural_number> ...]")
        sys.exit(1)

    with OUTPUT_PATH.open(encoding="utf-8") as f:
        records = json.load(f)

    cleared = 0
    for r in records:
        if r["kural_number"] in numbers:
            r["scenario_tags"] = []
            cleared += 1

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Cleared scenario_tags for {cleared} kurals: {sorted(numbers)}")
    print("Re-run generate_tags_full.py to regenerate them.")


if __name__ == "__main__":
    main()
