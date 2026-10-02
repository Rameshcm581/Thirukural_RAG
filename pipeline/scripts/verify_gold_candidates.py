"""
verify_gold_candidates.py

Phase 5 helper: prints the actual couplet/explanation/chapter for a list
of candidate kural numbers, so we can verify their real content matches
what we assumed before locking them into the gold evaluation set.

Usage:
    python3 scripts/verify_gold_candidates.py 1 305 421 1028
"""

import json
import sys
from pathlib import Path

INPUT_PATH = Path(__file__).resolve().parent.parent / "data/processed/kurals_flat.json"


def main():
    numbers = {int(n) for n in sys.argv[1:]}
    with INPUT_PATH.open(encoding="utf-8") as f:
        records = {r["kural_number"]: r for r in json.load(f)}

    for num in sorted(numbers):
        r = records.get(num)
        if not r:
            print(f"=== Kural {num}: NOT FOUND ===\n")
            continue
        print(f"=== Kural {num} ({r['book']} / {r['adikaram']}) ===")
        print(f"Couplet: {r['english_couplet']}")
        print(f"Explanation: {r['english_explanation']}")
        print()


if __name__ == "__main__":
    main()
