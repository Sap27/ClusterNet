"""Count communities with >= MIN_COMMUNITY_SIZE nodes (actually queried)."""
import json
from pathlib import Path

MIN_COMMUNITY_SIZE = 5

dirs = [
    Path(r"D:\cluster\BIologicalNetworks\results\partitions"),
    Path(r"D:\results-biological-networks-classical2\results-biological-networks-classical\results\partitions"),
]

results = []
seen = set()
for d in dirs:
    if not d.is_dir():
        continue
    for f in sorted(d.glob("*.json")):
        if f.name in seen:
            continue
        seen.add(f.name)
        with open(f) as fh:
            communities = json.load(fh)
        valid = sum(1 for c in communities if len(c) >= MIN_COMMUNITY_SIZE)
        results.append((valid, len(communities), f.name))

results.sort(reverse=True)
print(f"{'Valid(>=5)':>10} {'Total':>7}  File")
print("-" * 70)
for valid, total, name in results[:30]:
    print(f"{valid:>10} {total:>7}  {name}")
print(f"\nMax valid communities to query: {results[0][0]} ({results[0][2]})")
