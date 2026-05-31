"""Count communities in all partition JSONs to find the max."""
import json, os
from pathlib import Path

dirs = [
    Path(r"D:\cluster\BIologicalNetworks\results\partitions"),
    Path(r"D:\results-biological-networks-classical2\results-biological-networks-classical\results\partitions"),
]

results = []
for d in dirs:
    if not d.is_dir():
        continue
    for f in sorted(d.glob("*.json")):
        with open(f) as fh:
            communities = json.load(fh)
        results.append((len(communities), f.name))

results.sort(reverse=True)
print(f"{'Communities':>12}  File")
print("-" * 60)
for count, name in results[:30]:
    print(f"{count:>12}  {name}")
print(f"\nTotal partitions: {len(results)}")
print(f"Maximum communities: {results[0][0]} ({results[0][1]})")
