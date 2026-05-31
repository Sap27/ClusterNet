"""Show partition file counts per input directory."""
from pathlib import Path

dirs = [
    Path(r"D:\cluster\BIologicalNetworks\results\partitions"),
    Path(r"D:\results-biological-networks-classical2\results-biological-networks-classical\results\partitions"),
    Path(r"D:\results-perturb-gnn-biological\results\partitions"),
]

grand_total = 0
for i, d in enumerate(dirs, 1):
    files = sorted(d.glob("*.json")) if d.is_dir() else []
    nets = {}
    for f in files:
        for net in ["signor", "grn", "biogrid", "gtex_0.9"]:
            if f.stem.startswith(net + "_"):
                nets[net] = nets.get(net, 0) + 1
                break
    print(f"\nDir {i}: {d}")
    print(f"  Total files: {len(files)}")
    for net, cnt in sorted(nets.items()):
        print(f"    {net:<12} {cnt:>3} partitions")
    grand_total += len(files)

print(f"\n{'='*50}")
print(f"Grand total: {grand_total} partition files")
