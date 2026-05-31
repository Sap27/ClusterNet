import json

base = '/mnt/d/results-clusternet-classical-sca/networks'
for fname in ['metadata.json', 'metadata_gnn.json']:
    try:
        d = json.load(open(f'{base}/{fname}'))
        sizes = set()
        for n in d['networks']:
            s = n.get('size', n.get('N', n.get('nodes')))
            sizes.add(s)
        print(f"{fname}: {len(d['networks'])} networks, sizes={sorted(sizes)}")
    except FileNotFoundError:
        print(f"{fname}: not found")
