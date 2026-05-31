import json
d = json.load(open('/mnt/d/results-clusternet-classical-sca/networks/metadata.json'))
sizes = set()
for n in d['networks']:
    s = n.get('size', n.get('N', n.get('nodes')))
    sizes.add(s)
print("Available sizes:", sorted(sizes))
print("Total networks:", len(d['networks']))
