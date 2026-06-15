import json

with open("/mnt/d/cluster/BIologicalNetworks/results/partitions/signor_leiden.json") as f:
    d = json.load(f)
print(f"Type: {type(d)}, Num communities: {len(d)}")
print(f"Community 0 size: {len(d[0])}, first 5: {d[0][:5]}")
print(f"Community 1 size: {len(d[1])}, first 5: {d[1][:5]}")

with open("/mnt/d/cluster/BIologicalNetworks/data/signor/node_list.json") as f:
    nodes = json.load(f)
print(f"\nNode list type: {type(nodes)}, length: {len(nodes)}")
print(f"First 5 nodes: {nodes[:5]}")
print(f"Node at index {d[0][0]}: {nodes[d[0][0]]}")
