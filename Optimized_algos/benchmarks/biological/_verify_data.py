"""Quick verification that all preprocessed data is consistent."""
import json, os
import numpy as np
from pathlib import Path

data_dir = Path('/mnt/d/cluster/BIologicalNetworks/data')

for net_id in ['gtex_0.8', 'grn', 'biogrid', 'signor']:
    d = data_dir / net_id
    meta = json.load(open(d / 'metadata.json'))
    edges = np.load(d / 'edges.npy')
    features = np.load(d / 'features.npy')
    nodes = json.load(open(d / 'node_list.json'))

    checks = []
    checks.append(('nodes match', len(nodes) == meta['num_nodes']))
    checks.append(('edges match', len(edges) == meta['num_edges_undirected']))
    checks.append(('features rows', features.shape[0] == meta['num_nodes']))
    checks.append(('features cols', features.shape[1] == meta['feature_dim']))
    checks.append(('no NaN features', not np.isnan(features).any()))
    checks.append(('no Inf features', not np.isinf(features).any()))

    if (d / 'directed_edges.json').exists():
        de = json.load(open(d / 'directed_edges.json'))
        checks.append(('directed edges', len(de) == meta['num_edges_directed']))

    if (d / 'ffls.json').exists():
        ffls = json.load(open(d / 'ffls.json'))
        checks.append(('FFLs', len(ffls) == meta['num_ffls']))

    status = 'OK' if all(c[1] for c in checks) else 'FAIL'
    print(f"\n{net_id} [{status}]:")
    print(f"  {meta['num_nodes']:>7,} nodes | {meta['num_edges_undirected']:>10,} edges | "
          f"{meta['feature_dim']:>5} feat dims | avg_deg={meta['avg_degree']}")
    for name, ok in checks:
        mark = 'PASS' if ok else 'FAIL'
        print(f"  [{mark}] {name}")
