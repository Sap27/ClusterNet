"""
Smoke test: generate a small LFR network and run every registered algorithm.
Reports PASS / FAIL + AMI for each, catching import and runtime errors early.

Usage:
    python _smoke_test.py
"""
import sys, os, time, traceback
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithm_runner import AlgorithmRunner
from metrics import compute_all_metrics

# ---------------------------------------------------------------------------
# 1. Generate a small LFR network via networkx (no external binary needed)
# ---------------------------------------------------------------------------
print("Generating LFR network (N=500, mu=0.3)...", flush=True)
G = nx.LFR_benchmark_graph(
    n=500, tau1=2.5, tau2=1.5, mu=0.3,
    min_degree=10, max_degree=50,
    min_community=20, max_community=80,
    seed=42,
)
true_comms_map = {}
for node in G.nodes():
    for cid, comm in enumerate(G.nodes[node]['community']):
        pass
comms_dict = {}
for node in G.nodes():
    comm = frozenset(G.nodes[node]['community'])
    comms_dict.setdefault(comm, []).append(node)
true_communities = [sorted(v) for v in comms_dict.values()]
num_true = len(true_communities)
n = G.number_of_nodes()
gt = np.zeros(n, dtype=int)
for cid, comm in enumerate(true_communities):
    for node in comm:
        gt[node] = cid

print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, "
      f"Communities: {num_true}\n", flush=True)

# ---------------------------------------------------------------------------
# 2. Set up runner and algorithm lists
# ---------------------------------------------------------------------------
runner = AlgorithmRunner(verbose=False)
all_algos = sorted(runner.get_available_algorithms())

classical = [a for a in all_algos if not any(
    a.endswith(s) for s in ('_gnn', '_kmeans', '_supervised')
) and not a.startswith(('gcn_', 'gat_'))]
gnn_unsup = [a for a in all_algos if a in (
    'dmon_gnn', 'mincut_gnn', 'vgae_gnn', 'dgi_gnn', 'mlp_kmeans', 'node2vec_gnn')]
gnn_sup = [a for a in all_algos if a.startswith(('gcn_supervised', 'gat_supervised'))]

# Supervised kwargs
rng = np.random.RandomState(42)
train_mask = np.zeros(n, dtype=bool)
train_mask[rng.choice(n, size=max(1, int(0.2 * n)), replace=False)] = True

print(f"Registered algorithms ({len(all_algos)} total):")
print(f"  Classical ({len(classical)}): {', '.join(classical)}")
print(f"  GNN unsupervised ({len(gnn_unsup)}): {', '.join(gnn_unsup) or 'NONE (torch/pyg missing?)'}")
print(f"  GNN supervised ({len(gnn_sup)}): {', '.join(gnn_sup) or 'NONE (torch/pyg missing?)'}")
print(flush=True)

# ---------------------------------------------------------------------------
# 3. Run every algorithm
# ---------------------------------------------------------------------------
results = []

def run_one(name, kwargs):
    t0 = time.time()
    try:
        result = runner.run_algorithm(G, name, **kwargs)
        elapsed = time.time() - t0
        if result and result.success and result.communities:
            metrics = compute_all_metrics(G, true_communities, result.communities)
            ami = metrics['ami']
            nc = len(result.communities)
            status = 'PASS'
        elif result:
            ami, nc = 0.0, 0
            status = f'FAIL ({result.error_message or "no communities"})'
        else:
            ami, nc = 0.0, 0
            status = 'FAIL (returned None)'
    except Exception as e:
        elapsed = time.time() - t0
        ami, nc = 0.0, 0
        status = f'ERROR ({type(e).__name__}: {str(e)[:80]})'
    results.append((name, status, ami, nc, elapsed))

print(f"{'Algorithm':<30s} {'Status':<50s} {'AMI':>6s} {'K':>4s} {'Time':>8s}")
print("-" * 102)

for name in classical:
    run_one(name, {'num_clusters': num_true})
    s = results[-1]
    tag = 'PASS' if s[1] == 'PASS' else 'FAIL'
    print(f"{s[0]:<30s} {s[1]:<50s} {s[2]:>6.3f} {s[3]:>4d} {s[4]:>7.1f}s", flush=True)

if gnn_unsup:
    print(f"\n{'--- GNN Unsupervised ---':<30s}")
    for name in gnn_unsup:
        run_one(name, {'num_clusters': num_true, 'epochs': 50})
        s = results[-1]
        print(f"{s[0]:<30s} {s[1]:<50s} {s[2]:>6.3f} {s[3]:>4d} {s[4]:>7.1f}s", flush=True)

if gnn_sup:
    print(f"\n{'--- GNN Supervised ---':<30s}")
    for name in gnn_sup:
        run_one(name, {
            'num_clusters': num_true, 'epochs': 50,
            'labels': gt, 'train_mask': train_mask,
        })
        s = results[-1]
        print(f"{s[0]:<30s} {s[1]:<50s} {s[2]:>6.3f} {s[3]:>4d} {s[4]:>7.1f}s", flush=True)

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
passed = sum(1 for r in results if r[1] == 'PASS')
failed = len(results) - passed
print(f"\n{'='*60}")
print(f"TOTAL: {len(results)} algorithms  |  {passed} PASS  |  {failed} FAIL")
if failed:
    print(f"\nFailed algorithms:")
    for r in results:
        if r[1] != 'PASS':
            print(f"  {r[0]}: {r[1]}")
print(f"{'='*60}")
sys.exit(0 if failed == 0 else 1)
