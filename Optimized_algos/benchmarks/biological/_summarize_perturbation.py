import pandas as pd
import numpy as np

df = pd.read_csv('/mnt/d/results-perturb-gnn-biological/results/perturbation/grn_perturbation_results.csv')

print(f"Total rows: {len(df)}")
print(f"Algorithms: {sorted(df['algorithm'].unique())}")
print(f"Tests: {sorted(df['test'].unique())}")

# Classify algorithms
classical = ['leiden', 'louvain', 'rb_pots', 'fastgreedy', 'agdl']
gnn = ['dmon_gnn', 'mincut_gnn', 'dgi_gnn', 'gcn_supervised', 'mlp_kmeans']

def algo_type(a):
    return 'classical' if a in classical else 'GNN'

df['algo_type'] = df['algorithm'].apply(algo_type)

# =========================================================
# 1. MEAN METRICS PER TEST x ALGORITHM (averaged over seeds)
# =========================================================
print("\n" + "="*100)
print("MEAN METRICS PER TEST x ALGORITHM")
print("="*100)

for test in sorted(df['test'].unique()):
    tdf = df[df['test'] == test]
    grp = tdf.groupby('algorithm').agg({
        'nmi': 'mean', 'ami': 'mean', 'jaccard': 'mean',
        'modularity_change': 'mean', 'ffl_change': 'mean',
        'nodes_perturbed': 'first', 'edges_perturbed': 'first',
        'k_perturbed': 'mean',
    }).round(4)
    grp = grp.sort_values('nmi', ascending=False)
    n_orig = tdf['nodes_original'].iloc[0]
    e_orig = tdf['edges_original'].iloc[0]
    n_pert = tdf['nodes_perturbed'].iloc[0]
    e_pert = tdf['edges_perturbed'].iloc[0]
    print(f"\n--- {test} (N: {n_orig}->{n_pert}, E: {e_orig}->{e_pert}) ---")
    print(grp[['nmi', 'ami', 'jaccard', 'ffl_change', 'modularity_change', 'k_perturbed']].to_string())

# =========================================================
# 2. TF KNOCKOUT vs RANDOM CONTROL (the key comparison)
# =========================================================
print("\n\n" + "="*100)
print("KEY COMPARISON: TF KNOCKOUT vs RANDOM CONTROL")
print("="*100)

for k in [5, 10]:
    tf_test = f'tf_knockout_top{k}'
    ctrl_test = f'tf_knockout_control_random{k}'
    
    tf_df = df[df['test'] == tf_test].groupby('algorithm').agg({'nmi': 'mean', 'ffl_change': 'mean'}).round(4)
    ctrl_df = df[df['test'] == ctrl_test].groupby('algorithm').agg({'nmi': 'mean', 'ffl_change': 'mean'}).round(4)
    
    comparison = tf_df.join(ctrl_df, lsuffix='_tf', rsuffix='_ctrl')
    comparison['nmi_gap'] = (comparison['nmi_ctrl'] - comparison['nmi_tf']).round(4)
    comparison['ffl_gap'] = (comparison['ffl_change_ctrl'] - comparison['ffl_change_tf']).round(4)
    comparison = comparison.sort_values('nmi_gap', ascending=False)
    
    print(f"\n--- Top-{k} TFs removed vs {k} random genes ---")
    print(f"    NMI_tf = NMI under TF knockout, NMI_ctrl = NMI under random removal")
    print(f"    nmi_gap = how much MORE stable under random (positive = TF knockout is worse)")
    print(comparison.to_string())

# =========================================================
# 3. EDGE DROPOUT: DEGRADATION CURVE
# =========================================================
print("\n\n" + "="*100)
print("EDGE DROPOUT: STABILITY DEGRADATION (mean NMI across seeds)")
print("="*100)

ed_df = df[df['test'].str.startswith('edge_dropout')]
ed_grp = ed_df.groupby(['level', 'algorithm']).agg({'nmi': 'mean'}).round(4).unstack()
ed_grp.columns = ed_grp.columns.droplevel(0)
print(ed_grp.to_string())

# Classical vs GNN mean
print("\n--- Classical vs GNN mean NMI per dropout level ---")
ed_df2 = ed_df.copy()
ed_grp2 = ed_df2.groupby(['level', 'algo_type']).agg({'nmi': 'mean', 'ffl_change': 'mean'}).round(4).unstack()
print(ed_grp2.to_string())

# =========================================================
# 4. HUB GENE REMOVAL: CATASTROPHIC FAILURE POINT
# =========================================================
print("\n\n" + "="*100)
print("HUB GENE REMOVAL: NETWORK DESTRUCTION")
print("="*100)

hub_df = df[df['test'].str.startswith('gene_removal_hub')]
hub_grp = hub_df.groupby('test').agg({
    'nodes_perturbed': 'first', 'edges_perturbed': 'first',
    'nmi': 'mean', 'ffl_change': 'mean', 'modularity_change': 'mean',
}).round(4)
print(hub_grp.to_string())

print("\n--- Per algorithm at hub 1% ---")
h1 = hub_df[hub_df['test'] == 'gene_removal_hub_0.01']
h1_grp = h1.groupby('algorithm').agg({'nmi': 'mean', 'ffl_change': 'mean', 'modularity_change': 'mean'}).round(4)
h1_grp = h1_grp.sort_values('nmi', ascending=False)
print(h1_grp.to_string())

# =========================================================
# 5. OVERALL STABILITY RANKING
# =========================================================
print("\n\n" + "="*100)
print("OVERALL STABILITY RANKING (mean NMI across ALL perturbations)")
print("="*100)

overall = df.groupby('algorithm').agg({
    'nmi': 'mean', 'ami': 'mean', 'jaccard': 'mean',
    'ffl_change': 'mean', 'modularity_change': 'mean',
}).round(4)
overall['type'] = overall.index.map(algo_type)
overall = overall.sort_values('nmi', ascending=False)
print(overall.to_string())

# =========================================================
# 6. FFL PRESERVATION: WHO KEEPS MOTIFS BEST?
# =========================================================
print("\n\n" + "="*100)
print("FFL PRESERVATION UNDER PERTURBATION (mean ffl_preserved_pert)")
print("="*100)

# Exclude hub removal (too destructive) for fair comparison
mild_df = df[~df['test'].str.startswith('gene_removal_hub')]
ffl_rank = mild_df.groupby('algorithm').agg({
    'ffl_preserved_pert': 'mean', 'ffl_preserved_orig': 'mean', 'ffl_change': 'mean'
}).round(4)
ffl_rank = ffl_rank.sort_values('ffl_preserved_pert', ascending=False)
print(ffl_rank.to_string())
