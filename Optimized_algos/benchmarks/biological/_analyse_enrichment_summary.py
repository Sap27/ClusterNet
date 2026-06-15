"""Summarize enrichment results and extract key insights for the paper."""
import pandas as pd
import numpy as np

import platform
if platform.system() == 'Windows':
    csv_path = r"D:\cluster\BIologicalNetworks\results\bio_enrichment_results.csv"
else:
    csv_path = "/mnt/d/cluster/BIologicalNetworks/results/bio_enrichment_results.csv"
df = pd.read_csv(csv_path)

print("=" * 80)
print("BIOLOGICAL ENRICHMENT ANALYSIS - KEY INSIGHTS")
print("=" * 80)

# Separate classical vs GNN
gnn_algos = ['dgi_gnn', 'dmon_gnn', 'gat_supervised', 'gcn_supervised',
             'mincut_gnn', 'mlp_kmeans', 'node2vec_gnn', 'vgae_gnn']
df['type'] = df['algorithm'].apply(lambda x: 'GNN' if x in gnn_algos else 'Classical')

# Average over seeds for GNN
df_avg = df.groupby(['network', 'algorithm', 'type']).agg({
    'num_communities': 'mean',
    'go_enriched_frac': 'mean',
    'kegg_enriched_frac': 'mean',
    'communities_tested': 'mean',
    'mean_go_per_community': 'mean',
    'sign_coherence': 'mean',
}).reset_index()

for net in ['signor', 'grn', 'biogrid', 'gtex_0.9']:
    print(f"\n{'='*80}")
    print(f"  NETWORK: {net.upper()}")
    print(f"{'='*80}")
    
    sub = df_avg[df_avg['network'] == net].copy()
    
    # Top by GO enrichment fraction
    print(f"\n  Top 10 by GO enrichment fraction:")
    top_go = sub.nlargest(10, 'go_enriched_frac')
    for _, r in top_go.iterrows():
        tag = " [GNN]" if r['type'] == 'GNN' else ""
        print(f"    {r['algorithm']:<22} GO={r['go_enriched_frac']:.3f}  "
              f"KEGG={r['kegg_enriched_frac']:.3f}  "
              f"K={int(r['num_communities']):>5}  "
              f"tested={int(r['communities_tested']):>4}{tag}")
    
    # Top by KEGG
    print(f"\n  Top 10 by KEGG enrichment fraction:")
    top_kegg = sub.nlargest(10, 'kegg_enriched_frac')
    for _, r in top_kegg.iterrows():
        tag = " [GNN]" if r['type'] == 'GNN' else ""
        print(f"    {r['algorithm']:<22} KEGG={r['kegg_enriched_frac']:.3f}  "
              f"GO={r['go_enriched_frac']:.3f}  "
              f"K={int(r['num_communities']):>5}{tag}")
    
    # Classical vs GNN comparison
    cl = sub[sub['type'] == 'Classical']
    gn = sub[sub['type'] == 'GNN']
    if len(cl) > 0 and len(gn) > 0:
        print(f"\n  Classical vs GNN (mean across algorithms):")
        print(f"    Classical ({len(cl)} algos): GO={cl['go_enriched_frac'].mean():.3f}  "
              f"KEGG={cl['kegg_enriched_frac'].mean():.3f}")
        print(f"    GNN      ({len(gn)} algos): GO={gn['go_enriched_frac'].mean():.3f}  "
              f"KEGG={gn['kegg_enriched_frac'].mean():.3f}")
    
    # Sign coherence (SIGNOR only)
    if net == 'signor':
        print(f"\n  Sign coherence (SIGNOR-specific):")
        top_sign = sub.nlargest(10, 'sign_coherence')
        for _, r in top_sign.iterrows():
            tag = " [GNN]" if r['type'] == 'GNN' else ""
            print(f"    {r['algorithm']:<22} coherence={r['sign_coherence']:.3f}  "
                  f"GO={r['go_enriched_frac']:.3f}{tag}")

# Cross-network summary
print(f"\n\n{'='*80}")
print("  CROSS-NETWORK SUMMARY")
print("='*80")

print("\n  Mean GO enrichment by algorithm type per network:")
pivot = df_avg.groupby(['network', 'type'])['go_enriched_frac'].mean().unstack()
print(pivot.to_string())

print("\n  Mean KEGG enrichment by algorithm type per network:")
pivot2 = df_avg.groupby(['network', 'type'])['kegg_enriched_frac'].mean().unstack()
print(pivot2.to_string())

# Algorithms consistently in top-5 across networks
print("\n\n  Algorithms in top-5 GO enrichment across multiple networks:")
top5_counts = {}
for net in ['signor', 'grn', 'biogrid', 'gtex_0.9']:
    sub = df_avg[df_avg['network'] == net]
    top5 = sub.nlargest(5, 'go_enriched_frac')['algorithm'].tolist()
    for a in top5:
        top5_counts[a] = top5_counts.get(a, 0) + 1
for algo, cnt in sorted(top5_counts.items(), key=lambda x: -x[1]):
    if cnt >= 2:
        print(f"    {algo:<22} in top-5 on {cnt}/4 networks")

# Key insight: community size vs enrichment
print("\n\n  Community count vs enrichment (does fewer = better?):")
for net in ['signor', 'grn', 'biogrid', 'gtex_0.9']:
    sub = df_avg[df_avg['network'] == net]
    few = sub[sub['num_communities'] <= 20]
    many = sub[sub['num_communities'] > 100]
    if len(few) > 0 and len(many) > 0:
        print(f"    {net:<12} K<=20: GO={few['go_enriched_frac'].mean():.3f} ({len(few)} algos)  |  "
              f"K>100: GO={many['go_enriched_frac'].mean():.3f} ({len(many)} algos)")
