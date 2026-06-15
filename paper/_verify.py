import csv

rows = list(csv.DictReader(open(r'd:\cluster\BIologicalNetworks\results\bio_enrichment_results.csv')))

# GRN: what community counts give 100% enrichment?
grn = [r for r in rows if r['network']=='grn' and float(r['go_enriched_frac'])==1.0]
print("GRN algorithms with 100% GO enrichment:")
for r in sorted(grn, key=lambda x: int(x['num_communities'])):
    print(f"  {r['algorithm']:20s} K={r['num_communities']}")

# GRN: what K range gives best enrichment on average?
grn_all = [r for r in rows if r['network']=='grn' and r['go_enriched_frac']]
print("\nGRN all, sorted by K:")
for r in sorted(grn_all, key=lambda x: int(x['num_communities']))[:15]:
    print(f"  {r['algorithm']:20s} K={int(r['num_communities']):>5d} GO={float(r['go_enriched_frac']):.3f}")
