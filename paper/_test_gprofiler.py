from gprofiler import GProfiler
gp = GProfiler(return_dataframe=True)
r = gp.profile(organism='hsapiens', query=['TP53','BRCA1','CDK2'], sources=['GO:BP','KEGG'])
print(r[['source','native','name','p_value']].head(5))
print("SUCCESS")
