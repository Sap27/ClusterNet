import numpy as np
for net in ['signor', 'biogrid', 'grn', 'gtex_0.9']:
    path = f"/mnt/d/cluster/BIologicalNetworks/data/{net}/features.npy"
    try:
        f = np.load(path)
        print(f"{net}: features shape={f.shape}, dtype={f.dtype}, non-zero frac={f.astype(bool).mean():.3f}")
    except Exception as e:
        print(f"{net}: {e}")
