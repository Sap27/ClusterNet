# coding=utf-8
"""
SVT-based Community Detection - Cleaned Version

This is a cleaned version of tianle.py with:
- Better variable names
- Removed commented code
- Added docstrings
- Fixed duplicate parameter bug
- SAME algorithm logic as original
"""

import numpy as np
import time
import _pickle as pickle
import networkx as nx
from scipy.sparse.linalg import svds
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph


def svt_feature(mat, M=None, include_diagonal=False, svd_k=3, svd_maxiter=100, 
                svt_delta=1.5, e=0.0001, svt_maxiter=100, save_feature=False, 
                feature_filename=''):    
    """SVT feature extraction"""
    if M is None or include_diagonal:
        idx1, idx2 = mat.nonzero()
        M = mat[idx1, idx2]
    else:
        idx1 = np.array(M[:,0], dtype='int')
        idx2 = np.array(M[:,1], dtype='int')
        M = M[:,2]
    
    tic = time.time()
    # Efficient svd implementation from scipy
    U, s, V = svds(mat, k=svd_k, maxiter=svd_maxiter)
    i = 0
    err = 100    # Initial error set a large number
    # First svd approximation
    Y = U.dot(np.diag(s)).dot(V)
    while err > e and i <= svt_maxiter:
        Y[idx1, idx2] += svt_delta * (M - Y[idx1, idx2])
        U, s, V = svds(Y, k=svd_k, maxiter=svd_maxiter)
        Y = U.dot(np.diag(s)).dot(V)
        i += 1
        err = np.sum((M - Y[idx1, idx2])**2) / np.sum(M**2)
        if i % 10 == 0:
            c = np.corrcoef(U.dot(np.diag(np.sqrt(s))))
    X = U.dot(np.diag(np.sqrt(s)))
    if save_feature:
        pickle.dump(X, open(feature_filename, 'wb'), -1) 
    
    return X


def sub_cluster(X, nodes, module_ass, module_size=40, linkage='ward', 
                constraint=False, n_neighbors=100):
    """Second-level clustering"""
    if constraint:
        connectivity = kneighbors_graph(X[nodes], n_neighbors=n_neighbors, include_self=False)
        ward_res = AgglomerativeClustering(n_clusters=nodes.shape[0]//module_size, 
                                          connectivity=connectivity, 
                                          linkage=linkage).fit(X[nodes])
    else:
        ward_res = AgglomerativeClustering(n_clusters=nodes.shape[0]//module_size, 
                                          linkage=linkage).fit(X[nodes])
    label = ward_res.labels_
    c = np.bincount(label)
    num = max(module_ass) + 1
    for i in np.where(np.logical_and(c>2, c<101))[0]:
        module_ass[nodes[label == i]] = num
        num += 1
    for i in np.where(c>100)[0]:
        module_ass = sub_cluster(X, nodes[label == i], module_ass)
    return module_ass


def module_disc(X, output_filename, n_neighbors=100, n_clusters=1000, 
                linkage='ward', module_size=40, constraint2=False, n_neighbors2=100):
    """Module discovery based on learned features"""
    # Initial clustering with connectivity constraint
    connectivity = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False)    
    ward = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity, 
                                   linkage=linkage).fit(X)
    label = ward.labels_
    
    c = np.bincount(label) 
    nodes = np.array(range(label.shape[0]))
    module_ass = np.zeros(label.shape)
    num = 1
    # Identify modules of size 3 to 100 from initial clustering result
    for i in np.where(np.logical_and(c>2, c<101))[0]:
        module_ass[label == i] = num
        num += 1
    for i in np.where(c>100)[0]:
        module_ass = sub_cluster(X, nodes[label == i], module_ass, 
                                module_size=module_size, linkage=linkage, 
                                constraint=constraint2, n_neighbors=n_neighbors2)
    
    c = np.bincount(np.array(module_ass, dtype='int'))
    num_clus = np.where(np.logical_and(c>2, c<101))[0].shape[0] - np.logical_and(c[0]>2, c[0]<101)
    print('Total clusters:', num_clus)
    if output_filename != '':
        with open(output_filename, 'w') as f:
            for i in range(1, int(max(module_ass))+1):
                f.write(str(i) + '\t0.5\t')
                for j in np.where(module_ass==i)[0]:
                    f.write(str(j)+'\t')
                f.write('\n')
    
    communities = []
    for i in range(1, int(max(module_ass))+1):
        communities.append([])
        for j in np.where(module_ass==i)[0]:
            communities[-1].append(j)
    return communities


def tianle(output_filename, input_filename='', M=None, include_diagonal=False, 
           G=None, directed=False, svd_k=50, svd_maxiter=100, svt_delta=1.5, 
           e=0.0001, svt_maxiter=100, save_feature=False, feature_filename='', 
           n_neighbors=100, n_clusters=50, linkage='ward', module_size=40, 
           constraint2=False, n_neighbors2=100):
    """Main function for SVT-based community detection"""
    if M is None and G is None:
        M = np.loadtxt(open(input_filename, 'r'), delimiter='\t')
    if G is None:
        G = nx.DiGraph() if directed else nx.Graph()
        for g in M:
            G.add_edge(g[0], g[1], weight=float(g[2]))
        original_nodes = list(G.nodes())
        node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
        for edge in M:
            edge[0] = node_mapping[edge[0]]
            edge[1] = node_mapping[edge[1]]

        reverse_mapping = {idx: node for node, idx in node_mapping.items()}
    
        # Relabel the graph nodes with new sequential integers
        relabeled_G = nx.relabel_nodes(G, node_mapping)
        A = nx.adjacency_matrix(relabeled_G)
        mat = A.toarray()
    else:
        original_nodes = list(G.nodes())
        node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
        reverse_mapping = {idx: node for node, idx in node_mapping.items()}
        # Relabel the graph nodes with new sequential integers
        relabeled_G = nx.relabel_nodes(G, node_mapping)
        mat = nx.adjacency_matrix(relabeled_G)
      
    X = svt_feature(mat, M=M, include_diagonal=include_diagonal, svd_k=svd_k, 
                   svd_maxiter=svd_maxiter, svt_delta=svt_delta, e=e, 
                   svt_maxiter=svt_maxiter, save_feature=save_feature, 
                   feature_filename=feature_filename)
    communities = module_disc(X, output_filename, n_neighbors=n_neighbors, 
                             n_clusters=n_clusters, linkage=linkage, 
                             module_size=module_size, constraint2=constraint2, 
                             n_neighbors2=n_neighbors2)                          
    return [[int(reverse_mapping[j]) for j in i] for i in communities]


if __name__ == "__main__":
    communities = tianle('tianle2.txt', input_filename='network.dat', directed=False)
    print(f"Found {len(communities)} communities")

