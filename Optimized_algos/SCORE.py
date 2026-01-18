import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
import warnings

class SCOREAlgorithm:
    """
    Implementation of SCORE (Spectral Clustering On Ratios-of-Eigenvectors).
    
    Reference: Jin, J. (2015). "Fast Community Detection by SCORE".
    
    This algorithm is specifically robust for DIRECTED graphs with high degree heterogeneity.
    It uses the ratios of the leading eigenvector to normalize node degrees.
    """
    def __init__(self, G, n_clusters=None):
        self.G = G
        self.n_clusters = n_clusters
        
        # Determine number of nodes
        self.n = G.number_of_nodes()
        self.nodes = list(G.nodes())
        
        # Create Adjacency Matrix
        # SCORE works best on the raw Adjacency Matrix (A), not the Laplacian.
        # We use numpy for dense operations (SCORE is spectral, usually O(N^3) or O(N^2))
        self.A = nx.to_numpy_array(G, nodelist=self.nodes)

    def run(self):
        """
        Main execution pipeline.
        """
        # 1. Determine K (Number of Clusters)
        # If user didn't provide K, we estimate it using the "Eigengap" heuristic
        if self.n_clusters is None:
            k = self._estimate_k_eigengap()
        else:
            k = self.n_clusters
            
        # Safety check: K cannot be greater than N
        if k >= self.n:
            k = self.n // 2
        if k < 2:
            return [self.nodes] # One big cluster

        # 2. Eigen Decomposition
        # We need the first K+1 leading eigenvectors
        # 'LM' = Largest Magnitude
        try:
            vals, vecs = eigs(self.A, k=k+1, which='LM')
        except:
            # Fallback for very small graphs or convergence issues
            # We pad with zeros if eigen decomp fails
            return [self.nodes]

        # 3. Process Eigenvectors (The SCORE Logic)
        # SCORE typically looks at the "leading" eigenvector (dominant) 
        # and divides the others by it.
        
        # Sort by magnitude of eigenvalues (just to be safe, eigs usually sorts)
        idx = np.argsort(np.abs(vals))[::-1]
        vecs = vecs[:, idx]
        
        # The leading eigenvector is index 0. The others are 1..K
        leading_vec = vecs[:, 0]
        other_vecs = vecs[:, 1:k] # Take the next k-1 vectors (to form K dimensions total usually)
        
        # Note: In the original paper, they construct a matrix R of size n x (K-1)
        # where R_ij = V_{j+1}(i) / V_1(i)
        
        # Avoid division by zero
        # We add a tiny epsilon to the leading vector
        denominator = leading_vec.copy()
        denominator[np.abs(denominator) < 1e-10] = 1e-10
        
        # Calculate Ratios
        # We perform element-wise division for each column
        ratios = np.divide(other_vecs, denominator[:, None])
        
        # SCORE handles directed graphs which result in Complex numbers.
        # We project to Real numbers (taking the real part is standard)
        ratios = np.real(ratios)
        
        # 4. Clipping (Optional but recommended for stability)
        # Extreme outliers in ratios can ruin K-Means. 
        # We clip to roughly log(n) range or standard thresholds.
        threshold = np.log(self.n) if self.n > 1 else 10
        ratios = np.clip(ratios, -threshold, threshold)
        
        # 5. Clustering (K-Means on the Ratios)
        # We assume k clusters.
        # If we only extracted K-1 vectors, we are clustering in K-1 dim space.
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(ratios)
        
        return self._format_output(labels)

    def _estimate_k_eigengap(self):
        """
        Heuristic to find K if not provided.
        Looks for the largest drop in magnitude between sorted eigenvalues.
        """
        # Calculate top 20 eigenvalues (or N/2)
        limit = min(self.n - 1, 20)
        vals = eigs(self.A, k=limit, return_eigenvectors=False)
        vals = np.sort(np.abs(vals))[::-1] # Sort descending
        
        # Calculate diffs
        diffs = np.diff(vals)
        # The 'elbow' is usually where the diff is largest
        # We search after the first one (index 0) because the first gap is often trivial
        k_found = np.argmax(np.abs(diffs)) + 1
        
        return max(2, k_found)

    def _format_output(self, labels):
        clusters = {}
        for i, label in enumerate(labels):
            node_name = self.nodes[i]  # Keep original node type (int or str)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node_name)
        return list(clusters.values())

# ==========================================
# WRAPPER FUNCTION
# ==========================================

def score_clustering(input_data, n_clusters=None, is_directed=None):
    """
    Main entry point for SCORE.
    
    Args:
        input_data: NetworkX graph OR file path.
        n_clusters: Target number of communities (K). If None, estimates automatically.
        is_directed: Boolean. (If None, inferred from Graph type).
                     SCORE works on both, but is famous for Directed.
        
    Returns:
        List of lists: [['n1', 'n2'], ...]
    """
    
    # --- 1. INPUT HANDLING ---
    if isinstance(input_data, str):
        # Default to Directed if not specified, as SCORE is a directed algo
        if is_directed is False:
             create_using = nx.Graph()
        else:
             create_using = nx.DiGraph()

        try:
            G = nx.read_edgelist(input_data, create_using=create_using, nodetype=str, data=(('weight', float),))
        except TypeError:
            # Fallback requires re-instantiation
            if is_directed is False: create_using = nx.Graph()
            else: create_using = nx.DiGraph()
            G = nx.read_edgelist(input_data, create_using=create_using, nodetype=str)
            
    elif isinstance(input_data, (nx.Graph, nx.DiGraph)):
        G = input_data
    else:
        raise ValueError("Input must be a file path string or a NetworkX Graph object.")

    # --- 2. EXECUTION ---
    algo = SCOREAlgorithm(G, n_clusters=n_clusters)
    communities = algo.run()
    
    return communities