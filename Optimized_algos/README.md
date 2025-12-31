# ClusterNet: A Unified Framework for Community Detection

ClusterNet is a comprehensive Python package for community detection in graphs. It provides a unified interface for running various community detection algorithms on directed/undirected and weighted/unweighted graphs.

## 🚀 Features

- **Unified API**: Single interface for all algorithms
- **Multiple Algorithms**: 23+ state-of-the-art community detection algorithms
- **Algorithm Diversity**: Statistical, physics-based, diffusion, structural, and overlapping methods
- **Graph Flexibility**: Handles directed/undirected, weighted/unweighted graphs
- **Easy to Use**: Simple Python API and command-line interface
- **Comprehensive Metrics**: Built-in evaluation metrics (NMI, AMI, ARI, Modularity, etc.)
- **Production Ready**: Well-tested, documented, and optimized
- **CDlib Integration**: Access to 70+ algorithms from the cdlib library

## 📦 Installation

```bash
# Clone the repository
cd Optimized_algos

# Install core dependencies
pip install -r requirements-clusternet.txt

# Install the package
pip install -e .

# Or install with all dependencies
pip install -e ".[all]"

# Optional: Install graph-tool for SBM algorithms (requires conda)
conda install -c conda-forge graph-tool

# Optional: Install angel-cd for Angel algorithm
pip install angel-cd
```

## 🎯 Quick Start

### Python API

```python
import networkx as nx
from clusternet import CommunityDetector

# Load or create a graph
G = nx.karate_club_graph()

# Detect communities using Louvain
detector = CommunityDetector(algorithm='louvain', resolution=1.0)
communities = detector.detect(G)

print(f"Found {len(communities)} communities")
```

### Command-Line Interface

```bash
# Run Louvain on a network file
clusternet network.dat louvain --output communities.txt

# Run Leiden with custom parameters
clusternet network.dat leiden --resolution 1.5 --output output.txt

# Run statistical algorithms
clusternet network.dat sbm --output communities.txt
clusternet network.dat em --k 5

# Run physics-based algorithms
clusternet network.dat cpm --resolution_parameter 1.0
clusternet network.dat rb_pots --resolution_parameter 1.5

# Run diffusion algorithms
clusternet network.dat der --walk_len 3
clusternet network.dat async_fluid --k 5

# Run structural algorithms
clusternet network.dat scan --epsilon 0.5 --mu 3
clusternet network.dat agdl --number_communities 5 --kc 3

# Run overlapping algorithms
clusternet network.dat angel --threshold 0.25 --min_community_size 3
clusternet network.dat surprise_communities

# List all available algorithms
clusternet --list-algorithms

# Evaluate communities
clusternet network.dat louvain --evaluate
```

## 🧩 Available Algorithms

### Core Algorithms (9)

| Algorithm | Type | Directed | Weighted | Best For |
|-----------|------|----------|----------|----------|
| **Louvain** | Modularity | ✓ | ✓ | General purpose, fast |
| **Leiden** | Modularity | ✓ | ✓ | Guaranteed connectivity |
| **CSBIO-IITM2** | Ensemble | ✓ | ✓ | Robust, stable communities |
| **SCORE** | Spectral | ✓ | ✓ | Directed graphs, degree heterogeneity |
| **SVT** | Spectral | ✗ | ✓ | Feature-based clustering |
| **TeamCS** | Hybrid | ✗ | ✓ | Hierarchical structure |
| **SimNet** | Spectral | ✗ | ✓ | Noisy networks |
| **Tusk** | Hybrid | ✗ | ✓ | Complex networks |
| **BiGS2** | Hybrid | ✗ | ✓ | Multi-level communities |

### igraph/sklearn-Based Algorithms (7)

These algorithms use igraph's highly optimized C implementations or sklearn and match the main.py reference exactly:

| Algorithm | Type | Directed | Weighted | Function Used |
|-----------|------|----------|----------|---------------|
| **Walktrap** | Random Walk | ✓ | ✓ | `community_walktrap` |
| **Fast Greedy** | Modularity | ✓ | ✓ | `community_fastgreedy` |
| **Spin Glass** | Physics | ✓ | ✓ | `community_spinglass` |
| **Label Propagation** | Propagation | ✓ | ✓ | `community_label_propagation` |
| **Leading Eigenvector** | Spectral | ✓ | ✓ | `community_leading_eigenvector` |
| **Girvan-Newman** | Betweenness | ✓ | ✓ | `nx.community.girvan_newman` |
| **Spectral Clustering** | Spectral | ✓ | ✓ | `sklearn.SpectralClustering` |

> **Note**: All these implementations are identical to `main.py` - they use the same underlying igraph/networkx/sklearn functions with the same parameters.

### CDlib-Integrated Algorithms (14)

#### Statistical Inference (3)
| Algorithm | Type | Directed | Weighted | Best For |
|-----------|------|----------|----------|----------|
| **EM** | Statistical | ✗ | ✗ | Model-based clustering |
| **SBM** | Statistical | ✓ | ✓ | Stochastic block models |
| **Nested SBM** | Statistical | ✓ | ✓ | Hierarchical structure inference |

#### Physics-Based (3)
| Algorithm | Type | Directed | Weighted | Best For |
|-----------|------|----------|----------|----------|
| **CPM** | Physics | ✓ | ✓ | Resolution-based partitioning |
| **RB Potts** | Physics | ✓ | ✓ | Energy minimization |
| **RBER Potts** | Physics | ✗ | ✓ | ER null model |

#### Diffusion-Based (2)
| Algorithm | Type | Directed | Weighted | Best For |
|-----------|------|----------|----------|----------|
| **DER** | Diffusion | ✗ | ✓ | Entropy-based detection |
| **Async Fluid** | Diffusion | ✗ | ✗ | Fast, scalable networks |

#### Structural (3)
| Algorithm | Type | Directed | Weighted | Best For |
|-----------|------|----------|----------|----------|
| **SCAN** | Structural | ✗ | ✗ | Density-based clustering |
| **AGDL** | Structural | ✗ | ✓ | High-dimensional data |
| **GDMP2** | Structural | ✗ | ✓ | Graph decomposition |

#### Overlapping (2)
| Algorithm | Type | Directed | Weighted | Best For |
|-----------|------|----------|----------|----------|
| **Angel** | Overlapping | ✗ | ✗ | Node-centric, fast |
| **Surprise** | Overlapping | ✗ | ✓ | Quality function optimization |

## 📖 Usage Examples

### 1. Basic Usage

```python
from clusternet import CommunityDetector
import networkx as nx

# Create a graph
G = nx.karate_club_graph()

# Detect communities
detector = CommunityDetector(algorithm='louvain')
communities = detector.detect(G)
```

### 2. Load from File

```python
from clusternet import CommunityDetector, load_graph

# Load graph from edge list file
G = load_graph('network.dat', directed=False, weighted=True)

# Detect communities
detector = CommunityDetector(algorithm='leiden', resolution=1.5)
communities = detector.detect(G, output_file='communities.txt')
```

### 3. Compare Multiple Algorithms

```python
from clusternet import BatchDetector
import networkx as nx

G = nx.karate_club_graph()

# Run multiple algorithms
detector = BatchDetector(algorithms=['louvain', 'leiden', 'score'])
results = detector.detect_all(G)

for algo, communities in results.items():
    print(f"{algo}: {len(communities)} communities")
```

### 4. Evaluate Communities

```python
from clusternet import CommunityDetector
from clusternet.utils import evaluate_communities
import networkx as nx

G = nx.karate_club_graph()

detector = CommunityDetector(algorithm='louvain')
communities = detector.detect(G)

# Evaluate quality
metrics = evaluate_communities(G, communities, verbose=True)
print(f"Modularity: {metrics['modularity']:.4f}")
```

### 5. Compare Two Community Structures

```python
from clusternet.utils import compare_communities

# Detect with two different algorithms
detector1 = CommunityDetector(algorithm='louvain')
communities1 = detector1.detect(G)

detector2 = CommunityDetector(algorithm='leiden')
communities2 = detector2.detect(G)

# Compare
metrics = compare_communities(communities1, communities2, verbose=True)
print(f"NMI: {metrics['nmi']:.4f}")
```

## 🔧 Algorithm-Specific Parameters

### Louvain
```python
detector = CommunityDetector(
    algorithm='louvain',
    resolution=1.0  # Higher = more communities
)
```

### Leiden
```python
detector = CommunityDetector(
    algorithm='leiden',
    resolution=1.0,
    randomness=0.01  # Theta parameter
)
```

### CSBIO-IITM2 (Ensemble Louvain)
```python
detector = CommunityDetector(
    algorithm='csbio_iitm2',
    resolutions=None,    # Auto range 0.1-1.0
    strength=2           # Ensemble strength
)
```

### SVT
```python
detector = CommunityDetector(
    algorithm='svt',
    svd_k=50,           # Number of singular values
    n_clusters=50,      # Initial clusters
    module_size=40,     # Target module size
    n_neighbors=100     # Connectivity neighbors
)
```

### SCORE
```python
detector = CommunityDetector(
    algorithm='score',
    n_clusters=None  # Auto-detect if None
)
```

### SimNet
```python
detector = CommunityDetector(
    algorithm='simnet',
    initial_clusters=28,
    cluster_size_threshold=50,
    denoise_lambda=1.0,
    denoise_beta=0.1
)
```

### igraph-Based Algorithms

These algorithms match `main.py` exactly and use igraph's optimized C implementations:

#### Walktrap
```python
detector = CommunityDetector(
    algorithm='walktrap',
    steps=10,         # Number of steps in random walk
    weighted=True,    # Use edge weights
    directed=False    # Treat as undirected
)
```

#### Fast Greedy
```python
detector = CommunityDetector(
    algorithm='fastgreedy',
    weighted=True,
    directed=False
)
```

#### Spin Glass
```python
detector = CommunityDetector(
    algorithm='spinglass',
    spins=25,         # Number of spins (max communities)
    weighted=True,
    directed=False
)
```

#### Label Propagation
```python
detector = CommunityDetector(
    algorithm='label_propagation',
    weighted=True,
    directed=False
)
```

#### Leading Eigenvector
```python
detector = CommunityDetector(
    algorithm='leading_eigenvector',
    weighted=True,
    directed=False
)
```

#### Girvan-Newman
```python
detector = CommunityDetector(
    algorithm='girvan_newman',
    weighted=True,
    directed=False
)
```

#### Spectral Clustering
```python
detector = CommunityDetector(
    algorithm='spectral',
    n_clusters=30,      # Number of clusters
    n_components=10,    # Number of eigenvector components
    weighted=True,
    directed=False
)
```

### Statistical Algorithms

#### EM
```python
detector = CommunityDetector(
    algorithm='em',
    k=5  # Number of communities
)
```

#### SBM (Stochastic Block Model)
```python
detector = CommunityDetector(
    algorithm='sbm'  # Auto-detects number of communities
)
```

#### Nested SBM
```python
detector = CommunityDetector(
    algorithm='sbm_nested'  # Hierarchical structure
)
```

### Physics-Based Algorithms

#### CPM (Constant Potts Model)
```python
detector = CommunityDetector(
    algorithm='cpm',
    resolution_parameter=1.0
)
```

#### RB Potts
```python
detector = CommunityDetector(
    algorithm='rb_pots',
    resolution_parameter=1.0,
    weights=None  # Use edge weights if present
)
```

#### RBER Potts
```python
detector = CommunityDetector(
    algorithm='rber_pots',
    resolution_parameter=1.0
)
```

### Diffusion-Based Algorithms

#### DER (Diffusion Entropy Reducer)
```python
detector = CommunityDetector(
    algorithm='der',
    walk_len=3,
    threshold=0.00001
)
```

#### Async Fluid
```python
detector = CommunityDetector(
    algorithm='async_fluid',
    k=5  # Number of communities
)
```

### Structural Algorithms

#### SCAN
```python
detector = CommunityDetector(
    algorithm='scan',
    epsilon=0.5,  # Neighborhood radius
    mu=3  # Minimum neighbors
)
```

#### AGDL
```python
detector = CommunityDetector(
    algorithm='agdl',
    number_communities=5,
    kc=3  # Seed selection parameter
)
```

#### GDMP2
```python
detector = CommunityDetector(
    algorithm='gdmp2',
    min_threshold=0.75
)
```

### Overlapping Algorithms

#### Angel
```python
detector = CommunityDetector(
    algorithm='angel',
    threshold=0.25,
    min_community_size=3
)
```

#### Surprise Communities
```python
detector = CommunityDetector(
    algorithm='surprise_communities'
)
```

## 📊 Evaluation Metrics

ClusterNet provides comprehensive evaluation metrics:

- **Normalized Mutual Information (NMI)**: Similarity between two partitions
- **Adjusted Mutual Information (AMI)**: Chance-adjusted NMI (recommended over NMI)
- **Adjusted Rand Index (ARI)**: Similarity measure adjusted for chance
- **Fowlkes-Mallows Index (FMI)**: Geometric mean of precision and recall
- **Modularity**: Quality measure for community structure
- **Coverage**: Fraction of edges within communities
- **Performance**: Ratio of correctly identified edges
- **Surprise**: Statistical significance of community structure

> **Note**: We recommend using **AMI** instead of NMI for more robust comparisons, as it accounts for chance agreement.

```python
from clusternet.utils import evaluate_communities, compare_communities

# Evaluate single community structure
metrics = evaluate_communities(G, communities, verbose=True)

# Compare two structures
comparison = compare_communities(communities1, communities2, verbose=True)
```

## 🔄 Graph Preprocessing

ClusterNet automatically handles different graph types:

```python
from clusternet.utils import preprocess_graph

# Convert directed to undirected
G_undirected = preprocess_graph(G, directed=False)

# Remove self-loops
G_clean = preprocess_graph(G, remove_self_loops=True)

# Keep only largest connected component
G_connected = preprocess_graph(G, ensure_connected=True)
```

## 📁 File Formats

### Input Format (Edge List)
```
# Comments start with #
source target weight
1 2 1.5
2 3 2.0
3 4 1.0
```

### Output Format
```
# Simple (default)
1 2 3 4
5 6 7
8 9 10

# Detailed
Community 1 (size=4): 1 2 3 4
Community 2 (size=3): 5 6 7
Community 3 (size=3): 8 9 10

# ClusterNet format
1	0.5	1	2	3	4
2	0.5	5	6	7
3	0.5	8	9	10
```

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Building Documentation
```bash
cd docs
make html
```

### Code Formatting
```bash
black clusternet/
flake8 clusternet/
```

## 📚 API Reference

### CommunityDetector

Main class for community detection.

```python
CommunityDetector(algorithm: str, **kwargs)
```

**Methods:**
- `detect(G, input_file, directed, weighted, output_file)`: Detect communities
- `get_algorithm_info()`: Get algorithm metadata

### BatchDetector

Run multiple algorithms for comparison.

```python
BatchDetector(algorithms: List[str], **common_params)
```

**Methods:**
- `detect_all(G, input_file, directed, weighted)`: Run all algorithms

### Utility Functions

```python
# Graph I/O
load_graph(file_path, directed, weighted)
save_communities(communities, file_path, format)

# Preprocessing
preprocess_graph(G, directed, weighted, remove_self_loops, ensure_connected)
validate_graph(G, min_nodes, min_edges, allow_self_loops)

# Evaluation
compare_communities(communities1, communities2, verbose)
evaluate_communities(G, communities, verbose)
compute_modularity(G, communities)
compute_coverage(G, communities)

# Registry
list_algorithms(verbose)
get_algorithm_class(name)
```

## 📚 Algorithm Categories

Algorithms are organized into categories for easy discovery:

```python
from clusternet.registry import get_algorithms_by_category, get_algorithm_category

# Get algorithms by category
statistical = get_algorithms_by_category('statistical')
# ['em', 'sbm', 'sbm_nested']

physics = get_algorithms_by_category('physics')
# ['cpm', 'rb_pots', 'rber_pots']

diffusion = get_algorithms_by_category('diffusion')
# ['der', 'async_fluid']

structural = get_algorithms_by_category('structural')
# ['scan', 'agdl', 'gdmp2']

overlapping = get_algorithms_by_category('overlapping')
# ['angel', 'surprise_communities']

# Check algorithm category
category = get_algorithm_category('sbm')
# 'statistical'
```

## 🤝 Contributing

Contributions are welcome! To add a new algorithm:

1. Create a wrapper class inheriting from `BaseAlgorithm`
2. Implement the `run()` method
3. Register with `@register_algorithm` decorator
4. Add tests and documentation

```python
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm

@register_algorithm('myalgo', aliases=['my_algorithm'])
class MyAlgorithmWrapper(BaseAlgorithm):
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def run(self):
        # Your implementation
        communities = ...
        return communities
```

## 📄 License

MIT License - See LICENSE file for details

## 📧 Contact

For questions and support, please open an issue on GitHub.

## 🙏 Acknowledgments

This package integrates and provides a unified interface for multiple community detection algorithms developed by various researchers. 

**Special thanks to:**
- The [cdlib](https://github.com/GiulioRossetti/cdlib) project for providing access to 70+ community detection algorithms
- Original algorithm authors whose work we integrate

Please cite the original papers when using specific algorithms.

## 📖 References

### Core Algorithms
- Blondel et al. (2008). "Fast unfolding of communities in large networks"
- Traag et al. (2019). "From Louvain to Leiden"
- Jin, J. (2015). "Fast Community Detection by SCORE"

### igraph-Based Algorithms
- Pons & Latapy (2005). "Computing Communities in Large Networks Using Random Walks" (Walktrap)
- Clauset, Newman, Moore (2004). "Finding community structure in very large networks" (Fast Greedy)
- Reichardt & Bornholdt (2006). "Statistical mechanics of community detection" (Spin Glass)
- Raghavan, Albert, Kumara (2007). "Near linear time algorithm to detect community structures" (Label Propagation)
- Newman (2006). "Finding community structure using eigenvectors of matrices" (Leading Eigenvector)
- Girvan & Newman (2002). "Community structure in social and biological networks" (Girvan-Newman)
- Ng, Jordan, Weiss (2002). "On Spectral Clustering: Analysis and an algorithm" (Spectral Clustering)

### Statistical Algorithms
- Newman & Leicht (2007). "Mixture models and exploratory analysis in networks"
- Peixoto (2014). "Hierarchical block structures and high-resolution model selection"

### Physics-Based Algorithms
- Reichardt & Bornholdt (2006). "Statistical mechanics of community detection"
- Traag et al. (2011). "Narrow scope for resolution-limit-free community detection"

### Diffusion Algorithms
- Kozdoba & Mannor (2013). "Community detection via measure space embedding"
- Parés et al. (2017). "Fluid Communities: A Competitive and Highly Scalable Algorithm"

### Structural Algorithms
- Xu et al. (2007). "SCAN: A structural clustering algorithm for networks"
- Zhang et al. (2012). "Graph degree linkage: Agglomerative clustering"

### Overlapping Algorithms
- Rossetti (2019). "Exorcising the Demon: Angel, Efficient Node-Centric Community Discovery"
- Aldecoa & Marín (2013). "Surprise maximization reveals community structure"

### Framework
- Rossetti et al. (2019). "CDlib: a Python Library to Extract, Compare and Evaluate Communities"
