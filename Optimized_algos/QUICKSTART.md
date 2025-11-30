# ClusterNet Quick Start Guide

Get up and running with ClusterNet in 5 minutes!

## 🚀 Installation (30 seconds)

```bash
cd Optimized_algos
pip install -e .
```

## ✅ Verify (10 seconds)

```bash
clusternet --list-algorithms
```

## 🎯 Your First Detection (1 minute)

### Python API

```python
from clusternet import CommunityDetector
import networkx as nx

# Create a graph
G = nx.karate_club_graph()

# Detect communities
detector = CommunityDetector(algorithm='louvain')
communities = detector.detect(G)

print(f"Found {len(communities)} communities!")
```

### Command Line

```bash
# Run on your data
clusternet network.dat louvain --output communities.txt

# Evaluate quality
clusternet network.dat louvain --evaluate
```

## 📚 Common Tasks

### 1. Load Graph from File

```python
from clusternet import load_graph, CommunityDetector

G = load_graph('network.dat', directed=False, weighted=True)
detector = CommunityDetector(algorithm='leiden')
communities = detector.detect(G)
```

### 2. Try Different Algorithms

```python
from clusternet import BatchDetector

detector = BatchDetector(algorithms=['louvain', 'leiden', 'score'])
results = detector.detect_all(G)
```

### 3. Compare Results

```python
from clusternet.utils import compare_communities

# Run two algorithms
communities1 = CommunityDetector('louvain').detect(G)
communities2 = CommunityDetector('leiden').detect(G)

# Compare
metrics = compare_communities(communities1, communities2, verbose=True)
```

### 4. Save Results

```python
from clusternet.utils import save_communities

save_communities(communities, 'output.txt', format='detailed')
```

## 🎨 Available Algorithms

| Use Case | Algorithm | Command |
|----------|-----------|---------|
| **General purpose** | Louvain | `louvain` |
| **Better quality** | Leiden | `leiden` |
| **Directed graphs** | SCORE | `score` |
| **Spectral method** | SVT | `svt` |
| **Hierarchical** | TeamCS | `teamcs` |

## ⚙️ Common Parameters

```python
# Louvain/Leiden
detector = CommunityDetector('louvain', resolution=1.0)

# SVT
detector = CommunityDetector('svt', svd_k=50, n_clusters=50)

# SCORE
detector = CommunityDetector('score', n_clusters=10)
```

## 📊 Evaluate Quality

```python
from clusternet.utils import evaluate_communities

metrics = evaluate_communities(G, communities, verbose=True)
# Prints: modularity, coverage, performance, and statistics
```

## 🔍 Complete Example

```python
import networkx as nx
from clusternet import CommunityDetector
from clusternet.utils import evaluate_communities, save_communities

# Load graph
G = nx.karate_club_graph()
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Detect communities
detector = CommunityDetector(algorithm='louvain', resolution=1.0)
communities = detector.detect(G)

# Evaluate
metrics = evaluate_communities(G, communities, verbose=True)

# Save
save_communities(communities, 'communities.txt')

print(f"\n✓ Found {len(communities)} communities")
print(f"✓ Modularity: {metrics['modularity']:.3f}")
print(f"✓ Saved to communities.txt")
```

## 🎓 Next Steps

1. **Read full docs**: See `PACKAGE_README.md`
2. **Try examples**: Run `python examples/example_usage.py`
3. **API reference**: Check docstrings in code
4. **Your data**: Replace with your network file!

## 💡 Tips

- **Start simple**: Try Louvain first (fast, good quality)
- **Compare**: Use `BatchDetector` to try multiple algorithms
- **Evaluate**: Always check modularity and other metrics
- **Parameters**: Default parameters work well for most cases

## 🆘 Quick Help

```bash
# List all algorithms
clusternet --list-algorithms

# Get algorithm info
python -c "from clusternet import CommunityDetector; \
           d = CommunityDetector('louvain'); \
           print(d.get_algorithm_info())"

# Verbose mode
clusternet network.dat louvain --verbose
```

## 🐛 Common Issues

**ImportError**: Run `pip install -e .` again

**Algorithm not found**: Check spelling with `--list-algorithms`

**File not found**: Use absolute path or check current directory

## 📝 Quick Reference Card

```bash
# Installation
pip install -e .

# List algorithms
clusternet --list-algorithms

# Run detection
clusternet INPUT.dat ALGORITHM --output OUTPUT.txt

# Evaluate
clusternet INPUT.dat ALGORITHM --evaluate

# Python API
from clusternet import CommunityDetector
detector = CommunityDetector(algorithm='louvain')
communities = detector.detect(G)
```

---

**That's it!** 🎉 You're ready to detect communities!

Need more help? Check `PACKAGE_README.md` or `INSTALL.md`

