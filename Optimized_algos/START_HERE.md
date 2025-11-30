# 🚀 START HERE - ClusterNet Package

## 📦 What You Have

A **complete, production-ready Python package** for community detection in graphs!

---

## ⚡ Quick Install & Test (60 seconds)

```bash
# 1. Navigate to directory
cd /Users/sapnaraja/Downloads/ClusterNet-main/Optimized_algos

# 2. Install
pip install -e .

# 3. Test
python -c "from clusternet import CommunityDetector; print('✅ Package installed successfully!')"

# 4. List algorithms
clusternet --list-algorithms

# 5. Run on sample data
clusternet network.dat louvain --output test_output.txt --evaluate
```

**Expected output**: "✅ Package installed successfully!" and list of 8 algorithms

---

## 📚 Documentation Guide

| Read This | If You Want To... | Time |
|-----------|-------------------|------|
| **START_HERE.md** (this file) | Get oriented | 2 min |
| **QUICKSTART.md** | Start using immediately | 5 min |
| **PACKAGE_README.md** | Learn all features | 20 min |
| **INSTALL.md** | Fix installation issues | 10 min |
| **PACKAGE_STRUCTURE.md** | Understand architecture | 15 min |
| **examples/example_usage.py** | See code examples | 10 min |
| **CREATED_PACKAGE_SUMMARY.md** | See what was built | 5 min |

---

## 🎯 Your First Commands

### Test with Python:
```python
import networkx as nx
from clusternet import CommunityDetector

G = nx.karate_club_graph()
detector = CommunityDetector(algorithm='louvain')
communities = detector.detect(G)
print(f"Found {len(communities)} communities!")
```

### Test with Command Line:
```bash
clusternet network.dat louvain --evaluate
```

---

## 🎓 Learning Path

### 🟢 Beginner (Start Here!)
1. Run installation commands above
2. Read **QUICKSTART.md**
3. Try the Python example above
4. Run on your own data

### 🟡 Intermediate
1. Read **PACKAGE_README.md** sections 1-5
2. Try different algorithms
3. Compare results
4. Use evaluation metrics

### 🔴 Advanced
1. Read full **PACKAGE_README.md**
2. Understand **PACKAGE_STRUCTURE.md**
3. Customize parameters
4. Add your own algorithm

---

## 🎨 What Algorithms Are Available?

Run this to see all:
```bash
clusternet --list-algorithms
```

Or in Python:
```python
from clusternet import list_algorithms
list_algorithms(verbose=True)
```

**Available**: Louvain, Leiden, SCORE, SVT, TeamCS, SimNet, Tusk, BiGS2

---

## 🔥 Most Common Use Cases

### 1. Quick Detection
```bash
clusternet yournetwork.dat louvain --output communities.txt
```

### 2. Compare Algorithms
```python
from clusternet import BatchDetector
batch = BatchDetector(['louvain', 'leiden', 'score'])
results = batch.detect_all(G)
```

### 3. Evaluate Quality
```bash
clusternet network.dat louvain --evaluate
```

---

## 📁 Package Structure

```
Optimized_algos/
├── clusternet/                # 📦 Main package
│   ├── algorithms/           # 🔌 8 algorithm wrappers
│   ├── utils/                # 🛠️ Utilities (I/O, metrics)
│   ├── core.py               # 🎯 Main detector
│   ├── base.py               # 🏗️ Base class
│   ├── registry.py           # 📋 Algorithm registry
│   └── cli.py                # 💻 Command-line interface
├── examples/                  # 📖 Usage examples
├── setup.py                   # ⚙️ Installation config
├── requirements.txt           # 📝 Dependencies
└── *.md                       # 📚 Documentation
```

---

## ✅ Verification Checklist

After installation, verify everything works:

- [ ] `pip install -e .` completed without errors
- [ ] `clusternet --list-algorithms` shows 8 algorithms
- [ ] Python import works: `from clusternet import CommunityDetector`
- [ ] Can detect communities: Run quick Python example above
- [ ] CLI works: `clusternet network.dat louvain`

If any fail, see **INSTALL.md** for troubleshooting.

---

## 🆘 Quick Help

### "clusternet: command not found"
```bash
pip install -e .
```

### "No module named 'clusternet'"
```bash
cd /Users/sapnaraja/Downloads/ClusterNet-main/Optimized_algos
pip install -e .
```

### "Algorithm not found"
```bash
clusternet --list-algorithms  # Check spelling
```

### More help needed?
1. Check **INSTALL.md**
2. Read **PACKAGE_README.md**
3. Look at **examples/example_usage.py**

---

## 🎯 Next Steps

### Right Now (5 minutes)
1. ✅ Run installation commands above
2. ✅ Read **QUICKSTART.md**
3. ✅ Try first example

### Today (30 minutes)
1. Read **PACKAGE_README.md**
2. Run on your own data
3. Try different algorithms
4. Evaluate results

### This Week
1. Compare algorithms systematically
2. Tune parameters
3. Integrate into your workflow
4. Share with team

---

## 📊 Package Features

✅ **9 Algorithms**: Louvain, Leiden, CSBIO-IITM2, SCORE, SVT, TeamCS, SimNet, Tusk, BiGS2  
✅ **Unified API**: Same interface for all  
✅ **Graph Support**: Directed/Undirected, Weighted/Unweighted  
✅ **CLI Tool**: Command-line interface  
✅ **Metrics**: NMI, AMI, ARI, Modularity, Coverage, etc.  
✅ **Well Documented**: 7+ documentation files  
✅ **Examples**: Complete usage examples  
✅ **Production Ready**: Error handling, validation  

---

## 🎁 What You Can Do Now

1. **Detect Communities**: In any graph
2. **Compare Algorithms**: Try all 8 methods
3. **Evaluate Quality**: Using standard metrics
4. **Command Line**: Easy CLI interface
5. **Python API**: Full programmatic control
6. **Extend**: Add your own algorithms

---

## 💡 Pro Tips

- **Start simple**: Use Louvain first (fast, good quality)
- **Compare**: Try multiple algorithms on same data
- **Evaluate**: Always check modularity and other metrics
- **Visualize**: Use networkx to visualize results
- **Parameter tune**: Adjust resolution for Louvain/Leiden
- **Read docs**: Each algorithm has specific parameters

---

## 📞 Support

- **Documentation**: Start with QUICKSTART.md
- **Examples**: See examples/example_usage.py
- **Issues**: Open GitHub issue
- **Email**: Contact maintainers

---

## 🎉 You're All Set!

Run this now:
```bash
cd /Users/sapnaraja/Downloads/ClusterNet-main/Optimized_algos
pip install -e .
clusternet network.dat louvain --evaluate
```

**Next**: Read **QUICKSTART.md** for detailed usage

---

**Happy Community Detection!** 🚀

*Package created with ❤️ for the ClusterNet project*

