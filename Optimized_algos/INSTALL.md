# ClusterNet Installation Guide

## Prerequisites

- Python >= 3.8
- pip (Python package installer)

## Basic Installation

### Method 1: Install from source (Recommended for development)

```bash
# Navigate to the Optimized_algos directory
cd /path/to/ClusterNet-main/Optimized_algos

# Install in editable mode
pip install -e .
```

### Method 2: Install with all dependencies

```bash
# Install with optional dependencies
pip install -e ".[all]"
```

### Method 3: Install specific extras

```bash
# For GPU support
pip install -e ".[gpu]"

# For development
pip install -e ".[dev]"
```

## Step-by-Step Installation

### 1. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv clusternet_env

# Activate it (Linux/Mac)
source clusternet_env/bin/activate

# Activate it (Windows)
clusternet_env\Scripts\activate
```

### 2. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install ClusterNet

```bash
pip install -e .
```

### 4. Verify Installation

```python
# Test import
python -c "from clusternet import CommunityDetector; print('✓ ClusterNet installed successfully!')"

# List algorithms
clusternet --list-algorithms
```

## Dependencies

### Core Dependencies (Required)
- networkx >= 2.8
- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- infomap >= 2.0.0
- joblib >= 1.0.0

### Optional Dependencies

#### GPU Support (Optional)
For GPU-accelerated algorithms:
```bash
# For CUDA 11.x
pip install cupy-cuda11x>=10.0.0
pip install cudf-cu11>=22.0.0
pip install cugraph-cu11>=22.0.0

# For CUDA 12.x
pip install cupy-cuda12x>=10.0.0
pip install cudf-cu12>=22.0.0
pip install cugraph-cu12>=22.0.0
```

**Note**: GPU support requires NVIDIA GPU with CUDA installed.

#### Visualization (Optional)
```bash
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
```

#### Profiling (Optional)
```bash
pip install psutil>=5.8.0
pip install memory_profiler
```

## Platform-Specific Instructions

### Linux

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-dev

# Install ClusterNet
cd Optimized_algos
pip install -e .
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python if needed
brew install python@3.11

# Install ClusterNet
cd Optimized_algos
pip install -e .
```

### Windows

```bash
# Ensure Python is installed and in PATH
python --version

# Install ClusterNet
cd Optimized_algos
pip install -e .
```

## Common Installation Issues

### Issue 1: Permission Denied

**Solution**: Use `--user` flag
```bash
pip install --user -e .
```

### Issue 2: Missing Compiler

**On Linux**:
```bash
sudo apt-get install build-essential python3-dev
```

**On macOS**:
```bash
xcode-select --install
```

**On Windows**:
Install Visual Studio Build Tools from Microsoft

### Issue 3: Infomap Installation Fails

**Solution**: Install from source
```bash
pip install git+https://github.com/mapequation/infomap.git@v2.7.1#subdirectory=interfaces/python
```

### Issue 4: NumPy/SciPy Compatibility

**Solution**: Install specific versions
```bash
pip install numpy==1.24.0 scipy==1.10.0
```

## Verify Installation

### Quick Test

```python
import networkx as nx
from clusternet import CommunityDetector

# Create test graph
G = nx.karate_club_graph()

# Run detection
detector = CommunityDetector(algorithm='louvain')
communities = detector.detect(G)

print(f"✓ Found {len(communities)} communities")
```

### Run Examples

```bash
cd examples
python example_usage.py
```

### Run Tests (if installed with dev dependencies)

```bash
pytest tests/
```

## Uninstallation

```bash
pip uninstall clusternet
```

## Updating

```bash
cd Optimized_algos
git pull  # if using git
pip install -e . --upgrade
```

## Docker Installation (Advanced)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy package files
COPY requirements.txt .
COPY setup.py .
COPY clusternet/ ./clusternet/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Set entrypoint
ENTRYPOINT ["clusternet"]
```

Build and run:
```bash
docker build -t clusternet .
docker run -v $(pwd):/data clusternet /data/network.dat louvain
```

## Conda Installation (Alternative)

```bash
# Create conda environment
conda create -n clusternet python=3.11
conda activate clusternet

# Install dependencies
conda install networkx numpy scipy scikit-learn
pip install infomap joblib

# Install ClusterNet
cd Optimized_algos
pip install -e .
```

## GPU Setup (Detailed)

### 1. Check CUDA Version
```bash
nvidia-smi
```

### 2. Install RAPIDS (for GPU acceleration)
```bash
# For CUDA 11.x
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=23.02 cugraph=23.02 python=3.10 cudatoolkit=11.8

# Install CuPy
pip install cupy-cuda11x
```

### 3. Verify GPU
```python
import cupy as cp
print(cp.cuda.runtime.getDeviceCount())  # Should print number of GPUs
```

## Development Setup

For contributing to ClusterNet:

```bash
# Clone repository
git clone <repository_url>
cd ClusterNet-main/Optimized_algos

# Create dev environment
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Check code style
black clusternet/
flake8 clusternet/
```

## Troubleshooting

### Get Help
```bash
# Check version
python -c "import clusternet; print(clusternet.__version__)"

# List algorithms
clusternet --list-algorithms

# Verbose output
clusternet network.dat louvain --verbose
```

### Report Issues
If installation fails, please provide:
1. Python version: `python --version`
2. OS and version
3. Full error message
4. Output of `pip list`

## Support

For installation help:
- Check GitHub Issues
- Read documentation
- Contact maintainers

## Next Steps

After installation:
1. Read PACKAGE_README.md for usage guide
2. Try examples in examples/example_usage.py
3. Check API reference
4. Run on your own data!

---

**Installation successful?** 🎉

Try this:
```bash
clusternet network.dat louvain --evaluate
```

