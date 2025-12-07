from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clusternet",
    version="1.0.0",
    author="ClusterNet Team",
    author_email="clusternet@example.com",
    description="A unified framework for community detection in graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/clusternet/clusternet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "networkx>=2.8",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "infomap>=2.0.0",
        "joblib>=1.0.0",
        "igraph",
        "cdlib>=0.4.0",
        "python-louvain>=0.15",
        "leidenalg>=0.8.0",
        "wurlitzer>=3.0.0",
    ],
    extras_require={
        "gpu": [
            "cupy-cuda11x>=10.0.0",  # Adjust cuda version as needed
            "cudf-cu11>=22.0.0",
            "cugraph-cu11>=22.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "sphinx>=4.0.0",
        ],
        "all": [
            "psutil>=5.8.0",
            "matplotlib>=3.5.0",
            "pandas>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "clusternet=clusternet.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "clusternet": ["data/*.dat"],
    },
)

