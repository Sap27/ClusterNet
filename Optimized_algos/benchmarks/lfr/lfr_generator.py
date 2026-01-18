"""
LFR Network Generator
=====================

Generates LFR benchmark networks using the official C++ binaries.
Supports: standard, hierarchical, and overlapping networks.
"""

import os
import subprocess
import shutil
import tempfile
import networkx as nx
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


class LFRGenerator:
    """
    Generate LFR benchmark networks using official binaries.
    """
    
    def __init__(self, binary_dir: str):
        """
        Args:
            binary_dir: Path to LFRbenchmarks directory containing compiled binaries
        """
        self.binary_dir = Path(binary_dir)
        self._verify_binaries()
    
    def _verify_binaries(self):
        """Check that required binaries exist and are executable."""
        required = [
            'unweighted_undirected/benchmark',
            'hierarchical/hbenchmark',
        ]
        for binary in required:
            path = self.binary_dir / binary
            if not path.exists():
                print(f"Warning: Binary not found: {path}")
                print("You may need to compile it with 'make'")
    
    def generate_standard(
        self,
        N: int,
        mu: float,
        k: int = 15,
        maxk: int = 50,
        t1: float = 2,
        t2: float = 1,
        minc: int = 20,
        maxc: int = 50,
        on: int = 0,
        om: int = 0,
        seed: Optional[int] = None
    ) -> Tuple[nx.Graph, List[List[int]]]:
        """
        Generate a standard LFR network.
        
        Args:
            N: Number of nodes
            mu: Mixing parameter (fraction of inter-community edges)
            k: Average degree
            maxk: Maximum degree
            t1: Degree distribution exponent
            t2: Community size distribution exponent
            minc: Minimum community size
            maxc: Maximum community size
            on: Number of overlapping nodes
            om: Number of memberships for overlapping nodes
            seed: Random seed
            
        Returns:
            (graph, communities) tuple
        """
        binary = self.binary_dir / 'unweighted_undirected' / 'benchmark'
        if not binary.exists():
            raise FileNotFoundError(f"Binary not found: {binary}")
        
        # Create temp directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set seed if provided
            if seed is not None:
                seed_file = Path(tmpdir) / 'time_seed.dat'
                with open(seed_file, 'w') as f:
                    f.write(f"{seed}\n")
            
            # Build command
            cmd = [
                str(binary),
                '-N', str(N),
                '-k', str(k),
                '-maxk', str(maxk),
                '-mu', str(mu),
                '-t1', str(t1),
                '-t2', str(t2),
                '-minc', str(minc),
                '-maxc', str(maxc),
                '-on', str(on),
                '-om', str(om),
            ]
            
            # Run generator
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"LFR generation failed: {result.stderr}")
            
            # Parse output files
            network_file = Path(tmpdir) / 'network.dat'
            community_file = Path(tmpdir) / 'community.dat'
            
            G = self._parse_network(network_file)
            communities = self._parse_communities(community_file, overlapping=(on > 0))
            
            return G, communities
    
    def generate_hierarchical(
        self,
        N: int,
        mu1: float,
        mu2: float,
        k: int = 20,
        maxk: int = 100,
        t1: float = 2,
        t2: float = 1,
        minc: int = 10,
        maxc: int = 100,
        minC: int = 200,
        maxC: int = 1000,
        on: int = 0,
        om: int = 0,
        seed: Optional[int] = None
    ) -> Tuple[nx.Graph, List[List[int]], List[List[int]]]:
        """
        Generate a hierarchical LFR network with two levels.
        
        Args:
            N: Number of nodes
            mu1: Macro-community mixing parameter
            mu2: Micro-community mixing parameter
            k: Average degree
            maxk: Maximum degree
            t1, t2: Distribution exponents
            minc, maxc: Micro community size bounds
            minC, maxC: Macro community size bounds
            on: Number of overlapping nodes
            om: Number of memberships
            seed: Random seed
            
        Returns:
            (graph, micro_communities, macro_communities) tuple
        """
        binary = self.binary_dir / 'hierarchical' / 'hbenchmark'
        if not binary.exists():
            raise FileNotFoundError(f"Binary not found: {binary}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            if seed is not None:
                seed_file = Path(tmpdir) / 'time_seed.dat'
                with open(seed_file, 'w') as f:
                    f.write(f"{seed}\n")
            
            cmd = [
                str(binary),
                '-N', str(N),
                '-k', str(k),
                '-maxk', str(maxk),
                '-mu1', str(mu1),
                '-mu2', str(mu2),
                '-t1', str(t1),
                '-t2', str(t2),
                '-minc', str(minc),
                '-maxc', str(maxc),
                '-minC', str(minC),
                '-maxC', str(maxC),
                '-on', str(on),
                '-om', str(om),
            ]
            
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Hierarchical LFR generation failed: {result.stderr}")
            
            G = self._parse_network(Path(tmpdir) / 'network.dat')
            micro_comms = self._parse_communities(
                Path(tmpdir) / 'community_first_level.dat',
                overlapping=(on > 0)
            )
            macro_comms = self._parse_communities(
                Path(tmpdir) / 'community_second_level.dat',
                overlapping=False
            )
            
            return G, micro_comms, macro_comms
    
    def generate_overlapping(
        self,
        N: int,
        mu: float,
        overlap_fraction: float,
        om: int = 2,
        k: int = 15,
        maxk: int = 50,
        t1: float = 2,
        t2: float = 1,
        minc: int = 20,
        maxc: int = 50,
        seed: Optional[int] = None
    ) -> Tuple[nx.Graph, List[List[int]]]:
        """
        Generate an LFR network with overlapping communities.
        
        Args:
            N: Number of nodes
            mu: Mixing parameter
            overlap_fraction: Fraction of nodes that are overlapping (0 to 1)
            om: Number of memberships for overlapping nodes
            Other args: same as standard LFR
            
        Returns:
            (graph, communities) tuple with overlapping communities
        """
        on = int(N * overlap_fraction)
        return self.generate_standard(
            N=N, mu=mu, k=k, maxk=maxk, t1=t1, t2=t2,
            minc=minc, maxc=maxc, on=on, om=om, seed=seed
        )
    
    def _parse_network(self, filepath: Path) -> nx.Graph:
        """Parse network.dat file into NetworkX graph."""
        G = nx.Graph()
        edges_seen = set()
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    # LFR uses 1-indexed nodes, convert to 0-indexed
                    u, v = int(parts[0]) - 1, int(parts[1]) - 1
                    edge = tuple(sorted([u, v]))
                    if edge not in edges_seen:
                        G.add_edge(u, v)
                        edges_seen.add(edge)
        
        return G
    
    def _parse_communities(
        self,
        filepath: Path,
        overlapping: bool = False
    ) -> List[List[int]]:
        """
        Parse community.dat file.
        
        Format: node_id community_id [community_id ...]
        """
        if overlapping:
            # Node can belong to multiple communities
            comm_dict = {}
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        node = int(parts[0]) - 1  # 0-indexed
                        for comm_id in parts[1:]:
                            comm_id = int(comm_id)
                            if comm_id not in comm_dict:
                                comm_dict[comm_id] = []
                            comm_dict[comm_id].append(node)
            return [sorted(nodes) for nodes in comm_dict.values()]
        else:
            # Standard disjoint communities
            comm_dict = {}
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        node = int(parts[0]) - 1
                        comm_id = int(parts[1])
                        if comm_id not in comm_dict:
                            comm_dict[comm_id] = []
                        comm_dict[comm_id].append(node)
            return [sorted(nodes) for nodes in comm_dict.values()]


def save_network(G: nx.Graph, communities: List[List[int]], output_dir: str, name: str):
    """Save network and communities to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save graph
    nx.write_edgelist(G, os.path.join(output_dir, f'{name}_edges.txt'), data=False)
    
    # Save communities
    with open(os.path.join(output_dir, f'{name}_communities.txt'), 'w') as f:
        for i, comm in enumerate(communities):
            f.write(f"{i}: {' '.join(map(str, comm))}\n")
    
    # Save metadata
    metadata = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'num_communities': len(communities),
        'community_sizes': [len(c) for c in communities],
    }
    with open(os.path.join(output_dir, f'{name}_meta.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def load_network(input_dir: str, name: str) -> Tuple[nx.Graph, List[List[int]]]:
    """Load network and communities from files."""
    G = nx.read_edgelist(
        os.path.join(input_dir, f'{name}_edges.txt'),
        nodetype=int
    )
    
    communities = []
    with open(os.path.join(input_dir, f'{name}_communities.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                nodes = list(map(int, parts[1].split()))
                communities.append(nodes)
    
    return G, communities


if __name__ == '__main__':
    # Test the generator
    import sys
    
    # Find LFR binary directory
    script_dir = Path(__file__).parent
    lfr_dir = script_dir.parent.parent.parent.parent / 'LFRbenchmarks'
    
    print(f"LFR directory: {lfr_dir}")
    
    generator = LFRGenerator(str(lfr_dir))
    
    # Test standard generation
    print("\nGenerating standard LFR network (N=1000, μ=0.3)...")
    try:
        G, communities = generator.generate_standard(N=1000, mu=0.3, seed=42)
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Communities: {len(communities)}")
        print(f"  Community sizes: {[len(c) for c in communities]}")
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Test overlapping generation
    print("\nGenerating overlapping LFR network (N=1000, μ=0.3, 10% overlap)...")
    try:
        G, communities = generator.generate_overlapping(
            N=1000, mu=0.3, overlap_fraction=0.1, seed=42
        )
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Communities: {len(communities)}")
        
        # Check for overlapping nodes
        from collections import Counter
        node_memberships = Counter()
        for comm in communities:
            for node in comm:
                node_memberships[node] += 1
        overlapping_count = sum(1 for n, c in node_memberships.items() if c > 1)
        print(f"  Overlapping nodes: {overlapping_count}")
    except Exception as e:
        print(f"  Failed: {e}")




