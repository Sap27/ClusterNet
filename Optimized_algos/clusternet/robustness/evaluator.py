"""
Robustness Evaluator

Comprehensive framework for evaluating community detection
algorithm robustness under various perturbations.
"""

import numpy as np
import networkx as nx
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

from .perturbations import (
    perturb_edges_random,
    perturb_edges_targeted,
    perturb_features_mean,
    perturb_features_variance,
    perturb_features_dropout
)


@dataclass
class RobustnessResult:
    """Result of robustness evaluation."""
    algorithm: str
    perturbation_type: str
    perturbation_level: float
    base_score: float
    perturbed_score: float
    score_change: float
    relative_change: float
    recovery_rate: float
    runtime_base: float
    runtime_perturbed: float


class RobustnessEvaluator:
    """
    Evaluates community detection algorithm robustness.
    
    Features:
    - Multiple perturbation types
    - Multiple evaluation metrics
    - Systematic sweeps over perturbation levels
    """
    
    PERTURBATION_TYPES = [
        'edge_random',
        'edge_targeted',
        'feature_mean',
        'feature_variance',
        'feature_dropout'
    ]
    
    def __init__(
        self,
        G: nx.Graph,
        ground_truth: List[List[int]],
        features: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            G: Original graph
            ground_truth: Ground truth communities
            features: Node features (optional)
            seed: Random seed
        """
        self.G = G
        self.ground_truth = ground_truth
        self.features = features
        self.seed = seed
        
        # Convert ground truth to labels
        self.n = G.number_of_nodes()
        self.true_labels = np.zeros(self.n, dtype=int)
        for i, comm in enumerate(ground_truth):
            for node in comm:
                self.true_labels[node] = i
    
    def _compute_nmi(
        self,
        communities: List[List[int]]
    ) -> float:
        """Compute NMI with ground truth."""
        try:
            from sklearn.metrics import normalized_mutual_info_score
            
            pred_labels = np.zeros(self.n, dtype=int)
            for i, comm in enumerate(communities):
                for node in comm:
                    if node < self.n:
                        pred_labels[node] = i
            
            return normalized_mutual_info_score(self.true_labels, pred_labels)
        except:
            return 0.0
    
    def _apply_perturbation(
        self,
        perturbation_type: str,
        level: float
    ) -> Tuple[nx.Graph, Optional[np.ndarray]]:
        """Apply perturbation to graph/features."""
        G_pert = self.G
        feat_pert = self.features
        
        if perturbation_type == 'edge_random':
            G_pert = perturb_edges_random(self.G, remove_ratio=level, seed=self.seed)
        elif perturbation_type == 'edge_targeted':
            G_pert = perturb_edges_targeted(self.G, remove_ratio=level, seed=self.seed)
        elif perturbation_type == 'feature_mean' and self.features is not None:
            feat_pert = perturb_features_mean(self.features, shift=level, seed=self.seed)
        elif perturbation_type == 'feature_variance' and self.features is not None:
            feat_pert = perturb_features_variance(self.features, scale=1 + level)
        elif perturbation_type == 'feature_dropout' and self.features is not None:
            feat_pert = perturb_features_dropout(self.features, drop_ratio=level, seed=self.seed)
        
        return G_pert, feat_pert
    
    def evaluate_single(
        self,
        algorithm: Callable,
        perturbation_type: str,
        level: float,
        **algo_kwargs
    ) -> RobustnessResult:
        """
        Evaluate algorithm robustness for single perturbation.
        
        Args:
            algorithm: Community detection function (G, features) -> communities
            perturbation_type: Type of perturbation
            level: Perturbation level
            **algo_kwargs: Additional arguments for algorithm
            
        Returns:
            RobustnessResult
        """
        # Run on original
        t1 = time.time()
        base_communities = algorithm(self.G, features=self.features, **algo_kwargs)
        t2 = time.time()
        base_score = self._compute_nmi(base_communities)
        runtime_base = t2 - t1
        
        # Apply perturbation
        G_pert, feat_pert = self._apply_perturbation(perturbation_type, level)
        
        # Run on perturbed
        t1 = time.time()
        pert_communities = algorithm(G_pert, features=feat_pert, **algo_kwargs)
        t2 = time.time()
        pert_score = self._compute_nmi(pert_communities)
        runtime_pert = t2 - t1
        
        # Compute metrics
        score_change = pert_score - base_score
        relative_change = score_change / base_score if base_score > 0 else 0
        recovery_rate = pert_score / base_score if base_score > 0 else 1.0
        
        return RobustnessResult(
            algorithm=algorithm.__name__ if hasattr(algorithm, '__name__') else str(algorithm),
            perturbation_type=perturbation_type,
            perturbation_level=level,
            base_score=base_score,
            perturbed_score=pert_score,
            score_change=score_change,
            relative_change=relative_change,
            recovery_rate=recovery_rate,
            runtime_base=runtime_base,
            runtime_perturbed=runtime_pert
        )
    
    def evaluate_sweep(
        self,
        algorithm: Callable,
        perturbation_type: str,
        levels: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5],
        **algo_kwargs
    ) -> List[RobustnessResult]:
        """
        Evaluate algorithm over multiple perturbation levels.
        
        Args:
            algorithm: Community detection function
            perturbation_type: Type of perturbation
            levels: List of perturbation levels
            **algo_kwargs: Additional arguments for algorithm
            
        Returns:
            List of RobustnessResult
        """
        results = []
        
        for level in levels:
            result = self.evaluate_single(
                algorithm,
                perturbation_type,
                level,
                **algo_kwargs
            )
            results.append(result)
        
        return results
    
    def evaluate_comprehensive(
        self,
        algorithm: Callable,
        perturbation_types: Optional[List[str]] = None,
        levels: List[float] = [0.05, 0.1, 0.2, 0.3],
        **algo_kwargs
    ) -> Dict[str, List[RobustnessResult]]:
        """
        Comprehensive evaluation over all perturbation types.
        
        Args:
            algorithm: Community detection function
            perturbation_types: List of perturbation types (all if None)
            levels: Perturbation levels
            **algo_kwargs: Additional arguments
            
        Returns:
            Dictionary mapping perturbation type to results
        """
        if perturbation_types is None:
            perturbation_types = self.PERTURBATION_TYPES
            # Filter out feature perturbations if no features
            if self.features is None:
                perturbation_types = [p for p in perturbation_types if not p.startswith('feature')]
        
        all_results = {}
        
        for pert_type in perturbation_types:
            results = self.evaluate_sweep(
                algorithm,
                pert_type,
                levels,
                **algo_kwargs
            )
            all_results[pert_type] = results
        
        return all_results
    
    def compare_algorithms(
        self,
        algorithms: Dict[str, Callable],
        perturbation_type: str = 'edge_random',
        levels: List[float] = [0.05, 0.1, 0.2, 0.3],
        **common_kwargs
    ) -> Dict[str, List[RobustnessResult]]:
        """
        Compare multiple algorithms under same perturbations.
        
        Args:
            algorithms: Dictionary of algorithm name -> function
            perturbation_type: Type of perturbation
            levels: Perturbation levels
            **common_kwargs: Arguments passed to all algorithms
            
        Returns:
            Dictionary mapping algorithm name to results
        """
        all_results = {}
        
        for name, algo in algorithms.items():
            results = self.evaluate_sweep(
                algo,
                perturbation_type,
                levels,
                **common_kwargs
            )
            all_results[name] = results
        
        return all_results
    
    def summary_statistics(
        self,
        results: Dict[str, List[RobustnessResult]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics from results.
        
        Args:
            results: Results from evaluate_comprehensive or compare_algorithms
            
        Returns:
            Summary statistics per algorithm/perturbation
        """
        summary = {}
        
        for key, result_list in results.items():
            recovery_rates = [r.recovery_rate for r in result_list]
            score_changes = [r.score_change for r in result_list]
            
            summary[key] = {
                'mean_recovery': np.mean(recovery_rates),
                'min_recovery': np.min(recovery_rates),
                'std_recovery': np.std(recovery_rates),
                'mean_score_change': np.mean(score_changes),
                'robustness_score': np.mean(recovery_rates) * (1 - np.std(recovery_rates))
            }
        
        return summary


def run_robustness_benchmark(
    G: nx.Graph,
    ground_truth: List[List[int]],
    algorithms: Dict[str, Callable],
    features: Optional[np.ndarray] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run comprehensive robustness benchmark.
    
    Args:
        G: Graph
        ground_truth: Ground truth communities
        algorithms: Dictionary of algorithms to test
        features: Optional node features
        seed: Random seed
        
    Returns:
        Comprehensive results dictionary
    """
    evaluator = RobustnessEvaluator(G, ground_truth, features, seed)
    
    results = {
        'per_algorithm': {},
        'summaries': {},
        'rankings': {}
    }
    
    for algo_name, algo_func in algorithms.items():
        algo_results = evaluator.evaluate_comprehensive(algo_func)
        results['per_algorithm'][algo_name] = algo_results
        results['summaries'][algo_name] = evaluator.summary_statistics(algo_results)
    
    # Compute overall rankings
    robustness_scores = {}
    for algo_name, summaries in results['summaries'].items():
        scores = [s['robustness_score'] for s in summaries.values()]
        robustness_scores[algo_name] = np.mean(scores)
    
    # Sort by robustness
    rankings = sorted(robustness_scores.items(), key=lambda x: x[1], reverse=True)
    results['rankings'] = {
        name: {'rank': i + 1, 'score': score}
        for i, (name, score) in enumerate(rankings)
    }
    
    return results









