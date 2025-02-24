import networkx as nx
import numpy as np
from typing import List, Set, Dict, Any

def evaluate_partition(G: nx.Graph, partition: List[Set[int]]) -> Dict[str, Any]:
    """
    Evaluate partition quality metrics

    Args:
        G: Input graph
        partition: List of sets containing node indices for each partition

    Returns:
        Dictionary with metrics:
        - connectivity: Whether each partition is connected
        - cut_edges: Number of edges between partitions
        - balance: Measure of size balance between partitions
        - modularity: Newman-Girvan modularity (simplified)
    """
    metrics = {}

    # Check connectivity for each partition
    for part in partition:
        if not part:
            # Empty partition, mark as disconnected
            metrics['connectivity'] = False
            break
        subg = G.subgraph(part)
        if not nx.is_connected(subg):
            metrics['connectivity'] = False
            break
    else:
        metrics['connectivity'] = True

    # Count cut edges and compute modularity
    cut_edges = 0
    internal_edges = 0
    total_edges = G.number_of_edges()
    node_to_part = {}

    for i, part in enumerate(partition):
        for node in part:
            node_to_part[node] = i

    for u, v in G.edges():
        if node_to_part[u] != node_to_part[v]:
            cut_edges += 1
        else:
            internal_edges += 1

    metrics['cut_edges'] = cut_edges

    # Compute modularity Q
    Q = (internal_edges / total_edges) - sum(
        (len(part) / (2 * total_edges)) ** 2
        for part in partition
    )
    metrics['modularity'] = Q

    # Compute balance measure
    sizes = [len(part) for part in partition]
    avg_size = np.mean(sizes)
    max_dev = max(abs(size - avg_size) for size in sizes)
    metrics['balance'] = 1 - (max_dev / avg_size)

    return metrics

def compute_metrics(G: nx.Graph, partition: List[Set[int]], target_sizes: List[int]) -> Dict[str, float]:
    """
    Compute a comprehensive set of partition quality metrics

    Args:
        G: Input graph
        partition: List of sets containing node indices for each partition
        target_sizes: Target size for each partition

    Returns:
        Dictionary of quality metrics
    """
    metrics = evaluate_partition(G, partition)

    # Add size deviation metrics
    actual_sizes = [len(part) for part in partition]
    size_diffs = [abs(actual - target) for actual, target in zip(actual_sizes, target_sizes)]

    metrics.update({
        'max_size_deviation': max(size_diffs),
        'avg_size_deviation': np.mean(size_diffs),
        'size_dev_std': np.std(size_diffs)
    })

    # Add edge cut ratio
    if G.number_of_edges() > 0:
        metrics['cut_ratio'] = metrics['cut_edges'] / G.number_of_edges()
    else:
        metrics['cut_ratio'] = 0.0

    return metrics
