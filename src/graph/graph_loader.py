import networkx as nx
from typing import Tuple, Dict, Any


def load_graph(graph_file: str) -> nx.Graph:
    """
    Load graph from adjacency list format.
    First line: <num_nodes> <num_edges>
    Following lines: <neighbor1> <neighbor2> ...
    Note: Node IDs start from 0
    """
    print(f"\nAnalyzing file: {graph_file}")

    # Read the graph into memory first
    with open(graph_file, 'r') as f:
        lines = f.readlines()

    # Parse header
    n, m = map(int, lines[0].strip().split())
    print(f"Header specifies {n} nodes and {m} edges")

    # Initialize graph
    G = nx.Graph()
    G.add_nodes_from(range(n))  # Add exactly n nodes

    # Add edges
    edges = set()
    for node_id in range(n):
        if node_id < len(lines):  # Ensure we don't go past end of file
            neighbors = lines[node_id + 1].strip().split()
            if neighbors:  # If line is not empty
                for neighbor in map(int, neighbors):
                    if node_id != neighbor and neighbor < n:  # Skip self-loops and invalid nodes
                        edges.add(tuple(sorted([node_id, neighbor])))

    # Add all edges at once
    G.add_edges_from(edges)

    print(f"\nGraph Analysis:")
    print(f"Nodes: {G.number_of_nodes()} (specified: {n})")
    print(f"Edges: {G.number_of_edges()} (specified: {m})")
    print(f"Density: {nx.density(G):.6f}")

    # Check connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest_comp = max(components, key=len)
        print(f"\nConnectivity Analysis:")
        print(f"Found {len(components)} components")
        print(f"Largest component: {len(largest_comp)} nodes ({len(largest_comp) / n * 100:.1f}%)")

        # Extract largest component
        G = G.subgraph(largest_comp).copy()
        # Renumber nodes to be 0-based consecutive integers
        G = nx.convert_node_labels_to_integers(G)
        print(f"\nExtracted largest component:")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")

    return G


def get_graph_stats(G: nx.Graph) -> Dict[str, Any]:
    """
    Compute basic graph statistics
    """
    n = G.number_of_nodes()
    stats = {
        'num_nodes': n,
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_connected(G),
        'avg_degree': float(sum(dict(G.degree()).values()) / n),
        'min_degree': min(dict(G.degree()).values()),
        'max_degree': max(dict(G.degree()).values()),
    }

    # Get degree distribution
    degrees = [d for _, d in G.degree()]
    stats['degree_distribution'] = {
        'min': min(degrees),
        'max': max(degrees),
        'avg': sum(degrees) / len(degrees),
        'histogram': nx.degree_histogram(G)[:10]  # First 10 values
    }

    # Compute local statistics on a sample if graph is large
    if n > 1000:
        sample_size = min(1000, n)
        sample_nodes = list(G.nodes())[:sample_size]
        sample_graph = G.subgraph(sample_nodes)
        try:
            stats['sample_clustering'] = nx.average_clustering(sample_graph)
        except:
            stats['sample_clustering'] = None
    else:
        try:
            stats['clustering_coeff'] = nx.average_clustering(G)
            stats['diameter'] = nx.diameter(G)
        except:
            stats['clustering_coeff'] = None
            stats['diameter'] = None

    return stats
