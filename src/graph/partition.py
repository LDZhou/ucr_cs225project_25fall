import networkx as nx
import random
import multiprocessing
from typing import List, Set, Optional
from functools import partial
from tqdm import tqdm


class GraphPartitioner:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.adj_list = self._build_adjacency_list()

    def _build_adjacency_list(self) -> List[Set[int]]:
        """Convert a NetworkX graph to an adjacency list representation."""
        n = self.graph.number_of_nodes()
        adj = [set() for _ in range(n)]
        for u, v in self.graph.edges():
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _find_seed_nodes(self, k: int) -> List[int]:
        """
        Improved Gapless Random Seed Selection:
         - First seed: Randomly select from unassigned nodes, prioritizing nodes with degree ≥ 10.
         - Subsequent seeds: Randomly select from the neighbors of the previous seed;
           if no neighbors are available, backtrack to earlier seeds.
         - If all candidate neighbors have a degree below 5, expand to two-hop neighbors via BFS
           to ensure growth potential.
        """
        n = self.graph.number_of_nodes()
        available = set(range(n))
        seeds = []

        # First round: Prioritize nodes with degree ≥ 10
        high_degree_candidates = [node for node in available if len(self.adj_list[node]) >= 10]
        if high_degree_candidates:
            first_seed = random.choice(high_degree_candidates)
        else:
            first_seed = random.choice(list(available))
        seeds.append(first_seed)
        available.remove(first_seed)

        # Subsequent seeds
        for i in range(1, k):
            # Select from the last seed's neighbors (only unassigned nodes)
            last_seed = seeds[-1]
            candidate_set = self.adj_list[last_seed] & available

            # If no candidates, backtrack to earlier seeds
            fallback_index = len(seeds) - 2
            while not candidate_set and fallback_index >= 0:
                candidate_set = self.adj_list[seeds[fallback_index]] & available
                fallback_index -= 1

            if not candidate_set:
                # If still empty, select randomly
                candidate = random.choice(list(available))
            else:
                # If all candidates have low degree (< 5), expand to two-hop neighbors
                if all(len(self.adj_list[node]) < 5 for node in candidate_set):
                    two_hop = set()
                    visited = {last_seed}
                    frontier = {last_seed}
                    depth = 0
                    while frontier and depth < 2:
                        next_frontier = set()
                        for node in frontier:
                            for nb in self.adj_list[node]:
                                if nb in available and nb not in visited:
                                    two_hop.add(nb)
                                    next_frontier.add(nb)
                                    visited.add(nb)
                        frontier = next_frontier
                        depth += 1
                    if two_hop:
                        candidate_set = two_hop
                candidate = random.choice(list(candidate_set))
            seeds.append(candidate)
            available.remove(candidate)
        return seeds

    def _partition_attempt(self, attempt_id: int, k: int, sizes: List[int]) -> Optional[List[Set[int]]]:
        """
        Single partition attempt (executed in parallel).
        Returns partitions satisfying strict size and connectivity requirements; otherwise, returns None.
        """
        n = self.graph.number_of_nodes()
        if n == 0:
            return [set() for _ in range(k)]

        # Use the improved seed selection method
        seeds = self._find_seed_nodes(k)
        partitions = [set([seed]) for seed in seeds]
        unassigned = set(range(n)) - set(seeds)

        # Region growth: Assign unallocated nodes to partitions
        while unassigned:
            # Select the most underfilled partition (with the largest size deficit)
            part_idx = max(range(k), key=lambda i: sizes[i] - len(partitions[i]))
            needed = sizes[part_idx] - len(partitions[part_idx])
            if needed <= 0:
                overshoot = all(sizes[i] - len(partitions[i]) <= 0 for i in range(k))
                if overshoot:
                    return None
                continue

            # Find all unassigned neighbors of the current partition
            frontier = set()
            for node in partitions[part_idx]:
                frontier.update(nb for nb in self.adj_list[node] if nb in unassigned)

            if not frontier:
                node = random.choice(list(unassigned))
            else:
                # Improved selection: Score = (connections to current partition) - (connections to unassigned nodes)
                candidate_scores = {x: len(self.adj_list[x] & partitions[part_idx]) - len(self.adj_list[x] & unassigned)
                                    for x in frontier}
                node = max(candidate_scores, key=candidate_scores.get)
            partitions[part_idx].add(node)
            unassigned.remove(node)

        # Local optimization: Adjust nodes from oversize to undersize partitions
        max_moves = 300
        moves = 0
        while moves < max_moves:
            improved = False
            for i in range(k):
                diff_i = len(partitions[i]) - sizes[i]
                if diff_i > 0:
                    for j in range(k):
                        diff_j = sizes[j] - len(partitions[j])
                        if diff_j > 0:
                            candidates = [node for node in partitions[i] if any(nb in partitions[j] for nb in self.adj_list[node])]
                            if candidates:
                                node_to_move = random.choice(candidates)
                                partitions[i].remove(node_to_move)
                                partitions[j].add(node_to_move)
                                improved = True
                                break
                if improved:
                    break
            if not improved:
                break
            moves += 1

        # Final check: Ensure partition sizes strictly match
        for i in range(k):
            if len(partitions[i]) != sizes[i]:
                return None

        # Check connectivity of each partition
        for part in partitions:
            if not part:
                return None
            subg = self.graph.subgraph(part)
            if not nx.is_connected(subg):
                return None

        return partitions

    def partition(self, k: int, sizes: List[int], max_attempts: int = 50) -> List[Set[int]]:
        """
        Partition the graph into k parts, each strictly matching the sizes in the list.
        Uses multiprocessing for parallel execution and returns a valid partitioning.
        """
        n = self.graph.number_of_nodes()
        if k != len(sizes) or sum(sizes) != n:
            raise ValueError("Invalid partition sizes (sum mismatch or length mismatch).")

        worker = partial(self._partition_attempt, k=k, sizes=sizes)
        num_cores = min(multiprocessing.cpu_count(), 48)
        print(f"Using {num_cores} cores for parallel processing")

        with multiprocessing.Pool(num_cores) as pool:
            results = list(tqdm(
                pool.imap_unordered(worker, range(max_attempts)),
                total=max_attempts,
                desc="Partition attempts"
            ))

        valid_results = [r for r in results if r is not None]

        if not valid_results:
            raise RuntimeError("Failed to find valid partition (connectivity or size constraints not met).")

        return valid_results[0]
