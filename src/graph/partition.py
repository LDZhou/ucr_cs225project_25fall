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
        """Convert networkx graph to adjacency list representation."""
        n = self.graph.number_of_nodes()
        adj = [set() for _ in range(n)]
        for u, v in self.graph.edges():
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _find_seed_nodes(self, k: int) -> List[int]:
        """
        改进的Gapless Random Seed Selection：
         - 第一个种子：从未分配节点中随机选择，但优先选择度数 ≥ 10 的节点。
         - 后续种子：从前一分区（即上一个种子）的邻居中随机选择；
           若邻居为空，则回退到更早的种子；
         - 如果候选邻居节点的度数全部低于 5，则通过 BFS 扩展到两跳邻居，以确保生长潜力。
        """
        n = self.graph.number_of_nodes()
        available = set(range(n))
        seeds = []

        # 第一轮：优先选择度数 ≥ 10 的节点
        high_degree_candidates = [node for node in available if len(self.adj_list[node]) >= 10]
        if high_degree_candidates:
            first_seed = random.choice(high_degree_candidates)
        else:
            first_seed = random.choice(list(available))
        seeds.append(first_seed)
        available.remove(first_seed)

        # 后续种子
        for i in range(1, k):
            # 从最后一个种子邻居中挑选（仅考虑未分配节点）
            last_seed = seeds[-1]
            candidate_set = self.adj_list[last_seed] & available

            # 若候选集合为空，则依次回退到之前的种子
            fallback_index = len(seeds) - 2
            while not candidate_set and fallback_index >= 0:
                candidate_set = self.adj_list[seeds[fallback_index]] & available
                fallback_index -= 1

            if not candidate_set:
                # 若仍为空，则随机选择
                candidate = random.choice(list(available))
            else:
                # 如果候选节点全部度数较低（< 5），则扩展到两跳邻居
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
        单次分区尝试（可并行执行）。
        返回满足严格大小和连通性要求的分区（列表形式），否则返回 None。
        """
        n = self.graph.number_of_nodes()
        if n == 0:
            return [set() for _ in range(k)]

        # 使用改进后的种子选择方法
        seeds = self._find_seed_nodes(k)
        partitions = [set([seed]) for seed in seeds]
        unassigned = set(range(n)) - set(seeds)

        # 区域生长：不断将未分配节点分配给各区域
        while unassigned:
            # 选择当前最缺节点的区域（目标大小与当前区域大小的差值最大）
            part_idx = max(range(k), key=lambda i: sizes[i] - len(partitions[i]))
            needed = sizes[part_idx] - len(partitions[part_idx])
            if needed <= 0:
                overshoot = all(sizes[i] - len(partitions[i]) <= 0 for i in range(k))
                if overshoot:
                    return None
                continue

            # 从当前区域的节点中寻找所有未分配的邻居
            frontier = set()
            for node in partitions[part_idx]:
                frontier.update(nb for nb in self.adj_list[node] if nb in unassigned)

            if not frontier:
                node = random.choice(list(unassigned))
            else:
                # 改进选择：计算评分 = (与当前区域连接数) - (与未分配节点连接数)
                candidate_scores = {x: len(self.adj_list[x] & partitions[part_idx]) - len(self.adj_list[x] & unassigned)
                                    for x in frontier}
                node = max(candidate_scores, key=candidate_scores.get)
            partitions[part_idx].add(node)
            unassigned.remove(node)

        # 局部搜索调整：尝试将节点从超额区域转移到不足区域
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

        # 最终检查：每个分区大小是否严格匹配
        for i in range(k):
            if len(partitions[i]) != sizes[i]:
                return None

        # 检查每个分区连通性
        for part in partitions:
            if not part:
                return None
            subg = self.graph.subgraph(part)
            if not nx.is_connected(subg):
                return None

        return partitions

    def partition(self, k: int, sizes: List[int], max_attempts: int = 50) -> List[Set[int]]:
        """
        将图划分成 k 个部分，每个部分的节点数严格为 sizes 中对应的值。
        采用多进程并行尝试，返回满足大小和连通性要求的分区方案。
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
