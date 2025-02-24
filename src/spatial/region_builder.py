import random
from typing import List, Set, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import geopandas as gpd

from .utils import connected_components, is_articulation


def grow_region(unassigned: Set[int], target_size: int, adj: List[Set[int]],
                grow_attempts: int = 20, max_stalls: int = 50) -> Optional[Set[int]]:
    """
    Phase 1: Randomly grow a contiguous region from a seed until target_size.
    """
    for _ in range(grow_attempts):
        # 选择具有较多未分配邻居的种子点
        potential_seeds = list(unassigned)
        if len(potential_seeds) > 10:  # 如果候选太多，随机采样
            potential_seeds = random.sample(potential_seeds, 10)
        seed = max(potential_seeds, key=lambda x: len(adj[x] & unassigned))

        region = {seed}
        frontier = adj[seed] & unassigned
        stall_count = 0

        while frontier and len(region) < target_size and stall_count < max_stalls:
            # 优先选择非关节点
            candidates = [x for x in frontier if not is_articulation(x, region | {x}, adj)]
            chosen = random.choice(candidates) if candidates else random.choice(list(frontier))

            region.add(chosen)
            frontier.remove(chosen)
            frontier |= (adj[chosen] & unassigned)
            frontier -= region
            stall_count += 1
            if len(frontier) > 0:
                stall_count = 0

        if len(region) == target_size:
            return region
    return None


def merge_unassigned(region: Set[int], unassigned: Set[int],
                     adj: List[Set[int]]) -> Tuple[Set[int], Set[int]]:
    """
    Phase 2: Force the unassigned polygons to remain in exactly one component.
    """
    comps = connected_components(unassigned, adj)
    if len(comps) <= 1:
        return region, unassigned

    largest = max(comps, key=len)
    smaller_comps = set().union(*[c for c in comps if c != largest])

    region |= smaller_comps
    return region, largest


def split_or_adjust_region(region: Set[int], unassigned: Set[int], target_size: int,
                           adj: List[Set[int]], max_iter: int = 100) -> Optional[Set[int]]:
    """
    Phase 3: Ensure region ends with exactly target_size polygons.
    """
    rset = set(region)

    # Shrink if too big
    while len(rset) > target_size and max_iter > 0:
        max_iter -= 1
        boundary = [x for x in rset if adj[x] & unassigned] or list(rset)

        candidates = [x for x in boundary if not is_articulation(x, rset, adj)]
        chosen = random.choice(candidates) if candidates else random.choice(boundary)

        rset.remove(chosen)
        unassigned.add(chosen)

        comps = connected_components(rset, adj)
        if len(comps) > 1:
            largest = max(comps, key=len)
            for c in comps:
                if c != largest:
                    unassigned |= c
            rset = largest

    # Expand if too small
    while len(rset) < target_size and max_iter > 0:
        max_iter -= 1
        boundary = [u for u in unassigned if adj[u] & rset]
        if not boundary:
            break

        candidates = [u for u in boundary if not is_articulation(u, unassigned, adj)]
        chosen = random.choice(candidates) if candidates else random.choice(boundary)

        rset.add(chosen)
        unassigned.remove(chosen)

    return rset if len(rset) == target_size else None


def build_solution(dummy_id: int, gdf: gpd.GeoDataFrame, adj: List[Set[int]],
                   cardinalities: List[int], region_attempts: int = 30,
                   leftover_passes: int = 1) -> Optional[List[Set[int]]]:
    """Build a complete solution with specified number of regions."""
    sorted_cards = sorted(cardinalities, reverse=True)
    unassigned = set(range(len(gdf)))
    region_list = []

    for target in sorted_cards:
        built = False
        for _ in range(region_attempts):
            region = grow_region(unassigned, target, adj)
            if not region:
                continue

            leftover = unassigned - region
            region, leftover = merge_unassigned(region, leftover, adj)

            region_final = split_or_adjust_region(region, leftover, target, adj)
            if region_final and len(region_final) == target:
                region_list.append(region_final)
                unassigned = leftover
                built = True
                break

        if not built:
            return None

    if len(region_list) == len(cardinalities) and not unassigned:
        return region_list
    return None


def solve_prrp_parallel(gdf: gpd.GeoDataFrame, adj: List[Set[int]],
                        cardinalities: List[int], num_solutions: int = 10,
                        region_attempts: int = 30, leftover_passes: int = 1,
                        workers: Optional[int] = None, chunk_size: int = 1) -> List[List[Set[int]]]:
    """Generate multiple solutions in parallel."""
    n = len(gdf)
    if sum(cardinalities) != n:
        raise ValueError(f"Cardinalities must sum to {n}.")

    worker_func = partial(
        build_solution,
        gdf=gdf,
        adj=adj,
        cardinalities=cardinalities,
        region_attempts=region_attempts,
        leftover_passes=leftover_passes
    )

    workers = workers or cpu_count()
    chunk_size = max(1, num_solutions // (workers * 4))

    with Pool(workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(worker_func, range(num_solutions), chunksize=chunk_size),
            total=num_solutions,
            desc="Generating solutions",
            mininterval=0.5
        ))

    return [r for r in results if r is not None]