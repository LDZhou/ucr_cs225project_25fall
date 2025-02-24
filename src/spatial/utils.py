from typing import Set, List, Optional
import geopandas as gpd


def connected_components(areas: Set[int], adj: List[Set[int]]) -> List[Set[int]]:
    """
    Find connected components in a set of areas using BFS/DFS.
    """
    visited = set()
    components = []

    for start in areas:
        if start not in visited:
            stack = [start]
            comp = set()

            while stack:
                cur = stack.pop()
                if cur not in visited:
                    visited.add(cur)
                    comp.add(cur)
                    stack.extend([nb for nb in adj[cur] if nb in areas and nb not in visited])

            components.append(comp)

    return components


def is_articulation(poly_idx: int, region: Set[int], adj: List[Set[int]]) -> bool:
    """
    Check if removing a polygon would split the region.
    """
    if poly_idx not in region:
        return False

    region_minus = region - {poly_idx}
    if not region_minus:
        return False

    before = connected_components(region, adj)
    after = connected_components(region_minus, adj)
    return len(after) > len(before)


def compute_heterogeneity_score(gdf: gpd.GeoDataFrame, solution: List[Set[int]]) -> float:
    """
    Compute heterogeneity score for a solution.
    Lower scores indicate better regionalization.
    """
    total_score = 0
    for region in solution:
        region_geom = gdf.iloc[list(region)].geometry
        centroid = region_geom.centroid.unary_union
        score = sum(region_geom.distance(centroid))
        total_score += score
    return total_score


def validate_solution(gdf: gpd.GeoDataFrame, solution: List[Set[int]],
                      cardinalities: List[int], adj: List[Set[int]]) -> bool:
    """
    Validate a solution meets all requirements.
    """
    # Check number of regions
    if len(solution) != len(cardinalities):
        return False

    # Check region sizes
    for region, target in zip(solution, cardinalities):
        if len(region) != target:
            return False

    # Check each region is connected
    for region in solution:
        if len(connected_components(region, adj)) > 1:
            return False

    # Check all areas are assigned exactly once
    assigned = set().union(*solution)
    if len(assigned) != len(gdf) or any(len(r & assigned - r) > 0 for r in solution):
        return False

    return True