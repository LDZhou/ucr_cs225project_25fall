"""
Spatial regionalization module implementing PRRP algorithm.
"""

from .data_loader import load_data_and_build_adjacency, get_region_counts
from .region_builder import solve_prrp_parallel
from .utils import compute_heterogeneity_score, validate_solution
from .visualization import plot_solution, plot_multiple_solutions

__all__ = [
    'load_data_and_build_adjacency',
    'get_region_counts',
    'solve_prrp_parallel',
    'compute_heterogeneity_score',
    'validate_solution',
    'plot_solution',
    'plot_multiple_solutions'
]