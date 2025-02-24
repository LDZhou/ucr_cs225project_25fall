import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from ..spatial.data_loader import load_data_and_build_adjacency, get_region_counts
from ..spatial.region_builder import solve_prrp_parallel
from ..spatial.utils import compute_heterogeneity_score, validate_solution
from ..spatial.visualization import (
    plot_solution,
    plot_multiple_solutions,
    plot_heterogeneity_distribution,
    plot_convergence
)


def run_spatial_experiments(
        data_files: List[str],
        k_values: List[int],
        num_solutions: int = 100,
        num_workers: int = None,
        output_dir: str = "results/spatial"
) -> Dict[str, Any]:
    """
    Run comprehensive experiments on spatial regionalization.
    """
    results = {}

    for data_file in data_files:
        name = Path(data_file).stem
        results[name] = {}

        # Load data
        print(f"\nProcessing {name}")
        gdf, adj = load_data_and_build_adjacency(data_file)
        total_areas = len(gdf)

        for k in k_values:
            print(f"\nGenerating solutions for k={k}")

            # Calculate target cardinalities
            base_size = total_areas // k
            remainder = total_areas % k
            cardinalities = [base_size + 1] * remainder + [base_size] * (k - remainder)

            # Generate solutions
            start_time = time.time()
            solutions = solve_prrp_parallel(
                gdf=gdf,
                adj=adj,
                cardinalities=cardinalities,
                num_solutions=num_solutions,
                workers=num_workers
            )
            end_time = time.time()

            # Compute metrics
            runtimes = []
            heterogeneity_scores = []
            valid_solutions = []

            for solution in solutions:
                if solution and validate_solution(gdf, solution, cardinalities, adj):
                    score = compute_heterogeneity_score(gdf, solution)
                    heterogeneity_scores.append(score)
                    valid_solutions.append(solution)

            # Store results
            results[name][k] = {
                'runtime': end_time - start_time,
                'success_rate': len(valid_solutions) / num_solutions,
                'avg_heterogeneity': np.mean(heterogeneity_scores) if heterogeneity_scores else None,
                'std_heterogeneity': np.std(heterogeneity_scores) if heterogeneity_scores else None,
                'min_heterogeneity': min(heterogeneity_scores) if heterogeneity_scores else None,
                'max_heterogeneity': max(heterogeneity_scores) if heterogeneity_scores else None
            }

            # Generate visualizations
            if valid_solutions:
                os.makedirs(os.path.join(output_dir, name), exist_ok=True)

                # Plot best solution
                best_idx = np.argmin(heterogeneity_scores)
                best_solution = valid_solutions[best_idx]
                plot_solution(
                    gdf=gdf,
                    solution=best_solution,
                    title=f"Best Solution (k={k})",
                    save_path=os.path.join(output_dir, name, f"best_solution_k{k}.png")
                )

                # Plot multiple solutions
                plot_multiple_solutions(
                    gdf=gdf,
                    solutions=valid_solutions[:4],
                    scores=heterogeneity_scores[:4],
                    save_path=os.path.join(output_dir, name, f"solution_comparison_k{k}.png")
                )

                # Plot score distribution
                plot_heterogeneity_distribution(
                    scores=heterogeneity_scores,
                    target_score=min(heterogeneity_scores),
                    title=f"Heterogeneity Distribution (k={k})",
                    save_path=os.path.join(output_dir, name, f"score_distribution_k{k}.png")
                )

                # Plot convergence
                plot_convergence(
                    scores=sorted(heterogeneity_scores),
                    title=f"Solution Convergence (k={k})",
                    save_path=os.path.join(output_dir, name, f"convergence_k{k}.png")
                )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    # Example usage
    data_files = ["data/spatial/cb_2015_42_tract_500k.shp"]
    k_values = [5, 10, 20]
    results = run_spatial_experiments(data_files, k_values)

    # Print summary
    for name, dataset_results in results.items():
        print(f"\nResults for {name}:")
        for k, k_results in dataset_results.items():
            print(f"\nk = {k}:")
            print(f"Runtime: {k_results['runtime']:.2f}s")
            print(f"Success rate: {k_results['success_rate']:.2%}")
            print(f"Avg heterogeneity: {k_results['avg_heterogeneity']:.2f}")
            print(f"Min heterogeneity: {k_results['min_heterogeneity']:.2f}")