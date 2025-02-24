import geopandas as gpd
import matplotlib.pyplot as plt
from typing import List, Set, Optional
import numpy as np


def plot_solution(gdf: gpd.GeoDataFrame, solution: List[Set[int]],
                  title: str = "Regionalization Solution",
                  figsize: tuple = (12, 8),
                  save_path: Optional[str] = None) -> None:
    """
    Plot a regionalization solution.

    Args:
        gdf: GeoDataFrame with geometry
        solution: List of sets containing region assignments
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Create a copy to avoid modifying original
    plot_gdf = gdf.copy()

    # Assign region labels
    plot_gdf['region'] = -1
    for i, region in enumerate(solution):
        plot_gdf.loc[list(region), 'region'] = i

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    plot_gdf.plot(column='region', categorical=True, legend=True,
                  ax=ax, legend_kwds={'title': 'Region'})

    # Customize the plot
    ax.set_title(title)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    plt.close()


def plot_multiple_solutions(gdf: gpd.GeoDataFrame, solutions: List[List[Set[int]]],
                            scores: Optional[List[float]] = None,
                            max_plots: int = 4,
                            figsize: tuple = (15, 15),
                            save_path: Optional[str] = None) -> None:
    """
    Plot multiple solutions in a grid.

    Args:
        gdf: GeoDataFrame with geometry
        solutions: List of solutions to plot
        scores: Optional list of scores for each solution
        max_plots: Maximum number of solutions to plot
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    n_plots = min(len(solutions), max_plots)
    rows = int(np.ceil(np.sqrt(n_plots)))
    cols = int(np.ceil(n_plots / rows))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for idx, solution in enumerate(solutions[:n_plots]):
        row = idx // cols
        col = idx % cols

        # Create plot
        plot_gdf = gdf.copy()
        plot_gdf['region'] = -1
        for i, region in enumerate(solution):
            plot_gdf.loc[list(region), 'region'] = i

        plot_gdf.plot(column='region', categorical=True, legend=True,
                      ax=axes[row, col], legend_kwds={'title': 'Region'})

        title = f"Solution {idx + 1}"
        if scores is not None:
            title += f"\nScore: {scores[idx]:.2f}"
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    # Remove empty subplots
    for idx in range(n_plots, rows * cols):
        row = idx // cols
        col = idx % cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    plt.close()


def plot_heterogeneity_distribution(scores: List[float],
                                    target_score: float = None,
                                    title: str = "Solution Heterogeneity Distribution",
                                    figsize: tuple = (10, 6),
                                    save_path: Optional[str] = None) -> None:
    """
    Plot histogram of heterogeneity scores.

    Args:
        scores: List of heterogeneity scores
        target_score: Optional target score to highlight
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    plt.figure(figsize=figsize)

    plt.hist(scores, bins=30, edgecolor='black')
    plt.xlabel('Heterogeneity Score')
    plt.ylabel('Frequency')
    plt.title(title)

    if target_score is not None:
        plt.axvline(x=target_score, color='r', linestyle='--',
                    label=f'Target Score: {target_score:.2f}')
        plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    plt.close()


def plot_convergence(scores: List[float],
                     title: str = "Solution Convergence",
                     figsize: tuple = (10, 6),
                     save_path: Optional[str] = None) -> None:
    """
    Plot convergence of solution scores over iterations.

    Args:
        scores: List of scores in order of generation
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    plt.figure(figsize=figsize)

    iterations = range(1, len(scores) + 1)
    plt.plot(iterations, scores, 'b-')
    plt.plot(iterations, scores, 'bo')

    plt.xlabel('Iteration')
    plt.ylabel('Heterogeneity Score')
    plt.title(title)

    # Add trend line
    z = np.polyfit(iterations, scores, 1)
    p = np.poly1d(z)
    plt.plot(iterations, p(iterations), "r--", alpha=0.8,
             label=f'Trend: {z[0]:.2e}x + {z[1]:.2f}')

    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    plt.close()