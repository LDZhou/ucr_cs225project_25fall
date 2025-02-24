import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Set
import networkx as nx
import matplotlib.pyplot as plt

# 注意：下面的import路径需要根据你的实际目录结构进行调整
# 假设你在 src/ 下运行，且 graph 文件夹也在 src/ 下：
from ..graph.graph_loader import load_graph, get_graph_stats
from ..graph.partition import GraphPartitioner
from ..graph.evaluation import compute_metrics


def visualize_partition(G: nx.Graph, partition: List[Set[int]],
                        title: str, save_path: str = None):
    """
    Visualize graph partition with a fixed-seed spring layout.
    For large graphs (10k+ nodes), consider random sampling or
    a different layout approach if runtime is too long or visualization is too cluttered.
    """
    # 设置可重复的布局，以及更多迭代次数
    pos = nx.spring_layout(G, seed=42, iterations=100)
    plt.figure(figsize=(12, 8))

    # 调整节点大小、边透明度等
    colors = plt.cm.rainbow(np.linspace(0, 1, len(partition)))
    for idx, part in enumerate(partition):
        nx.draw_networkx_nodes(G, pos,
                               nodelist=list(part),
                               node_color=[colors[idx]],
                               node_size=50,   # 可根据需要调大/调小
                               alpha=0.9)

    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()


def run_graph_experiments(
        graph_files: List[str],
        k_values: List[int],
        num_trials: int = 10,
        output_dir: str = "results/graphs"
) -> Dict[str, Any]:
    """
    Run comprehensive experiments on graph partitioning.
    For each graph file, tries different k values, attempts multiple trials.
    """
    results = {}

    for graph_file in graph_files:
        graph_name = Path(graph_file).stem
        results[graph_name] = {}

        # Load graph
        G = load_graph(graph_file)
        stats = get_graph_stats(G)
        results[graph_name]['stats'] = stats

        partitioner = GraphPartitioner(G)
        n = G.number_of_nodes()

        for k in k_values:
            print(f"\nProcessing {graph_name} with k={k}")
            results[graph_name][k] = {
                'runtime': [],
                'cut_edges': [],
                'modularity': [],
                'balance': [],
                'success_rate': 0,
                'metrics': []
            }

            successes = 0
            all_metrics = []

            # Base partition sizes
            base_size = n // k
            remainder = n % k
            target_sizes = [base_size + 1] * remainder + [base_size] * (k - remainder)

            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}")

                try:
                    # Time the partitioning
                    start_time = time.time()
                    partition = partitioner.partition(k, target_sizes)
                    end_time = time.time()

                    # Compute metrics
                    metrics = compute_metrics(G, partition, target_sizes)
                    all_metrics.append(metrics)

                    # Record results
                    runtime = end_time - start_time
                    results[graph_name][k]['runtime'].append(runtime)
                    results[graph_name][k]['cut_edges'].append(metrics['cut_edges'])
                    results[graph_name][k]['modularity'].append(metrics['modularity'])
                    results[graph_name][k]['balance'].append(metrics['balance'])

                    successes += 1

                    # 只可视化第一次成功的trial
                    if trial == 0:
                        os.makedirs(os.path.join(output_dir, graph_name), exist_ok=True)
                        visualize_partition(
                            G,
                            partition,
                            f"Partition (k={k})",
                            save_path=os.path.join(output_dir, graph_name, f"partition_k{k}.png")
                        )

                except Exception as e:
                    print(f"  Failed: {e}")

            # Calculate aggregate statistics
            results[graph_name][k]['success_rate'] = successes / num_trials

            if successes > 0:
                for metric in ['runtime', 'cut_edges', 'modularity', 'balance']:
                    vals = results[graph_name][k][metric]
                    results[graph_name][k][f'avg_{metric}'] = np.mean(vals)
                    results[graph_name][k][f'std_{metric}'] = np.std(vals)

                # 选择cut_edges最小的作为best solution
                best_idx = np.argmin(results[graph_name][k]['cut_edges'])
                results[graph_name][k]['best_metrics'] = all_metrics[best_idx]

    # 保存结果到 JSON
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "graph_results.json")
    with open(out_file, "w", encoding='utf-8') as f:
        json_results = json.loads(
            json.dumps(results, default=lambda x: x.item() if hasattr(x, 'item') else x)
        )
        json.dump(json_results, f, indent=2)

    return results


def plot_metrics(results: Dict[str, Any], output_dir: str):
    """
    Plot experimental results from run_graph_experiments.
    Generates line charts of runtime, cut_edges, modularity, balance, and success_rate vs k.
    """
    for graph_name, graph_results in results.items():
        if 'stats' in graph_results:
            del graph_results['stats']

        k_values = sorted(int(k) for k in graph_results.keys())
        metrics = ['runtime', 'cut_edges', 'modularity', 'balance', 'success_rate']

        for metric in metrics:
            plt.figure(figsize=(10, 6))

            # Get values
            values = []
            errors = []
            for k in k_values:
                if metric == 'success_rate':
                    values.append(graph_results[k][metric])
                    errors.append(0)
                else:
                    values.append(graph_results[k][f'avg_{metric}'])
                    errors.append(graph_results[k][f'std_{metric}'])

            # Plot with error bars
            plt.errorbar(k_values, values, yerr=errors, marker='o', capsize=4)

            plt.xlabel('Number of Partitions (k)')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()} vs k - {graph_name}')
            plt.grid(True)

            save_path = os.path.join(output_dir, graph_name, f'{metric}_vs_k.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()


if __name__ == "__main__":
    # 示例配置
    graph_files = ["data/graphs/PGPgiantcompo.graph"]
    k_values = [5, 8, 10]
    num_trials = 1

    output_dir = "results/graphs"
    os.makedirs(output_dir, exist_ok=True)

    print("\nStarting graph partitioning experiments...")
    results_data = run_graph_experiments(graph_files, k_values, num_trials=num_trials, output_dir=output_dir)

    print("\nPlotting metrics...")
    plot_metrics(results_data, output_dir=output_dir)

    print(f"\nExperiments complete. Results and plots are saved to: {output_dir}")
