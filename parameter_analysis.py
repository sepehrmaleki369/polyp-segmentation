import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import stats
import pandas as pd
import seaborn as sns
from tabulate import tabulate
from pathlib import Path

def load_metrics(results_dir):
    """Load metrics from a model's result directory"""
    metrics_path = os.path.join(results_dir, "test_metrics.npy")
    if not os.path.exists(metrics_path):
        return None

    metrics = np.load(metrics_path, allow_pickle=True).item()
    return metrics

def statistical_comparison(model1_metrics, model2_metrics, alpha=0.05):
    """Perform statistical tests to compare the models"""
    results = {}

    for metric in model1_metrics.keys():
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(model1_metrics[metric], model2_metrics[metric])

        # Wilcoxon signed-rank test
        w_stat, w_pvalue = stats.wilcoxon(model1_metrics[metric], model2_metrics[metric])

        results[metric] = {
            't_stat': t_stat,
            't_pvalue': p_value,
            'w_stat': w_stat,
            'w_pvalue': w_pvalue,
            'significant_t': p_value < alpha,
            'significant_w': w_pvalue < alpha
        }

    return results

def main(args):
    # Load metrics for both models
    model1_metrics = load_metrics(args.model1_dir)
    model2_metrics = load_metrics(args.model2_dir)

    if model1_metrics is None or model2_metrics is None:
        print("Error: Could not load metrics for one or both models.")
        return

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Calculate mean and std for each metric
    model1_avg = {key: np.mean(model1_metrics[key]) for key in model1_metrics}
    model1_std = {key: np.std(model1_metrics[key]) for key in model1_metrics}

    model2_avg = {key: np.mean(model2_metrics[key]) for key in model2_metrics}
    model2_std = {key: np.std(model2_metrics[key]) for key in model2_metrics}

    # Create comparison table
    comparison_table = []
    for metric in model1_metrics:
        row = [
            metric.capitalize(),
            f"{model1_avg[metric]:.4f} ± {model1_std[metric]:.4f}",
            f"{model2_avg[metric]:.4f} ± {model2_std[metric]:.4f}",
            f"{(model2_avg[metric] - model1_avg[metric]):.4f}"
        ]
        comparison_table.append(row)

    # Print and save comparison table
    table_str = tabulate(
        comparison_table,
        headers=["Metric", args.model1_name, args.model2_name, "Difference"],
        tablefmt="grid"
    )
    print("\nModel Comparison:")
    print(table_str)

    with open(os.path.join(output_dir, "comparison_table.txt"), 'w') as f:
        f.write(table_str)

    # Perform statistical tests
    stat_results = statistical_comparison(model1_metrics, model2_metrics)

    # Print and save statistical test results
    stat_table = []
    for metric, results in stat_results.items():
        row = [
            metric.capitalize(),
            f"{results['t_stat']:.4f}",
            f"{results['t_pvalue']:.4f}",
            "Yes" if results['significant_t'] else "No",
            f"{results['w_stat']:.4f}",
            f"{results['w_pvalue']:.4f}",
            "Yes" if results['significant_w'] else "No"
        ]
        stat_table.append(row)

    stat_table_str = tabulate(
        stat_table,
        headers=["Metric", "t-statistic", "p-value (t)", "Significant?",
                "Wilcoxon", "p-value (W)", "Significant?"],
        tablefmt="grid"
    )
    print("\nStatistical Tests:")
    print(stat_table_str)

    with open(os.path.join(output_dir, "statistical_tests.txt"), 'w') as f:
        f.write(stat_table_str)

    # Create comparison plots
    for metric in model1_metrics.keys():
        plt.figure(figsize=(10, 6))

        data = {
            args.model1_name: model1_metrics[metric],
            args.model2_name: model2_metrics[metric]
        }

        df = pd.DataFrame(data)

        # Create box plot
        sns.boxplot(data=df)
        plt.title(f'Comparison of {metric.capitalize()} Score')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'), dpi=300)
        plt.close()

    # Create a violin plot for Dice score
    plt.figure(figsize=(10, 6))
    data = {
        args.model1_name: model1_metrics['dice'],
        args.model2_name: model2_metrics['dice']
    }
    df = pd.DataFrame(data)

    sns.violinplot(data=df)
    plt.title('Distribution of Dice Scores')
    plt.ylabel('Dice Score')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'violin_dice.png'), dpi=300)
    plt.close()

    print(f"\nComparison results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two polyp segmentation models")
    parser.add_argument("--model1_dir", type=str, required=True,
                        help="Path to the first model's test results directory")
    parser.add_argument("--model2_dir", type=str, required=True,
                        help="Path to the second model's test results directory")
    parser.add_argument("--model1_name", type=str, default="TransUNet",
                        help="Name of the first model")
    parser.add_argument("--model2_name", type=str, default="MSRAFormer",
                        help="Name of the second model")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                        help="Directory to save comparison results")
    args = parser.parse_args()

    main(args)
