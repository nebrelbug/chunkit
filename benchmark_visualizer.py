#!/usr/bin/env python3
"""
Simplified Benchmark Visualizer

Creates one grouped bar chart per dataset group with performance stats.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class BenchmarkVisualizer:
    """Creates visualizations for tokenizer benchmark results."""

    def __init__(self, output_dir: str = "./benchmarks"):
        # Create nested directory structure
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up matplotlib style
        plt.style.use("default")
        sns.set_palette("husl")

    def create_all_visualizations(self, results: List[Dict[str, Any]]) -> None:
        """Create all visualizations for the benchmark results."""
        if not results:
            logger.warning("No results to visualize")
            return

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Filter out failed results
        df = df[df["chars_per_token"] > 0]

        if df.empty:
            logger.warning("No successful results to visualize")
            return

        logger.info(f"Creating visualizations for {len(df)} results...")

        # Get unique groups
        groups = df["group"].unique()

        # Create one grouped bar chart per group
        for group in groups:
            self._create_group_visualization(df, group)

        # Create overall summary
        self._create_overall_summary(df)

        logger.info(f"All visualizations saved to {self.output_dir}")

    def _create_group_visualization(self, df: pd.DataFrame, group_name: str) -> None:
        """Create a grouped bar chart for a specific dataset group."""
        group_data = df[df["group"] == group_name]

        if group_data.empty:
            logger.warning(f"No data for group: {group_name}")
            return

        # Prepare data for grouped bar chart
        # Group by subset (dataset) and tokenizer, then get mean chars_per_token
        pivot_data = (
            group_data.groupby(["subset", "tokenizer"])["chars_per_token"]
            .mean()
            .unstack(fill_value=0)
        )

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Create grouped bar chart
        pivot_data.plot(kind="bar", ax=ax, width=0.8, rot=45)

        # Customize the plot
        ax.set_title(
            f"{group_name}\nCompression Efficiency by Dataset and Tokenizer",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
        ax.set_ylabel(
            "Characters per Token (Higher = Better)", fontsize=12, fontweight="bold"
        )

        # Improve legend
        ax.legend(
            title="Tokenizer", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10
        )

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, rotation=90, padding=3)

        # Add grid for better readability
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Calculate detailed statistics
        group_stats = (
            group_data.groupby("tokenizer")
            .agg(
                {
                    "chars_per_token": ["mean", "std", "count"],
                    "tokens_per_second": ["mean", "std"],
                    "samples_processed": "sum",
                }
            )
            .round(3)
        )

        # Flatten column names
        group_stats.columns = ["_".join(col).strip() for col in group_stats.columns]

        # Create stats table - make it full width and larger text
        stats_text = self._format_stats_table(group_stats)

        # Position stats box to span full width
        fig.text(
            0.1,
            0.02,
            stats_text,
            fontsize=10,
            fontfamily="monospace",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgray", alpha=0.9),
            transform=fig.transFigure,
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.35)  # Make more room for larger stats table

        # Save the plot
        safe_name = group_name.lower().replace(" ", "_").replace("-", "_")
        filename = f"benchmark_{safe_name}.png"
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Created grouped bar chart: {filename}")

    def _format_stats_table(self, stats_df: pd.DataFrame) -> str:
        """Format statistics as a readable table with larger text."""
        lines = ["DETAILED STATISTICS"]
        lines.append("=" * 100)
        lines.append(
            f"{'Tokenizer':<25} {'Avg C/T':<10} {'Std C/T':<10} {'Avg T/s':<10} {'Std T/s':<10} {'Samples':<10}"
        )
        lines.append("=" * 100)

        for tokenizer, row in stats_df.iterrows():
            lines.append(
                f"{tokenizer:<25} "
                f"{row['chars_per_token_mean']:<10.3f} "
                f"{row['chars_per_token_std']:<10.3f} "
                f"{row['tokens_per_second_mean']:<10.0f} "
                f"{row['tokens_per_second_std']:<10.0f} "
                f"{int(row['samples_processed_sum']):<10}"
            )

        return "\n".join(lines)

    def _create_overall_summary(self, df: pd.DataFrame) -> None:
        """Create an overall summary visualization."""
        # Overall performance across all groups
        overall_stats = (
            df.groupby("tokenizer")
            .agg(
                {
                    "chars_per_token": ["mean", "std"],
                    "tokens_per_second": ["mean", "std"],
                    "samples_processed": "sum",
                }
            )
            .round(3)
        )

        # Create summary chart
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Overall compression efficiency
        compression_data = (
            df.groupby("tokenizer")["chars_per_token"]
            .mean()
            .sort_values(ascending=False)
        )
        bars = ax.bar(
            range(len(compression_data)),
            compression_data.values,
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][
                : len(compression_data)
            ],
        )

        ax.set_title(
            "Overall Tokenizer Performance\n(Average Compression Efficiency Across All Groups)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_ylabel("Characters per Token (Higher = Better)", fontsize=12)
        ax.set_xticks(range(len(compression_data)))
        ax.set_xticklabels(compression_data.index, rotation=45, ha="right", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, compression_data.values)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Add summary stats
        best_tokenizer = compression_data.index[0]
        best_score = compression_data.iloc[0]
        total_samples = df["samples_processed"].sum()

        summary_text = f"""
BENCHMARK SUMMARY:
• Best Overall: {best_tokenizer} ({best_score:.3f} chars/token)
• Total Samples: {total_samples:,}
• Groups Tested: {df["group"].nunique()}
• Tokenizers: {df["tokenizer"].nunique()}
        """.strip()

        ax.text(
            0.02,
            0.98,
            summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()

        # Save the plot
        filename = "benchmark_overall_summary.png"
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Created overall summary: {filename}")

        # Also save detailed CSV
        overall_stats.to_csv(self.output_dir / "overall_performance.csv")
        df.to_csv(self.output_dir / "detailed_results.csv", index=False)
        logger.info("Saved detailed results to CSV files")
