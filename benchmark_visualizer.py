#!/usr/bin/env python3
"""
Benchmark Visualizer - Creates charts and visualizations for tokenizer benchmark results.
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

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

    def create_all_visualizations(
        self, results: List[Dict[str, Any]], output_dir: str
    ) -> None:
        """Create all visualizations for benchmark results."""
        output_path = Path(output_dir)

        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Filter successful results
        successful_df = df[df["chars_per_token"] > 0]

        if successful_df.empty:
            logger.warning("No successful results to visualize")
            return

        logger.info(
            f"Creating visualizations with {len(successful_df)} successful results"
        )

        # Create grouped bar charts by dataset group
        self._create_grouped_charts(successful_df, output_path)

        # Create overall comparison chart
        self._create_overall_comparison(successful_df, output_path)

        # Create performance heatmap
        self._create_performance_heatmap(successful_df, output_path)

        # Create summary table
        self._create_summary_table(successful_df, output_path)

        logger.info("All visualizations created successfully")

    def _create_grouped_charts(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create grouped bar charts for each dataset group."""
        if "group" not in df.columns:
            logger.warning("No group information available for grouped charts")
            return

        groups = df["group"].unique()

        for group in groups:
            group_df = df[df["group"] == group]

            # Create pivot table for plotting
            pivot_data = group_df.pivot_table(
                index="subset",
                columns="tokenizer",
                values="chars_per_token",
                aggfunc="mean",
            )

            if pivot_data.empty:
                continue

            # Create grouped bar chart
            fig, ax = plt.subplots(figsize=(12, 8))

            pivot_data.plot(kind="bar", ax=ax, width=0.8)

            ax.set_title(
                f"Tokenizer Performance - {group}", fontsize=16, fontweight="bold"
            )
            ax.set_xlabel("Dataset", fontsize=12)
            ax.set_ylabel(
                "Characters per Token (Higher = Better Compression)", fontsize=12
            )
            ax.legend(title="Tokenizer", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # Save chart
            filename = (
                f"performance_{group.lower().replace(' ', '_').replace('-', '_')}.png"
            )
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Created grouped chart: {filename}")

    def _create_overall_comparison(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create overall tokenizer comparison scatter plot: vocab size vs compression."""
        # Calculate average performance per tokenizer
        tokenizer_avg = df.groupby("tokenizer")["chars_per_token"].mean()

        # Get vocabulary sizes for each tokenizer
        vocab_sizes = {}
        tokenizer_names = tokenizer_avg.index.tolist()

        # We need to get vocab sizes from the actual tokenizers
        # This requires access to the tokenizer manager, so we'll estimate based on known tokenizers
        known_vocab_sizes = {
            "GPT-2": 50257,
            "GPT-4": 100256,
            "LLaMA-3.3": 128000,
            "Gemma": 256000,
            "Mixtral": 32000,
        }

        # Extract vocab sizes, handling custom tokenizers
        for tokenizer_name in tokenizer_names:
            if tokenizer_name in known_vocab_sizes:
                vocab_sizes[tokenizer_name] = known_vocab_sizes[tokenizer_name]
            elif tokenizer_name.startswith("Custom-"):
                # For custom tokenizers, estimate based on typical BPE sizes
                vocab_sizes[tokenizer_name] = 32000  # Common BPE vocab size
            else:
                # Default estimate for unknown tokenizers
                vocab_sizes[tokenizer_name] = 50000

        # Create scatter plot data
        x_values = [vocab_sizes[name] for name in tokenizer_names]
        y_values = [tokenizer_avg[name] for name in tokenizer_names]

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create scatter plot with different colors for each tokenizer
        colors = plt.cm.Set3(range(len(tokenizer_names)))
        scatter = ax.scatter(
            x_values,
            y_values,
            c=colors,
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
        )

        # Add labels for each point
        for i, name in enumerate(tokenizer_names):
            ax.annotate(
                name,
                (x_values[i], y_values[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(
            "Tokenizer Performance: Vocabulary Size vs Compression Efficiency",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Vocabulary Size", fontsize=12)
        ax.set_ylabel(
            "Average Characters per Token (Higher = Better Compression)", fontsize=12
        )
        ax.grid(True, alpha=0.3)

        # Add trend line if we have enough points
        if len(x_values) > 2:
            import numpy as np

            z = np.polyfit(x_values, y_values, 1)
            p = np.poly1d(z)
            ax.plot(
                x_values, p(x_values), "r--", alpha=0.8, linewidth=2, label="Trend line"
            )
            ax.legend()

        # Format x-axis to show vocabulary sizes nicely
        ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

        plt.tight_layout()
        plt.savefig(output_dir / "overall_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("Created vocabulary size vs compression scatter plot")

    def _create_performance_heatmap(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create performance heatmap."""
        # Create pivot table for heatmap
        pivot_data = df.pivot_table(
            index="tokenizer",
            columns="dataset",
            values="chars_per_token",
            aggfunc="mean",
        )

        if pivot_data.empty:
            return

        fig, ax = plt.subplots(figsize=(16, 8))

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=pivot_data.mean().mean(),
            ax=ax,
            cbar_kws={"label": "Characters per Token"},
        )

        ax.set_title("Tokenizer Performance Heatmap", fontsize=16, fontweight="bold")
        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel("Tokenizer", fontsize=12)

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            output_dir / "performance_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info("Created performance heatmap")

    def _create_summary_table(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create summary statistics table."""
        # Calculate summary statistics
        summary_stats = (
            df.groupby("tokenizer")
            .agg(
                {
                    "chars_per_token": ["mean", "std", "min", "max"],
                    "tokens_per_second": ["mean", "std"],
                    "samples_processed": "sum",
                }
            )
            .round(3)
        )

        # Flatten column names
        summary_stats.columns = [
            "_".join(col).strip() for col in summary_stats.columns.values
        ]

        # Create figure for table
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis("tight")
        ax.axis("off")

        # Create table
        table = ax.table(
            cellText=summary_stats.values,
            rowLabels=summary_stats.index,
            colLabels=summary_stats.columns,
            cellLoc="center",
            loc="center",
            colWidths=[0.12] * len(summary_stats.columns),
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style the table
        for i in range(len(summary_stats.columns)):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        for i in range(len(summary_stats.index)):
            table[(i + 1, -1)].set_facecolor("#E8F5E8")

        ax.set_title(
            "Tokenizer Performance Summary", fontsize=16, fontweight="bold", pad=20
        )

        plt.tight_layout()
        plt.savefig(output_dir / "summary_table.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("Created summary table")
