#!/usr/bin/env python3
"""
Tokenizer Benchmark Visualizer

Generate charts and tables from benchmark results for easy comparison.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class BenchmarkVisualizer:
    """Visualize tokenizer benchmark results."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.df = None
        self.summary = None

        # Load data
        self._load_data()

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

    def _load_data(self):
        """Load benchmark results and summary."""
        # Load detailed results
        csv_path = self.results_dir / "benchmark_results.csv"
        if csv_path.exists():
            self.df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(self.df)} benchmark results")
        else:
            raise FileNotFoundError(f"Results file not found: {csv_path}")

        # Load summary
        summary_path = self.results_dir / "benchmark_summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                self.summary = json.load(f)
            print(
                f"✅ Loaded summary with {len(self.summary['tokenizers'])} tokenizers"
            )

    def create_all_visualizations(self):
        """Create all visualization charts."""
        output_dir = self.results_dir / "visualizations"
        output_dir.mkdir(exist_ok=True)

        print("📊 Creating visualizations...")

        # 1. Overall compression comparison
        self.plot_compression_by_tokenizer(output_dir / "compression_by_tokenizer.png")

        # 2. Language-specific performance
        self.plot_compression_by_language(output_dir / "compression_by_language.png")

        # 3. Domain-specific performance
        self.plot_compression_by_domain(output_dir / "compression_by_domain.png")

        # 4. Heatmap of tokenizer vs language performance
        self.plot_performance_heatmap(output_dir / "performance_heatmap.png")

        # 5. Encoding speed comparison
        self.plot_encoding_speed(output_dir / "encoding_speed.png")

        # 6. Detailed comparison table
        self.create_comparison_table(output_dir / "comparison_table.html")

        print(f"✅ All visualizations saved to: {output_dir}")

    def plot_compression_by_tokenizer(self, output_path: Path):
        """Plot average compression ratio by tokenizer."""
        plt.figure(figsize=(12, 6))

        # Calculate average compression by tokenizer
        avg_compression = (
            self.df.groupby("tokenizer_name")["compression_ratio"]
            .agg(["mean", "std"])
            .reset_index()
        )
        avg_compression = avg_compression.sort_values("mean", ascending=False)

        # Create bar plot with error bars
        plt.bar(
            avg_compression["tokenizer_name"],
            avg_compression["mean"],
            yerr=avg_compression["std"],
            capsize=5,
            alpha=0.8,
        )

        plt.title(
            "Average Compression Ratio by Tokenizer\n(Higher = Better)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Tokenizer", fontsize=12)
        plt.ylabel("Compression Ratio (chars/token)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📈 Saved compression comparison: {output_path}")

    def plot_compression_by_language(self, output_path: Path):
        """Plot compression performance by language."""
        plt.figure(figsize=(14, 8))

        # Create box plot
        sns.boxplot(
            data=self.df, x="language", y="compression_ratio", hue="tokenizer_name"
        )

        plt.title(
            "Compression Ratio by Language and Tokenizer",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Language", fontsize=12)
        plt.ylabel("Compression Ratio (chars/token)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📈 Saved language comparison: {output_path}")

    def plot_compression_by_domain(self, output_path: Path):
        """Plot compression performance by domain."""
        plt.figure(figsize=(10, 6))

        # Create violin plot
        sns.violinplot(
            data=self.df, x="domain", y="compression_ratio", hue="tokenizer_name"
        )

        plt.title(
            "Compression Ratio by Domain and Tokenizer", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Domain", fontsize=12)
        plt.ylabel("Compression Ratio (chars/token)", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📈 Saved domain comparison: {output_path}")

    def plot_performance_heatmap(self, output_path: Path):
        """Plot heatmap of tokenizer performance across languages."""
        plt.figure(figsize=(12, 8))

        # Create pivot table
        pivot_data = (
            self.df.groupby(["tokenizer_name", "language"])["compression_ratio"]
            .mean()
            .unstack()
        )

        # Create heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            cbar_kws={"label": "Compression Ratio (chars/token)"},
        )

        plt.title(
            "Tokenizer Performance Heatmap\n(Higher values = Better compression)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Language", fontsize=12)
        plt.ylabel("Tokenizer", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📈 Saved performance heatmap: {output_path}")

    def plot_encoding_speed(self, output_path: Path):
        """Plot encoding speed comparison."""
        plt.figure(figsize=(12, 6))

        # Calculate average encoding time by tokenizer
        avg_speed = (
            self.df.groupby("tokenizer_name")["encoding_time_ms"]
            .agg(["mean", "std"])
            .reset_index()
        )
        avg_speed = avg_speed.sort_values("mean")

        # Create bar plot with error bars
        plt.bar(
            avg_speed["tokenizer_name"],
            avg_speed["mean"],
            yerr=avg_speed["std"],
            capsize=5,
            alpha=0.8,
            color="lightblue",
        )

        plt.title(
            "Average Encoding Speed by Tokenizer\n(Lower = Faster)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Tokenizer", fontsize=12)
        plt.ylabel("Encoding Time (ms)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📈 Saved encoding speed comparison: {output_path}")

    def create_comparison_table(self, output_path: Path):
        """Create detailed comparison table."""
        # Calculate comprehensive statistics
        stats = (
            self.df.groupby("tokenizer_name")
            .agg(
                {
                    "compression_ratio": ["mean", "std", "min", "max"],
                    "tokens_per_char": ["mean", "std"],
                    "encoding_time_ms": ["mean", "std"],
                    "vocab_size": "first",
                }
            )
            .round(4)
        )

        # Flatten column names
        stats.columns = ["_".join(col).strip() for col in stats.columns]
        stats = stats.reset_index()

        # Rename columns for readability
        column_mapping = {
            "tokenizer_name": "Tokenizer",
            "compression_ratio_mean": "Avg Compression",
            "compression_ratio_std": "Compression Std",
            "compression_ratio_min": "Min Compression",
            "compression_ratio_max": "Max Compression",
            "tokens_per_char_mean": "Avg Tokens/Char",
            "tokens_per_char_std": "Tokens/Char Std",
            "encoding_time_ms_mean": "Avg Encoding Time (ms)",
            "encoding_time_ms_std": "Encoding Time Std",
            "vocab_size_first": "Vocab Size",
        }
        stats = stats.rename(columns=column_mapping)

        # Sort by average compression ratio (descending)
        stats = stats.sort_values("Avg Compression", ascending=False)

        # Create HTML table with styling
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tokenizer Benchmark Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .best {{ background-color: #d4edda !important; }}
                .summary {{ background-color: #e7f3ff; padding: 15px; margin: 20px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Tokenizer Benchmark Results</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Tests:</strong> {self.summary["total_tests"]:,}</p>
                <p><strong>Tokenizers Tested:</strong> {", ".join(self.summary["tokenizers"])}</p>
                <p><strong>Languages:</strong> {", ".join(self.summary["languages"])}</p>
                <p><strong>Best Overall:</strong> {self.summary["best_tokenizer_overall"]}</p>
            </div>
            
            <h2>Detailed Comparison</h2>
            {stats.to_html(index=False, classes="table", escape=False)}
            
            <h2>Best Tokenizer by Language</h2>
            <ul>
        """

        for lang, tokenizer in self.summary["best_tokenizer_by_language"].items():
            html += f"<li><strong>{lang}:</strong> {tokenizer}</li>"

        html += """
            </ul>
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html)

        print(f"📋 Saved comparison table: {output_path}")

    def print_summary(self):
        """Print benchmark summary to console."""
        print("\n" + "=" * 60)
        print("📊 TOKENIZER BENCHMARK SUMMARY")
        print("=" * 60)

        print(f"Total tests run: {self.summary['total_tests']:,}")
        print(f"Tokenizers tested: {len(self.summary['tokenizers'])}")
        print(f"Languages tested: {len(self.summary['languages'])}")

        print(f"\n🏆 BEST TOKENIZER OVERALL: {self.summary['best_tokenizer_overall']}")

        print("\n📈 AVERAGE COMPRESSION BY TOKENIZER:")
        for tokenizer, ratio in sorted(
            self.summary["avg_compression_by_tokenizer"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"  {tokenizer:15s}: {ratio:.3f} chars/token")

        print("\n🌍 BEST TOKENIZER BY LANGUAGE:")
        for lang, tokenizer in self.summary["best_tokenizer_by_language"].items():
            print(f"  {lang:12s}: {tokenizer}")

        print("\n" + "=" * 60)


def main():
    """Main CLI interface for benchmark visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize tokenizer benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_visualizer.py --results ./benchmark_results
  python benchmark_visualizer.py --results ./my_benchmark --summary-only
        """,
    )

    parser.add_argument(
        "--results",
        "-r",
        type=str,
        required=True,
        help="Path to benchmark results directory",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary, don't create visualizations",
    )

    args = parser.parse_args()

    try:
        # Initialize visualizer
        visualizer = BenchmarkVisualizer(args.results)

        # Print summary
        visualizer.print_summary()

        # Create visualizations unless summary-only
        if not args.summary_only:
            visualizer.create_all_visualizations()

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
