#!/usr/bin/env python3
"""
Results Manager

Handles saving, loading, and summarizing benchmark results.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


class ResultsManager:
    """Manages benchmark results storage and analysis."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_results(self, df: pd.DataFrame) -> None:
        """Save benchmark results to files."""
        print("💾 Saving benchmark results...")

        # Save detailed results
        self._save_detailed_results(df)

        # Save summary statistics
        self._save_summary(df)

        # Save aggregated results
        self._save_aggregated_results(df)

        print(f"✅ All results saved to: {self.output_dir}")

    def _save_detailed_results(self, df: pd.DataFrame) -> None:
        """Save detailed CSV results."""
        csv_path = self.output_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"📊 Detailed results saved to: {csv_path}")

    def _save_summary(self, df: pd.DataFrame) -> None:
        """Save summary statistics as JSON."""
        summary = self._generate_summary(df)
        summary_path = self.output_dir / "benchmark_summary.json"

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📈 Summary saved to: {summary_path}")

    def _save_aggregated_results(self, df: pd.DataFrame) -> None:
        """Save aggregated results by tokenizer/language/domain."""
        agg_path = self.output_dir / "benchmark_aggregated.csv"

        aggregated = (
            df.groupby(["tokenizer_name", "language", "domain"])
            .agg(
                {
                    "token_count": ["mean", "std"],
                    "compression_ratio": ["mean", "std"],
                    "tokens_per_char": ["mean", "std"],
                    "encoding_time_ms": ["mean", "std"],
                }
            )
            .round(4)
        )

        aggregated.to_csv(agg_path)
        print(f"📋 Aggregated results saved to: {agg_path}")

    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "total_tests": len(df),
            "tokenizers": df["tokenizer_name"].unique().tolist(),
            "languages": df["language"].unique().tolist(),
            "domains": df["domain"].unique().tolist(),
            "avg_compression_by_tokenizer": df.groupby("tokenizer_name")[
                "compression_ratio"
            ]
            .mean()
            .to_dict(),
            "avg_compression_by_language": df.groupby("language")["compression_ratio"]
            .mean()
            .to_dict(),
            "avg_compression_by_domain": df.groupby("domain")["compression_ratio"]
            .mean()
            .to_dict(),
            "best_tokenizer_overall": df.groupby("tokenizer_name")["compression_ratio"]
            .mean()
            .idxmax(),
            "best_tokenizer_by_language": df.groupby(["language", "tokenizer_name"])[
                "compression_ratio"
            ]
            .mean()
            .groupby("language")
            .idxmax()
            .to_dict(),
        }
        return summary

    def load_results(self) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Load existing benchmark results."""
        # Load detailed results
        csv_path = self.output_dir / "benchmark_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Results file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Load summary
        summary_path = self.output_dir / "benchmark_summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary = json.load(f)
        else:
            summary = self._generate_summary(df)

        return df, summary

    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Print benchmark summary to console."""
        print("\n" + "=" * 60)
        print("📊 TOKENIZER BENCHMARK SUMMARY")
        print("=" * 60)

        print(f"Total tests run: {summary['total_tests']:,}")
        print(f"Tokenizers tested: {len(summary['tokenizers'])}")
        print(f"Languages tested: {len(summary['languages'])}")

        print(f"\n🏆 BEST TOKENIZER OVERALL: {summary['best_tokenizer_overall']}")

        print("\n📈 AVERAGE COMPRESSION BY TOKENIZER:")
        for tokenizer, ratio in sorted(
            summary["avg_compression_by_tokenizer"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"  {tokenizer:15s}: {ratio:.3f} chars/token")

        print("\n🌍 BEST TOKENIZER BY LANGUAGE:")
        for lang, tokenizer in summary["best_tokenizer_by_language"].items():
            print(f"  {lang:12s}: {tokenizer}")

        print("\n" + "=" * 60)
