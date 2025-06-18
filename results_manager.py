#!/usr/bin/env python3
"""
Results Manager - Handles saving and loading benchmark results.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class ResultsManager:
    """Manages benchmark results storage and retrieval."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.benchmark_config = config.get("benchmark", {})
        self.output_dir = None

    def save_results(self, results: List[Dict[str, Any]]) -> str:
        """Save benchmark results to files and return output directory."""
        # Create timestamped output directory
        base_dir = Path(self.benchmark_config.get("output_dir", "./benchmarks"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = base_dir / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to: {self.output_dir}")

        # Save detailed results as JSON
        results_file = self.output_dir / "detailed_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Create and save summary DataFrame
        df = pd.DataFrame(results)
        csv_file = self.output_dir / "results_summary.csv"
        df.to_csv(csv_file, index=False)

        # Generate and save summary stats
        summary = self._generate_summary(df)
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved: {len(results)} records")
        return str(self.output_dir)

    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics from results DataFrame."""
        # Filter out failed results
        successful_df = df[df["chars_per_token"] > 0]

        if successful_df.empty:
            return {"error": "No successful benchmark results"}

        # Overall statistics
        summary = {
            "total_tests": len(df),
            "successful_tests": len(successful_df),
            "failed_tests": len(df) - len(successful_df),
            "timestamp": datetime.now().isoformat(),
        }

        # Best performers by metric
        best_compression = successful_df.loc[successful_df["chars_per_token"].idxmax()]
        best_speed = successful_df.loc[successful_df["tokens_per_second"].idxmax()]

        summary["best_compression"] = {
            "tokenizer": best_compression["tokenizer"],
            "dataset": best_compression["dataset"],
            "chars_per_token": float(best_compression["chars_per_token"]),
        }

        summary["best_speed"] = {
            "tokenizer": best_speed["tokenizer"],
            "dataset": best_speed["dataset"],
            "tokens_per_second": float(best_speed["tokens_per_second"]),
        }

        # Tokenizer rankings
        tokenizer_stats = (
            successful_df.groupby("tokenizer")
            .agg(
                {
                    "chars_per_token": ["mean", "std"],
                    "tokens_per_second": ["mean", "std"],
                }
            )
            .round(3)
        )

        # Flatten column names
        tokenizer_stats.columns = [
            "_".join(col).strip() for col in tokenizer_stats.columns.values
        ]

        summary["tokenizer_rankings"] = tokenizer_stats.to_dict("index")

        # Dataset group performance
        if "group" in successful_df.columns:
            group_stats = (
                successful_df.groupby("group")["chars_per_token"]
                .agg(["mean", "std", "count"])
                .round(3)
            )
            summary["group_performance"] = group_stats.to_dict("index")

        return summary

    def load_results(self, results_dir: str) -> Dict[str, Any]:
        """Load results from a previous benchmark run."""
        results_path = Path(results_dir)

        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        # Load detailed results
        results_file = results_path / "detailed_results.json"
        with open(results_file, "r") as f:
            detailed_results = json.load(f)

        # Load summary
        summary_file = results_path / "summary.json"
        with open(summary_file, "r") as f:
            summary = json.load(f)

        return {"results": detailed_results, "summary": summary}

    def get_results_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis."""
        return pd.DataFrame(results)
