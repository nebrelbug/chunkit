#!/usr/bin/env python3
"""
Simplified Tokenizer Benchmark System

Compares tokenizer performance across language groups with one grouped bar chart per group.

Usage:
    python tokenizer_benchmark.py
"""

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from benchmark_runner import BenchmarkRunner
from benchmark_visualizer import BenchmarkVisualizer
from tokenizer_manager import TokenizerManager


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Simplified tokenizer benchmark system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python tokenizer_benchmark.py
        """,
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config_path = "configs/benchmark.yaml"
        logger.info(f"Loading configuration from {config_path}")
        config = OmegaConf.load(config_path)

        # Create nested output directory structure
        base_output_dir = config.benchmark.get("output_dir", "./benchmarks")
        # Use timestamp or config name for nested directory
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("benchmarks") / f"run_{timestamp}"

        logger.info(f"Results will be saved to: {output_dir}")

        # Print configuration summary
        logger.info("Configuration Summary:")
        logger.info(f"  Total samples: {config.benchmark.get('total_samples', 'N/A')}")
        logger.info(f"  Dataset groups: {len(config.get('dataset_groups', []))}")
        logger.info(
            f"  Enabled tokenizers: {sum(1 for t in config.get('tokenizers', []) if t.get('enabled', True))}"
        )
        logger.info(
            f"  Custom tokenizers: {'enabled' if config.get('custom_tokenizers', {}).get('enabled', False) else 'disabled'}"
        )

        # Initialize components
        logger.info("Initializing tokenizer manager...")
        tokenizer_manager = TokenizerManager(config)

        logger.info("Initializing benchmark runner...")
        benchmark_runner = BenchmarkRunner(config)

        # Load datasets
        logger.info("Loading datasets...")
        datasets = benchmark_runner.load_datasets()

        if not datasets:
            logger.error("No datasets loaded successfully!")
            return 1

        logger.info(f"Loaded {len(datasets)} datasets")

        # Load tokenizers
        tokenizers = tokenizer_manager.get_all_tokenizers()

        if not tokenizers:
            logger.error("No tokenizers loaded successfully!")
            return 1

        logger.info(f"Loaded {len(tokenizers)} tokenizers: {list(tokenizers.keys())}")

        # Run benchmark
        logger.info("Starting benchmark...")
        results = benchmark_runner.run_benchmark(tokenizer_manager, datasets)

        if not results:
            logger.error("No benchmark results generated!")
            return 1

        logger.info(f"Benchmark completed with {len(results)} results")

        # Create visualizations
        logger.info("Creating visualizations...")
        visualizer = BenchmarkVisualizer(str(output_dir))
        visualizer.create_all_visualizations(results)

        # Print summary
        successful_results = [r for r in results if "error" not in r]
        failed_results = [r for r in results if "error" in r]

        logger.info("✅ Benchmark completed successfully!")
        logger.info(
            f"📊 Results: {len(successful_results)} successful, {len(failed_results)} failed"
        )
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info("📋 Generated files:")

        # List the actual generated files
        if output_dir.exists():
            for file in output_dir.glob("*.png"):
                logger.info(f"  - {file.name}")
            for file in output_dir.glob("*.csv"):
                logger.info(f"  - {file.name}")

        if failed_results:
            logger.warning(
                f"⚠️  {len(failed_results)} tests failed - check logs for details"
            )

        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
