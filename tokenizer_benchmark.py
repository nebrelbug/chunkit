#!/usr/bin/env python3
"""
Comprehensive Tokenizer Benchmark System

Evaluates tokenizer performance across multiple languages and programming domains.
Supports external tokenizers (GPT-2, GPT-4, LLaMA, Gemma, etc.) and custom tokenizers.

Usage:
    python tokenizer_benchmark.py --config configs/benchmark.yaml
    python tokenizer_benchmark.py --config configs/benchmark.yaml benchmark.total_samples=5000
"""

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from benchmark_runner import BenchmarkRunner
from benchmark_visualizer import BenchmarkVisualizer
from results_manager import ResultsManager
from tokenizer_manager import TokenizerManager


def setup_logging() -> None:
    """Configure logging for the benchmark system."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress verbose third-party logging
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def load_config(config_path: str, overrides: list = None) -> OmegaConf:
    """Load and validate configuration file."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load base configuration
    config = OmegaConf.load(config_path)

    # Apply CLI overrides
    if overrides:
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Invalid override format: {override}. Use key=value")
            key, value = override.split("=", 1)
            OmegaConf.set(config, key, value)

    return config


def print_config_summary(config: OmegaConf) -> None:
    """Print configuration summary."""
    print("🔧 Configuration Summary:")
    print(f"  Total samples: {config.benchmark.total_samples:,}")
    print(f"  Max text length: {config.benchmark.max_text_length:,}")
    print(f"  Output directory: {config.benchmark.output_dir}")

    # Count tokenizers (all listed tokenizers are enabled)
    external_count = len(config.tokenizers) if hasattr(config, "tokenizers") else 0
    custom_enabled = config.custom_tokenizers.get("enabled", True)

    print(f"  External tokenizers: {external_count}")
    print(f"  Custom tokenizers: {'Enabled' if custom_enabled else 'Disabled'}")

    # Count dataset groups
    group_count = len(config.dataset_groups)
    total_datasets = sum(
        len(dataset.get("subsets", []))
        for group in config.dataset_groups
        for dataset in group.get("datasets", [])
    )

    print(f"  Dataset groups: {group_count}")
    print(f"  Total datasets: {total_datasets}")
    print()


def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive tokenizer benchmark system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tokenizer_benchmark.py --config configs/benchmark.yaml
  python tokenizer_benchmark.py --config configs/benchmark.yaml benchmark.total_samples=5000
  python tokenizer_benchmark.py --config configs/benchmark.yaml custom_tokenizers.enabled=false

CLI overrides use dot notation:
  benchmark.total_samples=5000
  benchmark.output_dir=./custom-results
  custom_tokenizers.enabled=false
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "overrides",
        nargs="*",
        help="Configuration overrides in key=value format",
    )

    args = parser.parse_args()

    try:
        # Setup
        setup_logging()
        config = load_config(args.config, args.overrides)

        print("🚀 Starting Comprehensive Tokenizer Benchmark")
        print("=" * 60)

        if args.overrides:
            print(f"📝 CLI overrides: {args.overrides}")

        print_config_summary(config)

        # Initialize components
        print("🔧 Initializing benchmark components...")
        tokenizer_manager = TokenizerManager(config)
        benchmark_runner = BenchmarkRunner(config)
        results_manager = ResultsManager(config)

        if tokenizer_manager.count() == 0:
            print("❌ No tokenizers loaded. Check your configuration.")
            sys.exit(1)

        # Load datasets
        print("📊 Loading datasets...")
        datasets = benchmark_runner.load_datasets()

        if not datasets:
            print("❌ No datasets loaded. Check your configuration.")
            sys.exit(1)

        print(f"✅ Loaded {len(datasets)} datasets")

        # Run benchmark
        print("🏃 Running benchmark...")
        results = benchmark_runner.run_benchmark(tokenizer_manager, datasets)

        # Save results
        print("💾 Saving results...")
        output_dir = results_manager.save_results(results)

        # Generate visualizations
        print("📈 Generating visualizations...")
        visualizer = BenchmarkVisualizer(config)
        visualizer.create_all_visualizations(results, output_dir)

        print("✅ Benchmark completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
