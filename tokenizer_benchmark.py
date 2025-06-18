#!/usr/bin/env python3
"""
Tokenizer Benchmark System

Compare different tokenizers across multiple languages and domains.
Measures compression efficiency, vocabulary utilization, and other metrics.
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from config import load_config, parse_datasets_from_config
from tokenizers import Tokenizer


@dataclass
class TokenizerResult:
    """Results for a single tokenizer on a single sample."""

    tokenizer_name: str
    dataset_name: str
    language: str
    domain: str
    sample_id: int
    original_text: str
    original_chars: int
    token_count: int
    tokens_per_char: float
    compression_ratio: float  # chars/tokens
    encoding_time_ms: float
    vocab_size: int


@dataclass
class BenchmarkConfig:
    """Configuration for tokenizer benchmark."""

    samples_per_dataset: int = 1000
    max_text_length: int = 2048  # Limit very long texts
    output_dir: str = "./benchmark_results"
    include_token_details: bool = False  # Save actual tokens (large files)


class TokenizerBenchmark:
    """Main benchmark class for comparing tokenizers."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[TokenizerResult] = []
        self.tokenizers: Dict[str, Any] = {}

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def add_tokenizer(
        self, name: str, tokenizer_path_or_name: str, tokenizer_type: str = "auto"
    ):
        """Add a tokenizer to benchmark."""
        print(f"Loading tokenizer: {name}")

        try:
            if tokenizer_type == "auto":
                # HuggingFace tokenizer with trust_remote_code for models that need it
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name)
                except Exception as first_error:
                    # If it fails, try with trust_remote_code=True
                    if "trust_remote_code" in str(first_error):
                        print(f"  🔒 Retrying {name} with trust_remote_code=True...")
                        tokenizer = AutoTokenizer.from_pretrained(
                            tokenizer_path_or_name, trust_remote_code=True
                        )
                    else:
                        raise first_error
            elif tokenizer_type == "local":
                # Local tokenizer.json file
                tokenizer = Tokenizer.from_file(tokenizer_path_or_name)
            else:
                raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

            self.tokenizers[name] = {
                "tokenizer": tokenizer,
                "type": tokenizer_type,
                "vocab_size": len(tokenizer.get_vocab())
                if hasattr(tokenizer, "get_vocab")
                else getattr(tokenizer, "vocab_size", "unknown"),
            }
            print(
                f"✅ Loaded {name} (vocab size: {self.tokenizers[name]['vocab_size']})"
            )

        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")

    def load_test_data(
        self, datasets_config: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load test samples from configured datasets."""
        test_data = defaultdict(list)

        for dataset_cfg in datasets_config:
            print(
                f"Loading test data from {dataset_cfg['path']} ({dataset_cfg.get('name', 'default')})"
            )

            try:
                # Load dataset
                load_kwargs = {
                    "path": dataset_cfg["path"],
                    "split": dataset_cfg["split"],
                    "streaming": True,
                }

                # Handle different dataset types
                if dataset_cfg.get("name"):
                    if "fineweb" in dataset_cfg["path"]:
                        load_kwargs["name"] = dataset_cfg["name"]
                    elif (
                        "stack" in dataset_cfg["path"]
                        or "starcoder" in dataset_cfg["path"]
                    ):
                        load_kwargs["data_dir"] = dataset_cfg["name"]

                dataset = load_dataset(**load_kwargs)

                # Sample data
                samples = []
                count = 0
                for sample in dataset:
                    if count >= self.config.samples_per_dataset:
                        break

                    # Get text from appropriate column
                    text = (
                        sample.get("text", "")
                        or sample.get("content", "")
                        or sample.get("code", "")
                    )

                    if text and len(text.strip()) > 10:
                        # Truncate very long texts
                        if len(text) > self.config.max_text_length:
                            text = text[: self.config.max_text_length]

                        samples.append(
                            {
                                "text": text,
                                "dataset_name": dataset_cfg["name"],
                                "language": self._infer_language(dataset_cfg),
                                "domain": self._infer_domain(dataset_cfg),
                                "sample_id": count,
                            }
                        )
                        count += 1

                test_data[dataset_cfg["name"]] = samples
                print(f"✅ Loaded {len(samples)} samples from {dataset_cfg['name']}")

            except Exception as e:
                print(f"❌ Failed to load {dataset_cfg['path']}: {e}")

        return dict(test_data)

    def _infer_language(self, dataset_cfg: Dict[str, Any]) -> str:
        """Infer language from dataset configuration."""
        name = dataset_cfg.get("name", "")

        # Language mappings
        lang_map = {
            "sample-10BT": "English",
            "rus_Cyrl": "Russian",
            "cmn_Hani": "Chinese",
            "deu_Latn": "German",
            "jpn_Jpan": "Japanese",
            "spa_Latn": "Spanish",
            "fra_Latn": "French",
            "python": "Python",
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "java": "Java",
        }

        return lang_map.get(name, name.title())

    def _infer_domain(self, dataset_cfg: Dict[str, Any]) -> str:
        """Infer domain from dataset configuration."""
        path = dataset_cfg.get("path", "")
        name = dataset_cfg.get("name", "")

        if "fineweb" in path:
            return "Web"
        elif "stack" in path or "starcoder" in path:
            return "Code"
        else:
            return "Other"

    def run_benchmark(self, test_data: Dict[str, List[Dict[str, Any]]]):
        """Run benchmark on all tokenizers and test data."""
        total_tests = len(self.tokenizers) * sum(
            len(samples) for samples in test_data.values()
        )
        current_test = 0

        print(
            f"🚀 Starting benchmark: {len(self.tokenizers)} tokenizers × {sum(len(s) for s in test_data.values())} samples = {total_tests} tests"
        )

        for tokenizer_name, tokenizer_info in self.tokenizers.items():
            print(f"\n📊 Testing tokenizer: {tokenizer_name}")
            tokenizer = tokenizer_info["tokenizer"]

            for dataset_name, samples in test_data.items():
                for sample in samples:
                    current_test += 1
                    if current_test % 100 == 0:
                        print(
                            f"  Progress: {current_test}/{total_tests} ({current_test / total_tests * 100:.1f}%)"
                        )

                    # Benchmark this sample
                    result = self._benchmark_sample(
                        tokenizer_name=tokenizer_name,
                        tokenizer=tokenizer,
                        vocab_size=tokenizer_info["vocab_size"],
                        sample=sample,
                    )

                    if result:
                        self.results.append(result)

        print(f"✅ Benchmark complete! {len(self.results)} results collected.")

    def _benchmark_sample(
        self,
        tokenizer_name: str,
        tokenizer: Any,
        vocab_size: int,
        sample: Dict[str, Any],
    ) -> Optional[TokenizerResult]:
        """Benchmark a single sample with a single tokenizer."""
        try:
            text = sample["text"]

            # Time the encoding
            start_time = time.time()

            if hasattr(tokenizer, "encode"):
                # HuggingFace tokenizer
                tokens = tokenizer.encode(text)
                token_count = len(tokens)
            elif hasattr(tokenizer, "encode"):
                # tokenizers library
                encoding = tokenizer.encode(text)
                token_count = len(encoding.tokens)
            else:
                return None

            encoding_time = (time.time() - start_time) * 1000  # Convert to ms

            # Calculate metrics
            char_count = len(text)
            tokens_per_char = token_count / char_count if char_count > 0 else 0
            compression_ratio = char_count / token_count if token_count > 0 else 0

            return TokenizerResult(
                tokenizer_name=tokenizer_name,
                dataset_name=sample["dataset_name"],
                language=sample["language"],
                domain=sample["domain"],
                sample_id=sample["sample_id"],
                original_text=text[:100] + "..."
                if len(text) > 100
                else text,  # Truncate for storage
                original_chars=char_count,
                token_count=token_count,
                tokens_per_char=tokens_per_char,
                compression_ratio=compression_ratio,
                encoding_time_ms=encoding_time,
                vocab_size=vocab_size,
            )

        except Exception as e:
            print(
                f"❌ Error benchmarking {tokenizer_name} on {sample['dataset_name']}: {e}"
            )
            return None

    def save_results(self):
        """Save benchmark results to files."""
        # Convert results to DataFrame
        df = pd.DataFrame([asdict(result) for result in self.results])

        # Save detailed results
        csv_path = Path(self.config.output_dir) / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"📊 Detailed results saved to: {csv_path}")

        # Save summary statistics
        summary = self._generate_summary(df)
        summary_path = Path(self.config.output_dir) / "benchmark_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📈 Summary saved to: {summary_path}")

        # Save aggregated results by language/domain
        agg_path = Path(self.config.output_dir) / "benchmark_aggregated.csv"
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


def main():
    """Main CLI interface for tokenizer benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark tokenizers across multiple languages and domains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tokenizer_benchmark.py --config configs/simple_500k.yaml --samples 100
  python tokenizer_benchmark.py --config configs/simple_50m.yaml --samples 500 --output ./my_benchmark
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to dataset configuration file",
    )
    parser.add_argument(
        "--samples", type=int, default=100, help="Samples per dataset (default: 100)"
    )
    parser.add_argument(
        "--output", type=str, default="./benchmark_results", help="Output directory"
    )
    parser.add_argument(
        "--max-length", type=int, default=2048, help="Maximum text length"
    )

    args = parser.parse_args()

    # Load dataset configuration
    cfg = load_config(args.config)
    datasets = parse_datasets_from_config(cfg)

    # Create benchmark config
    benchmark_config = BenchmarkConfig(
        samples_per_dataset=args.samples,
        max_text_length=args.max_length,
        output_dir=args.output,
    )

    # Initialize benchmark
    benchmark = TokenizerBenchmark(benchmark_config)

    # Add tokenizers to test
    print("🔧 Adding tokenizers to benchmark...")

    # Popular legacy tokenizers
    benchmark.add_tokenizer("GPT-2", "gpt2")
    benchmark.add_tokenizer("GPT-4", "Xenova/gpt-4")
    benchmark.add_tokenizer("CodeT5", "Salesforce/codet5-base")
    benchmark.add_tokenizer("StarCoder", "bigcode/starcoder")

    # Modern state-of-the-art tokenizers
    benchmark.add_tokenizer("LLaMA-3.3", "meta-llama/Llama-3.3-70B-Instruct")

    # Try Mixtral with sentencepiece handling
    try:
        import sentencepiece

        benchmark.add_tokenizer("Mixtral", "mistralai/Mixtral-8x22B-Instruct-v0.1")
    except ImportError:
        print(
            "⚠️  Skipping Mixtral: sentencepiece not installed. Run: pip install sentencepiece"
        )
    except Exception as e:
        print(f"⚠️  Skipping Mixtral: {e}")

    benchmark.add_tokenizer("Gemma", "google/gemma-7b")
    benchmark.add_tokenizer("Qwen2", "Qwen/Qwen2-72B")
    benchmark.add_tokenizer("DeepSeek-V3", "deepseek-ai/DeepSeek-V3")
    benchmark.add_tokenizer("Yi", "01-ai/Yi-34B")
    benchmark.add_tokenizer("Phi-3", "microsoft/phi-3-small-128k-instruct")
    benchmark.add_tokenizer("BLOOM", "bigscience/bloom")

    # Automatically discover and add all trained tokenizers
    print("🔍 Discovering trained tokenizers...")
    tokenizers_dir = Path("./tokenizers")
    if tokenizers_dir.exists():
        for tokenizer_path in tokenizers_dir.rglob("tokenizer.json"):
            # Extract a meaningful name from the path
            # e.g., "./tokenizers/train-simple-500k/tokenizer.json" -> "Custom-Simple-500K"
            relative_path = tokenizer_path.relative_to(tokenizers_dir)
            folder_name = (
                relative_path.parent.name
                if relative_path.parent.name != "."
                else "Custom"
            )

            # Clean up the name for display
            display_name = folder_name.replace("train-", "").replace("-", "-").title()
            if not display_name.startswith("Custom"):
                display_name = f"Custom-{display_name}"

            benchmark.add_tokenizer(display_name, str(tokenizer_path), "local")
    else:
        print("📁 No ./tokenizers directory found - skipping custom tokenizers")

    # Generate test datasets config
    from dataset_utils import generate_datasets_config

    datasets_config = generate_datasets_config(
        datasets=datasets,
        total_samples=args.samples * len(datasets),
        temperature=0.3,
        min_samples_per_lang=args.samples,
        max_samples_per_lang=args.samples,
    )

    # Load test data
    print("📚 Loading test data...")
    test_data = benchmark.load_test_data(datasets_config)

    # Run benchmark
    benchmark.run_benchmark(test_data)

    # Save results
    benchmark.save_results()

    print(f"🎉 Benchmark complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
