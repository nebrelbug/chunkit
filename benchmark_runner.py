#!/usr/bin/env python3
"""
Benchmark Runner

Core benchmarking logic for tokenizer performance evaluation.
"""

import logging
import time
from typing import Any, Dict, List

from datasets import load_dataset

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Handles the core benchmarking logic."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.benchmark_config = config.get("benchmark", {})

    def _calculate_group_samples(self) -> Dict[str, int]:
        """Calculate sample allocation for each dataset group."""
        total_samples = self.benchmark_config.get("total_samples", 1000)
        dataset_groups = self.config.get("dataset_groups", [])

        group_samples = {}
        allocated_samples = 0

        for group in dataset_groups:
            group_name = group["name"]
            allocation = group.get("sample_allocation", 0.0)
            samples = int(total_samples * allocation)
            group_samples[group_name] = samples
            allocated_samples += samples

        # Handle any remaining samples due to rounding
        remaining = total_samples - allocated_samples
        if remaining > 0 and dataset_groups:
            # Add remaining samples to the first group
            first_group = dataset_groups[0]["name"]
            group_samples[first_group] += remaining

        logger.info(f"Sample allocation: {group_samples}")
        return group_samples

    def _calculate_subset_samples(self, group_samples: int, num_subsets: int) -> int:
        """Calculate samples per subset within a group."""
        return max(1, group_samples // num_subsets)

    def load_datasets(self) -> Dict[str, Any]:
        """Load datasets based on group configuration."""
        dataset_groups = self.config.get("dataset_groups", [])
        group_samples = self._calculate_group_samples()

        datasets = {}

        for group in dataset_groups:
            group_name = group["name"]
            total_group_samples = group_samples.get(group_name, 0)

            # Count total subsets in this group
            total_subsets = sum(
                len(dataset.get("subsets", [])) for dataset in group.get("datasets", [])
            )

            if total_subsets == 0:
                logger.warning(f"No subsets found in group '{group_name}'")
                continue

            samples_per_subset = self._calculate_subset_samples(
                total_group_samples, total_subsets
            )

            logger.info(
                f"Group '{group_name}': {total_group_samples} samples across {total_subsets} subsets ({samples_per_subset} each)"
            )

            for dataset_config in group.get("datasets", []):
                path = dataset_config["path"]

                for subset_config in dataset_config.get("subsets", []):
                    subset_name = subset_config["name"]
                    text_column = subset_config.get(
                        "text_column", "text"
                    )  # Support custom columns

                    dataset_key = f"{group_name}:{subset_name}"

                    try:
                        # Load dataset with proper subset handling
                        if path == "bigcode/starcoderdata":
                            # Special handling for StarCoder
                            dataset = load_dataset(
                                path,
                                data_dir=subset_name,
                                streaming=True,
                                split="train",
                            )
                        else:
                            dataset = load_dataset(
                                path, subset_name, streaming=True, split="train"
                            )

                        datasets[dataset_key] = {
                            "dataset": dataset,
                            "samples": samples_per_subset,
                            "text_column": text_column,
                            "group": group_name,
                            "subset": subset_name,
                            "description": f"{group['description']} - {subset_name}",
                        }

                        logger.info(
                            f"Loaded dataset: {dataset_key} ({samples_per_subset} samples, column: '{text_column}')"
                        )

                    except Exception as e:
                        logger.error(
                            f"Failed to load dataset {path}:{subset_name}: {e}"
                        )
                        continue

        return datasets

    def run_benchmark(
        self, tokenizer_manager, datasets: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run benchmark across all tokenizers and datasets."""
        results = []
        tokenizers = tokenizer_manager.get_all_tokenizers()

        total_tests = len(tokenizers) * len(datasets)
        test_count = 0

        logger.info(
            f"Starting benchmark: {len(tokenizers)} tokenizers × {len(datasets)} datasets = {total_tests} tests"
        )

        for tokenizer_name, tokenizer in tokenizers.items():
            for dataset_key, dataset_info in datasets.items():
                test_count += 1
                logger.info(
                    f"[{test_count}/{total_tests}] Testing {tokenizer_name} on {dataset_key}"
                )

                try:
                    result = self._benchmark_single(
                        tokenizer_name, tokenizer, dataset_key, dataset_info
                    )
                    results.append(result)

                    logger.info(
                        f"Result: {result['chars_per_token']:.3f} chars/token, {result['tokens_per_second']:.1f} tok/s"
                    )

                except Exception as e:
                    logger.error(
                        f"Benchmark failed for {tokenizer_name} on {dataset_key}: {e}"
                    )
                    # Add failed result
                    results.append(
                        {
                            "tokenizer": tokenizer_name,
                            "dataset": dataset_key,
                            "group": dataset_info["group"],
                            "subset": dataset_info["subset"],
                            "chars_per_token": 0.0,
                            "tokens_per_second": 0.0,
                            "total_chars": 0,
                            "total_tokens": 0,
                            "samples_processed": 0,
                            "error": str(e),
                        }
                    )

        return results

    def _benchmark_single(
        self,
        tokenizer_name: str,
        tokenizer,
        dataset_key: str,
        dataset_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Benchmark a single tokenizer on a single dataset."""
        dataset = dataset_info["dataset"]
        samples_to_process = dataset_info["samples"]
        text_column = dataset_info["text_column"]
        max_length = self.benchmark_config.get("max_text_length", 2048)

        total_chars = 0
        total_tokens = 0
        samples_processed = 0

        start_time = time.time()

        try:
            for i, example in enumerate(dataset):
                if i >= samples_to_process:
                    break

                # Get text from the specified column
                text = example.get(text_column, "")
                if not text or not isinstance(text, str):
                    continue

                # Truncate if too long
                if len(text) > max_length:
                    text = text[:max_length]

                # Tokenize
                tokens = tokenizer.encode(text)

                total_chars += len(text)
                total_tokens += (
                    len(tokens.ids) if hasattr(tokens, "ids") else len(tokens)
                )
                samples_processed += 1

        except Exception as e:
            logger.error(f"Error processing dataset {dataset_key}: {e}")
            raise

        end_time = time.time()

        if total_tokens == 0:
            raise ValueError(f"No tokens generated for {dataset_key}")

        chars_per_token = total_chars / total_tokens
        tokens_per_second = total_tokens / (end_time - start_time)

        return {
            "tokenizer": tokenizer_name,
            "dataset": dataset_key,
            "group": dataset_info["group"],
            "subset": dataset_info["subset"],
            "chars_per_token": chars_per_token,
            "tokens_per_second": tokens_per_second,
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "samples_processed": samples_processed,
        }
