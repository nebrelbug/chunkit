#!/usr/bin/env python3
"""
Dataset configuration for multilingual tokenizer training.
Defines datasets, splits, and sampling parameters.
"""

# Constants - modify these to adjust training
TOTAL_SAMPLES = 55000  # Total number of samples across all datasets
STREAMING_ENABLED = True

# Dataset configuration for tokenizer training
DATASETS_CONFIG = [
    {
        "path": "HuggingFaceFW/fineweb",
        "name": "sample-10BT",  # English
        "split": "train",
        "bias": 3,  # English gets higher weight
    },
    {
        "path": "HuggingFaceFW/fineweb-2",
        "name": "rus_Cyrl",  # Russian Cyrillic
        "split": "train",
        "bias": 2,  # Standard weight
    },
    {
        "path": "HuggingFaceFW/fineweb-2",
        "name": "cmn_Hani",  # Mandarin Chinese
        "split": "train",
        "bias": 2,  # Standard weight
    },
    {
        "path": "HuggingFaceFW/fineweb-2",
        "name": "deu_Latn",  # German
        "split": "train",
        "bias": 2,  # Standard weight
    },
    {
        "path": "HuggingFaceFW/fineweb-2",
        "name": "jpn_Jpan",  # Japanese
        "split": "train",
        "bias": 2,  # Standard weight
    },
]

# Calculate percentages from bias weights
total_bias = sum(config["bias"] for config in DATASETS_CONFIG)
print(f"Dataset bias configuration (total bias weight: {total_bias}):")

for config in DATASETS_CONFIG:
    config["percent"] = config["bias"] / total_bias
    config["samples"] = int(config["percent"] * TOTAL_SAMPLES)
    print(
        f"  {config['name']}: bias={config['bias']} -> {config['percent']:.1%} ({config['samples']:,} samples)"
    )

# Interleaving probabilities (same as calculated percentages)
INTERLEAVE_PROBABILITIES = [config["percent"] for config in DATASETS_CONFIG]
