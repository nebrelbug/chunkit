#!/usr/bin/env python3
"""
Improved multilingual tokenizer training with streaming support.
Unicode-friendly BPE with byte fallback, proper normalization, and multiple datasets.
"""

import os
from pathlib import Path
from typing import Iterator

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from datasets import interleave_datasets, load_dataset
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)

from dataset_config import (
    DATASETS_CONFIG,
    STREAMING_ENABLED,
    TOTAL_SAMPLES,
)


def train_tokenizer():
    """Train a multilingual BPE tokenizer with proper normalization and streaming."""

    print("Loading datasets with streaming...")

    # Load datasets from configuration
    streaming_datasets = []
    for config in DATASETS_CONFIG:
        print(f"Loading {config['path']} ({config['name']}) - bias: {config['bias']}")
        try:
            # Load dataset with proper name handling
            load_kwargs = {
                "path": config["path"],
                "split": config["split"],
                "streaming": STREAMING_ENABLED,
            }
            if config.get("name"):
                load_kwargs["name"] = config["name"]

            dataset = load_dataset(**load_kwargs)
            # Take specified number of samples
            dataset = dataset.take(config["samples"])
            streaming_datasets.append(dataset)
        except Exception as e:
            print(f"Warning: Could not load {config['path']}: {e}")
            continue

    # Fallback to just first dataset if others fail
    if not streaming_datasets:
        print("Falling back to single dataset...")
        config = DATASETS_CONFIG[0]
        load_kwargs = {
            "path": config["path"],
            "split": config["split"],
            "streaming": STREAMING_ENABLED,
        }
        if config.get("name"):
            load_kwargs["name"] = config["name"]

        dataset = load_dataset(**load_kwargs).take(TOTAL_SAMPLES)
        streaming_datasets = [dataset]

    print(f"Successfully loaded {len(streaming_datasets)} datasets")

    # Use HuggingFace's native interleaving
    print("Interleaving datasets with native HuggingFace support...")
    if len(streaming_datasets) > 1:
        # Calculate interleaving probabilities from dataset config
        interleave_probabilities = [
            config["percent"] for config in DATASETS_CONFIG[: len(streaming_datasets)]
        ]
        interleaved_dataset = interleave_datasets(
            streaming_datasets, probabilities=interleave_probabilities, seed=42
        )
    else:
        interleaved_dataset = streaming_datasets[0]
        interleave_probabilities = [1.0]

    print(f"Interleaved dataset ready with probabilities: {interleave_probabilities}")

    # BPE tokenizer with byte fallback
    print("Initializing tokenizer with byte fallback...")
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # The decoder must be the corresponding ByteLevel decoder to ensure
    # the process is perfectly reversible.
    tokenizer.decoder = decoders.ByteLevel()

    # Conservative normalization for multilingual support
    print("Setting up conservative normalization...")
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Strip(),  # Remove leading/trailing whitespace only
            normalizers.NFC(),  # Canonical composition (safest Unicode normalization)
        ]
    )

    # Special tokens for various tasks (256 total)
    special_tokens = [
        # Core tokens
        "<|startoftext|>",  # Start of text/document
        "<|endoftext|>",  # End of text/document
        "<|pad|>",  # Padding token
        "<|mask|>",  # Masked language modeling
        "<|sep|>",  # Separator for multi-sequence tasks
        "<|user|>",  # User turn (for chat/instruction tuning)
        "<|assistant|>",  # Assistant turn (for chat/instruction tuning)
        "<|system|>",  # System prompt (for chat/instruction tuning)
    ]

    # Add 248 reserved special tokens (for future use)
    special_tokens.extend([f"<|reserved_special_token_{i}|>" for i in range(248)])

    print(f"Total special tokens: {len(special_tokens)}")

    # Note: Post-processor will be set up after training when we have actual token IDs

    # BPE trainer with large vocab for multilingual support
    # Total vocab: 128,000 = 256 special + 256 byte fallbacks + 127,488 trained BPE
    trainer = trainers.BpeTrainer(
        vocab_size=128000,  # Large vocab for comprehensive multilingual coverage
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,  # Only keep tokens that appear at least twice
    )

    print(f"Training vocab size: {128000}")
    print(f"Special tokens: {len(special_tokens)}")
    print(f"Byte alphabet: {len(pre_tokenizers.ByteLevel.alphabet())}")
    print(
        f"Trained BPE tokens: {128000 - len(special_tokens) - len(pre_tokenizers.ByteLevel.alphabet())}"
    )

    print("Training tokenizer on multilingual data...")
    print("This may take several minutes depending on dataset size...")

    def multilingual_text_iterator() -> Iterator[str]:
        """Iterate through the interleaved dataset."""
        count = 0
        for sample in interleaved_dataset:
            text = sample.get("text", "")
            if text and len(text.strip()) > 10:  # Filter very short texts
                count += 1
                if count % 5000 == 0:
                    print(f"  Processed {count:,} samples...")
                yield text

    # Train the tokenizer
    print("Starting BPE training...")
    tokenizer.train_from_iterator(multilingual_text_iterator(), trainer)
    print("BPE training completed!")

    # Set up post-processor with correct token IDs after training
    print("Setting up post-processor with trained token IDs...")
    vocab = tokenizer.get_vocab()
    try:
        start_token_id = vocab["<|startoftext|>"]
        end_token_id = vocab["<|endoftext|>"]

        tokenizer.post_processor = processors.TemplateProcessing(
            single="<|startoftext|> $A <|endoftext|>",
            special_tokens=[
                ("<|startoftext|>", start_token_id),
                ("<|endoftext|>", end_token_id),
            ],
        )
        print("Post-processor configured successfully")
    except KeyError as e:
        print(f"Warning: Could not set up post-processor, missing token: {e}")
        print("Tokenizer will work but won't automatically add special tokens")

    # Save tokenizer
    output_dir = Path("./train-500k")
    output_dir.mkdir(exist_ok=True)

    tokenizer_file = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))

    # Also save a readable vocab file for inspection
    vocab_file = output_dir / "vocab.txt"
    vocab = tokenizer.get_vocab()
    with open(vocab_file, "w", encoding="utf-8") as f:
        # Sort by token ID
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        for token, token_id in sorted_vocab:
            f.write(f"{token_id}\t{token}\n")

    print(f"Tokenizer saved to {tokenizer_file}")
    print(f"Vocabulary saved to {vocab_file}")
    print(f"Final vocabulary size: {tokenizer.get_vocab_size()}")
    print("Training complete!")

    # Test the tokenizer
    print("\n--- Tokenizer Test ---")
    test_texts = [
        "Hello, how are you today?",
        "Bonjour, comment allez-vous?",
        "Hola, ¿cómo estás?",
        "Hallo, wie geht es dir?",
    ]

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"Original: {text}")
        print(f"Tokens: {encoded.tokens}")
        print(f"Decoded: {decoded}")
        print(f"Round-trip match: {text == decoded}")
        print()


if __name__ == "__main__":
    train_tokenizer()
