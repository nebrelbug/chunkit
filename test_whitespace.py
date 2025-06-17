#!/usr/bin/env python3
"""
Quick test of whitespace handling with different pre-tokenizer configurations.
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def test_whitespace_handling():
    """Test different pre-tokenizer configurations for whitespace handling."""

    # Test texts with various whitespace scenarios
    test_texts = [
        "Hello world",
        "  Leading spaces",
        "Trailing spaces  ",
        "  Both sides  ",
        "Multiple   spaces",
        "Tab\tseparated",
        "Newline\nseparated",
        "Mixed\t  whitespace\n  here",
        "Normal sentence with punctuation.",
        "",  # Empty string
        "   ",  # Only whitespace
    ]

    print(
        "=== Testing Split(preserve whitespace) + ByteLevel(add_prefix_space=False) ==="
    )

    # Create a simple tokenizer for testing
    tokenizer = Tokenizer(models.BPE())

    # Use Split with a pattern that captures whitespace
    # This regex splits on word boundaries but keeps the whitespace
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=r"(\s+)",  # Capture whitespace in groups
                behavior="isolated",  # Keep captured groups as separate tokens
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()

    # Create a minimal trainer just for testing
    trainer = trainers.BpeTrainer(
        vocab_size=1000,
        special_tokens=["<|startoftext|>", "<|endoftext|>"],
        show_progress=False,
    )

    # Train on our test texts
    tokenizer.train_from_iterator(test_texts, trainer)

    print("Training completed. Testing whitespace preservation:")
    print()

    for text in test_texts:
        try:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded.ids)
            match = "✓" if text == decoded else "✗"

            print(f"{match} Original: {repr(text)}")
            print(f"   Tokens: {encoded.tokens}")
            print(f"   Decoded: {repr(decoded)}")
            if text != decoded:
                print("   ⚠️  MISMATCH!")
                print(f"   Original bytes: {text.encode('utf-8')}")
                print(f"   Decoded bytes:  {decoded.encode('utf-8')}")
            print()
        except Exception as e:
            print(f"❌ Error with {repr(text)}: {e}")
            print()

    print("\n=== Testing ByteLevel(add_prefix_space=True) approach ===")

    # Test the original approach for comparison
    tokenizer2 = Tokenizer(models.BPE())
    tokenizer2.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer2.decoder = decoders.ByteLevel()

    trainer2 = trainers.BpeTrainer(
        vocab_size=1000,
        special_tokens=["<|startoftext|>", "<|endoftext|>"],
        show_progress=False,
    )

    tokenizer2.train_from_iterator(test_texts, trainer2)

    print("Testing ByteLevel with add_prefix_space=True:")
    print()

    for text in test_texts[:3]:  # Just test a few
        try:
            encoded = tokenizer2.encode(text)
            decoded = tokenizer2.decode(encoded.ids)
            match = "✓" if text == decoded else "✗"

            print(f"{match} Original: {repr(text)}")
            print(f"   Tokens: {encoded.tokens}")
            print(f"   Decoded: {repr(decoded)}")
            if text != decoded:
                print(f"   Leading space issue: {repr(decoded)} vs {repr(text)}")
            print()
        except Exception as e:
            print(f"❌ Error with {repr(text)}: {e}")
            print()


if __name__ == "__main__":
    test_whitespace_handling()
