#!/usr/bin/env python3
"""
Compare different tokenizer approaches for multilingual content.
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def test_approaches():
    """Compare Split approach vs add_prefix_space approach."""

    # Multilingual test cases
    test_texts = [
        "Hello world!",  # English
        "Hola mundo!",  # Spanish
        "你好世界！",  # Chinese (no spaces)
        "こんにちは世界！",  # Japanese (no spaces)
        "Привет мир!",  # Russian
        "مرحبا بالعالم!",  # Arabic (RTL)
        "  Leading space test",
        "Multiple   spaces   test",
        "Tab\ttest\there",
        "Newline\ntest\nhere",
    ]

    print("=== Approach 1: Split Pattern (Current) ===")

    # Our current approach
    tokenizer1 = Tokenizer(models.BPE())
    tokenizer1.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(pattern=r"(\s+)", behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tokenizer1.decoder = decoders.ByteLevel()

    trainer1 = trainers.BpeTrainer(
        vocab_size=1000,
        special_tokens=["<|startoftext|>", "<|endoftext|>"],
        show_progress=False,
    )

    tokenizer1.train_from_iterator(test_texts, trainer1)

    print("Results:")
    for text in test_texts:
        encoded = tokenizer1.encode(text)
        decoded = tokenizer1.decode(encoded.ids)
        match = "✓" if text == decoded else "✗"
        print(f"{match} {repr(text)} -> {len(encoded.tokens)} tokens")
        if text != decoded:
            print(f"   Decoded: {repr(decoded)}")

    print("\n=== Approach 2: add_prefix_space=True + Strip Decoder ===")

    # SOTA approach with post-processing
    tokenizer2 = Tokenizer(models.BPE())
    tokenizer2.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer2.decoder = decoders.Sequence(
        [
            decoders.ByteLevel(),
            decoders.Strip(content=" ", left=True, right=False),  # Remove leading space
        ]
    )

    trainer2 = trainers.BpeTrainer(
        vocab_size=1000,
        special_tokens=["<|startoftext|>", "<|endoftext|>"],
        show_progress=False,
    )

    tokenizer2.train_from_iterator(test_texts, trainer2)

    print("Results:")
    for text in test_texts:
        encoded = tokenizer2.encode(text)
        decoded = tokenizer2.decode(encoded.ids)
        match = "✓" if text == decoded else "✗"
        print(f"{match} {repr(text)} -> {len(encoded.tokens)} tokens")
        if text != decoded:
            print(f"   Decoded: {repr(decoded)}")

    print("\n=== Token Consistency Analysis ===")

    # Test token consistency - key advantage of add_prefix_space=True
    consistency_tests = [
        "world",
        " world",  # With leading space
        "The world is big",
        "Hello world",
    ]

    print("Approach 1 (Split) - 'world' tokenization:")
    for text in consistency_tests:
        encoded = tokenizer1.encode(text)
        world_tokens = [token for token in encoded.tokens if "world" in token.lower()]
        print(f"  {repr(text)} -> world tokens: {world_tokens}")

    print("\nApproach 2 (add_prefix_space) - 'world' tokenization:")
    for text in consistency_tests:
        encoded = tokenizer2.encode(text)
        world_tokens = [token for token in encoded.tokens if "world" in token.lower()]
        print(f"  {repr(text)} -> world tokens: {world_tokens}")


def test_metaspace_approach():
    """Demonstrate the robust Metaspace approach."""

    test_texts = [
        "Hello world!",
        "Hola mundo!",
        "你好世界！",
        "こんにちは世界！",
        "Привет мир!",
        "مرحبا بالعالم!",
        "  Leading space test",  # The failing case
        "Multiple   spaces   test",
        "Tab\ttest\there",
        "Newline\ntest\nhere",
    ]

    print("\n=== Approach 3: Metaspace (Corrected) ===")

    tokenizer = Tokenizer(models.BPE())

    # The pre-tokenizer replaces spaces and adds a prefix if needed.
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="always")

    # CRITICAL FIX: The decoder must ALSO know the prepend_scheme to reverse it correctly.
    tokenizer.decoder = decoders.Metaspace(prepend_scheme="always")

    trainer = trainers.BpeTrainer(
        vocab_size=1000,
        special_tokens=["<|startoftext|>", "<|endoftext|>"],
        show_progress=False,
    )

    tokenizer.train_from_iterator(test_texts, trainer)

    print("Results:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        match = "✓" if text == decoded else "✗"
        print(f"{match} {repr(text)} -> {len(encoded.tokens)} tokens")
        if text != decoded:
            print(f"   Decoded: {repr(decoded)}")


if __name__ == "__main__":
    test_approaches()
    test_metaspace_approach()
