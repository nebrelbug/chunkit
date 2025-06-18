#!/usr/bin/env python3
"""
Tokenizer Manager - Handles loading and managing different types of tokenizers.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import DictConfig
from transformers import AutoTokenizer

from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class TokenizerManager:
    """Manages loading and storing tokenizers for benchmarking."""

    def __init__(self, config: DictConfig):
        self.tokenizers: Dict[str, Any] = {}
        self.config = config
        self._load_all_tokenizers()

    def _load_all_tokenizers(self) -> None:
        """Load all tokenizers specified in configuration."""
        logger.info("Loading tokenizers from configuration...")

        # Load external tokenizers
        if hasattr(self.config, "tokenizers"):
            for tokenizer_cfg in self.config.tokenizers:
                self._load_external_tokenizer(tokenizer_cfg)

        # Load custom tokenizers
        custom_config = getattr(self.config, "custom_tokenizers", {})
        if custom_config.get("enabled", True):
            self._load_custom_tokenizers(custom_config)

        logger.info(f"Loaded {len(self.tokenizers)} tokenizers total")

    def _load_external_tokenizer(self, tokenizer_cfg: DictConfig) -> None:
        """Load a single external tokenizer."""
        name = tokenizer_cfg.name
        model = tokenizer_cfg.model

        logger.info(f"Loading tokenizer: {name}")

        try:
            # Prepare loading arguments
            load_kwargs = {}
            if tokenizer_cfg.get("trust_remote_code", False):
                load_kwargs["trust_remote_code"] = True

            # Load tokenizer with retry logic
            try:
                tokenizer = AutoTokenizer.from_pretrained(model, **load_kwargs)
            except Exception as first_error:
                # Retry with trust_remote_code if needed
                if "trust_remote_code" in str(first_error) and not load_kwargs.get(
                    "trust_remote_code"
                ):
                    logger.info(f"Retrying {name} with trust_remote_code=True...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        model, trust_remote_code=True
                    )
                else:
                    raise first_error

            self.tokenizers[name] = tokenizer
            logger.info(f"Successfully loaded {name}")

        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

    def _load_custom_tokenizers(self, custom_cfg: DictConfig) -> None:
        """Load custom tokenizers from directory."""
        logger.info("Discovering custom tokenizers...")

        tokenizers_dir = Path(custom_cfg.get("directory", "./tokenizers"))
        name_pattern = custom_cfg.get("name_pattern", "Custom-{folder_name}")

        if not tokenizers_dir.exists():
            logger.info(
                f"No {tokenizers_dir} directory found - skipping custom tokenizers"
            )
            return

        # Search for tokenizer.json files
        for tokenizer_path in tokenizers_dir.rglob("tokenizer.json"):
            try:
                # Extract folder name for naming
                relative_path = tokenizer_path.relative_to(tokenizers_dir)
                folder_name = (
                    relative_path.parent.name
                    if relative_path.parent.name != "."
                    else "Custom"
                )

                # Generate display name
                display_name = self._format_tokenizer_name(folder_name, name_pattern)

                # Load tokenizer
                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                self.tokenizers[display_name] = tokenizer

                logger.info(f"Successfully loaded custom tokenizer: {display_name}")

            except Exception as e:
                logger.error(f"Failed to load custom tokenizer {tokenizer_path}: {e}")

    def _format_tokenizer_name(self, folder_name: str, pattern: str) -> str:
        """Format tokenizer name based on folder and pattern."""
        # Clean up folder name
        clean_name = folder_name.replace("train-", "").replace("-", "-").title()

        # Apply naming pattern
        if "{folder_name}" in pattern:
            return pattern.format(folder_name=clean_name)
        else:
            return f"Custom-{clean_name}"

    def get_all_tokenizers(self) -> Dict[str, Any]:
        """Get all loaded tokenizers."""
        return self.tokenizers

    def get_tokenizer_names(self) -> List[str]:
        """Get list of all tokenizer names."""
        return list(self.tokenizers.keys())

    def count(self) -> int:
        """Get number of loaded tokenizers."""
        return len(self.tokenizers)
