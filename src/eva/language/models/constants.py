"""Constants for language models."""

import os

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 32768))
"""Default maximum number of new tokens to generate for language models."""
