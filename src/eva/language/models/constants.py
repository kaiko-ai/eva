"""Constants for language models."""

import os

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 4096))
"""Default maximum number of new tokens to generate for language models."""
