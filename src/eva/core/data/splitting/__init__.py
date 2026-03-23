"""Dataset splitting API."""

from eva.core.data.splitting.random import random_split
from eva.core.data.splitting.stratified import stratified_split

__all__ = ["random_split", "stratified_split"]
