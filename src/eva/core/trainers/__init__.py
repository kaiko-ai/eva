"""Trainers API."""

from eva.core.trainers.functional import infer_model, run_evaluation_session
from eva.core.trainers.trainer import Trainer

__all__ = ["infer_model", "run_evaluation_session", "Trainer"]
