"""Trainers API."""

from eva.trainers.functional import infer_model, run_evaluation_session
from eva.trainers.trainer import Trainer

__all__ = ["infer_model", "run_evaluation_session", "Trainer"]
