"""Baseline unsupervised classification tasks for comparison against RERC."""

from .BaselineTask import GMMTask, NearestCentroidTask

__all__ = [
    "NearestCentroidTask",
    "GMMTask",
]
