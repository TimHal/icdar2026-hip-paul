"""Lightning task modules for training class-wise autoencoders."""

from .ClasswiseAETask import ClasswiseAETask
from .JITClasswiseAETask import JITClasswiseAETask
from .MAETask import MAETask
from .SelfSupervisedAETask import SelfSupervisedAETask

__all__ = [
    "ClasswiseAETask",
    "JITClasswiseAETask",
    "MAETask",
    "SelfSupervisedAETask",
]
