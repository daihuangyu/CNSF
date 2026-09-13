"""Neural recursive Track-MT3 experiments.

This package is intentionally independent from :mod:`track_mt3.models`.  The
v9 sliding-window tracker and the v16 encoded-cache ablation must remain
loadable from their original checkpoints while the recursive design evolves.
"""

from .config import V17AConfig, V17BConfig
from .model import (
    EndToEndRecursiveTracker,
    JointAssociationTracker,
    OracleRecursiveTracker,
    SeparatedLifecycleTracker,
)
from .state import RecursiveTrackState

__all__ = [
    "EndToEndRecursiveTracker",
    "JointAssociationTracker",
    "OracleRecursiveTracker",
    "RecursiveTrackState",
    "SeparatedLifecycleTracker",
    "V17AConfig",
    "V17BConfig",
]
