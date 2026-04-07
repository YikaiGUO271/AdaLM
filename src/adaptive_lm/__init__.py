"""Adaptive Levenberg-Marquardt related optimization package."""

from .base import BaseOptimizer
from .models import (
    HighDimRosenbrock,
    LogSumExpFunction,
    MultivariateTMLE,
    PolytopeFeasibility,
    PowellSingularFunction,
    WorstInstancesFunction,
    ZakharovFunction,
)
from .optimizers import AdaN, Algorithm1, ARC, CR, CubicMM, ECME, SuperUniversalNewton

__version__ = "0.1.0"

__all__ = [
    "BaseOptimizer",
    "HighDimRosenbrock",
    "LogSumExpFunction",
    "MultivariateTMLE",
    "PolytopeFeasibility",
    "PowellSingularFunction",
    "WorstInstancesFunction",
    "ZakharovFunction",
    "AdaN",
    "Algorithm1",
    "ARC",
    "CR",
    "CubicMM",
    "ECME",
    "SuperUniversalNewton",
]
