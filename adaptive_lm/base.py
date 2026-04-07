"""Base abstractions shared by all optimizers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
np.random.seed(42)


class BaseOptimizer(ABC):
    """Abstract base class for second-order optimization methods.

    Each optimizer stores a lightweight iteration history containing the
    objective value, gradient norm, and elapsed wall-clock time.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.history = {"loss": [], "grad_norm": [], "time": []}

    @abstractmethod
    def optimize(
        self,
        model: Any,
        dim: int,
        initial_theta: jnp.ndarray | np.ndarray | None = None,
        max_iter: int = 100,
        **kwargs: Any,
    ):
        """Run the optimization method and return the final iterate."""

    def reset_history(self) -> None:
        """Clear the stored optimization history."""
        self.history = {"loss": [], "grad_norm": [], "time": []}

    def _record(self, theta: jnp.ndarray, model: Any, start_time: float) -> None:
        """Store diagnostics for the current iterate."""
        loss_val = model.loss(theta)
        grad_val = model.gradient(theta)
        self.history["loss"].append(loss_val)
        self.history["grad_norm"].append(float(jnp.linalg.norm(grad_val)))
        self.history["time"].append(time.time() - start_time)
