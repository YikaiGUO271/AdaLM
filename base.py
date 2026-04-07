import time
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import numpy as np

# 启用JAX的64位精度
jax.config.update("jax_enable_x64", True)
np.random.seed(42)


class BaseOptimizer(ABC):
    """优化器基类"""
    def __init__(self, name):
        self.name = name
        self.history = {'loss': [], 'grad_norm': [], 'time': []}
    
    @abstractmethod
    def optimize(self, model, dim, initial_theta=None, max_iter=100, **kwargs):
        pass
    
    def reset_history(self):
        self.history = {'loss': [], 'grad_norm': [], 'time': []}
    
    def _record(self, theta, model, start_time):
        loss_val = model.loss(theta)
        grad_val = model.gradient(theta)
        self.history['loss'].append(loss_val)
        self.history['grad_norm'].append(jnp.linalg.norm(grad_val).item())
        self.history['time'].append(time.time() - start_time)
