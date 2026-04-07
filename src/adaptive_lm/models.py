"""Benchmark objective functions and statistical estimation models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, hessian, jit


class HighDimRosenbrock:
    """High-dimensional Rosenbrock objective."""

    def __init__(self, dim: int = 200, a: float = 1.0, b: float = 100.0) -> None:
        self.dim = dim
        self.a = a
        self.b = b

        self._loss_jitted = jax.jit(self._loss)
        self._gradient_jitted = jax.jit(jax.grad(self._loss))
        self._hessian_jitted = jax.jit(jax.hessian(self._loss))

    def _loss(self, x: jnp.ndarray) -> jnp.ndarray:
        terms = self.b * (x[1:] - x[:-1] ** 2) ** 2 + (self.a - x[:-1]) ** 2
        return jnp.sum(terms)

    def loss(self, x: jnp.ndarray) -> float:
        return float(self._loss_jitted(x))

    def gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._gradient_jitted(x)

    def hessian(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._hessian_jitted(x)


class PolytopeFeasibility:
    """Penalty model for polytope feasibility problems."""

    def __init__(self, dim: int = 100, m: int = 1000, p: float = 2.0) -> None:
        self.dim = dim
        self.m = m
        self.p = p
        self.A = jnp.array(np.random.randn(m, dim))
        self.b = jnp.array(np.random.randn(m))

        self._loss_jitted = jax.jit(self._loss)
        self._gradient_jitted = jax.jit(jax.grad(self._loss))
        self._hessian_jitted = jax.jit(jax.hessian(self._loss))

    def _loss(self, x: jnp.ndarray) -> jnp.ndarray:
        residuals = self.A @ x - self.b
        positive_residuals = jnp.maximum(0, residuals)
        return jnp.sum(positive_residuals ** self.p)

    def loss(self, x: jnp.ndarray) -> float:
        return float(self._loss_jitted(x))

    def gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._gradient_jitted(x)

    def hessian(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._hessian_jitted(x)


class WorstInstancesFunction:
    """Nonsmooth worst-instance objective used in complexity studies."""

    def __init__(self, dim: int = 200, q: float = 3.0) -> None:
        self.dim = dim
        self.q = q

        self._loss_jitted = jax.jit(self._loss)
        self._gradient_jitted = jax.jit(jax.grad(self._loss))
        self._hessian_jitted = jax.jit(jax.hessian(self._loss))

    def _loss(self, x: jnp.ndarray) -> jnp.ndarray:
        diff_terms = jnp.sum(jnp.abs(x[1:] - x[:-1]) ** self.q)
        last_term = jnp.abs(x[-1]) ** self.q
        return (diff_terms + last_term) / self.q

    def loss(self, x: jnp.ndarray) -> float:
        return float(self._loss_jitted(x))

    def gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._gradient_jitted(x)

    def hessian(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._hessian_jitted(x)


class ZakharovFunction:
    """Zakharov benchmark objective."""

    def __init__(self, dim: int = 200) -> None:
        self.dim = dim
        self.i_vec = 0.5 * jnp.arange(1, dim + 1)

        self._loss_jitted = jax.jit(self._loss)
        self._gradient_jitted = jax.jit(jax.grad(self._loss))
        self._hessian_jitted = jax.jit(jax.hessian(self._loss))

    def _loss(self, x: jnp.ndarray) -> jnp.ndarray:
        linear_term = jnp.sum(self.i_vec * x)
        return jnp.sum(x**2) + linear_term**2 + linear_term**4

    def loss(self, x: jnp.ndarray) -> float:
        return float(self._loss_jitted(x))

    def gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._gradient_jitted(x)

    def hessian(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._hessian_jitted(x)


class PowellSingularFunction:
    """Powell singular function in block form."""

    def __init__(self, dim: int = 200) -> None:
        if dim % 4 != 0:
            raise ValueError("dim must be divisible by 4 for the Powell singular function.")

        self.dim = dim
        self.groups = dim // 4

        self._loss_jitted = jax.jit(self._loss)
        self._gradient_jitted = jax.jit(jax.grad(self._loss))
        self._hessian_jitted = jax.jit(jax.hessian(self._loss))

    def _loss(self, x: jnp.ndarray) -> jnp.ndarray:
        total = 0.0
        for i in range(self.groups):
            x1 = x[4 * i]
            x2 = x[4 * i + 1]
            x3 = x[4 * i + 2]
            x4 = x[4 * i + 3]

            term1 = (x1 + 10 * x2) ** 2
            term2 = 5 * (x3 - x4) ** 2
            term3 = (x2 - 2 * x3) ** 4
            term4 = 10 * (x1 - x4) ** 4

            total += term1 + term2 + term3 + term4
        return total

    def loss(self, x: jnp.ndarray) -> float:
        return float(self._loss_jitted(x))

    def gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._gradient_jitted(x)

    def hessian(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._hessian_jitted(x)


class LogSumExpFunction:
    """Smooth log-sum-exp test problem."""

    def __init__(self, n: int = 500, d: int = 200, rho: float = 0.5, random_state: int = 42) -> None:
        self.n = n
        self.d = d
        self.rho = rho

        key = jax.random.PRNGKey(random_state)
        key_A, key_b = jax.random.split(key)
        self.A = jax.random.normal(key_A, (n, d))
        self.b = jax.random.normal(key_b, (n,))

        self._loss_jitted = jax.jit(self._loss)
        self._gradient_jitted = jax.jit(jax.grad(self._loss))
        self._hessian_jitted = jax.jit(jax.hessian(self._loss))

    def _loss(self, x: jnp.ndarray) -> jnp.ndarray:
        linear_terms = jnp.dot(self.A, x) - self.b
        max_term = jnp.max(linear_terms)
        shifted_terms = (linear_terms - max_term) / self.rho
        exp_terms = jnp.exp(shifted_terms)
        return jnp.log(jnp.sum(exp_terms)) + max_term

    def loss(self, x: jnp.ndarray) -> float:
        return float(self._loss_jitted(x))

    def gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._gradient_jitted(x)

    def hessian(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._hessian_jitted(x)


class MultivariateTMLE:
    """Negative log-likelihood model for multivariate Student-t estimation.

    Parameters are stored in an unconstrained form:
    ``theta = [mu, vec(L), nu_tilde]``, where ``L`` is lower triangular and the
    diagonal entries are exponentiated to guarantee positive definiteness.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        dim: int = 10,
        df_true: float = 5.0,
        random_state: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.dim = dim
        self.df_true = df_true

        key = jax.random.PRNGKey(random_state)
        key1, key2, key3 = jax.random.split(key, 3)

        self.mu_true = jax.random.normal(key1, (dim,))
        L_true = jax.random.normal(key2, (dim, dim))
        self.Sigma_true = L_true @ L_true.T + jnp.eye(dim)
        self.X = self._generate_multivariate_t(key3, self.mu_true, self.Sigma_true, df_true, n_samples)

        self.n_L_params = self.dim * (self.dim + 1) // 2
        self.total_params = self.dim + self.n_L_params + 1
        self.L_indices = self._precompute_L_indices()

        self._loss_jitted = jit(self._loss)
        self._gradient_jitted = jit(grad(self._loss))
        self._hessian_jitted = jit(hessian(self._loss))

    def _precompute_L_indices(self) -> list[tuple[int, int, int]]:
        """Precompute the index map from the packed vector to the Cholesky factor."""
        indices: list[tuple[int, int, int]] = []
        idx = 0
        for i in range(self.dim):
            for j in range(i + 1):
                indices.append((i, j, idx))
                idx += 1
        return indices

    def _generate_multivariate_t(
        self,
        key: jax.Array,
        mu: jnp.ndarray,
        Sigma: jnp.ndarray,
        df: float,
        n_samples: int,
    ) -> jnp.ndarray:
        """Generate multivariate Student-t samples."""
        key_gamma, key_normal = jax.random.split(key)

        shape = df / 2.0
        rate = df / 2.0
        u = jax.random.gamma(key_gamma, a=shape, shape=(n_samples,)) / rate

        L = jnp.linalg.cholesky(Sigma)
        z = jax.random.normal(key_normal, (n_samples, self.dim))
        normal_samples = mu + jnp.dot(z, L.T)
        scaling = jnp.sqrt(1.0 / u)[:, jnp.newaxis]
        return normal_samples * scaling

    def _unpack_parameters(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Split the unconstrained parameter vector into its components."""
        mu = theta[: self.dim]
        L_flat = theta[self.dim : self.dim + self.n_L_params]
        nu_tilde = theta[self.dim + self.n_L_params]
        return mu, L_flat, nu_tilde

    def _reconstruct_L(self, L_flat: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct the lower-triangular Cholesky factor from packed form."""
        L = jnp.zeros((self.dim, self.dim))
        for i, j, flat_idx in self.L_indices:
            value = L_flat[flat_idx]
            if i == j:
                value = jnp.exp(value)
            L = L.at[i, j].set(value)
        return L

    def _loss(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the unconstrained negative log-likelihood."""
        mu, L_flat, nu_tilde = self._unpack_parameters(theta)

        L = self._reconstruct_L(L_flat)
        Sigma = L @ L.T
        nu = jnp.exp(nu_tilde)

        log_gamma_term = jax.scipy.special.gammaln((nu + self.dim) / 2) - jax.scipy.special.gammaln(nu / 2)
        log_det_term = 0.5 * jnp.linalg.slogdet(Sigma)[1]
        log_const = log_gamma_term - 0.5 * self.dim * jnp.log(nu * jnp.pi) - log_det_term

        diff = self.X - mu
        L_inv_diff = jax.vmap(lambda x: jax.scipy.linalg.solve_triangular(L, x, lower=True))(diff)
        mahalanobis_sq = jnp.sum(L_inv_diff**2, axis=1)
        data_terms = (nu + self.dim) / 2 * jnp.log(1 + mahalanobis_sq / nu)
        total_data_term = jnp.sum(data_terms)

        nll = -self.n_samples * log_const + total_data_term
        reg_term = 0.0 * 1e-8 * (jnp.sum(mu**2) + jnp.sum(L_flat**2) + nu_tilde**2)
        return nll + reg_term

    def loss(self, theta: jnp.ndarray | np.ndarray) -> float:
        """Return the negative log-likelihood value."""
        theta = jnp.asarray(theta)
        return float(self._loss_jitted(theta))

    def gradient(self, theta: jnp.ndarray | np.ndarray) -> np.ndarray:
        """Return the gradient of the negative log-likelihood."""
        theta = jnp.asarray(theta)
        return np.array(self._gradient_jitted(theta))

    def hessian(self, theta: jnp.ndarray | np.ndarray) -> np.ndarray:
        """Return the Hessian of the negative log-likelihood."""
        theta = jnp.asarray(theta)
        return np.array(self._hessian_jitted(theta))

    def get_initial_guess(self) -> np.ndarray:
        """Construct a moment-based initial guess."""
        sample_mean = jnp.mean(self.X, axis=0)
        sample_cov = jnp.cov(self.X, rowvar=False)
        sample_cov_reg = sample_cov + jnp.eye(self.dim) * 0.0

        try:
            L0 = jnp.linalg.cholesky(sample_cov_reg)
        except Exception:
            L0 = jnp.eye(self.dim)

        L_flat = []
        for i in range(self.dim):
            for j in range(i + 1):
                if i == j:
                    L_flat.append(float(jnp.log(jnp.maximum(L0[i, j], 1e-8))))
                else:
                    L_flat.append(float(L0[i, j]))
        nu_tilde_0 = np.log(10.0)
        return np.concatenate([np.array(sample_mean), np.array(L_flat), np.array([nu_tilde_0])])

    def get_parameter_count(self) -> int:
        """Return the total number of free parameters."""
        return self.total_params

    def evaluate_estimation(self, theta: jnp.ndarray | np.ndarray) -> dict[str, object]:
        """Compare estimated parameters against the data-generating truth."""
        theta = jnp.asarray(theta)
        mu_est, L_flat_est, nu_tilde_est = self._unpack_parameters(theta)
        nu_est = jnp.exp(nu_tilde_est)
        L_est = self._reconstruct_L(L_flat_est)
        Sigma_est = L_est @ L_est.T
        d = self.dim

        mu_error = float(jnp.linalg.norm(mu_est - self.mu_true) / d)
        sigma_error = float(jnp.linalg.norm(Sigma_est - self.Sigma_true) / (((1 + d) * d) / 2))
        nu_error = float(jnp.abs(nu_est - self.df_true))

        return {
            "mu_error": mu_error,
            "Sigma_error": sigma_error,
            "nu_error": nu_error,
            "mu_estimated": np.array(mu_est),
            "Sigma_estimated": np.array(Sigma_est),
            "nu_estimated": float(nu_est),
        }
