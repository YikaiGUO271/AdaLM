"""Optimization algorithms implemented in JAX/NumPy form."""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import digamma, gammaln
from scipy.optimize import brentq, minimize_scalar

from .base import BaseOptimizer


def _default_initial_theta(dim: int, value: float = 1.5) -> jnp.ndarray:
    """Return a deterministic default initialization used in the experiments."""
    return jnp.array(np.full(dim, value, dtype=float))


class AdaN(BaseOptimizer):
    """Adaptive Newton method with a regularized inner loop."""

    def __init__(self, H0: float = 1.0, max_inner_iter: int = 20) -> None:
        super().__init__("AdaN")
        self.H0 = H0
        self.max_inner_iter = max_inner_iter
        self.history_H: list[float] = []
        self.history_lambda: list[float] = []
        self.history_inner_iters: list[int] = []

    def optimize(
        self,
        model: Any,
        dim: int,
        initial_theta: jnp.ndarray | np.ndarray | None = None,
        max_iter: int = 100,
        tol: float = 1e-8,
        **kwargs: Any,
    ) -> jnp.ndarray:
        self.reset_history()
        self.history_H = []
        self.history_lambda = []
        self.history_inner_iters = []

        theta = jnp.asarray(initial_theta) if initial_theta is not None else _default_initial_theta(dim)
        start_time = time.time()
        H_k = self.H0

        for k in range(max_iter):
            self._record(theta, model, start_time)

            grad = model.gradient(theta)
            grad_norm = jnp.linalg.norm(grad)
            if grad_norm < tol:
                break

            inner_success = False
            inner_iters = 0
            lambda_k = 0.0
            theta_plus = None
            hess = model.hessian(theta)

            if k > 0 and self.history_H:
                H_k = max(self.history_H[-1] / 4, self.H0)

            for n in range(self.max_inner_iter):
                inner_iters = n + 1
                lambda_k = jnp.sqrt(H_k * grad_norm)

                try:
                    A = hess + lambda_k * jnp.eye(dim)
                    L = jax.scipy.linalg.cho_factor(A, lower=True)
                    delta_theta = jax.scipy.linalg.cho_solve(L, -grad)
                    theta_plus = theta + delta_theta
                except Exception:
                    H_k *= 2
                    continue

                r_plus = jnp.linalg.norm(delta_theta)
                grad_plus = model.gradient(theta_plus)
                grad_plus_norm = jnp.linalg.norm(grad_plus)
                f_plus = model.loss(theta_plus)
                f_current = model.loss(theta)

                condition1 = grad_plus_norm <= 2 * lambda_k * r_plus
                condition2 = f_plus <= f_current - (2 / 3) * lambda_k * (r_plus**2)

                if condition1 and condition2:
                    inner_success = True
                    break

                H_k *= 2

            self.history_H.append(float(H_k))
            self.history_lambda.append(float(lambda_k))
            self.history_inner_iters.append(inner_iters)

            if inner_success and theta_plus is not None:
                theta = theta_plus
            else:
                step_size = 1.0 / (grad_norm + 1e-12)
                theta = theta - step_size * grad

        return theta

    def get_detailed_history(self) -> dict[str, list[float] | list[int]]:
        """Return the basic history together with inner-loop diagnostics."""
        return {
            "loss": self.history.get("loss", []),
            "grad_norm": self.history.get("grad_norm", []),
            "time": self.history.get("time", []),
            "H_values": self.history_H,
            "lambda_values": self.history_lambda,
            "inner_iterations": self.history_inner_iters,
        }


class CR(BaseOptimizer):
    """Adaptive cubic regularization with a strict inner loop."""

    def __init__(self, sigma0: float = 1.0, max_inner_iter: int = 20) -> None:
        super().__init__("CR")
        self.sigma = sigma0
        self.max_inner_iter = max_inner_iter

    def optimize(
        self,
        model: Any,
        dim: int,
        initial_theta: jnp.ndarray | np.ndarray | None = None,
        max_iter: int = 100,
        tol: float = 1e-7,
        **kwargs: Any,
    ) -> jnp.ndarray:
        self.reset_history()
        theta = jnp.asarray(initial_theta) if initial_theta is not None else _default_initial_theta(dim)
        start_time = time.time()

        iter_count = 0
        inner_iter_total = 0

        while iter_count < max_iter:
            self._record(theta, model, start_time)
            grad = model.gradient(theta)
            hess = model.hessian(theta)

            if jnp.linalg.norm(grad) < tol:
                break

            step_accepted = False

            for _ in range(self.max_inner_iter):
                inner_iter_total += 1

                try:
                    lambda_min = jnp.linalg.eigvalsh(hess)[0]
                    r_low = jnp.maximum(0, -2 * lambda_min / self.sigma) + 1e-8

                    def phi(r: float | jnp.ndarray) -> jnp.ndarray:
                        A = hess + (self.sigma * r / 2) * jnp.eye(dim)
                        try:
                            L = jax.scipy.linalg.cho_factor(A, lower=True)
                            d = -jax.scipy.linalg.cho_solve(L, grad)
                            return jnp.linalg.norm(d) - r
                        except Exception:
                            return jnp.inf

                    r_high = r_low + 1.0
                    while phi(r_high) > 0:
                        r_high *= 2
                        if r_high > 1e12:
                            break

                    try:
                        r_opt = brentq(
                            lambda r: phi(r).item(),
                            r_low.item(),
                            r_high.item(),
                            rtol=1e-6,
                            maxiter=100,
                        )
                    except Exception:
                        r_opt = (r_low + r_high) / 2
                        r_opt = r_opt.item() if hasattr(r_opt, "item") else r_opt

                    A = hess + (self.sigma * r_opt / 2 + 1e-8) * jnp.eye(dim)
                    L = jax.scipy.linalg.cho_factor(A, lower=True)
                    delta_theta = -jax.scipy.linalg.cho_solve(L, grad)

                    loss_val = model.loss(theta)
                    loss_new = model.loss(theta + delta_theta)
                    actual_reduction = loss_val - loss_new

                    predicted_reduction = -0.5 * jnp.dot(grad, delta_theta)
                    predicted_reduction -= 0.5 * jnp.dot(delta_theta, jnp.dot(hess, delta_theta))
                    predicted_reduction -= (self.sigma / 6) * jnp.linalg.norm(delta_theta) ** 3

                    rho = actual_reduction - predicted_reduction

                    if rho >= 0.0 and actual_reduction > 0:
                        theta = theta + delta_theta
                        self.sigma = max(self.sigma * 0.5, 1e-8)
                        step_accepted = True
                        iter_count += 1
                        break

                    self.sigma *= 2

                except Exception:
                    self.sigma *= 2

                self.sigma = min(self.sigma, 1e12)

            if not step_accepted:
                self.sigma *= 2
                iter_count += 1

            self.sigma = min(self.sigma, 1e12)

            if inner_iter_total >= max_iter * self.max_inner_iter:
                print(f"CR: reached the maximum number of inner iterations ({inner_iter_total}).")
                break

        return theta


class ARC(BaseOptimizer):
    """Adaptive regularization by cubics based on a Cauchy-point solver."""

    def __init__(
        self,
        eta1: float = 0.1,
        eta2: float = 0.9,
        sigma0: float = 1.0,
        gamma1: float = 2.0,
        gamma2: float = 0.5,
        sigma_min: float = 1e-6,
        max_solver_iter: int = 200,
        solver_tol: float = 1e-5,
    ) -> None:
        super().__init__("ARC")
        self.eta1 = eta1
        self.eta2 = eta2
        self.sigma = sigma0
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.sigma_min = sigma_min
        self.max_solver_iter = max_solver_iter
        self.solver_tol = solver_tol
        self.f_prev: float | None = None

    def _cauchy_point(self, g: np.ndarray, H: np.ndarray, M: float) -> np.ndarray:
        """Compute a Cauchy step for the cubic model."""
        if np.linalg.norm(g) == 0 or M == 0:
            return np.zeros_like(g)

        g_norm = np.linalg.norm(g)
        g_dir = g / g_norm
        H_gg = g_dir.T @ H @ g_dir
        discriminant = H_gg**2 - 4 * M * g_norm

        if discriminant >= 0:
            r1 = (-H_gg + np.sqrt(discriminant)) / (2 * M)
            r2 = (-H_gg - np.sqrt(discriminant)) / (2 * M)
            r = max(r1, r2) if r1 > 0 and r2 > 0 else (r1 if r1 > 0 else r2)
        else:
            r = g_norm / (np.abs(H_gg) + M * g_norm)

        return -r * g_dir

    def _cubic_subsolver(
        self,
        x: np.ndarray,
        g: np.ndarray,
        H: np.ndarray,
        M: float,
        model: Any,
    ) -> tuple[np.ndarray, int]:
        """Approximately solve the cubic regularization subproblem."""
        dim = len(g)
        cauchy_step = self._cauchy_point(g, H, M)
        r_min = np.linalg.norm(cauchy_step)

        try:
            newton_step = -np.linalg.solve(H, g)
            r_max = np.linalg.norm(newton_step)
        except Exception:
            newton_step = -g / (np.linalg.norm(g) + 1e-12)
            r_max = np.linalg.norm(newton_step)

        if M == 0:
            return x + newton_step, 1

        def convergence_criterion(s: np.ndarray, r: float) -> float:
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-12:
                return -1.0
            return 1 / s_norm - 1 / r

        identity = np.eye(dim)
        best_step = newton_step
        best_crit = float("inf")

        for solver_iter in range(self.max_solver_iter):
            r_try = (r_min + r_max) / 2
            lambda_try = M * r_try

            try:
                A = H + lambda_try * jnp.eye(dim)
                step_try = -np.linalg.solve(A, g)
            except Exception:
                lambda_try *= 2
                A = H + lambda_try * identity + 1e-8 * jnp.eye(dim)
                step_try = -np.linalg.solve(A, g)

            crit = convergence_criterion(step_try, r_try)

            if abs(crit) < abs(best_crit):
                best_step = step_try
                best_crit = crit

            if abs(crit) < self.solver_tol:
                break

            if crit < 0:
                r_min = r_try
            else:
                r_max = r_try

            if r_max - r_min < self.solver_tol:
                break

        return x + best_step, solver_iter + 1

    def optimize(
        self,
        model: Any,
        dim: int,
        initial_theta: jnp.ndarray | np.ndarray | None = None,
        max_iter: int = 100,
        tol: float = 1e-8,
        **kwargs: Any,
    ) -> jnp.ndarray:
        self.reset_history()
        theta = jnp.asarray(initial_theta) if initial_theta is not None else _default_initial_theta(dim)
        start_time = time.time()

        self.f_prev = model.loss(theta)

        for _ in range(max_iter):
            self._record(theta, model, start_time)
            grad = model.gradient(theta)
            hess = model.hessian(theta)
            grad_norm = np.linalg.norm(grad)

            if grad_norm < tol:
                break

            theta_candidate, _ = self._cubic_subsolver(theta, grad, hess, self.sigma, model)
            f_new = model.loss(theta_candidate)

            delta_theta = theta_candidate - theta
            model_decrease = (
                np.dot(grad, delta_theta)
                + 0.5 * delta_theta.T @ hess @ delta_theta
                + self.sigma / 3 * np.linalg.norm(delta_theta) ** 3
            )

            actual_reduction = self.f_prev - f_new
            predicted_reduction = -model_decrease
            rho = 0.0 if abs(predicted_reduction) < 1e-12 else actual_reduction / abs(predicted_reduction)

            if rho > self.eta1:
                theta = theta_candidate
                self.f_prev = f_new
                if rho > self.eta2:
                    self.sigma = max(self.sigma_min, self.sigma * self.gamma2)
            else:
                self.sigma *= self.gamma1

        return theta


class Algorithm1(BaseOptimizer):
    """Adaptive LM method"""

    def __init__(self, alpha: float = 0.5, beta: float = 1 / 6, H0: float = 1.0, max_inner_iter: int = 200) -> None:
        super().__init__("ALM")
        self.alpha = alpha
        self.beta = beta
        self.H0 = H0
        self.max_inner_iter = max_inner_iter
        self.history_theta: list[jnp.ndarray] = []
        self.history_grad: list[jnp.ndarray] = []

    def optimize(
        self,
        model: Any,
        dim: int,
        initial_theta: jnp.ndarray | np.ndarray | None = None,
        max_iter: int = 100,
        tol: float = 1e-8,
        **kwargs: Any,
    ) -> jnp.ndarray:
        self.reset_history()
        theta = jnp.asarray(initial_theta) if initial_theta is not None else jnp.zeros(dim)
        start_time = time.time()
        H_t = self.H0

        for _ in range(max_iter):
            self._record(theta, model, start_time)
            grad = model.gradient(theta)
            grad_norm = jnp.linalg.norm(grad)
            if grad_norm < tol:
                break

            j_t = 0
            success = False
            hess = model.hessian(theta)

            for _ in range(self.max_inner_iter):
                lambda_jt = (2**j_t) * H_t
                try:
                    A = hess + lambda_jt * jnp.eye(dim)
                    L = jax.scipy.linalg.cho_factor(A, lower=True)
                    d_plus = jax.scipy.linalg.cho_solve(L, -grad)

                    theta_candidate = theta + d_plus
                    candidate_loss = model.loss(theta_candidate)
                    current_loss = model.loss(theta)
                    expected_reduction = self.alpha * jnp.dot(grad, d_plus) - lambda_jt * self.beta * jnp.dot(d_plus, d_plus)
                    condition1 = candidate_loss <= current_loss + expected_reduction

                    if condition1:
                        theta = theta_candidate
                        self.history_theta.append(theta.copy())
                        self.history_grad.append(grad.copy())
                        H_t = max((2**j_t * H_t) / 2, 1e-4 * grad_norm**0.5)
                        success = True
                        break

                    j_t += 1

                except Exception:
                    j_t += 1

            if not success:
                if "theta_candidate" in locals():
                    theta = theta_candidate
                self.history_theta.append(theta.copy())
                self.history_grad.append(grad.copy())

        return theta


class SuperUniversalNewton(BaseOptimizer):
    """Super-universal regularized Newton method ."""

    def __init__(self, H_0: float = 1.0, alpha: float = 0.75, adaptive_search: bool = True, H_min: float = 1e-5) -> None:
        super().__init__("SUN")
        self.H_0 = H_0
        self.alpha = alpha
        self.adaptive_search = adaptive_search
        self.H_min = H_min

    def optimize(
        self,
        model: Any,
        dim: int,
        initial_theta: jnp.ndarray | np.ndarray | None = None,
        max_iter: int = 100,
        tol: float = 1e-8,
        **kwargs: Any,
    ) -> jnp.ndarray:
        self.reset_history()
        theta = jnp.asarray(initial_theta) if initial_theta is not None else _default_initial_theta(dim)
        start_time = time.time()

        g_k = model.gradient(theta)
        g_k_norm = jnp.linalg.norm(g_k)
        H_k = self.H_0

        for _ in range(max_iter):
            self._record(theta, model, start_time)
            if g_k_norm < tol:
                break

            Hess_k = model.hessian(theta)

            adaptive_search_max_iter = 40
            for i in range(adaptive_search_max_iter + 1):
                if i == adaptive_search_max_iter:
                    break

                lambda_k = H_k * (g_k_norm**self.alpha)

                try:
                    A = Hess_k + lambda_k * jnp.eye(dim)
                    L = jax.scipy.linalg.cho_factor(A, lower=True)
                    delta_theta = jax.scipy.linalg.cho_solve(L, -g_k)
                except Exception:
                    H_k *= 4
                    continue

                theta_new = theta + delta_theta
                g_new = model.gradient(theta_new)
                g_new_norm = jnp.linalg.norm(g_new)

                if not self.adaptive_search:
                    break

                lhs = jnp.dot(g_new, -delta_theta)
                rhs = (g_new_norm**2) / (4 * lambda_k)

                if lhs >= rhs:
                    H_k = jnp.maximum(H_k * 0.25, self.H_min)
                    break

                H_k *= 4

            theta = theta_new
            g_k = g_new
            g_k_norm = g_new_norm

        return theta


class CubicMM(BaseOptimizer):
    """Cubic majorization-minimization with a fixed Lipschitz estimate."""

    def __init__(self, L_fixed: float = 10.0) -> None:
        super().__init__("CMM")
        self.L_fixed = L_fixed
        self.history_L: list[float] = []
        self.history_success: list[int] = []

    def optimize(
        self,
        model: Any,
        dim: int,
        initial_theta: jnp.ndarray | np.ndarray | None = None,
        max_iter: int = 100,
        tol: float = 1e-8,
        **kwargs: Any,
    ) -> jnp.ndarray:
        self.reset_history()
        self.history_L = []
        self.history_success = []

        theta = jnp.asarray(initial_theta) if initial_theta is not None else jnp.array(np.random.randn(dim))
        start_time = time.time()
        L_k = self.L_fixed

        for _ in range(max_iter):
            self._record(theta, model, start_time)
            grad = model.gradient(theta)
            grad_norm = jnp.linalg.norm(grad)
            hess = model.hessian(theta)

            if grad_norm < tol:
                break

            try:
                lambda_min = jax.scipy.linalg.eigh(hess, subset_by_index=[0, 0])[0][0]
                lambda_val = lambda_min - 1e-3 if lambda_min <= 0 else 0.0
            except Exception:
                lambda_val = 0.0

            inner_success = False
            c = jnp.sqrt(3.0 / L_k)
            g = jnp.sqrt(grad_norm)
            r = c * g
            d = jnp.maximum(L_k * c / 3, 1.0 / c)
            H = hess + (-lambda_val + d * g) * jnp.eye(dim)

            try:
                L_chol = jax.scipy.linalg.cho_factor(H, lower=True)
                v = jax.scipy.linalg.cho_solve(L_chol, -grad)
                step_norm = jnp.linalg.norm(v)

                if step_norm <= r + 1e-8:
                    theta_candidate = theta + v
                    f_current = model.loss(theta)
                    f_candidate = model.loss(theta_candidate)

                    if f_candidate < f_current - 1e-12:
                        theta = theta_candidate
                        inner_success = True
            except Exception:
                L_k = L_k * 2

            self.history_L.append(float(L_k))
            self.history_success.append(1 if inner_success else 0)

        return theta

    def get_detailed_history(self) -> dict[str, list[float] | list[int]]:
        """Return the optimization history with cubic-MM specific metadata."""
        return {
            "loss": self.history.get("loss", []),
            "grad_norm": self.history.get("grad_norm", []),
            "time": self.history.get("time", []),
            "L_values": self.history_L,
            "success_flags": self.history_success,
        }


class ECME(BaseOptimizer):
    """ECME algorithm for multivariate Student-t maximum likelihood estimation."""

    def __init__(
        self,
        nu_min: float = 0.01,
        nu_max: float = 100.0,
        nu_init: float = 10.0,
        max_nu_iters: int = 500,
        nu_tol: float = 1e-8,
        verbose: bool = False,
        print_every: int = 25,
    ) -> None:
        super().__init__("ECME")
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.nu_init = nu_init
        self.max_nu_iters = max_nu_iters
        self.nu_tol = nu_tol
        self.verbose = verbose
        self.print_every = print_every

    def _e_step(self, X: jnp.ndarray, mu: jnp.ndarray, Sigma: jnp.ndarray, nu: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the conditional weights used by ECME."""
        d = X.shape[1]
        diff = X - mu

        try:
            L = jnp.linalg.cholesky(Sigma)
            solved = jax.vmap(lambda x: jax.scipy.linalg.solve_triangular(L, x, lower=True))(diff)
            mahal_sq = jnp.sum(solved**2, axis=1)
        except Exception:
            Sigma_reg = Sigma + 1e-6 * jnp.eye(d)
            mahal_sq = jnp.sum(diff @ jnp.linalg.inv(Sigma_reg) * diff, axis=1)

        weights = (nu + d) / (nu + mahal_sq)
        return weights, mahal_sq

    def _cm_step_1(self, X: jnp.ndarray, weights: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Update the location and scale parameters."""
        n = X.shape[0]
        d = X.shape[1]

        weight_sum = jnp.sum(weights)
        mu_new = jnp.sum(weights[:, None] * X, axis=0) / weight_sum

        diff = X - mu_new
        Sigma_new = jnp.zeros((d, d))
        for i in range(n):
            Sigma_new += weights[i] * jnp.outer(diff[i], diff[i])
        Sigma_new = Sigma_new / n

        Sigma_new = (Sigma_new + Sigma_new.T) / 2
        min_eig = jnp.linalg.eigvalsh(Sigma_new)[0]
        if min_eig < 1e-8:
            Sigma_new += (1e-8 - min_eig + 1e-8) * jnp.eye(d)

        return mu_new, Sigma_new

    def _nu_equation(self, nu: float, d: int, n: int, weights: jnp.ndarray, mahal_sq: jnp.ndarray) -> jnp.ndarray:
        """Scalar equation for the degrees-of-freedom update."""
        term1 = -digamma(nu / 2) + jnp.log(nu / 2) + 1
        term2 = jnp.mean(jnp.log(weights) - weights)
        term3 = digamma((nu + d) / 2) - jnp.log((nu + d) / 2)
        return term1 + term2 + term3

    def _cm_step_2(self, X: jnp.ndarray, mu: jnp.ndarray, Sigma: jnp.ndarray, nu_current: float, d: int) -> float:
        """Update the degrees of freedom parameter."""
        n = X.shape[0]
        weights, mahal_sq = self._e_step(X, mu, Sigma, nu_current)

        def nu_objective(nu: float) -> float:
            if nu <= self.nu_min or nu >= self.nu_max:
                return 1e10
            term1 = n * (gammaln((nu + d) / 2) - gammaln(nu / 2))
            term2 = -n * d / 2 * jnp.log(nu * jnp.pi)
            term3 = jnp.sum(((nu + d) / 2) * jnp.log(1 + mahal_sq / nu))
            return float(-(term1 + term2 - term3))

        try:
            def f(nu: float) -> jnp.ndarray:
                return self._nu_equation(nu, d, n, weights, mahal_sq)

            if f(self.nu_min) * f(self.nu_max) < 0:
                nu_new = brentq(f, self.nu_min, self.nu_max, xtol=self.nu_tol)
            else:
                result = minimize_scalar(nu_objective, bounds=(self.nu_min, self.nu_max), method="bounded")
                nu_new = result.x
        except Exception:
            result = minimize_scalar(nu_objective, bounds=(self.nu_min, self.nu_max), method="bounded")
            nu_new = result.x

        return float(nu_new)

    def _compute_loss_direct(self, X: jnp.ndarray, mu: jnp.ndarray, Sigma: jnp.ndarray, nu: float) -> float:
        """Evaluate the negative log-likelihood directly from structured parameters."""
        n, d = X.shape
        log_gamma_term = gammaln((nu + d) / 2) - gammaln(nu / 2)
        log_det_term = 0.5 * jnp.linalg.slogdet(Sigma)[1]
        log_const = log_gamma_term - 0.5 * d * jnp.log(nu * jnp.pi) - log_det_term
        mahal_sq = self._e_step(X, mu, Sigma, nu)[1]
        data_terms = (nu + d) / 2 * jnp.log(1 + mahal_sq / nu)
        nll = -n * log_const + jnp.sum(data_terms)
        return float(nll)

    def optimize(
        self,
        model: Any,
        dim: int,
        initial_theta: jnp.ndarray | np.ndarray | None = None,
        max_iter: int = 100,
        tol: float = 1e-8,
        **kwargs: Any,
    ) -> np.ndarray:
        """Run ECME directly on the structured Student-t parameters."""
        del dim, tol, kwargs
        self.reset_history()

        X = model.X
        d = model.dim

        if initial_theta is not None:
            mu, L_flat, nu_tilde = model._unpack_parameters(initial_theta)
            L = model._reconstruct_L(L_flat)
            Sigma = L @ L.T
            nu = jnp.exp(nu_tilde)
        else:
            mu = jnp.mean(X, axis=0)
            Sigma = jnp.cov(X, rowvar=False) + 1e-6 * jnp.eye(d)
            nu = self.nu_init

        start_time = time.time()

        if self.verbose:
            print("\n" + "=" * 60)
            print("ECME optimization for multivariate Student-t MLE")
            print("=" * 60)
            print(f"{'Iter':>6} {'Loss':>15} {'nu':>10} {'mu_change':>12} {'Sigma_change':>12}")
            print("-" * 67)

        for iteration in range(max_iter):
            mu_old, Sigma_old = mu, Sigma

            weights, _ = self._e_step(X, mu, Sigma, nu)
            mu, Sigma = self._cm_step_1(X, weights)
            nu = self._cm_step_2(X, mu, Sigma, float(nu), d)

            mu_change = jnp.linalg.norm(mu - mu_old)
            sigma_change = jnp.linalg.norm(Sigma - Sigma_old) / (d * d)
            loss = self._compute_loss_direct(X, mu, Sigma, float(nu))

            self.history["loss"].append(loss)
            self.history["grad_norm"].append(0.0)
            self.history["time"].append(time.time() - start_time)

            if self.verbose and (iteration < 5 or iteration % self.print_every == 0):
                print(f"{iteration:6d} {loss:15.6e} {float(nu):10.4f} {float(mu_change):12.4e} {float(sigma_change):12.4e}")

            if mu_change < 1e-16 and sigma_change < 1e-16:
                if self.verbose:
                    print(f"\nConverged at iteration {iteration}: parameter changes are negligible.")
                break

        if self.verbose:
            print("-" * 67)
            print(f"Final: loss={loss:.6e}, nu={float(nu):.4f}")

        return self._pack_parameters(model, mu, Sigma, float(nu))

    def _pack_parameters(self, model: Any, mu: jnp.ndarray, Sigma: jnp.ndarray, nu: float) -> np.ndarray:
        """Pack structured Student-t parameters back into the unconstrained vector form."""
        try:
            L = jnp.linalg.cholesky(Sigma)
        except Exception:
            Sigma_reg = Sigma + 1e-6 * jnp.eye(model.dim)
            L = jnp.linalg.cholesky(Sigma_reg)

        L_flat = []
        for i in range(model.dim):
            for j in range(i + 1):
                if i == j:
                    L_flat.append(float(jnp.log(jnp.maximum(L[i, j], 1e-8))))
                else:
                    L_flat.append(float(L[i, j]))

        nu_tilde = float(jnp.log(jnp.maximum(nu, 1e-8)))
        return np.concatenate([np.array(mu), np.array(L_flat), np.array([nu_tilde])])


__all__ = [
    "AdaN",
    "Algorithm1",
    "ARC",
    "CR",
    "CubicMM",
    "ECME",
    "SuperUniversalNewton",
]
