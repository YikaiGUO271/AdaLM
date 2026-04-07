import jax.numpy as jnp

from adaptive_lm import AdaN, HighDimRosenbrock, MultivariateTMLE


def test_package_import_and_basic_optimization():
    model = HighDimRosenbrock(dim=4)
    optimizer = AdaN(H0=1.0, max_inner_iter=5)
    theta0 = jnp.ones(4) * 1.2
    theta = optimizer.optimize(model=model, dim=4, initial_theta=theta0, max_iter=3)

    assert theta.shape == (4,)
    assert len(optimizer.history["loss"]) >= 1


def test_multivariate_t_model_shapes():
    model = MultivariateTMLE(n_samples=50, dim=3, random_state=0)
    theta0 = model.get_initial_guess()
    grad = model.gradient(theta0)
    hess = model.hessian(theta0)

    assert theta0.ndim == 1
    assert grad.shape == theta0.shape
    assert hess.shape == (theta0.size, theta0.size)
