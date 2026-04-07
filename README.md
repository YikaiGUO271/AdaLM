# adaptive-lm

`adaptive-lm` is a research-oriented Python package for adaptive Levenberg-Marquardt-type methods and related second-order optimization algorithms. 

### Involved optimization methods
- `Algorithm1`:Our adaptive LM algorithm(ALM)
- [`AdaN`](https://doi.org/10.1137/22M1488752): Regularized Newton Method with Global O(1/k2) Convergence (Mishchenko, 2023)
- [`ARC`](https://doi.org/10.1007/s10107-009-0286-5): Adaptive regularization with cubics (Cartis et,al, 2011)
- [`CR`](https://doi.org/10.1007/s10107-006-0706-8): Cubic regularized Newton method(Nesterov & Polyak, 2007)
- [`SUN`](https://doi.org/10.1137/22M1519444): Super universal regularized Newton method(Dokiov et.al, 2024)
- [`CubicMM`](https://doi.org/10.1073/pnas.2303168120): MM algorithm with cubic surrogate (Landeros et,al, 2023)
- [`ECME`](https://www.jstor.org/stable/41430942): Expectation–conditional maximization either’algorithm for t-mle solving (He & Liu, 2012)

### Benchmark and statistical models
- `HighDimRosenbrock`
- `PolytopeFeasibility`
- `WorstInstancesFunction`
- `ZakharovFunction`
- `PowellSingularFunction`
- `LogSumExpFunction`
- `MultivariateTMLE`

## Installation

### Editable installation

```bash
pip install -e .
```

### Development dependencies

```bash
pip install -e .[dev]
```

### Experiment dependencies

```bash
pip install -e .[experiments]
```

## Build distribution artifacts

```bash
python -m build
```

The build command produces both:
- a source distribution (`.tar.gz`)
- a wheel (`.whl`)

## Minimal example

```python
import jax.numpy as jnp
from adaptive_lm import AdaN, HighDimRosenbrock

model = HighDimRosenbrock(dim=20)
optimizer = AdaN(H0=1.0, max_inner_iter=20)
initial_theta = jnp.ones(20) * 1.5
solution = optimizer.optimize(
    model=model,
    dim=20,
    initial_theta=initial_theta,
    max_iter=50,
)

print("final loss:", model.loss(solution))
print("final gradient norm:", optimizer.history["grad_norm"][-1])
```
## Repository layout

```text
.
├── .github/workflows/ci.yml
├── LICENSE.md
├── MANIFEST.in
├── README.md
├── pyproject.toml
├── src/
│   └── adaptive_lm/
│       ├── Experiments.ipynb           # 📘 Sample experiment results in the paper
│       ├── __init__.py
│       ├── base.py
│       ├── models.py
│       └── optimizers.py
└── test/
    └── test_smoke.py
```


## Reproducibility and archival notes

- The installable package lives under `src/adaptive_lm/`.
- The notebook `Experiments.ipynb` is preserved as the primary experiment record.
- Thin compatibility wrappers are kept at the repository root so older notebook imports continue to work.
- Basic smoke tests are provided in `test/`.
- Basic project metadata and release files are included for packaging and archival use.

## Suggested release workflow

1. Fill all placeholder metadata.
2. Run `pytest -q`.
3. Run `python -m build`.
4. Commit the repository to GitHub.
5. Create a GitHub release and attach the files in `dist/`.
6. Archive the release on Zenodo or another DOI-granting service.

