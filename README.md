# adaptive-lm

`adaptive-lm` is a research-oriented Python package for adaptive Levenberg-Marquardt-type methods and related second-order optimization algorithms. 

## Scope

This repository preserves the original experimental design while improving packaging, metadata, reproducibility support, and English-language documentation.

### Involved optimization methods
- `Algorithm1`:Our adaptive LM algorithm(ALM)
- [`AdaN`]()
- [`ARC`](https://arxiv.org/abs/xxxx.xxxxx)
- [`CR`](https://doi.org/10.1007/s10107-006-0706-8): cubic regularized Newton method(Nesterov & Polyak, 2007)
- [`SuperUniversalNewton`](https://doi.org/10.1137/22M1519444): super universal regularized Newton method(Dokiov et.al, 2024)
- [`CubicMM`](https://arxiv.org/abs/xxxx.xxxxx)
- [`ECME`](https://arxiv.org/abs/xxxx.xxxxx)

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
├── Experiments.ipynb
├── src/
│   └── adaptive_lm/
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

## Backward compatibility

The legacy top-level files (`base.py`, `models.py`, `optimizers.py`, `__init__.py`) are retained as thin compatibility wrappers for local notebook usage.
