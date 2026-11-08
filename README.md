<div align="center">

<h1>HyperbolicMomentum</h1>
<p>Curvature-Adaptive Manifold Optimization</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-15857B?style=flat-square)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)

<br />

<img src="assets/intuition-readme.png" alt="Three steps: learn curvature for each parameter group, move along the curved space, and rescale both parameters and momentum when curvature changes." width="100%" />

<sub>Learnable curvature · Geodesic updates · Momentum transport</sub>

</div>

## Overview

HyperbolicMomentum is an optimizer for models that learn in hyperbolic space. Each group of parameters can learn its own curvature, which controls the space's geometric scale. When curvature changes, the optimizer rescales both the parameters and their momentum so that the next update uses a consistent state.

This repository includes the optimizer, experiment code, saved results, figure code and data, and the paper's LaTeX sources. The paper studies convergence and generalization, with full proofs and experimental comparisons.

## Method

For each parameter group, the optimizer stores a point $x$, a momentum vector $m$, and a curvature magnitude $K>0$. The space has curvature $-K$. The tangent space gives the allowed directions of movement at a point. A geodesic is the curved-space equivalent of a straight line.

| Step | What it does |
| :--- | :--- |
| **Build momentum** | Combine the current gradient with momentum from earlier updates. Both vectors must lie in the tangent space at the current point. |
| **Move the parameters** | Scale the step using curvature and momentum size, then move along a geodesic. Parallel transport carries momentum to the new point. |
| **Update curvature** | Adjust curvature using the loss gradient, on a slower timescale than the parameter update. Rescale both the point and its momentum to the new curvature. |

The step size is

$$
\eta_{\mathrm{geo}}=\frac{\eta}{1+\gamma\sqrt{K}\lVert\widetilde m\rVert_x},
$$

where $\eta$ is the base step size, $\gamma$ controls normalization, and $\widetilde m$ is the updated momentum. When curvature changes, apply the same scale factor to the point and momentum:

$$
(x,m)\mapsto(a x,a m),
\qquad a=\sqrt{\frac{K_{\mathrm{old}}}{K_{\mathrm{new}}}}.
$$

This keeps $(\sqrt{K}x,\sqrt{K}m)$ unchanged in exact arithmetic: the point and momentum have the same coordinates when expressed at unit curvature. The code also projects momentum back onto the tangent space to correct numerical error.

This description assumes that each group has its own hyperbolic space. The paper discusses models that share entity representations separately.

## Quick start

The benchmark environment used Python **3.11** and PyTorch **2.5**. The geometry checks and small synthetic example below run on CPU. The full benchmark used CUDA.

```bash
git clone https://github.com/Hik289/manifold-hyperbolic-optimization.git
cd manifold-hyperbolic-optimization
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run these commands from the repository root:

```bash
# Check the geometry operations and curvature derivatives.
python experiments/revision/validate_geometry.py

# Run a small synthetic example without downloading data.
python scripts/run_experiment.py --smoke --device cpu

# Check the saved benchmark results without running training.
python scripts/summarize_results.py
```

The small example saves its output in `artifacts/reproduction/runs/fixture_mechanism_B5/`. Its results are marked `fixture_only` because the example checks that the code runs; it is not a full benchmark experiment.

## Experiments

Install the additional dependencies, list the available configurations, then run one:

```bash
python -m pip install -r requirements-full.txt
python scripts/run_experiment.py --list --family mechanism
python scripts/run_experiment.py \
  --id mechanism_smooth8_independent_lorentz_3_B5 \
  --device cuda --work-dir artifacts/mechanism
```

Use `--data-root data` or the `MANIFOLD_DATA` environment variable to choose a dataset directory. Each configuration specifies the training length, random seed, data split and evaluation rules.

| Task | What is compared | Code |
| :--- | :--- | :--- |
| Synthetic hierarchies | Tree shapes, sizes and embedding dimensions | [`structured.py`](experiments/revision/structured.py) |
| Fixed and changing distance scales | Curvature learning under stationary and drifting conditions | [`mechanism.py`](experiments/revision/mechanism.py) |
| Classification and continual learning | Graph, text and image models | [`runner.py`](experiments/revision/runner.py) |
| Knowledge graphs and recommendation | Ranking models with the specified data splits | [`structured.py`](experiments/revision/structured.py) |
| Taxonomy, link prediction and retrieval | Settings chosen on validation data, then evaluated on test data | [`new_tasks.py`](experiments/revision/new_tasks.py) |

The [experiment guide](docs/EXPERIMENTS.md) explains data preparation, baselines, evaluation and which runs were completed. The [recorded environment](requirements-reproduction.txt) lists the package versions used for the benchmark. Earlier experiments remain in `experiments/`; their settings differ from the revised benchmark.

<details>
<summary><strong>Run a group of experiments</strong></summary>

Run all configurations retained in the study for one task family, one after another:

```bash
python scripts/run_experiment.py \
  --all-retained --family mechanism --device cuda
```

The launcher skips runs that already have a complete, matching result. If a run stopped partway through, choose a new `--work-dir` to start again without mixing logs. Configurations marked `deferred` were left out when the study's compute budget was reduced. You can run them with `--id`; doing so does not change the saved published results.

</details>

## Reproduce the paper and figures

Rebuild the loss curves and comparison tables from the included data:

```bash
python paper/scripts/reproduce_figures.py
python paper/scripts/reproduce_positive_tables.py
```

These commands need no training, GPU or network access. The figure script checks that its input files have not changed, then saves **PNG and PDF** images. Use the [recorded plotting environment](paper/scripts/requirements-figures.txt) to match the included images. The README illustration is a separate conceptual diagram; its [creation notes](assets/intuition_readme_prompt.txt) are also included.

<details>
<summary><strong>Compile the paper locally or on Overleaf</strong></summary>

Install TeX Live and `latexmk`, then run:

```bash
python paper/scripts/build_paper.py
```

The repository contains LaTeX sources and PNG figures. Compiled PDFs stay on your computer and are excluded from Git.

For Overleaf, upload the contents of `paper/`, choose `main.tex` as the main document, and select **pdfLaTeX**. The current template is an ICML 2027 submission draft based on the ICML 2026 style, not an official 2027 template. The [template notes](paper/TEMPLATE_PROVENANCE.txt) record its source.

</details>

## Repository structure

<details>
<summary><strong>Where to find each part</strong></summary>

```text
assets/                    README illustration and creation notes
experiments/               Experiment code and optimizers
  revision/                Geometry, benchmark configurations and evaluation
scripts/                   Commands to run experiments and check results
results/                   Completed results, deferred runs and earlier results
paper/                     LaTeX sources, proofs, references and figures
  figure_data/             Data used to draw the paper's figures
  scripts/                 Code to rebuild figures, tables and the paper
figures/                   Earlier plotting scripts
docs/                      Experiment guide and original research plan
tests/                     Checks for geometry, data splits and execution
```

The repository includes all experiment source files, saved configurations, completed result records and data for the paper's plots. Training downloads or prepares the datasets and writes logs and model checkpoints locally. See the [release notes](docs/RELEASE.md) for the changes made to run the code outside the original environment.

</details>

## Citation

If you use HyperbolicMomentum in your research, please cite:

```bibtex
@misc{hyperbolicmomentum2026,
  title        = {HyperbolicMomentum: Curvature-Adaptive Manifold Optimization},
  author       = {{HyperbolicMomentum Contributors}},
  year         = {2026},
  howpublished = {GitHub},
  url          = {https://github.com/Hik289/manifold-hyperbolic-optimization}
}
```
