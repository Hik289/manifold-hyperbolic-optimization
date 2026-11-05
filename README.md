<div align="center">

<h1>HyperbolicMomentum</h1>
<p>Curvature-Adaptive Manifold Optimization</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-15857B?style=flat-square)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)

<br />

<img src="assets/intuition-readme.png" alt="Method schematic: learn group curvature, take a curvature-normalized geodesic step, and jointly map the parameters and tangent momentum when curvature changes." width="100%" />

<sub>Learnable curvature · Geodesic updates · Momentum transport</sub>

</div>

## Overview

HyperbolicMomentum studies Riemannian optimization with learnable, group-specific curvature. As the geometry changes, the optimizer must update both the parameter point and the tangent space in which its momentum lives. The method combines curvature-normalized geodesic steps with a joint transformation of parameters and momentum across curvature changes.

This repository contains the optimizers, experimental protocols and results, reproducible figure inputs, and the manuscript with convergence and generalization analysis.

## Method

Each group maintains a point on a Lorentz manifold, its tangent momentum, and a curvature magnitude $K>0$. The update separates motion on the current manifold from adaptation of its geometry.

| Component | Update |
| :--- | :--- |
| **Geodesic step** | Average tangent gradients into momentum and normalize the step by curvature and momentum magnitude. |
| **Momentum transport** | Move the parameter with the exponential map and parallel-transport momentum to the new base point. |
| **Curvature adaptation** | Update curvature on a slower timescale using the total curvature derivative at fixed canonical coordinates, then jointly map the parameter and momentum. |

The normalized step size and the cross-curvature map take the form

$$
\eta_{\mathrm{geo}}=\frac{\eta}{1+\gamma\sqrt{K}\lVert\widetilde m\rVert_x},
\qquad
(x,m)\mapsto(a x,a m),
\quad a=\sqrt{\frac{K_{\mathrm{old}}}{K_{\mathrm{new}}}}.
$$

The joint map preserves the canonical state $(\sqrt{K}x,\sqrt{K}m)$ in exact arithmetic; the implementation restores tangency after mapping. This interpretation applies to architectures with a dedicated manifold per group. Shared-entity models are treated separately in the manuscript.

## Quick start

Use Python **3.11** and PyTorch **2.5**. Geometry checks and the synthetic smoke test run on CPU; benchmark training uses CUDA.

```bash
git clone https://github.com/Hik289/manifold-hyperbolic-optimization.git
cd manifold-hyperbolic-optimization
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Verify the implementation and inspect the bundled results:

```bash
# Geometry and curvature-derivative checks.
python experiments/revision/validate_geometry.py

# Synthetic smoke test; no dataset download.
python scripts/run_experiment.py --smoke --device cpu

# Audit completed result records without retraining.
python scripts/summarize_results.py
```

Smoke-test outputs are saved under `artifacts/reproduction/runs/fixture_mechanism_B5/` and marked `fixture_only`, separate from benchmark results. Run all commands from the repository root.

## Experiments

Install the benchmark dependencies, inspect the registered settings, and launch a configuration:

```bash
python -m pip install -r requirements-full.txt
python scripts/run_experiment.py --list --family mechanism
python scripts/run_experiment.py \
  --id mechanism_smooth8_independent_lorentz_3_B5 \
  --device cuda --work-dir artifacts/mechanism
```

Set the dataset directory with `--data-root data` or `MANIFOLD_DATA`. The launcher preserves the manifest's training budgets, seeds, data splits and tuning rules.

| Evaluation setting | Protocol | Implementation |
| :--- | :--- | :--- |
| Synthetic hierarchies | Tree structure, size and embedding dimension | [`structured.py`](experiments/revision/structured.py) |
| Stationary and drifting metrics | Matched initialization searches | [`mechanism.py`](experiments/revision/mechanism.py) |
| Node classification, text, vision and continual learning | Task-specific endpoints and registered training budgets | [`runner.py`](experiments/revision/runner.py) |
| Knowledge graphs and recommendation | Filtered ranking and chronological splits | [`structured.py`](experiments/revision/structured.py) |
| Taxonomy, link prediction and retrieval | Validation-based selection followed by delayed test evaluation | [`new_tasks.py`](experiments/revision/new_tasks.py) |

The [experiment guide](docs/EXPERIMENTS.md) describes datasets, baselines, ablations and evaluation. [`requirements-reproduction.txt`](requirements-reproduction.txt) records the benchmark environment. Original exploratory implementations are retained under `experiments/` with their own protocols.

<details>
<summary><strong>Batch runs and output handling</strong></summary>

Run all retained configurations in a family sequentially:

```bash
python scripts/run_experiment.py \
  --all-retained --family mechanism --device cuda
```

Completed runs are reused. For an incomplete trajectory, choose a new work directory to avoid mixing runs. Deferred configurations can be launched with `--id`; these runs remain separate from the published completion record.

</details>

## Reproduce the paper and figures

Regenerate the loss figures and comparison tables from bundled inputs:

```bash
python paper/scripts/reproduce_figures.py
python paper/scripts/reproduce_positive_tables.py
```

These commands require no training, GPU or network access. The figure script verifies input hashes and exports **PNG and PDF**. Use the [recorded plotting environment](paper/scripts/requirements-figures.txt) to match the published rendering. The conceptual method illustration has a separate [generation specification](assets/intuition_readme_prompt.txt).

<details>
<summary><strong>Build the manuscript locally or on Overleaf</strong></summary>

With TeX Live and `latexmk` installed:

```bash
python paper/scripts/build_paper.py
```

The repository distributes LaTeX sources and PNG figures. PDF outputs are generated locally and excluded from version control.

For Overleaf, upload the contents of `paper/`, set `main.tex` as the main document, and select **pdfLaTeX**. The manuscript uses a local ICML 2027 draft wrapper derived from ICML 2026; [template provenance](paper/TEMPLATE_PROVENANCE.txt) documents its source and status.

</details>

## Repository structure

<details>
<summary><strong>Source, results and manuscript files</strong></summary>

```text
assets/                    Conceptual method illustration
experiments/               Original experiments and optimizers
  revision/                Geometry, benchmark suite, manifests and evaluation
scripts/                   Experiment launcher and result audit
results/                   Completed records, deferred records and history
paper/                     Manuscript, theory, bibliography and figures
  figure_data/             Exact plotting inputs with source hashes
  scripts/                 Figure, table and manuscript reproduction
figures/                   Original plotting implementations
docs/                      Experiment guide and research proposal
tests/                     Geometry, protocol and smoke-test checks
```

All experiment source files, registered configurations, completed final-result records and paper-figure inputs are included. Datasets, checkpoints and training logs are stored locally. The [release notes](docs/RELEASE.md) document source coverage and portability changes.

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
