<div align="center">

# HyperbolicMomentum
### Curvature-Adaptive Manifold Optimization

[![Checks](https://github.com/Hik289/manifold-hyperbolic-optimization/actions/workflows/checks.yml/badge.svg)](https://github.com/Hik289/manifold-hyperbolic-optimization/actions/workflows/checks.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)

<img src="assets/intuition.png" alt="HyperbolicMomentum intuition: adapt group curvature, take a normalized geodesic step, and map both the parameter point and its tangent momentum when curvature changes." width="100%" />

*Curvature adaptation, controlled geodesic steps, and joint mapping of parameters and momentum.*

</div>

HyperbolicMomentum optimizes parameters and **learns the geometry they move in**. It combines group-specific curvature, curvature-normalized geodesic steps, and a joint map of parameters and tangent momentum as curvature changes.

This repository includes the implementation, experiment configurations and results, figure reproduction tools, and the full manuscript with convergence and generalization analysis.

## At a glance

| Component | Included |
| :--- | :--- |
| Geometry | Lorentz exponential/log maps, parallel transport, tangent projection, cross-curvature maps |
| Optimization | Euclidean and fixed-curvature controls, learned curvature, normalization and transport ablations |
| Experiments | Synthetic hierarchies, metric drift, node classification, text, vision, KG, recommendation and continual learning |
| Additional tasks | Taxonomy, held-out link prediction and class-invariant retrieval, with delayed test evaluation |
| Reproduction | Registered configurations, evaluation protocols and published result records |
| Paper assets | Editable LaTeX, complete theoretical appendix, tables, loss-plot code and exact plotting inputs |

## Method in one minute

For curvature magnitude $K>0$, the Lorentz model satisfies $\langle x,x\rangle_{\mathcal L}=-1/K$. Each step maintains a point **and** a momentum vector in its current tangent space.

1. **Control the geometric step.** Average tangent gradients into $\widetilde m$, then use $\eta_{\mathrm{geo}}=\eta/(1+\gamma\sqrt K\lVert\widetilde m\rVert)$.
2. **Move on the current manifold.** Apply the exponential map and parallel-transport momentum to the new base point.
3. **Adapt curvature slowly.** Take the total curvature derivative at a fixed canonical point, with bounded drift in the implementation.
4. **Map the joint state.** With $a=\sqrt{K_{\mathrm{old}}/K_{\mathrm{new}}}$, map $x\leftarrow ax$ and $m\leftarrow am$, then restore numerical tangency.

The canonical point $\sqrt Kx$ and momentum $\sqrt Km$ are preserved by this ideal joint scaling. The interpretation applies to architectures with a dedicated manifold per group; the manuscript distinguishes this setting from shared-entity models.

## Quick start

Use Python **3.11**. The geometry checks and synthetic smoke test run on CPU; the full benchmark uses CUDA. Run the following commands from the repository root.

```bash
git clone https://github.com/Hik289/manifold-hyperbolic-optimization.git
cd manifold-hyperbolic-optimization
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Check geometry, total curvature derivatives and the transport ablation.
python experiments/revision/validate_geometry.py

# Run a small synthetic smoke test without downloading data.
python scripts/run_experiment.py --smoke --device cpu

# Audit the published final records without training or network access.
python scripts/summarize_results.py
```

The smoke test writes to `artifacts/reproduction/runs/fixture_mechanism_B5/` and is marked `fixture_only`, separate from benchmark results. The launcher reuses completed runs and does not append to incomplete trajectories.

## Run the experiments

Install the optional dataset and model dependencies for the full suite:

```bash
python -m pip install -r requirements-full.txt

# Inspect exact registered settings before launching a run.
python scripts/run_experiment.py --list --family mechanism

# Use the full registered training budget.
python scripts/run_experiment.py \
  --id mechanism_smooth8_independent_lorentz_3_B5 \
  --device cuda --work-dir artifacts/mechanism

# All retained configurations in one family, executed sequentially.
python scripts/run_experiment.py \
  --all-retained --family mechanism --device cuda
```

The launcher preserves the manifest's training budgets, seeds, data splits and tuning rules. Set the dataset directory with `--data-root data` or the `MANIFOLD_DATA` environment variable. Deferred configurations can be run with `--id`; these runs remain separate from the published completion record.

| Task family | Main implementation | Protocol |
| :--- | :--- | :--- |
| Synthetic hierarchies | [`structured.py`](experiments/revision/structured.py) | Tree type × size × dimension grid |
| Stationary and drifting metrics | [`mechanism.py`](experiments/revision/mechanism.py) | Matched initialization searches under stationary and drifting metrics |
| Node, text, vision and continual learning | [`runner.py`](experiments/revision/runner.py) | Full registered lengths; task-specific endpoints |
| Knowledge graphs and recommendation | [`structured.py`](experiments/revision/structured.py) | Filtered ranking and chronological recommendation splits |
| Taxonomy, link prediction, retrieval | [`new_tasks.py`](experiments/revision/new_tasks.py) | Validation-only search; test released after the complete block |
| Original exploratory studies | [`experiments/`](experiments) | Separate historical implementations and protocols |

The [experiment guide](docs/EXPERIMENTS.md) covers dataset layouts, baselines, ablations and evaluation. Benchmark software versions are recorded in [`requirements-reproduction.txt`](requirements-reproduction.txt). Historical exploratory scripts use separate protocols from the revised suite.

## Reproduce the paper and figures

Regenerate the published loss figures and comparison tables from bundled data:

```bash
python paper/scripts/reproduce_figures.py
python paper/scripts/reproduce_positive_tables.py
```

These commands require no training, GPU or network access. The figure script verifies the CSV hashes and exports **PNG and PDF**. Use the [recorded plotting environment](paper/scripts/requirements-figures.txt) to match the published rendering. The conceptual illustration above has a separate [generation specification](assets/intuition_prompt.txt).

To compile the complete paper with TeX Live and `latexmk`:

```bash
python paper/scripts/build_paper.py
```

For Overleaf, upload the contents of `paper/`, select `main.tex`, and choose **pdfLaTeX**. The year-specific style is a **local ICML 2027 draft wrapper based on ICML 2026**, not an official 2027 release; see [template provenance](paper/TEMPLATE_PROVENANCE.txt).

## Repository map

```text
assets/                    PNG intuition figure and generation specification
experiments/               All original experiment and optimizer implementations
  revision/                Revised suite, geometry, manifests and evaluation
scripts/                   Portable experiment launcher and result audit
results/                   Complete final records, deferred records and history
paper/                     Full manuscript, theory, bibliography and figures
  figure_data/             Exact CSV inputs with source hashes
  scripts/                 Figure/table reproduction and local paper build
figures/                   Original plotting implementations
docs/                      Detailed reproduction guide and research proposal
tests/                     Geometry, protocol and end-to-end fixture checks
```

The repository includes all experiment source files, registered configurations, completed final-result records and inputs for the paper figures. Datasets, checkpoints and training logs are stored locally. See the [release notes](docs/RELEASE.md) for source coverage and portability details.

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
