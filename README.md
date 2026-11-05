<div align="center">

# HyperbolicMomentum
### Curvature-Adaptive Manifold Optimization

**Learn the geometry. Carry the momentum.**

[![Paper](https://img.shields.io/badge/Paper-PDF-243457?style=for-the-badge&logo=readthedocs&logoColor=white)](paper/main.pdf)
[![Quick start](https://img.shields.io/badge/Quick_Start-5_minutes-15857B?style=for-the-badge&logo=python&logoColor=white)](#quick-start)
[![Experiments](https://img.shields.io/badge/Experiments-Code-6153A5?style=for-the-badge&logo=pytorch&logoColor=white)](docs/EXPERIMENTS.md)
[![Results](https://img.shields.io/badge/Results-705_complete-B57C27?style=for-the-badge&logo=googlesheets&logoColor=white)](results/README.md)

[![Checks](https://github.com/Hik289/manifold-hyperbolic-optimization/actions/workflows/checks.yml/badge.svg)](https://github.com/Hik289/manifold-hyperbolic-optimization/actions/workflows/checks.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)
![Protocol](https://img.shields.io/badge/Protocol-seed_42-687A86)

[English](README.md) · [简体中文](README_zh.md) · [Method](#method-in-one-minute) · [Reproduce figures](#reproduce-the-paper-and-figures) · [Citation](#citation)

<img src="assets/intuition.png" alt="HyperbolicMomentum intuition: adapt group curvature, take a normalized geodesic step, and map both the parameter point and its tangent momentum when curvature changes." width="100%" />

*A conceptual illustration for independent Lorentz-manifold groups; it is not an experimental result.*

</div>

HyperbolicMomentum studies optimization when **curvature is itself a learned state**. It combines group-specific curvature, curvature-normalized geodesic steps, and a compatible map of both the parameters and their tangent momentum. This repository contains the research implementation, every experimental source file, registered configurations, completed numerical records, and the full manuscript with detailed convergence and generalization analysis.

## At a glance

| Component | Included |
| :--- | :--- |
| Geometry | Lorentz exponential/log maps, parallel transport, tangent projection, cross-curvature maps |
| Optimization | Euclidean and fixed-curvature controls, learned curvature, normalization and transport ablations |
| Experiments | Synthetic hierarchies, metric drift, node classification, text, vision, KG, recommendation and continual learning |
| Additional tasks | Taxonomy, held-out link prediction and class-invariant retrieval, with delayed test evaluation |
| Reproduction | **893 registered configurations; 705 completed; 188 explicitly deferred** |
| Paper assets | Editable LaTeX, complete theoretical appendix, tables, loss-plot code and exact plotting inputs |

## Method in one minute

For curvature magnitude $K>0$, the Lorentz model satisfies $\langle x,x\rangle_{\mathcal L}=-1/K$. Each step maintains a point **and** a momentum vector in its current tangent space.

1. **Control the geometric step.** Average tangent gradients into $\widetilde m$, then use $\eta_{\mathrm{geo}}=\eta/(1+\gamma\sqrt K\lVert\widetilde m\rVert)$.
2. **Move on the current manifold.** Apply the exponential map and parallel-transport momentum to the new base point.
3. **Adapt curvature slowly.** Take the total curvature derivative at a fixed canonical point, with bounded drift in the implementation.
4. **Map the joint state.** With $a=\sqrt{K_{\mathrm{old}}/K_{\mathrm{new}}}$, map $x\leftarrow ax$ and $m\leftarrow am$, then restore numerical tangency.

The canonical point $\sqrt Kx$ and momentum $\sqrt Km$ are preserved by this ideal joint scaling. The interpretation applies to architectures with a dedicated manifold per group; the manuscript distinguishes this setting from shared-entity models.

**Start reading:** [geometry implementation](experiments/revision/geometry.py) · [Lorentz operators](experiments/lorentz.py) · [algorithm and derivations](paper/sections/4_methodology.tex).

## Quick start

Python **3.11** is recommended. The numerical checks and small synthetic fixture run on CPU; the complete benchmark was run with CUDA. All commands below are executed from the repository root.

```bash
git clone https://github.com/Hik289/manifold-hyperbolic-optimization.git
cd manifold-hyperbolic-optimization
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Check geometry, total curvature derivatives and the transport ablation.
python experiments/revision/validate_geometry.py

# A three-step CPU fixture; no dataset download and no benchmark claim.
python scripts/run_experiment.py --smoke --device cpu

# Audit the published final records without training or network access.
python scripts/summarize_results.py
```

The fixture writes to `artifacts/reproduction/runs/fixture_mechanism_B5/`. It is explicitly marked `fixture_only` and is separate from the registered experiments. Existing complete runs are reused; incomplete trajectories are never silently appended to.

## Run the experiments

Install the optional dataset and model dependencies for the full suite:

```bash
python -m pip install -r requirements-full.txt

# Inspect exact registered settings before launching a run.
python scripts/run_experiment.py --list --family mechanism

# Full registered budget: 2,000 updates, seed 42.
python scripts/run_experiment.py \
  --id mechanism_smooth8_independent_lorentz_3_B5 \
  --device cuda --work-dir artifacts/mechanism

# All retained configurations in one family, executed sequentially.
python scripts/run_experiment.py \
  --all-retained --family mechanism --device cuda
```

The entry point uses the saved manifest without changing epochs, seeds, splits or tuning rules. `--id` can also address an explicitly deferred configuration; such a new run is separate from the published completion record. Use `--data-root data` or the `MANIFOLD_DATA` environment variable to choose a dataset directory. No account-specific paths or services are required.

| Task family | Main implementation | Protocol |
| :--- | :--- | :--- |
| Synthetic hierarchies | [`structured.py`](experiments/revision/structured.py) | Tree type × size × dimension grid |
| Stationary and drifting metrics | [`mechanism.py`](experiments/revision/mechanism.py) | Six conditions; matched five-initialization comparison |
| Node, text, vision and continual learning | [`runner.py`](experiments/revision/runner.py) | Full registered lengths; task-specific endpoints |
| Knowledge graphs and recommendation | [`structured.py`](experiments/revision/structured.py) | Filtered ranking and chronological recommendation splits |
| Taxonomy, link prediction, retrieval | [`new_tasks.py`](experiments/revision/new_tasks.py) | Validation-only search; test released after the complete block |
| Original exploratory studies | [`experiments/`](experiments) | Separate historical implementations and protocols |

See **[Experiment guide](docs/EXPERIMENTS.md)** for dataset layouts, baseline definitions, ablations, queue tools and evaluation commands. The observed benchmark software versions are recorded in [`requirements-reproduction.txt`](requirements-reproduction.txt); the historical exploratory scripts are not interchangeable with the revised protocol.

## Selected evidence

The strongest matched comparisons are the **validation-selected mechanism experiments**, with the same five initializations and full 2,000-update budget. Lower held-out distortion is better.

| Condition | Best fixed-curvature control (B35) | HyperbolicMomentum (B5) | Relative reduction |
| :--- | ---: | ---: | ---: |
| Stationary hierarchy | 0.2011 | **0.1952** | **2.94%** |
| Smooth 2× metric drift | 0.1823 | **0.1762** | **3.31%** |
| Smooth 8× metric drift | 0.2757 | **0.2619** | **5.02%** |

These are condition-specific, single-seed numerical comparisons, not significance tests or three independent task-family wins. The matched point-only control remains close. The paper also reports all 48 original synthetic B5–B3 contrasts, five small positive final-checkpoint node-classification contrasts, and other explicitly named-control comparisons. **All ties, negative outcomes and stronger controls remain in the appendix and published records.** The six-task extension does not meet the stated family-level advantage criterion.

**Inspect the evidence:** [complete records](results/README.md) · [matched mechanism data](paper/tables/revision_evidence.json) · [additional comparisons](paper/evidence/positive_contrasts.json) · [full paper](paper/main.pdf).

## Reproduce the paper and figures

No training, GPU or network access is needed to regenerate the three published loss figures or the two added comparison tables:

```bash
python paper/scripts/reproduce_figures.py
python paper/scripts/reproduce_positive_tables.py
```

The figure script reads the bundled CSV files, verifies their hashes, and writes both **PNG and PDF**. The release check reproduces the existing figure pixels in the recorded plotting environment. [`paper/scripts/requirements-figures.txt`](paper/scripts/requirements-figures.txt) records that environment; newer plotting libraries may change rendering without changing the underlying values. The new README intuition image is a conceptual illustration, with its generation specification in [`assets/intuition_prompt.txt`](assets/intuition_prompt.txt).

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

Large datasets, model checkpoints, console logs and full per-step training archives are generated locally rather than committed. The repository includes every experiment source file, every registered configuration, all completed final-result records and all data needed for the included paper figures. [Release notes](docs/RELEASE.md) describe the portability changes and source coverage.

## Citation

If you use this research code, please cite the repository and the accompanying manuscript:

```bibtex
@misc{hyperbolicmomentum2026,
  title        = {HyperbolicMomentum: Curvature-Adaptive Manifold Optimization},
  author       = {{HyperbolicMomentum Contributors}},
  year         = {2026},
  howpublished = {GitHub},
  url          = {https://github.com/Hik289/manifold-hyperbolic-optimization}
}
```

Bundled conference styles retain their upstream notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
