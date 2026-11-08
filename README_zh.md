<div align="center">

<h1>HyperbolicMomentum</h1>
<p>曲率自适应流形优化</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-15857B?style=flat-square)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)

<img src="assets/intuition-readme.png" alt="三个步骤：为每组参数学习曲率，沿曲面更新参数，并在曲率变化时同步缩放参数和动量。" width="100%" />

<sub>可学习曲率 · 测地线更新 · 动量传输</sub>

</div>

## 概述

HyperbolicMomentum 是用于双曲空间模型的优化器。每组参数可以学习自己的曲率，从而调整空间的几何尺度。曲率变化时，优化器同步缩放参数和动量，使下一次更新使用的状态在几何上保持一致。

仓库包含优化器、实验代码、已保存的结果、绘图代码与数据，以及论文的 LaTeX 源码。论文分析了收敛性和泛化能力，并提供完整证明和实验对照。

## 方法

每组参数维护一个点 $x$、一个动量向量 $m$ 和曲率大小 $K>0$，对应空间的曲率为 $-K$。切空间表示某一点允许的移动方向，测地线相当于曲面上的直线。

| 步骤 | 作用 |
| :--- | :--- |
| **累积动量** | 将当前梯度与之前的动量加权相加。两个向量都要位于当前点的切空间。 |
| **更新参数** | 根据曲率和动量大小调整步长，沿测地线移动参数，再将动量平行移动到新位置。 |
| **更新曲率** | 根据损失梯度调整曲率，更新速度慢于参数。随后同步缩放参数点和动量，使它们对应新的曲率。 |

步长为

$$
\eta_{\mathrm{geo}}=\frac{\eta}{1+\gamma\sqrt{K}\lVert\widetilde m\rVert_x},
$$

其中，$\eta$ 是基础步长，$\gamma$ 控制归一化强度，$\widetilde m$ 是更新后的动量。曲率变化时，对参数和动量使用相同的缩放系数：

$$
(x,m)\mapsto(a x,a m),
\qquad a=\sqrt{\frac{K_{\mathrm{old}}}{K_{\mathrm{new}}}}.
$$

在精确计算下，这样可以保持 $(\sqrt{K}x,\sqrt{K}m)$ 不变，也就是保持它们在单位曲率下的坐标不变。代码还会将动量投影回切空间，以修正数值误差。

以上说明适用于每组参数拥有独立双曲空间的模型。共享实体表示的模型在论文中另行讨论。

## 快速开始

完整实验使用 Python **3.11** 和 PyTorch **2.5**。下面的几何检查和小型合成示例可以在 CPU 上运行，完整实验使用 CUDA。

```bash
git clone https://github.com/Hik289/manifold-hyperbolic-optimization.git
cd manifold-hyperbolic-optimization
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

在仓库根目录执行：

```bash
# 检查几何运算和曲率导数。
python experiments/revision/validate_geometry.py

# 运行小型合成示例，无需下载数据。
python scripts/run_experiment.py --smoke --device cpu

# 检查已保存的实验结果，无需重新训练。
python scripts/summarize_results.py
```

小型示例的输出位于 `artifacts/reproduction/runs/fixture_mechanism_B5/`。它带有 `fixture_only` 标记，只用于检查代码能否运行，不计入正式实验。

## 实验

安装额外依赖，查看已有配置，然后运行其中一项：

```bash
python -m pip install -r requirements-full.txt
python scripts/run_experiment.py --list --family mechanism
python scripts/run_experiment.py \
  --id mechanism_smooth8_independent_lorentz_3_B5 \
  --device cuda --work-dir artifacts/mechanism
```

用 `--data-root data` 或环境变量 `MANIFOLD_DATA` 指定数据目录。每个配置都规定了训练轮数、随机种子、数据划分和评估规则。

| 任务 | 比较内容 | 代码 |
| :--- | :--- | :--- |
| 合成层级结构 | 不同树形、节点数和嵌入维度 | [`structured.py`](experiments/revision/structured.py) |
| 固定和变化的距离尺度 | 平稳与漂移条件下的曲率学习 | [`mechanism.py`](experiments/revision/mechanism.py) |
| 分类与持续学习 | 图、文本和图像模型 | [`runner.py`](experiments/revision/runner.py) |
| 知识图谱与推荐 | 按指定数据划分评估排序模型 | [`structured.py`](experiments/revision/structured.py) |
| 分类层级、链接预测与检索 | 先用验证集选参数，再在测试集上评估 | [`new_tasks.py`](experiments/revision/new_tasks.py) |

[实验指南](docs/EXPERIMENTS.md)说明了数据准备、基线、评估方式和已完成范围。[环境记录](requirements-reproduction.txt)列出了完整实验使用的依赖版本。早期实验代码仍保留在 `experiments/` 中，其设置与后续基准实验不同。

<details>
<summary><strong>批量运行实验</strong></summary>

按顺序运行一个任务族中纳入研究的全部配置：

```bash
python scripts/run_experiment.py \
  --all-retained --family mechanism --device cuda
```

启动器会跳过已有完整且匹配结果的运行。如果某次运行中途停止，请指定新的 `--work-dir` 重新开始，避免混合日志。标记为 `deferred` 的配置是在缩减计算预算时暂缓的实验。仍可通过 `--id` 运行它们，但不会改变仓库里已发布的结果记录。

</details>

## 论文与图表复现

使用仓库内的数据重新生成 loss 曲线和比较表：

```bash
python paper/scripts/reproduce_figures.py
python paper/scripts/reproduce_positive_tables.py
```

这两项无需训练、GPU 或网络。绘图脚本先检查输入文件是否改变，再导出 **PNG 和 PDF**。使用[记录的绘图环境](paper/scripts/requirements-figures.txt)可以得到与随包图片一致的输出。README 中的方法示意图另附[制作说明](assets/intuition_readme_prompt.txt)。

<details>
<summary><strong>本地编译或上传 Overleaf</strong></summary>

安装 TeX Live 和 `latexmk` 后执行：

```bash
python paper/scripts/build_paper.py
```

仓库提供 LaTeX 源码与 PNG 图表，编译后的 PDF 保存在本地，不提交到 Git。

将 `paper/` 的内容上传到 Overleaf，主文件选择 `main.tex`，编译器选择 **pdfLaTeX**。当前模板是基于 ICML 2026 样式制作的 ICML 2027 投稿草稿，并非官方 2027 模板。来源见[模板说明](paper/TEMPLATE_PROVENANCE.txt)。

</details>

## 仓库内容

<details>
<summary><strong>文件在哪里</strong></summary>

```text
assets/                    README 示意图与制作说明
experiments/               实验代码和优化器
  revision/                几何运算、实验配置和评估代码
scripts/                   实验启动与结果检查脚本
results/                   完成结果、暂缓记录和早期结果
paper/                     LaTeX 源码、证明、参考文献和图片
  figure_data/             论文绘图数据
  scripts/                 生成图片、表格和论文的代码
figures/                   早期绘图脚本
docs/                      实验指南和初始研究计划
tests/                     几何运算、数据划分和运行流程检查
```

仓库包含全部实验源码、保存的配置、已完成的结果记录和论文绘图数据。运行训练时，程序下载或准备数据集，并在本地保存日志和模型权重。[发布说明](docs/RELEASE.md)介绍了为方便其他环境运行而做的调整。

</details>
