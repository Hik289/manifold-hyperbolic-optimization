# HyperbolicMomentum

**可学习曲率、几何归一化步长，以及与曲率变化一致的动量映射。**

[English](README.md) · [论文 PDF](paper/main.pdf) · [实验说明](docs/EXPERIMENTS.md) · [全部结果](results/README.md)

![方法直觉](assets/intuition.png)

本仓库包含全部原始及修订实验源码、优化器、消融、评估、完整论文、绘图代码与输入数据。原始清单注册了 **893 组**配置；修订预算保留的 **705 组已全部完成**，另有 **188 组明确暂缓**。暂缓配置没有被当作完成结果或数值失败。

## 安装与快速验证

建议使用 Python 3.11，在仓库根目录运行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/revision/validate_geometry.py
python scripts/run_experiment.py --smoke --device cpu
python scripts/summarize_results.py
```

快速检查只运行三步人工构造样例，无需下载数据；它带有 `fixture_only` 标记，不计入论文实验。正式实验仍使用清单中的完整轮数、seed 42、数据划分和选参规则。

## 正式实验

```bash
python -m pip install -r requirements-full.txt
python scripts/run_experiment.py --list --family mechanism
python scripts/run_experiment.py \
  --id mechanism_smooth8_independent_lorentz_3_B5 \
  --device cuda --work-dir artifacts/mechanism
```

数据默认放在 `data/`，也可以通过 `--data-root` 或 `MANIFOLD_DATA` 指定。所有路径均可本地配置，无需原作者机器或私人服务。数据准备、各任务入口和延迟测试评估见 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 论文与图表

```bash
python paper/scripts/reproduce_figures.py
python paper/scripts/reproduce_positive_tables.py
python paper/scripts/build_paper.py
```

前两项无需训练或 GPU；最后一项需要本地 TeX Live 和 `latexmk`。上传 `paper/` 内容到 Overleaf 后，主文件选择 `main.tex`、编译器选择 pdfLaTeX。当前模板是基于 ICML 2026 的本地 ICML 2027 草稿封装，未声称是官方 2027 模板。

匹配五种初始化、按验证集选参的机制实验中，静态、2× 漂移和 8× 漂移条件相对 B35 的失真分别下降 **2.94%、3.31%、5.02%**。这些是有明确比较条件的单种子数值结果；全部持平、负面和更强对照也保留在论文附录与结果目录中。

全部代码均包含在仓库中。大型数据集、模型权重、控制台日志和完整逐步训练日志由运行脚本生成，不随源码分发；论文中三张 loss 图所需的完整绘图数据已包含。
