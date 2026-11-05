# HyperbolicMomentum

**可学习曲率、几何归一化步长，以及与曲率变化一致的动量映射。**

![方法直觉](assets/intuition.png)

本仓库包含全部原始及修订实验源码、优化器、消融、评估、完整论文、绘图代码与输入数据。注册配置、评估协议和完整记录均包含在相应目录中。

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

快速检查运行小型人工构造样例，无需下载数据；它带有 `fixture_only` 标记，不计入论文实验。正式实验仍使用清单中的完整轮数、随机种子、数据划分和选参规则。

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

全部代码均包含在仓库中。大型数据集、模型权重、控制台日志和完整逐步训练日志由运行脚本生成，不随源码分发；论文 loss 图所需的完整绘图数据已包含。
