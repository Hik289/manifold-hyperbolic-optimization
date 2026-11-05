# KG G1 Main Table 草稿 (供 paper_writer)

> 两个 KG 数据集: 可学分组曲率 (B4) vs 固定曲率 (B3)。
> 配置统一: Lorentz distance model, per-relation 独立 K_r, K_max=2, dim20, Adam lr=3e-3,
> ρ=1e-2, λ_K=1e-2, M1 每步投影, seed=42, 2000ep。
> 指标 = 末端 8-eval 移动平均 (压 epoch 间震荡)。

## Table: Learnable grouped curvature vs fixed curvature on KG link prediction

| Dataset | 层级性 | Method | MRR | Hits@1 | Hits@10 | Train loss | K_std |
|---|---|---|---:|---:|---:|---:|---:|
| WN18RR | 强 (taxonomic) | B3 (fixed K) | 0.1320 | 0.0278 | 0.3477 | 2.127 | 0 (fixed) |
| WN18RR | 强 | **B4 (learned K)** | **0.1350** | 0.0272 | **0.3672** | **2.068** | 0.295 |
| | | Δ (rel) | **+2.3%** | −2.2% | **+5.6%** | **−2.8%** | — |
| FB15k-237 | 弱 (complex) | B3 (fixed K) | 0.1853 | 0.1070 | 0.3523 | 2.207 | 0 (fixed) |
| FB15k-237 | 弱 | **B4 (learned K)** | **0.1893** | **0.1103** | 0.3528 | **2.187** | 0.071 |
| | | Δ (rel) | **+2.2%** | **+3.1%** | +0.1% | −0.9% | — |

## 跨数据集诚实 claim (措辞校准, Director 定)

**核心 (可辩护)**: 可学分组曲率在两个 KG 上都提升 ranking 质量，**MRR 一致 +2.2~2.3%**。

**指标画像因数据集而异 (不能混成统一数字)**:
- **WN18RR (强层级)**: 稳健信号 = **Hits@10 +5.6% + loss −2.8%**；MRR +2.3%；Hits@1 平/略负。
- **FB15k-237 (弱层级)**: 稳健信号 = **MRR +2.2% + Hits@1 +3.1%**；Hits@10 平 (+0.1%)；loss 增益弱 (−0.9%)。
- **唯一两数据集都正向的指标 = MRR**。Hits@10 在 WN 强、FB 平；Hits@1 在 FB 强、WN 平。
- ⚠️ 不要写"Hits@10 一致 +4%" (FB 上 H@10 是平的，那是错的)。各数据集报各自稳健指标 + 共同 MRR。

## G3 机理对照 (曲率分化强度随层级性缩放)

| Dataset | K_std | K_range | 分化驱动因素 | top-K 关系 |
|---|---:|---|---|---|
| WN18RR | 0.295 | [1.01, 2.00] | 清晰分类层级 | _derivationally_related_form (2.00), _member_meronym (1.58), _hypernym (1.42) |
| FB15k-237 | 0.071 | [0.96, 1.84] | 关系复杂度/基数 | award_nomination (1.84), film_release (1.35), award_won (1.35), profession/genre (~1.18) |

- **K_std WN 0.30 vs FB 0.07 (4×)**: 曲率分化强度随数据集层级性强度缩放 → 支持 H0 机理。
- **分化来源不同但都符合"复杂/层级关系要更负曲率"**: WN 由 taxonomic hypernym/meronym 驱动；FB 由高频多对多复合关系 (award/film/genre) 驱动，清晰层级如 location/contains 反而近平 (K=0.985)。
- **诚实 nuance**: "曲率按关系语义分化"在 WN 是清晰层级驱动，在 FB 是复杂度驱动。跨数据集统一说法 = "曲率捕捉关系的几何复杂度需求，分化强度随层级性缩放"，不是"总按 taxonomic 层级分化"。

## 数值可信度 (纪律关)
- 两数据集 B4 曲率均收敛 plateau (WN K_std 0.28→0.295, FB 0.070→0.071)，K rail 到 K_max=2 内，全程无 NaN。
- gap 用 8-eval 移动平均，非单点终值 (压 MRR 震荡)。
- gap 收敛后维持 (非中途数)，区别于 synthetic 上 B3 长跑追平 (那是 G8 收敛速度型)。
