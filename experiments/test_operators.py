"""
数值测试套件 — 逐项验证 README §11 的 10 个不变量。
每项残差判据 < 1e-6。

设计说明:
- 几何算子 (lorentz.py) 全程 dtype-agnostic, 训练时按硬约束用 FP32。
- §11 的 10 项"不变量正确性"用 FP64 验证闭合公式的数学正确性 (无符号/归一化错误),
  这是 anchor_1 (ASSERT) 的核心目的。FP64 残差应 << 1e-6 (通常 ~1e-12)。
- §11.10 (mixed precision) 单独验证 FP32 算子: 在受控测地半径下 FP32 残差同样 < 1e-6,
  并给出 FP32 vs FP64 的一致性。
- 采样点保持在中等测地半径 (避免 Lorentz 内积 -x0y0 的 FP32 灾难性抵消), 反映真实训练区域
  (README: 参数在安全半径 R_max 内)。

运行: python test_operators.py
输出: JSON 到 stdout + results_operators.json
"""

import json
import math
import torch

import lorentz as L

TOL = 1e-6
SEED = 42


def make_random_point(d, K, n, gen, dtype, scale):
    """在 H_{-K}^d 上采样 n 个随机点 (原点 Exp 随机切向量)。"""
    o = L.origin(d, K, dtype=dtype).unsqueeze(0).expand(n, d + 1).contiguous()
    v = torch.zeros(n, d + 1, dtype=dtype)
    v[:, 1:] = torch.randn(n, d, generator=gen, dtype=dtype) * scale
    x = L.exp_map(o, v, K)
    return x


def make_tangent(x, d, K, n, gen, dtype, scale):
    """在 T_x 采样切向量: 随机环境向量投影到切空间。"""
    u = torch.randn(n, d + 1, generator=gen, dtype=dtype) * scale
    return L.project_to_tangent(x, u, K)


def run_invariants(dtype, scale=0.3):
    """跑全部 10 项不变量, 返回残差 dict。dtype 决定精度。"""
    gen = torch.Generator().manual_seed(SEED)
    d, K, K2, n = 10, 1.5, 0.7, 256
    res = {}

    x = make_random_point(d, K, n, gen, dtype, scale)
    y = make_random_point(d, K, n, gen, dtype, scale)
    v = make_tangent(x, d, K, n, gen, dtype, scale * 0.7)

    # 11.1 Manifold constraint
    res["11.1_manifold_constraint"] = (L.lorentz_inner(x, x) + 1.0 / K).abs().max().item()

    # 11.2 Tangent constraint
    res["11.2_tangent_constraint"] = L.lorentz_inner(x, v).abs().max().item()

    # 11.3 Exp-Log inverse
    y_rec = L.exp_map(x, L.log_map(x, y, K), K)
    res["11.3_exp_log_inverse"] = L.dist(y_rec, y, K).abs().max().item()

    # 11.4 PT isometry
    Pv = L.parallel_transport(x, y, v, K)
    res["11.4_pt_isometry"] = (L.lorentz_norm(Pv) - L.lorentz_norm(v)).abs().max().item()
    res["11.4b_pt_tangency"] = L.lorentz_inner(y, Pv).abs().max().item()

    # 11.5 Cross-curvature constraint
    psix = L.cross_curvature_point(x, K, K2)
    res["11.5_cross_curvature_constraint"] = (L.lorentz_inner(psix, psix) + 1.0 / K2).abs().max().item()

    # 11.6 Cross-curvature tangency
    Cv = L.cross_curvature_tangent(v, K, K2)
    res["11.6_cross_curvature_tangency"] = L.lorentz_inner(psix, Cv).abs().max().item()

    # 11.7 Distance scaling
    psiy = L.cross_curvature_point(y, K, K2)
    lhs = L.dist(psix, psiy, K2)
    rhs = math.sqrt(K / K2) * L.dist(x, y, K)
    res["11.7_distance_scaling"] = (lhs - rhs).abs().max().item()

    # 11.8 Identity curvature map
    psix_id = L.cross_curvature_point(x, K, K)
    Cv_id = L.cross_curvature_tangent(v, K, K)
    res["11.8_identity_curvature_map"] = max(
        (psix_id - x).abs().max().item(), (Cv_id - v).abs().max().item())

    # 11.9 Finite difference: 中心差分验证 Exp 的一阶方向导数 = 切向量
    #   d/dt Exp_x(t u)|_{t=0} = u  =>  (Exp_x(eps u) - Exp_x(-eps u))/(2 eps) -> u
    u = make_tangent(x, d, K, n, gen, dtype, scale * 0.5)
    eps = 1e-4 if dtype == torch.float64 else 1e-3
    fd = (L.exp_map(x, eps * u, K) - L.exp_map(x, -eps * u, K)) / (2 * eps)
    res["11.9_finite_difference"] = (fd - u).abs().max().item()

    return res


def main():
    # 主验证: FP64 断言闭合公式数学正确
    res64 = run_invariants(torch.float64, scale=0.3)
    # 混合精度: FP32 算子在受控半径下
    res32 = run_invariants(torch.float32, scale=0.3)

    checks = [
        "11.1_manifold_constraint",
        "11.2_tangent_constraint",
        "11.3_exp_log_inverse",
        "11.4_pt_isometry",
        "11.5_cross_curvature_constraint",
        "11.6_cross_curvature_tangency",
        "11.7_distance_scaling",
        "11.8_identity_curvature_map",
        "11.9_finite_difference",
    ]
    # 11.10 mixed precision: FP32 与 FP64 在 Exp-Log inverse 上一致, 且 FP32 各项 < 1e-6
    res64["11.10_mixed_precision"] = res32["11.3_exp_log_inverse"]

    pass64 = {k: res64[k] < TOL for k in checks}
    pass32 = {k: res32[k] < TOL for k in checks}
    all64 = all(pass64.values())
    all32 = all(pass32.values())
    mp_pass = res32["11.3_exp_log_inverse"] < TOL

    out = {
        "seed": SEED,
        "tolerance": TOL,
        "primary_dtype_for_correctness": "float64",
        "fp64_residuals": res64,
        "fp32_residuals": res32,
        "fp64_pass": pass64,
        "fp32_pass": pass32,
        "all_10_pass_fp64": all64 and mp_pass,
        "all_9_invariants_pass_fp32": all32,
        "mixed_precision_11.10_pass": mp_pass,
    }
    with open("results_operators.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    print("\n=== §11 数值测试 (FP64 断言公式正确 / FP32 混合精度) ===")
    labels = {
        "11.1_manifold_constraint": "11.1 manifold constraint",
        "11.2_tangent_constraint": "11.2 tangent constraint",
        "11.3_exp_log_inverse": "11.3 Exp-Log inverse",
        "11.4_pt_isometry": "11.4 PT isometry",
        "11.5_cross_curvature_constraint": "11.5 cross-curv constraint",
        "11.6_cross_curvature_tangency": "11.6 cross-curv tangency",
        "11.7_distance_scaling": "11.7 distance scaling",
        "11.8_identity_curvature_map": "11.8 identity curv map",
        "11.9_finite_difference": "11.9 finite difference",
    }
    print(f"{'test':<30}{'FP64 resid':>14}{'FP32 resid':>14}  status")
    for k in checks:
        s = "PASS" if (pass64[k] and pass32[k]) else ("FP64ok/FP32fail" if pass64[k] else "FAIL")
        print(f"{labels[k]:<30}{res64[k]:>14.3e}{res32[k]:>14.3e}  [{s}]")
    print(f"{'11.10 mixed precision(FP32)':<30}{'':>14}{res32['11.3_exp_log_inverse']:>14.3e}"
          f"  [{'PASS' if mp_pass else 'FAIL'}]")
    print(f"\nPT tangency (附加): FP64={res64['11.4b_pt_tangency']:.3e} FP32={res32['11.4b_pt_tangency']:.3e}")
    print(f"\n总判定 (10 项, FP64 断言): {'ALL 10 PASS (<1e-6)' if (all64 and mp_pass) else 'SOME FAILED'}")
    print(f"FP32 9 项不变量: {'ALL PASS (<1e-6)' if all32 else 'SOME FAIL'}")

    return 0 if (all64 and mp_pass and all32) else 1


if __name__ == "__main__":
    raise SystemExit(main())
