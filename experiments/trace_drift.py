"""
TRACE: 定位 manifold residual 漂移放大的来源。

核心分离手段: 对**同一批 FP32 存储的点 x**, 同时计算 residual = |<x,x>_L + 1/K|:
  - r_fp32: 在 FP32 里算内积 (含 -x0^2+||space||^2 的抵消)
  - r_fp64: 把同样的 bits upcast 到 FP64 再算内积 (无抵消, 反映点真实偏离流形的程度)
  - r_recompute_x0: 用 project_to_manifold 重算 x0 (FP32) 后的残差 (纯存储地板)

判读:
  - 若 r_fp32 >> r_fp64  → 主要是"测量抵消地板", 点其实还在流形附近, 漂移被 FP32 内积放大了。
  - 若 r_fp32 ≈ r_fp64   → 点真的偏离流形 (Exp/PT 累积的真实几何漂移)。

同时输出 residual vs mean_radius / max_radius, 看 √K·R 机制。
无重投影 (纯漂移观测), 400 epoch。
"""
import math
import json
import torch
import lorentz as L
from data import make_toy_binary_tree
from optimizers import B5_HyperbolicMomentum

SEED = 42


def residual_fp32(x, K):
    return (L.lorentz_inner(x, x) + 1.0 / K).abs().max().item()


def residual_fp64(x, K):
    xd = x.double()
    return (L.lorentz_inner(xd, xd) + 1.0 / K).abs().max().item()


def residual_after_x0_reproj_fp32(x, K):
    xp = L.project_to_manifold(x, K)
    return (L.lorentz_inner(xp, xp) + 1.0 / K).abs().max().item()


def loss_and_grad(X, Dg, K, mask):
    Xv = X.detach().clone().requires_grad_(True)
    n = Xv.shape[0]
    xi = Xv.unsqueeze(1); xj = Xv.unsqueeze(0); diff = xi - xj
    sq = -diff[..., 0] ** 2 + (diff[..., 1:] ** 2).sum(-1)
    s = torch.clamp(0.5 * K * sq, min=0.0)
    dh = torch.log1p(s + torch.sqrt(s * s + 2.0 * s + 1e-12)) / math.sqrt(K)
    ratio = dh / (Dg + torch.eye(n))
    loss = (((ratio - 1.0) ** 2) * mask).sum() / mask.sum()
    loss.backward()
    return loss.item(), Xv.grad.detach().clone()


def curv_egrad(X, Dg, K, mask, eps=1e-3):
    def L_at(Kv):
        n = X.shape[0]
        xi = X.unsqueeze(1); xj = X.unsqueeze(0); diff = xi - xj
        sq = -diff[..., 0] ** 2 + (diff[..., 1:] ** 2).sum(-1)
        s = torch.clamp(0.5 * Kv * sq, min=0.0)
        dh = torch.log1p(s + torch.sqrt(s * s + 2.0 * s + 1e-12)) / math.sqrt(Kv)
        ratio = dh / (Dg + torch.eye(n))
        return (((ratio - 1.0) ** 2) * mask).sum().item() / mask.sum().item()
    return (L_at(K + eps) - L_at(K - eps)) / (2 * eps)


def main():
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    n, edges, parent, Dg = make_toy_binary_tree(depth=8)
    d, K_min = 10, 0.1
    c_init = L.coord_from_curvature(torch.tensor(1.0), torch.tensor(K_min)).item()
    opt = B5_HyperbolicMomentum(K_min, c_init, lr=100.0, beta=0.9, gamma=1.0,
                                rho=1e-3, delta_K=1e-2, lam_K=1e-4, K_ref=1.0)
    K = opt.K()
    o = L.origin(d, K).unsqueeze(0)
    X = L.exp_map(o.expand(n, d + 1).contiguous(),
                  torch.cat([torch.zeros(n, 1), torch.randn(n, d, generator=gen) * 1e-3], 1), K)
    mask = torch.triu(torch.ones(n, n), diagonal=1)

    rows = []
    for epoch in range(400):
        K = opt.K()
        loss, eg = loss_and_grad(X, Dg, K, mask)
        cg = curv_egrad(X, Dg, K, mask)
        X, info = opt.step(X, eg, cg)
        K = opt.K()
        with torch.no_grad():
            radii = L.dist(X, o.expand_as(X), K)
            r32 = residual_fp32(X, K)
            r64 = residual_fp64(X, K)
            rx0 = residual_after_x0_reproj_fp32(X, K)
        if epoch % 20 == 19 or epoch < 5:
            rows.append({
                "epoch": epoch, "mean_radius": radii.mean().item(),
                "max_radius": radii.max().item(), "K": K,
                "r_fp32": r32, "r_fp64": r64, "r_x0reproj_fp32": rx0,
                "loss": loss,
            })
            print(f"ep{epoch:3d} R_mean={radii.mean().item():.3f} R_max={radii.max().item():.3f} "
                  f"| r_fp32={r32:.2e} r_fp64={r64:.2e} r_x0reproj={rx0:.2e} "
                  f"ratio32/64={r32/max(r64,1e-30):.1f}")
    json.dump(rows, open("trace_drift.json", "w"), indent=2)
    print("\n=== 判读 ===")
    last = rows[-1]
    print(f"末端(ep{last['epoch']}) R_max={last['max_radius']:.2f}: "
          f"r_fp32={last['r_fp32']:.2e}, r_fp64={last['r_fp64']:.2e}, "
          f"x0重投影后={last['r_x0reproj_fp32']:.2e}")


if __name__ == "__main__":
    main()
