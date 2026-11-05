"""
测试两个缓解方案对 pre-reproj manifold residual 的效果 (B5, 400ep, seed42):
  M0: baseline (每 50ep 重投影, 当前 train.py)
  M1: 每步轻量投影 (每步 project_to_manifold 重算 x0, O(d))
  M2: 每步轻量投影 + 几何算子 FP64 内部累加 (Exp/PT/inner 用 double 再回 FP32 存储)

输出每个方案的: max residual (整条轨迹的重投影前峰值), residual vs epoch 是否单调爆炸。
"""
import math, json
import torch
import lorentz as L
from data import make_toy_binary_tree
from optimizers import B5_HyperbolicMomentum

SEED = 42


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


def resid(X, K):
    return (L.lorentz_inner(X, X) + 1.0 / K).abs().max().item()


def run(mode, epochs=400):
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

    fp64_ops = (mode == "M2")
    per_step_proj = (mode in ("M1", "M2"))

    max_pre = 0.0
    series = []
    for epoch in range(epochs):
        K = opt.K()
        if fp64_ops:
            # 几何步在 FP64 内部做: 把 X 升到 double, step 后回 float
            Xd = X.double()
            loss, eg = loss_and_grad(Xd, Dg.double(), K, mask.double())
            cg = curv_egrad(Xd, Dg.double(), K, mask.double())
            Xd_new, info = opt.step(Xd, eg, cg)
            X = Xd_new.float()
        else:
            loss, eg = loss_and_grad(X, Dg, K, mask)
            cg = curv_egrad(X, Dg, K, mask)
            X, info = opt.step(X, eg, cg)
        K = opt.K()

        # pre-reproj residual (每步都测, 记峰值)
        with torch.no_grad():
            r_pre = resid(X, K)
            max_pre = max(max_pre, r_pre)
            if per_step_proj:
                X = L.project_to_manifold(X.double(), K).float() if fp64_ops \
                    else L.project_to_manifold(X, K)
                if opt.m is not None:
                    mr, _, _ = L.reproject_momentum(X, opt.m, K)
                    opt.m = mr
            elif (epoch + 1) % 50 == 0:
                X = L.project_to_manifold(X, K)
                if opt.m is not None:
                    mr, _, _ = L.reproject_momentum(X, opt.m, K)
                    opt.m = mr
        if epoch % 40 == 39:
            with torch.no_grad():
                radii = L.dist(X, o.expand_as(X).to(X.dtype) if False else o.expand_as(X), K)
            series.append({"epoch": epoch, "r_pre": r_pre,
                           "max_radius": radii.max().item(), "loss": loss})
    return {"mode": mode, "max_pre_reproj_residual": max_pre,
            "final_loss": series[-1]["loss"], "series": series}


def main():
    out = {}
    for mode in ["M0", "M1", "M2"]:
        r = run(mode)
        out[mode] = r
        print(f"\n=== {mode} ===  max_pre_reproj_residual = {r['max_pre_reproj_residual']:.3e}  "
              f"final_loss={r['final_loss']:.5f}")
        for s in r["series"][::2]:
            print(f"  ep{s['epoch']:3d} r_pre={s['r_pre']:.2e} R_max={s['max_radius']:.2f} loss={s['loss']:.4f}")
    json.dump(out, open("test_mitigation.json", "w"), indent=2)
    print("\n=== 汇总 (max pre-reproj residual) ===")
    for m in ["M0", "M1", "M2"]:
        print(f"  {m}: {out[m]['max_pre_reproj_residual']:.3e}  (final_loss {out[m]['final_loss']:.5f})")


if __name__ == "__main__":
    main()
