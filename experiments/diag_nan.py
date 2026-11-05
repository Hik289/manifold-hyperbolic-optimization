"""
NaN/半径爆炸根因诊断 (ASSERT/TRACE)。

复现 mg_G4_var_d5 的 B3 (fixed K NaN) 和 B5shared (半径爆炸)，
逐步 instrument，捕获第一个非有限值出现在哪个量:
  - Exp arg (sqrt(K)*||v||): cosh/sinh 溢出?
  - lorentz_norm 内的 sqrt(负数)?
  - dist/log 的 arcosh arg < 1 域外?
  - loss 里 arcosh?
  - momentum ||m|| 发散 (optimizer diverge) vs 纯数值?

对每组每步记录: max Exp-arg, max ||m_tilde||, mc residual, Rmax。
第一个非有限出现即 dump 前 3 步的完整量 + 定位。
"""
import math, json
import numpy as np
import torch
import lorentz as L
from optimizers_mg import B3_MG, B5_MG
from train_multigroup import gen_subtree, graph_dist, group_loss

SEED = 42
REPROJ_EVERY = 50


def finite(t):
    if isinstance(t, torch.Tensor):
        return bool(torch.isfinite(t).all())
    return math.isfinite(t)


def run_diag(method, G, branchings, dim, npg, epochs, device):
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    d = dim
    Dgs, masks, eyes, ns = [], [], [], []
    for g in range(G):
        edges, n = gen_subtree(branchings[g], npg, seed=SEED + g)
        Dgs.append(graph_dist(n, edges, device))
        masks.append(torch.triu(torch.ones(n, n, device=device), diagonal=1))
        eyes.append(torch.eye(n, device=device))
        ns.append(n)

    K_min = 0.1
    c_init = L.coord_from_curvature(torch.tensor(1.0), torch.tensor(K_min)).item()
    if method == "B3":
        opt = B3_MG(G, 1.0, 100.0, 0.9)
    elif method == "B5shared":
        opt = B5_MG(G, K_min, c_init, 100.0, 0.9, 1.0, 1e-3, 1e-2, 1e-4, 1.0, shared_K=True)
    Ks = opt.K()
    Xs = []
    for g in range(G):
        o = L.origin(d, Ks[g], device=device).unsqueeze(0)
        X = L.exp_map(o.expand(ns[g], d + 1).contiguous(),
                      torch.cat([torch.zeros(ns[g], 1, device=device),
                                 torch.randn(ns[g], d, generator=gen).to(device) * 1e-3], 1), Ks[g])
        Xs.append(X)

    learnable = method in ("B5shared",)
    recent = []  # 最近几步的诊断
    for epoch in range(epochs):
        Ks = opt.K()
        egrads, curv_egrads = [], []
        step_diag = {"epoch": epoch, "Ks": list(Ks), "groups": []}
        nan_where = None
        for g in range(G):
            Xv = Xs[g].detach().clone().requires_grad_(True)
            if learnable:
                Kt = torch.tensor(float(Ks[g]), device=device, requires_grad=True)
                loss, dist = group_loss(Xv, Dgs[g], Kt, masks[g], eyes[g])
                if not finite(loss):
                    nan_where = f"g{g} LOSS non-finite (arcosh in group_loss?)"
                eg, gK = torch.autograd.grad(loss, [Xv, Kt])
                curv_egrads.append(gK.item())
            else:
                Kt = torch.tensor(float(Ks[g]), device=device)
                loss, dist = group_loss(Xv, Dgs[g], Kt, masks[g], eyes[g])
                if not finite(loss):
                    nan_where = f"g{g} LOSS non-finite"
                eg = torch.autograd.grad(loss, Xv)[0]
                curv_egrads.append(0.0)
            egrads.append(eg.detach())
            # 诊断量
            with torch.no_grad():
                gg = L.egrad_to_rgrad(Xs[g], eg.detach(), Ks[g])
                gnorm = L.lorentz_norm(gg).max().item()
                mc = (L.lorentz_inner(Xs[g], Xs[g]) + 1.0 / Ks[g]).abs().max().item()
                oK = L.origin(d, Ks[g], device=device).unsqueeze(0)
                # dist arg = -K<x,o>: 检查是否 <1
                alpha = (-Ks[g] * L.lorentz_inner(Xs[g], oK.expand_as(Xs[g]))).min().item()
            gd = {"g": g, "b": branchings[g], "loss": loss.item(), "dist": dist.item(),
                  "rgrad_max": gnorm, "mc": mc, "min_dist_alpha": alpha,
                  "egrad_finite": finite(eg), "loss_finite": finite(loss)}
            if hasattr(opt, "m") and opt.m[g] is not None:
                gd["m_norm"] = L.lorentz_norm(opt.m[g]).max().item()
                gd["m_finite"] = finite(opt.m[g])
            step_diag["groups"].append(gd)
            if not finite(gg):
                nan_where = nan_where or f"g{g} RGRAD non-finite (mc={mc:.2e} alpha={alpha:.4f})"
            if alpha < 1.0:
                nan_where = nan_where or f"g{g} dist arcosh arg alpha={alpha:.4f} < 1 (off-manifold, mc={mc:.2e})"

        recent.append(step_diag)
        if len(recent) > 4:
            recent.pop(0)

        if nan_where:
            print(f"\n!!! 第一个异常 @ep{epoch}: {nan_where}")
            print("=== 前 4 步诊断 ===")
            for sd in recent:
                print(f"ep{sd['epoch']} Ks={[round(k,4) for k in sd['Ks']]}")
                for gd in sd["groups"]:
                    print(f"   g{gd['g']}(b{gd['b']}): loss={gd['loss']:.4f} dist={gd['dist']:.4f} "
                          f"rgrad_max={gd['rgrad_max']:.3e} mc={gd['mc']:.3e} "
                          f"alpha={gd['min_dist_alpha']:.4f}"
                          + (f" m_norm={gd.get('m_norm',0):.3e}" if 'm_norm' in gd else ""))
            return {"nan_epoch": epoch, "where": nan_where, "recent": recent}

        Xs_new, infos = opt.step(Xs, egrads, curv_egrads if learnable else None)
        Ks_new = opt.K()
        # 重投影
        if (epoch + 1) % REPROJ_EVERY == 0:
            with torch.no_grad():
                for g in range(G):
                    Xs_new[g] = L.project_to_manifold(Xs_new[g], Ks_new[g])
                    if hasattr(opt, "m") and opt.m[g] is not None:
                        mr, _, _ = L.reproject_momentum(Xs_new[g], opt.m[g], Ks_new[g])
                        opt.m[g] = mr
        # 检查 step 后是否已非有限
        with torch.no_grad():
            for g in range(G):
                if not finite(Xs_new[g]):
                    oK = L.origin(d, Ks_new[g], device=device).unsqueeze(0)
                    print(f"\n!!! step 后 X 非有限 @ep{epoch} g{g}")
                    return {"nan_epoch": epoch, "where": f"g{g} X after step non-finite"}
                rmax = L.dist(Xs_new[g], L.origin(d, Ks_new[g], device=device).unsqueeze(0).expand_as(Xs_new[g]), Ks_new[g]).max().item()
                if rmax > 20:
                    print(f"\n!!! 半径爆炸 @ep{epoch} g{g}(b{branchings[g]}): Rmax={rmax:.2f}")
                    mnorm = L.lorentz_norm(opt.m[g]).max().item() if hasattr(opt,'m') and opt.m[g] is not None else 0
                    print(f"   同步 m_norm={mnorm:.3e} (发散? vs 数值)")
                    print("=== 前几步 ===")
                    for sd in recent[-3:]:
                        for gd in sd["groups"]:
                            if gd["g"] == g:
                                print(f"   ep{sd['epoch']} rgrad_max={gd['rgrad_max']:.3e} "
                                      f"mc={gd['mc']:.3e}" + (f" m_norm={gd.get('m_norm',0):.3e}" if 'm_norm' in gd else ""))
                    return {"radius_explosion_epoch": epoch, "group": g, "rmax": rmax}
        Xs = Xs_new
        if epoch % 200 == 0:
            print(f"ep{epoch} OK Ks={[round(k,4) for k in Ks_new]} "
                  f"maxRmax={max(L.dist(Xs[g], L.origin(d,Ks_new[g],device=device).unsqueeze(0).expand_as(Xs[g]), Ks_new[g]).max().item() for g in range(G)):.2f}", flush=True)
    print("\n跑完无异常")
    return {"status": "no_nan"}


if __name__ == "__main__":
    import sys
    dev = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    method = sys.argv[1] if len(sys.argv) > 1 else "B3"
    print(f"=== 诊断 {method} (G4 var 2,4,6,8 dim5 npg2000) ===")
    r = run_diag(method, 4, [2, 4, 6, 8], 5, 2000, 2000, dev)
    json.dump(r, open(f"diag_nan_{method}.json", "w"), indent=2, default=str)
