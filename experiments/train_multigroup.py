"""
Multi-group synthetic hierarchy 训练 — 检验 H0 / Corollary 2.1。

构造曲率异质性: G 个子树, 每组不同分支率 b_g → 每组不同容量匹配曲率 K_cap。
每组独立状态 (X_g, m_g, c_g), 独立六步更新。loss = sum_g 组内 distortion。

对比:
  B3_MG: 全局固定单一 K (H0 主对比 baseline)
  B4_MG: 每组学 K, 只映射点 (G4 对照)
  B5_MG: 每组学 K + PT 动量 + 点&动量跨曲率 (主算法)
  B5_MG(shared_K): A3 消融, 单一共享可学习 K

H0 判据: B5 vs B3 vs B4 total distortion + learned K 分组散度 std(K_g)。

seed=42, 2000ep, 24列/组轨迹 + 每组 K, radius 监控。
用法:
  python train_multigroup.py --method B5 --G 4 --branchings 2 4 6 8 --dim 5 \
     --nodes_per_group 2000 --epochs 2000 --out runs/mg_G4_var_d5_B5
"""
import argparse, json, math, time, csv, os
import numpy as np
import torch

import lorentz as L
from optimizers_mg import B3_MG, B4_MG, B5_MG

SEED = 42
REPROJ_EVERY = 50


def gen_subtree(branching, target_nodes, seed):
    import random
    random.seed(seed)
    edges = []; frontier = [0]; n = 1
    while n < target_nodes and frontier:
        nf = []
        for p in frontier:
            if n >= target_nodes:
                break
            for _ in range(branching):
                edges.append((p, n)); nf.append(n); n += 1
                if n >= target_nodes:
                    break
        frontier = nf
    return edges, n


def graph_dist(n, edges, device):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    if edges:
        r = np.array([a for a, b in edges] + [b for a, b in edges])
        c = np.array([b for a, b in edges] + [a for a, b in edges])
        A = csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n))
    else:
        A = csr_matrix((n, n))
    D = shortest_path(A, method="D", unweighted=True)
    return torch.from_numpy(D.astype("float32")).to(device)


def group_loss(X, Dg, K, mask, eye):
    n = X.shape[0]
    xi = X.unsqueeze(1); xj = X.unsqueeze(0); diff = xi - xj
    sq = -diff[..., 0] ** 2 + (diff[..., 1:] ** 2).sum(-1)
    s = torch.clamp(0.5 * K * sq, min=0.0)
    dh = torch.log1p(s + torch.sqrt(s * s + 2.0 * s + 1e-12)) / torch.sqrt(K)
    ratio = dh / (Dg + eye)
    loss = (((ratio - 1.0) ** 2) * mask).sum() / mask.sum()
    with torch.no_grad():
        dist = ((ratio - 1.0).abs() * mask).sum() / mask.sum()
    return loss, dist


def pick_gpu():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best, bu = 0, 1e18
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        if total - free < bu:
            bu, best = total - free, i
    return torch.device(f"cuda:{best}")


def make_opt(method, G, args):
    K_min = 0.1
    c_init = L.coord_from_curvature(torch.tensor(args.init_K), torch.tensor(K_min)).item()
    if method == "B3":
        return B3_MG(G, args.init_K, args.lr, args.beta)
    if method == "B3.5":
        # 固定曲率 + 曲率归一化测地步长但不学曲率 (隔离归一化步长机制)
        return B3_MG(G, args.init_K, args.lr, args.beta, geo_step=True, gamma=args.gamma)
    if method == "B4":
        return B4_MG(G, K_min, c_init, args.lr, args.rho, args.delta_K, args.lam_K, args.init_K)
    if method == "B5":
        return B5_MG(G, K_min, c_init, args.lr, args.beta, args.gamma, args.rho,
                     args.delta_K, args.lam_K, args.init_K, shared_K=False, mom_transport=True)
    if method == "B5_noMomT":
        # A5 消融: 有动量但曲率变时只映射点、动量不跨曲率映射 (G4 干净对照)
        return B5_MG(G, K_min, c_init, args.lr, args.beta, args.gamma, args.rho,
                     args.delta_K, args.lam_K, args.init_K, shared_K=False, mom_transport=False)
    if method == "B5shared":
        return B5_MG(G, K_min, c_init, args.lr, args.beta, args.gamma, args.rho,
                     args.delta_K, args.lam_K, args.init_K, shared_K=True)
    raise ValueError(method)


def run(args):
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    device = torch.device(args.device) if args.device else pick_gpu()
    G = args.G
    branchings = args.branchings if args.branchings else [2, 4, 6, 8][:G]
    assert len(branchings) == G, "branchings 数量须等于 G"
    # 每组维度: --dims 优先 (per-group), 否则统一 --dim
    if args.dims:
        assert len(args.dims) == G, "dims 数量须等于 G"
        ds = list(args.dims)
    else:
        ds = [args.dim] * G
    d = args.dim  # 兼容 (轨迹列等用统一名; 实际逐组用 ds[g])

    # 每组独立子树 + 图距离
    Dgs, masks, eyes, ns = [], [], [], []
    for g in range(G):
        edges, n = gen_subtree(branchings[g], args.nodes_per_group, seed=SEED + g)
        Dgs.append(graph_dist(n, edges, device))
        masks.append(torch.triu(torch.ones(n, n, device=device), diagonal=1))
        eyes.append(torch.eye(n, device=device))
        ns.append(n)

    opt = make_opt(args.method, G, args)
    Ks = opt.K()
    # 每组初始化 embedding (逐组 dim)
    Xs = []
    for g in range(G):
        dg = ds[g]
        o = L.origin(dg, Ks[g], device=device).unsqueeze(0)
        X = L.exp_map(o.expand(ns[g], dg + 1).contiguous(),
                      torch.cat([torch.zeros(ns[g], 1, device=device),
                                 torch.randn(ns[g], dg, generator=gen).to(device) * 1e-3], 1), Ks[g])
        Xs.append(X)

    os.makedirs(args.out, exist_ok=True)
    # 每组一份轨迹 + 一份汇总(含 total distortion + K 散度)
    gwriters, gfiles = [], []
    base_cols = ["epoch", "group", "branching", "train_loss", "distortion", "curvature",
                 "curv_grad", "log_curv_step", "mean_radius", "max_radius",
                 "mom_norm", "tangent_residual", "manifold_res",
                 "reproj_event", "reproj_manifold_pre", "wall_clock"]
    tf = open(os.path.join(args.out, f"traj_{args.method}_seed{SEED}.csv"), "w", newline="")
    gw = csv.writer(tf); gw.writerow(base_cols)
    sf = open(os.path.join(args.out, f"summary_{args.method}_seed{SEED}.csv"), "w", newline="")
    sw = csv.writer(sf)
    sw.writerow(["epoch", "total_distortion", "total_loss", "K_mean", "K_std",
                 "max_radius_all", "max_manifold_res"] +
                [f"K_g{g}" for g in range(G)])

    learnable = args.method in ("B4", "B5", "B5_noMomT", "B5shared")
    t0 = time.time()
    RADIUS_ABORT = args.radius_abort
    cur_branchings = list(branchings)

    # ⭐ 平滑连续曲率漂移 (G4 round2): 固定树结构, 每组图距离乘平滑时变因子 s_g(t)。
    # D_t = D_base · s_g(t), s_g(t) 每组不同速率线性 ramp (正比缩放保三角不等式=合法度量)。
    # 距离缩放 → 最优 K 平滑连续移动 (K_cap ∝ 1/Δr²) 贯穿全程。测 B5(一致transport动量)
    # vs A5(每步损坏动量) 的累积追踪差异。突变(shift_epoch)是 round1; 漂移是 round2。
    Dgs_base = [D.clone() for D in Dgs]
    drift_on = args.drift_scale_end != 1.0 and args.drift_epochs > 0
    # 每组漂移速率不同 (制造异质漂移): 组 g 的终点缩放在 [1, drift_scale_end] 线性分布
    if drift_on:
        drift_ends = [1.0 + (args.drift_scale_end - 1.0) * (g + 1) / G for g in range(G)]

    for epoch in range(args.epochs):
        # 平滑漂移: 每 epoch 更新各组距离缩放
        if drift_on:
            alpha = min(1.0, epoch / args.drift_epochs)          # 0->1 贯穿 drift_epochs
            for g in range(G):
                s_g = 1.0 + (drift_ends[g] - 1.0) * alpha         # 平滑 ramp
                Dgs[g] = Dgs_base[g] * s_g
        # ⭐ phase-shift (G4 INTERVENE): 在 epoch T 突变各组分支率, 强迫最优曲率 shift。
        # 图距离矩阵重建 (节点数不变, 结构变), embedding/动量/曲率状态保留 → 迫使
        # 已收敛的曲率+动量适应新目标。B5(跨曲率映射动量) vs A5(只映射点) 恢复对比。
        if args.shift_epoch > 0 and epoch == args.shift_epoch:
            new_br = args.shift_branchings if args.shift_branchings else list(reversed(branchings))
            print(f"[PHASE-SHIFT] epoch {epoch}: branchings {cur_branchings} -> {new_br}", flush=True)
            for g in range(G):
                edges, n = gen_subtree(new_br[g], args.nodes_per_group, seed=SEED + g + 1000)
                Dgs[g] = graph_dist(n, edges, device)
                # 节点数应一致 (nodes_per_group 固定); 若变则重建 mask/eye
                if n != ns[g]:
                    masks[g] = torch.triu(torch.ones(n, n, device=device), diagonal=1)
                    eyes[g] = torch.eye(n, device=device)
                    ns[g] = n
            cur_branchings = list(new_br)
        Ks = opt.K()
        egrads, curv_egrads, losses, dists = [], [], [], []
        for g in range(G):
            Xv = Xs[g].detach().clone().requires_grad_(True)
            if learnable:
                Kt = torch.tensor(float(Ks[g]), device=device, requires_grad=True)
                loss, dist = group_loss(Xv, Dgs[g], Kt, masks[g], eyes[g])
                eg, gK = torch.autograd.grad(loss, [Xv, Kt])
                curv_egrads.append(gK.item())
            else:
                Kt = torch.tensor(float(Ks[g]), device=device)
                loss, dist = group_loss(Xv, Dgs[g], Kt, masks[g], eyes[g])
                eg = torch.autograd.grad(loss, Xv)[0]
                curv_egrads.append(0.0)
            egrads.append(eg.detach())
            losses.append(loss.item()); dists.append(dist.item())

        Xs_new, infos = opt.step(Xs, egrads, curv_egrads if learnable else None)
        Ks_new = opt.K()

        # 重投影护栏 (每组)。--per_step_reproj: 每步轻量投影(M1, 深子树大半径必需,
        # 断 FP32 单步复利防几何破裂); 否则每 REPROJ_EVERY 触发。
        reproj_event = 0; reproj_pre = [0.0] * G
        do_reproj = args.per_step_reproj or ((epoch + 1) % REPROJ_EVERY == 0)
        if do_reproj:
            reproj_event = 1
            with torch.no_grad():
                for g in range(G):
                    reproj_pre[g] = L.manifold_residual(Xs_new[g], Ks_new[g])
                    Xs_new[g] = L.project_to_manifold(Xs_new[g], Ks_new[g])
                    if hasattr(opt, "m") and opt.m[g] is not None:
                        mr, _, _ = L.reproject_momentum(Xs_new[g], opt.m[g], Ks_new[g])
                        opt.m[g] = mr

        # 诊断 + 落盘
        max_r_all = 0.0; max_mc = 0.0
        with torch.no_grad():
            for g in range(G):
                oK = L.origin(ds[g], Ks_new[g], device=device).unsqueeze(0)
                radii = L.dist(Xs_new[g], oK.expand_as(Xs_new[g]), Ks_new[g])
                mr, xr = radii.mean().item(), radii.max().item()
                mc = (L.lorentz_inner(Xs_new[g], Xs_new[g]) + 1.0 / Ks_new[g]).abs().max().item()
                max_r_all = max(max_r_all, xr); max_mc = max(max_mc, mc)
                gw.writerow([epoch, g, branchings[g], losses[g], dists[g], Ks_new[g],
                             infos[g].get("curv_grad", 0.0), infos[g].get("log_curv_step", 0.0),
                             mr, xr, infos[g].get("mom_norm", 0.0),
                             infos[g].get("tangent_residual", 0.0), mc,
                             reproj_event, reproj_pre[g], time.time() - t0])
        total_dist = sum(dists) / G
        total_loss = sum(losses) / G
        Karr = np.array(Ks_new)
        sw.writerow([epoch, total_dist, total_loss, float(Karr.mean()), float(Karr.std()),
                     max_r_all, max_mc] + list(Ks_new))
        Xs = Xs_new

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            tf.flush(); sf.flush()
            print(f"[{args.method} G{G}] ep{epoch:4d} total_dist={total_dist:.4f} "
                  f"K_mean={Karr.mean():.4f} K_std={Karr.std():.4f} "
                  f"Ks={[round(k,3) for k in Ks_new]} Rmax={max_r_all:.2f} mc={max_mc:.1e}",
                  flush=True)

        if max_r_all > RADIUS_ABORT:
            tf.close(); sf.close()
            msg = f"RADIUS_EXPLOSION ep{epoch} max_r={max_r_all:.2f} > {RADIUS_ABORT}"
            print("!!! " + msg, flush=True)
            json.dump({"aborted": True, "reason": msg}, open(os.path.join(args.out, "ABORT.json"), "w"))
            return {"status": "ABORTED", "epoch": epoch}
    tf.close(); sf.close()

    Karr = np.array(Ks_new)
    summary = {
        "status": "OK", "method": args.method, "G": G, "branchings": branchings,
        "dim": d, "nodes_per_group": args.nodes_per_group, "epochs": args.epochs,
        "final_total_distortion": total_dist, "final_total_loss": total_loss,
        "learned_K": list(Ks_new), "K_mean": float(Karr.mean()), "K_std": float(Karr.std()),
        "final_max_radius": max_r_all, "max_manifold_res": max_mc, "seed": SEED,
    }
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="B5", choices=["B3", "B3.5", "B4", "B5", "B5_noMomT", "B5shared"])
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--branchings", type=int, nargs="+", default=None)
    ap.add_argument("--dim", type=int, default=5)
    ap.add_argument("--dims", type=int, nargs="+", default=None,
                    help="每组独立 dim (数量=G); 覆盖 --dim")
    ap.add_argument("--nodes_per_group", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=100.0)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=1e-3)
    ap.add_argument("--delta_K", type=float, default=1e-2)
    ap.add_argument("--lam_K", type=float, default=1e-4)
    ap.add_argument("--init_K", type=float, default=1.0)
    ap.add_argument("--radius_abort", type=float, default=20.0)
    ap.add_argument("--shift_epoch", type=int, default=0,
                    help="G4 phase-shift(突变): 在此 epoch 突变分支率 (0=不 shift)")
    ap.add_argument("--shift_branchings", type=int, nargs="+", default=None,
                    help="shift 后的分支率 (默认 reversed(branchings))")
    ap.add_argument("--drift_scale_end", type=float, default=1.0,
                    help="G4 平滑漂移(round2): 距离缩放终点 (1.0=不漂移; 如 2.0=距离翻倍)")
    ap.add_argument("--drift_epochs", type=int, default=0,
                    help="平滑漂移贯穿的 epoch 数 (alpha 0->1)")
    ap.add_argument("--per_step_reproj", action="store_true",
                    help="每步轻量投影(M1); 深子树大半径防 FP32 破裂必需")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/mg_run")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
