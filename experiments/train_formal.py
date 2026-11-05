"""
正式 synthetic hierarchy 2000-epoch 训练脚本 (GPU, autograd 曲率梯度)。

相对 train.py 的改进:
- GPU 支持 (自动选空闲卡, 或 CUDA_VISIBLE_DEVICES 指定)
- 曲率梯度用 autograd (∂loss/∂K), 不再每步 2× O(n²) 有限差 → 可跑 10K 节点 2000ep
- 配置化: --tree {binary,4ary,8ary,variable} --nodes --dim
- REPROJ_EVERY=50 (Director 裁决)
- 24 列轨迹 (§5.3 20 列 + epoch 索引 + 3 重投影列)
- ⭐ radius 爆炸监控 (§12.2 failure criterion 6): mean/max radius 若持续单调增长破阈值 → abort 报警

复用 lorentz.py 的几何算子 + optimizers.py 的 B1..B5。
硬约束: seed=42, FP32 几何, 无 early stopping, 每 epoch 落盘。

用法:
  python train_formal.py --method B5 --tree binary --nodes 10000 --dim 10 --epochs 2000 \
      --out runs/binary_10k_d10_B5
"""
import argparse, json, math, time, csv, os
import torch
import torch.nn.functional as F

import lorentz as L

SEED = 42
REPROJ_EVERY = 50
PERM_CKPT = set([0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] + list(range(1100, 100000, 100)))

TRAJ_COLUMNS = [
    "epoch", "train_loss", "val_metric", "test_metric", "learning_rate",
    "curvature", "curvature_grad", "log_curvature_step", "geodesic_radius",
    "max_radius", "riemannian_grad_norm", "momentum_norm",
    "transported_momentum_norm", "tangent_residual", "manifold_constraint_residual",
    "parallel_transport_norm_error", "exp_map_argument", "curvature_adjusted_lr",
    "clipping_events", "wall_clock", "peak_memory_mb",
    "reproj_event", "reproj_manifold_residual_pre", "reproj_momentum_tangent_pre",
]


# ---------------------------------------------------------------------------
# 数据: 树生成 + 图距离 (GPU)
# ---------------------------------------------------------------------------
def gen_tree(kind, target_nodes):
    """按类型生成树 (BFS 编号), 逼近 target_nodes。返回 edges, parent, n。"""
    import random
    random.seed(SEED)
    edges = []
    parent = [-1]
    frontier = [0]
    n = 1
    if kind == "binary":
        bfun = lambda: 2
    elif kind == "4ary":
        bfun = lambda: 4
    elif kind == "8ary":
        bfun = lambda: 8
    elif kind == "variable":
        bfun = lambda: random.choice([2, 3, 4, 5])
    else:
        raise ValueError(kind)
    while n < target_nodes and frontier:
        new_frontier = []
        for p in frontier:
            if n >= target_nodes:
                break
            for _ in range(bfun()):
                c = n
                edges.append((p, c)); parent.append(p); new_frontier.append(c); n += 1
                if n >= target_nodes:
                    break
        frontier = new_frontier
    return edges, parent, n


def graph_dist_matrix(n, edges, device):
    """全源最短路 (树, 无权)。scipy 稀疏 BFS (C 优化), 返回 [n,n] float32 on device。"""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    if len(edges) > 0:
        rows = np.array([a for a, b in edges] + [b for a, b in edges])
        cols = np.array([b for a, b in edges] + [a for a, b in edges])
        data = np.ones(len(rows), dtype=np.float32)
        A = csr_matrix((data, (rows, cols)), shape=(n, n))
    else:
        A = csr_matrix((n, n))
    D = shortest_path(A, method="D", unweighted=True)  # [n,n] float64
    return torch.from_numpy(D.astype("float32")).to(device)


# ---------------------------------------------------------------------------
# 损失: pairwise distortion (可选分块, 大 n 省显存)
# ---------------------------------------------------------------------------
def distortion_loss(X, Dg, K, mask, eye):
    """loss = mean_{i<j} (d_hyp/d_graph - 1)^2。X requires_grad, K requires_grad。"""
    n = X.shape[0]
    xi = X.unsqueeze(1); xj = X.unsqueeze(0); diff = xi - xj
    sq = -diff[..., 0] ** 2 + (diff[..., 1:] ** 2).sum(-1)
    s = torch.clamp(0.5 * K * sq, min=0.0)
    dh = torch.log1p(s + torch.sqrt(s * s + 2.0 * s + 1e-12)) / torch.sqrt(K)
    ratio = dh / (Dg + eye)
    resid = ((ratio - 1.0) ** 2) * mask
    loss = resid.sum() / mask.sum()
    with torch.no_grad():
        distortion = ((ratio - 1.0).abs() * mask).sum() / mask.sum()
    return loss, distortion


def make_optimizer(method, K_init, args):
    from optimizers import (B1_Euclidean, B2_RSGD, B3_RiemannianMomentum,
                            B4_LearnableCurvatureRSGD, B5_HyperbolicMomentum)
    K_min = 0.1
    c_init = L.coord_from_curvature(torch.tensor(K_init), torch.tensor(K_min)).item()
    if method == "B1":
        return B1_Euclidean(K_init, args.lr, args.beta)
    if method == "B2":
        return B2_RSGD(K_init, args.lr)
    if method == "B3":
        return B3_RiemannianMomentum(K_init, args.lr, args.beta)
    if method == "B4":
        return B4_LearnableCurvatureRSGD(K_min, c_init, args.lr, args.rho,
                                         args.delta_K, args.lam_K, K_init)
    if method == "B5":
        return B5_HyperbolicMomentum(K_min, c_init, args.lr, args.beta, args.gamma,
                                     args.rho, args.delta_K, args.lam_K, K_init)
    raise ValueError(method)


def pick_gpu():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best, best_used = 0, 1e18
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        used = total - free
        if used < best_used:
            best_used, best = used, i
    return torch.device(f"cuda:{best}")


def run(args):
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    device = torch.device(args.device) if args.device else pick_gpu()

    edges, parent, n = gen_tree(args.tree, args.nodes)
    Dg = graph_dist_matrix(n, edges, device)
    d = args.dim
    mask = torch.triu(torch.ones(n, n, device=device), diagonal=1)
    eye = torch.eye(n, device=device)

    opt = make_optimizer(args.method, args.init_K, args)
    K_cur = opt.K() if hasattr(opt, "K") else args.init_K

    o = L.origin(d, K_cur, device=device).unsqueeze(0)
    X = L.exp_map(o.expand(n, d + 1).contiguous(),
                  torch.cat([torch.zeros(n, 1, device=device),
                             torch.randn(n, d, generator=gen).to(device) * 1e-3], 1), K_cur)

    os.makedirs(args.out, exist_ok=True)
    traj_path = os.path.join(args.out, f"traj_{args.method}_seed{SEED}.csv")
    tf = open(traj_path, "w", newline="")
    writer = csv.writer(tf); writer.writerow(TRAJ_COLUMNS)

    learnable_K = args.method in ("B4", "B5")
    t0 = time.time()
    history_tail = []
    # radius 爆炸监控状态
    radius_window = []
    RADIUS_ABORT = args.radius_abort   # max_radius 上界 (超过判半径爆炸)

    for epoch in range(args.epochs):
        K_cur = opt.K() if hasattr(opt, "K") else args.init_K

        # --- autograd: egrad(X) 与 (learnable) ∂loss/∂K ---
        Xv = X.detach().clone().requires_grad_(True)
        if learnable_K:
            Kt = torch.tensor(float(K_cur), device=device, requires_grad=True)
        else:
            Kt = torch.tensor(float(K_cur), device=device)
        loss, distortion = distortion_loss(Xv, Dg, Kt, mask, eye)
        if learnable_K:
            egrad_X, gK = torch.autograd.grad(loss, [Xv, Kt])
            curv_egrad = gK.item()
        else:
            egrad_X = torch.autograd.grad(loss, Xv)[0]
            curv_egrad = 0.0
        egrad_X = egrad_X.detach()
        loss_val = loss.item(); dist_val = distortion.item()

        # PT norm error 诊断
        with torch.no_grad():
            v_probe = L.project_to_tangent(X[:1], torch.randn(1, d + 1, generator=gen).to(device), K_cur)
            Pv = L.parallel_transport(X[:1], X[1:2], v_probe, K_cur)
            pt_err = abs(L.lorentz_norm(Pv).item() - L.lorentz_norm(v_probe).item())
            g_probe = L.egrad_to_rgrad(X, egrad_X, K_cur)

        # optimizer step
        if learnable_K:
            X_new, info = opt.step(X, egrad_X, curv_egrad)
        else:
            X_new, info = opt.step(X, egrad_X)
        K_new = opt.K() if hasattr(opt, "K") else args.init_K

        # 周期性重投影护栏
        reproj_event = 0; reproj_mc_pre = 0.0; reproj_mom_tan_pre = 0.0
        if (epoch + 1) % REPROJ_EVERY == 0:
            reproj_event = 1
            with torch.no_grad():
                reproj_mc_pre = L.manifold_residual(X_new, K_new)
                X_new = L.project_to_manifold(X_new, K_new)
                if hasattr(opt, "m") and opt.m is not None:
                    m_reproj, mom_tan_pre, _ = L.reproject_momentum(X_new, opt.m, K_new)
                    reproj_mom_tan_pre = mom_tan_pre
                    opt.m = m_reproj

        with torch.no_grad():
            oK = L.origin(d, K_new, device=device).unsqueeze(0)
            radii = L.dist(X_new, oK.expand_as(X_new), K_new)
            mean_r = radii.mean().item(); max_r = radii.max().item()
            mc_res = (L.lorentz_inner(X_new, X_new) + 1.0 / K_new).abs().max().item()
            tan_res = info.get("tangent_residual", 0.0)

        row = [
            epoch, loss_val, dist_val, dist_val, args.lr, K_new,
            info.get("curv_grad", 0.0), info.get("log_curv_step", 0.0),
            mean_r, max_r, info.get("rgrad_norm", L.lorentz_norm(g_probe).mean().item()),
            info.get("mom_norm", 0.0), info.get("transported_mom_norm", 0.0),
            tan_res, mc_res, pt_err, info.get("exp_arg", 0.0),
            info.get("curv_adj_lr", args.lr), info.get("clip_event", 0),
            time.time() - t0,
            (torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0),
            reproj_event, reproj_mc_pre, reproj_mom_tan_pre,
        ]
        writer.writerow(row)
        X = X_new

        if epoch in PERM_CKPT or epoch % 100 == 0:
            tf.flush()
            print(f"[{args.method}/{args.tree}/n{n}/d{d}] ep{epoch:4d} loss={loss_val:.5f} "
                  f"dist={dist_val:.4f} K={K_new:.4f} R̄={mean_r:.3f} Rmax={max_r:.3f} "
                  f"mc={mc_res:.2e}", flush=True)

        # ⭐ radius 爆炸监控 (§12.2 crit 6)
        radius_window.append(max_r)
        if max_r > RADIUS_ABORT:
            tf.flush(); tf.close()
            msg = (f"RADIUS_EXPLOSION at epoch {epoch}: max_radius={max_r:.2f} "
                   f"> abort={RADIUS_ABORT}. (§12.2 failure criterion 6)")
            print("!!! " + msg, flush=True)
            json.dump({"aborted": True, "reason": msg, "epoch": epoch,
                       "method": args.method, "tree": args.tree, "nodes": n, "dim": d},
                      open(os.path.join(args.out, "ABORT.json"), "w"), indent=2)
            return {"status": "ABORTED_RADIUS_EXPLOSION", "epoch": epoch, "max_radius": max_r}

        history_tail.append((mc_res, tan_res, reproj_event, reproj_mc_pre, mean_r, max_r))
        if len(history_tail) > 200:
            history_tail.pop(0)
    tf.close()

    summary = {
        "status": "OK", "method": args.method, "tree": args.tree, "nodes": n, "dim": d,
        "epochs": args.epochs, "final_loss": loss_val, "final_distortion": dist_val,
        "final_curvature": K_new, "final_mean_radius": mean_r, "final_max_radius": max_r,
        "device": str(device), "traj_path": traj_path, "traj_columns": len(TRAJ_COLUMNS),
        "reproj_every": REPROJ_EVERY, "seed": SEED,
    }
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="B5", choices=["B1", "B2", "B3", "B4", "B5"])
    ap.add_argument("--tree", default="binary", choices=["binary", "4ary", "8ary", "variable"])
    ap.add_argument("--nodes", type=int, default=10000)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=100.0)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=1e-3)
    ap.add_argument("--delta_K", type=float, default=1e-2)
    ap.add_argument("--lam_K", type=float, default=1e-4)
    ap.add_argument("--init_K", type=float, default=1.0)
    ap.add_argument("--radius_abort", type=float, default=20.0,
                    help="max_radius 超过此值判半径爆炸 abort")
    ap.add_argument("--device", default=None, help="cuda:0 等; 默认自动选空闲卡")
    ap.add_argument("--out", default="runs/run")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
