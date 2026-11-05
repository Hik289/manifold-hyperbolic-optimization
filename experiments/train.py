"""
Toy synthetic tree 双曲 embedding — anchor_2 (baseline 行为 COMPARE) + anchor_4 (20 列轨迹)。

任务: 把树节点 embed 到 H_{-K}^d, 最小化双曲距离与图距离的失真。
loss = mean_{i<j} ( d_hyp(x_i,x_j) / d_graph(i,j) - 1 )^2   (scale-free distortion)

支持 5 个 method: B1..B5。所有实验 seed=42, FP32 几何。
每 epoch 落盘 §5.3 的 20 列轨迹。

用法:
  python train.py --method B5 --epochs 300 --out results.json
"""

import argparse
import json
import math
import time
import csv
import os
import torch

import lorentz as L
from data import make_toy_binary_tree
from optimizers import (
    B1_Euclidean, B2_RSGD, B3_RiemannianMomentum,
    B4_LearnableCurvatureRSGD, B5_HyperbolicMomentum,
)

SEED = 42

# §5.3 permanent checkpoint epoch 集合
PERM_CKPT = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# §5.3 20 列轨迹 schema (顺序即列序)
TRAJ_COLUMNS = [
    "epoch",                        # 隐含索引, 但也写入
    "train_loss",                   # 1
    "val_metric",                   # 2
    "test_metric",                  # 3
    "learning_rate",                # 4
    "curvature",                    # 5 (per-group; toy=单组)
    "curvature_grad",               # 6
    "log_curvature_step",           # 7
    "geodesic_radius",              # 8 (mean radius from origin)
    "max_radius",                   # 9
    "riemannian_grad_norm",         # 10
    "momentum_norm",                # 11
    "transported_momentum_norm",    # 12
    "tangent_residual",             # 13
    "manifold_constraint_residual", # 14
    "parallel_transport_norm_error",# 15
    "exp_map_argument",             # 16
    "curvature_adjusted_lr",        # 17
    "clipping_events",              # 18
    "wall_clock",                   # 19
    "peak_memory_mb",               # 20
    # --- 数值卫生增补列 (§5.3 之外, 显式记录周期性重投影, 不静默修正) ---
    "reproj_event",                 # 21: 该 epoch 是否触发周期性重投影 (0/1)
    "reproj_manifold_residual_pre", # 22: 重投影前 manifold constraint residual 峰值 (Theorem4 e_geom 实测)
    "reproj_momentum_tangent_pre",  # 23: 重投影前动量切空间残差 |<x,m>_L| 峰值
]

# 每 N epoch 触发一次周期性重投影 (数值卫生护栏)
REPROJ_EVERY = 50


def init_embedding(n, d, K, gen):
    """在原点附近初始化 n 个双曲点。"""
    o = L.origin(d, K).unsqueeze(0).expand(n, d + 1).contiguous()
    v = torch.zeros(n, d + 1)
    v[:, 1:] = torch.randn(n, d, generator=gen) * 1e-3
    X = L.exp_map(o, v, K)
    return X


def distortion_loss_and_grad(X, Dg, K, mask):
    """distortion loss + 环境梯度 (autograd)。

    loss = mean_{i<j} (d_hyp/d_graph - 1)^2
    返回 loss, egrad(X), 以及 distortion metric。
    """
    Xv = X.detach().clone().requires_grad_(True)
    # pairwise hyperbolic distance via stable dist: 用 broadcasting
    n = Xv.shape[0]
    # 计算上三角 pair 的距离 (避免 O(n^2) 显存爆炸, toy n~500 可全量)
    xi = Xv.unsqueeze(1)   # [n,1,d+1]
    xj = Xv.unsqueeze(0)   # [1,n,d+1]
    diff = xi - xj
    # <diff,diff>_L
    sq = -diff[..., 0] ** 2 + (diff[..., 1:] ** 2).sum(-1)
    s = 0.5 * K * sq
    s = torch.clamp(s, min=0.0)
    dh = torch.log1p(s + torch.sqrt(s * s + 2.0 * s + 1e-12)) / math.sqrt(K)
    ratio = dh / (Dg + torch.eye(n))    # 对角 Dg=0 加 eye 防除零
    resid = (ratio - 1.0) ** 2
    resid = resid * mask                # 只算 i<j (上三角)
    loss = resid.sum() / mask.sum()
    loss.backward()
    egrad = Xv.grad.detach().clone()
    # distortion metric (avg |ratio-1|)
    with torch.no_grad():
        distortion = ((ratio - 1.0).abs() * mask).sum() / mask.sum()
    return loss.item(), egrad, distortion.item()


def make_optimizer(method, K_init, args):
    K_min = 0.1
    K_ref = K_init
    c_init = L.coord_from_curvature(torch.tensor(K_init), torch.tensor(K_min)).item()
    if method == "B1":
        return B1_Euclidean(K_init, args.lr, args.beta)
    if method == "B2":
        return B2_RSGD(K_init, args.lr)
    if method == "B3":
        return B3_RiemannianMomentum(K_init, args.lr, args.beta)
    if method == "B4":
        return B4_LearnableCurvatureRSGD(K_min, c_init, args.lr, args.rho,
                                         args.delta_K, args.lam_K, K_ref)
    if method == "B5":
        return B5_HyperbolicMomentum(K_min, c_init, args.lr, args.beta, args.gamma,
                                     args.rho, args.delta_K, args.lam_K, K_ref)
    raise ValueError(method)


def curvature_egrad(X, Dg, K, mask, eps=1e-3):
    """对曲率 K 的环境梯度 (有限差, toy 用)。仅 B4/B5 需要。"""
    def loss_at(Kv):
        n = X.shape[0]
        xi = X.unsqueeze(1); xj = X.unsqueeze(0); diff = xi - xj
        sq = -diff[..., 0] ** 2 + (diff[..., 1:] ** 2).sum(-1)
        s = torch.clamp(0.5 * Kv * sq, min=0.0)
        dh = torch.log1p(s + torch.sqrt(s * s + 2.0 * s + 1e-12)) / math.sqrt(Kv)
        ratio = dh / (Dg + torch.eye(n))
        return (((ratio - 1.0) ** 2) * mask).sum().item() / mask.sum().item()
    return (loss_at(K + eps) - loss_at(K - eps)) / (2 * eps)


def run(args):
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    n, edges, parent, Dg = make_toy_binary_tree(depth=args.depth)
    d = args.dim
    K_init = args.init_K
    mask = torch.triu(torch.ones(n, n), diagonal=1)

    opt = make_optimizer(args.method, K_init, args)
    K_cur = opt.K() if hasattr(opt, "K") else K_init
    X = init_embedding(n, d, K_cur, gen)

    os.makedirs(args.traj_dir, exist_ok=True)
    traj_path = os.path.join(args.traj_dir, f"traj_{args.method}_seed{SEED}.csv")
    tf = open(traj_path, "w", newline="")
    writer = csv.writer(tf)
    writer.writerow(TRAJ_COLUMNS)

    o = L.origin(d, K_cur).unsqueeze(0)
    t0 = time.time()
    history = []
    prev_X = None

    for epoch in range(args.epochs):
        K_cur = opt.K() if hasattr(opt, "K") else K_init
        loss, egrad, distortion = distortion_loss_and_grad(X, Dg, K_cur, mask)

        # PT norm error 诊断 (用当前 batch: 移动 grad 从 origin 到 X[0])
        with torch.no_grad():
            g_probe = L.egrad_to_rgrad(X, egrad, K_cur)
            v_probe = L.project_to_tangent(X[:1], torch.randn(1, d + 1, generator=gen), K_cur)
            Pv = L.parallel_transport(X[:1], X[1:2], v_probe, K_cur)
            pt_err = abs(L.lorentz_norm(Pv).item() - L.lorentz_norm(v_probe).item())

        # optimizer step
        if args.method in ("B4", "B5"):
            cg = curvature_egrad(X, Dg, K_cur, mask)
            X_new, info = opt.step(X, egrad, cg)
        else:
            X_new, info = opt.step(X, egrad)

        K_new = opt.K() if hasattr(opt, "K") else K_init

        # -------------------------------------------------------------------
        # 周期性重投影护栏 (每 REPROJ_EVERY epoch, 数值卫生, 不改 §3 数学)
        # 显式记录重投影前的 manifold/动量残差, 不静默修正。
        # -------------------------------------------------------------------
        reproj_event = 0
        reproj_mc_pre = 0.0
        reproj_mom_tan_pre = 0.0
        if (epoch + 1) % REPROJ_EVERY == 0:
            reproj_event = 1
            with torch.no_grad():
                # 1) 记录重投影前 manifold constraint residual 峰值
                reproj_mc_pre = L.manifold_residual(X_new, K_new)
                # 2) 点重投影回流形约束面 <x,x>_L = -1/K
                X_new = L.project_to_manifold(X_new, K_new)
                # 3) 动量切空间重投影 (仅有状态动量的 optimizer: B1/B3/B5)
                if hasattr(opt, "m") and opt.m is not None:
                    m_reproj, mom_tan_pre, _ = L.reproject_momentum(X_new, opt.m, K_new)
                    reproj_mom_tan_pre = mom_tan_pre
                    opt.m = m_reproj

        with torch.no_grad():
            radii = L.dist(X_new, o.expand_as(X_new), K_new)
            mc_res = (L.lorentz_inner(X_new, X_new) + 1.0 / K_new).abs().max().item()
            tan_res = info.get("tangent_residual", 0.0)

        row = {
            "epoch": epoch,
            "train_loss": loss,
            "val_metric": distortion,
            "test_metric": distortion,
            "learning_rate": args.lr,
            "curvature": K_new,
            "curvature_grad": info.get("curv_grad", 0.0),
            "log_curvature_step": info.get("log_curv_step", 0.0),
            "geodesic_radius": radii.mean().item(),
            "max_radius": radii.max().item(),
            "riemannian_grad_norm": info.get("rgrad_norm", L.lorentz_norm(g_probe).mean().item()),
            "momentum_norm": info.get("mom_norm", 0.0),
            "transported_momentum_norm": info.get("transported_mom_norm", 0.0),
            "tangent_residual": tan_res,
            "manifold_constraint_residual": mc_res,
            "parallel_transport_norm_error": pt_err,
            "exp_map_argument": info.get("exp_arg", 0.0),
            "curvature_adjusted_lr": info.get("curv_adj_lr", args.lr),
            "clipping_events": info.get("clip_event", 0),
            "wall_clock": time.time() - t0,
            "peak_memory_mb": (torch.cuda.max_memory_allocated() / 1e6
                               if torch.cuda.is_available() else 0.0),
            "reproj_event": reproj_event,
            "reproj_manifold_residual_pre": reproj_mc_pre,
            "reproj_momentum_tangent_pre": reproj_mom_tan_pre,
        }
        writer.writerow([row[c] for c in TRAJ_COLUMNS])
        history.append(row)
        X = X_new

        if epoch in PERM_CKPT or epoch % 100 == 0:
            print(f"[{args.method}] epoch {epoch:4d} loss={loss:.5f} "
                  f"distortion={distortion:.4f} K={K_new:.4f} "
                  f"radius={radii.mean().item():.3f} mc_res={mc_res:.2e} "
                  f"tan_res={tan_res:.2e}")
    tf.close()

    final = history[-1]
    summary = {
        "method": args.method,
        "final_loss": final["train_loss"],
        "final_distortion": final["val_metric"],
        "final_curvature": final["curvature"],
        "final_mean_radius": final["geodesic_radius"],
        "final_max_radius": final["max_radius"],
        "max_manifold_residual": max(h["manifold_constraint_residual"] for h in history),
        "max_tangent_residual": max(h["tangent_residual"] for h in history),
        "total_clip_events": sum(h["clipping_events"] for h in history),
        "reproj_count": sum(h["reproj_event"] for h in history),
        "max_reproj_manifold_residual_pre": max(h["reproj_manifold_residual_pre"] for h in history),
        "manifold_residual_after_last_reproj": next(
            (h["manifold_constraint_residual"] for h in reversed(history) if h["reproj_event"] == 1), None),
        "epochs": args.epochs,
        "n_nodes": n,
        "dim": d,
        "traj_path": traj_path,
        "traj_columns": len(TRAJ_COLUMNS),
    }
    return summary, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="B5", choices=["B1", "B2", "B3", "B4", "B5"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--depth", type=int, default=8)      # 2^9-1=511 nodes
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=1e-3)
    ap.add_argument("--delta_K", type=float, default=1e-2)
    ap.add_argument("--lam_K", type=float, default=1e-4)
    ap.add_argument("--init_K", type=float, default=1.0)
    ap.add_argument("--traj_dir", default="trajectories")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--all", action="store_true", help="跑全部 5 个 method")
    args = ap.parse_args()

    if args.all:
        results = []
        for method in ["B1", "B2", "B3", "B4", "B5"]:
            args.method = method
            summary, _ = run(args)
            results.append(summary)
            print(f"  -> {method}: distortion={summary['final_distortion']:.4f} "
                  f"loss={summary['final_loss']:.5f}\n")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("\n=== 全部 method 汇总 ===")
        for r in results:
            print(f"{r['method']}: final_distortion={r['final_distortion']:.4f} "
                  f"final_loss={r['final_loss']:.5f} K={r['final_curvature']:.4f} "
                  f"mc_res={r['max_manifold_residual']:.2e}")
    else:
        summary, _ = run(args)
        with open(args.out, "w") as f:
            json.dump([summary], f, indent=2)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
