from pathlib import Path
"""
Rotated MNIST 连续版 — G4 真实任务验证 (Director round2)。

连续旋转: 旋转角度 θ_t = θ_max·(t/T_drift) 线性 ramp 贯穿全程 (非离散 task 边界)。
数据分布连续漂移 → (假设) 最优曲率连续漂移 → G4 平滑漂移场景。

模型: 小 CNN backbone (欧氏, 低 lr) → 双曲分类头 (Lorentz distance to 10 类 prototypes)。
分组曲率: 每类一组独立 K_c (10 类 10 个 K), 可学 (B4/B5)。
score(x, c) = -d_{K_c}(embed(x) mapped to H_{-K_c}, prototype_c)

方法: B3(固定K)/B4(可学K无动量)/B5(可学K+跨曲率动量)/B5_noMomT(A5)。
双曲头参数 (prototypes + 曲率) 用 optimizers_mg 逻辑; backbone 用 Adam 低 lr。

⭐ 执行: 先 B4 诊断 learned K 是否随旋转平滑漂移。通过才 B5 vs A5。
seed=42, 0→180°。
"""
import argparse, json, math, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lorentz as L

SEED = 42


def rotate_batch(x, angle_deg):
    """旋转一个 batch 的图像 (angle 度)。x: [B,1,28,28]。用 affine grid。"""
    a = math.radians(angle_deg)
    theta = torch.tensor([[math.cos(a), -math.sin(a), 0.0],
                          [math.sin(a), math.cos(a), 0.0]], dtype=x.dtype, device=x.device)
    theta = theta.unsqueeze(0).expand(x.shape[0], 2, 3)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


class CNNBackbone(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, d))

    def forward(self, x):
        return self.net(x)


class HypHead(nn.Module):
    """双曲分类头: 每类 prototype (origin 切空间参数) + 每类曲率 K_c。"""
    def __init__(self, n_class, d, K_min, init_K, learnable_K, device):
        super().__init__()
        self.n_class, self.d = n_class, d
        self.K_min = K_min; self.device = device; self.learnable_K = learnable_K
        self.proto_tan = nn.Parameter(torch.randn(n_class, d, device=device) * 1e-2)
        c0 = L.coord_from_curvature(torch.tensor(init_K), torch.tensor(K_min)).item()
        if learnable_K:
            self.class_c = nn.Parameter(torch.full((n_class,), c0, device=device))
        else:
            self.register_buffer("class_c", torch.full((n_class,), c0, device=device))

    def K_c(self):
        return self.K_min + F.softplus(self.class_c)   # [n_class]

    def _on_manifold(self, tan, K):
        """切参数 tan [N,d] Exp 到曲率 K 流形。K 标量 tensor。"""
        N = tan.shape[0]
        vfull = torch.cat([torch.zeros(N, 1, device=self.device), tan], 1)
        Kb = K.reshape(1, 1).expand(N, 1) if K.dim() == 0 else K.reshape(-1, 1)
        o = torch.cat([1.0 / torch.sqrt(Kb), torch.zeros(N, self.d, device=self.device)], 1)
        return L.exp_map(o, vfull, Kb)

    def forward(self, feat):
        """feat: [B,d] backbone 输出 (欧氏切向量). 返回 logits [B,n_class] = -d_{K_c}(feat, proto_c)。"""
        Ks = self.K_c()                                # [n_class]
        B = feat.shape[0]
        logits = torch.empty(B, self.n_class, device=self.device)
        for c in range(self.n_class):
            K = Ks[c]
            # 把 feat 当切向量 Exp 到该类曲率流形
            xf = self._on_manifold(feat, K)            # [B,d+1]
            pc = self._on_manifold(self.proto_tan[c:c+1], K)  # [1,d+1]
            logits[:, c] = -L.dist(xf, pc.expand(B, self.d + 1), K)
        return logits


def pick_gpu():
    if not torch.cuda.is_available(): return torch.device("cpu")
    best, bu = 0, 1e18
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        if total - free < bu: bu, best = total - free, i
    return torch.device(f"cuda:{best}")


def run(args):
    import torchvision
    from torchvision import transforms
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device(args.device) if args.device else pick_gpu()

    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(args.data_dir, train=True, download=True, transform=tf)
    test_ds = torchvision.datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)
    # 子集加速 (诊断/对比够用)
    tr_idx = torch.randperm(len(train_ds))[:args.n_train]
    te_idx = torch.randperm(len(test_ds))[:args.n_test]
    Xtr = torch.stack([train_ds[i][0] for i in tr_idx]).to(device)
    Ytr = torch.tensor([train_ds[i][1] for i in tr_idx]).to(device)
    Xte = torch.stack([test_ds[i][0] for i in te_idx]).to(device)
    Yte = torch.tensor([test_ds[i][1] for i in te_idx]).to(device)

    backbone = CNNBackbone(args.dim).to(device)
    learnable_K = args.method in ("B4", "B5", "B5_noMomT")
    head = HypHead(10, args.dim, 0.1, args.init_K, learnable_K, device).to(device)

    # ⭐ 正确 frozen-CL 协议: 先 0° 预训练 backbone(+头) 到收敛, 再冻结 backbone。
    # 冻结的是训练好的特征提取器(非随机)，旋转适应只能走双曲头 → 曲率承载负载。
    if args.pretrain_epochs > 0:
        pre_bb_opt = torch.optim.Adam(backbone.parameters(), lr=1e-3)
        pre_pr_opt = torch.optim.Adam([head.proto_tan], lr=1e-3)
        bs0 = args.batch_size
        for pe in range(args.pretrain_epochs):
            backbone.train(); head.train()
            perm = torch.randperm(len(Xtr), device=device)
            for b in range(0, len(Xtr), bs0):
                bi = perm[b:b+bs0]
                pre_bb_opt.zero_grad(); pre_pr_opt.zero_grad()
                logits = head(backbone(Xtr[bi]))       # 0° (无旋转) 预训练
                lp = F.cross_entropy(logits, Ytr[bi])
                lp.backward(); pre_bb_opt.step(); pre_pr_opt.step()
                with torch.no_grad():
                    nrm = head.proto_tan.norm(dim=-1, keepdim=True)
                    head.proto_tan.mul_((args.max_tan_norm / nrm).clamp(max=1.0))
        with torch.no_grad():
            acc0 = eval_acc(backbone, head, Xte, Yte, 0.0, bs0, device)
        print(f"[pretrain] {args.pretrain_epochs}ep done, 0° test acc={acc0:.4f}", flush=True)
        if args.freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)
            print("[freeze] backbone 冻结 (预训练后)，旋转适应只走双曲头", flush=True)

    opt_bb = torch.optim.Adam(backbone.parameters(), lr=args.lr_backbone)
    opt_proto = torch.optim.Adam([head.proto_tan], lr=args.lr_proto)
    if learnable_K:
        opt_c = torch.optim.SGD([head.class_c], lr=args.rho)
    # ⭐ G4 真实现: proto 用显式动量 buffer (非 Adam), 每类 K_c 变时:
    #   B5: 动量做跨曲率缩放 C_{K→K'}(m) = √(K_old/K_new)·m (§3 Step6, 保动量幅度)
    #   A5(B5_noMomT): 动量不做跨曲率缩放 (每步曲率变化处动量幅度失配衰减)
    #   B4: 无动量; B3: 无动量固定 K。
    # 这样 B5 vs A5 才真正隔离"跨曲率映射动量", 不重蹈 KG B4=B5 覆辙。
    use_mom = args.method in ("B5", "B5_noMomT")
    do_cross_transport = (args.method == "B5")
    proto_mom = torch.zeros_like(head.proto_tan)   # 显式动量 buffer
    beta = args.beta

    os.makedirs(args.out, exist_ok=True)
    log = []
    t0 = time.time()
    bs = args.batch_size

    for epoch in range(args.epochs):
        # 连续旋转角度
        angle = args.theta_max * min(1.0, epoch / max(1, args.drift_epochs))
        # 训练
        backbone.train(); head.train()
        perm = torch.randperm(len(Xtr), device=device)
        ep_loss = 0.0; nb = 0
        for b in range(0, len(Xtr), bs):
            bi = perm[b:b+bs]
            xb = rotate_batch(Xtr[bi], angle); yb = Ytr[bi]
            opt_bb.zero_grad(); opt_proto.zero_grad()
            if learnable_K: opt_c.zero_grad()
            feat = backbone(xb)
            logits = head(feat)
            loss = F.cross_entropy(logits, yb)
            # 曲率正则
            if learnable_K and args.lam_K > 0:
                Kc = head.K_c()
                loss = loss + args.lam_K * ((torch.log(Kc) - math.log(args.init_K)) ** 2).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + [head.proto_tan], args.grad_clip)
            if learnable_K:
                torch.nn.utils.clip_grad_norm_([head.class_c], args.grad_clip)
            if not args.freeze_backbone:
                opt_bb.step()
            K_before = head.K_c().detach().clone() if learnable_K else None
            # --- proto 更新: 显式动量 (B5/A5) 或纯梯度 (B3/B4) ---
            with torch.no_grad():
                g = head.proto_tan.grad
                if g is not None:
                    if use_mom:
                        proto_mom.mul_(beta).add_(g, alpha=1 - beta)
                        head.proto_tan.add_(proto_mom, alpha=-args.lr_proto)
                    else:
                        head.proto_tan.add_(g, alpha=-args.lr_proto)   # B4/B3 无动量
            if learnable_K:
                if head.class_c.grad is not None and torch.isfinite(head.class_c.grad).all():
                    opt_c.step()
            # 护栏 + 跨曲率动量映射
            with torch.no_grad():
                nrm = head.proto_tan.norm(dim=-1, keepdim=True)
                head.proto_tan.mul_((args.max_tan_norm / nrm).clamp(max=1.0))
                if learnable_K:
                    bad = ~torch.isfinite(head.class_c)
                    if bad.any():
                        head.class_c[bad] = L.coord_from_curvature(torch.tensor(args.init_K), torch.tensor(0.1)).item()
                    Kc = head.K_c()
                    Kc = torch.clamp(Kc, 0.1 + 1e-4, args.K_max)
                    head.class_c.copy_(L.coord_from_curvature(Kc, torch.tensor(0.1)))
                    # ⭐ 跨曲率动量映射 (per-class): K 变时对该类动量做 C_{K→K'}
                    if use_mom and K_before is not None:
                        K_after = head.K_c()
                        scale = torch.sqrt(K_before / K_after)   # [n_class] √(K_old/K_new)
                        if do_cross_transport:
                            proto_mom.mul_(scale.unsqueeze(-1))   # B5: 保动量幅度
                        # A5: 不缩放 (do_cross_transport=False), 动量幅度失配
            ep_loss += loss.item(); nb += 1
        ep_loss /= max(nb, 1)

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            backbone.eval(); head.eval()
            with torch.no_grad():
                # 当前角度 test acc
                acc_cur = eval_acc(backbone, head, Xte, Yte, angle, bs, device)
                # 原始 0° test acc (forgetting: 保持初始角度性能)
                acc_orig = eval_acc(backbone, head, Xte, Yte, 0.0, bs, device)
                Kc = head.K_c().detach().cpu().numpy()
                pn = head.proto_tan.norm(dim=-1).mean().item()
                mom_norm = proto_mom.norm(dim=-1).mean().item()   # 机理证据: 动量保持
            row = {"epoch": epoch, "angle": angle, "loss": ep_loss,
                   "acc_cur": acc_cur, "acc_orig": acc_orig,
                   "K_mean": float(Kc.mean()), "K_std": float(Kc.std()),
                   "K_per_class": Kc.tolist(), "proto_norm": pn,
                   "mom_norm": mom_norm, "wall": time.time() - t0}
            log.append(row)
            print(f"[{args.method}] ep{epoch:3d} ang={angle:.1f} loss={ep_loss:.3f} "
                  f"acc_cur={acc_cur:.3f} acc_orig={acc_orig:.3f} K_mean={Kc.mean():.3f} "
                  f"K_std={Kc.std():.3f} K_range=[{Kc.min():.3f},{Kc.max():.3f}]", flush=True)

    json.dump({"method": args.method, "seed": SEED, "dim": args.dim, "log": log,
               "final": log[-1]}, open(os.path.join(args.out, "rotmnist_result.json"), "w"), indent=2)
    return log[-1]


@torch.no_grad()
def eval_acc(backbone, head, X, Y, angle, bs, device):
    correct = 0
    for b in range(0, len(X), bs):
        xb = rotate_batch(X[b:b+bs], angle)
        logits = head(backbone(xb))
        correct += (logits.argmax(1) == Y[b:b+bs]).sum().item()
    return correct / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="B4", choices=["B3", "B4", "B5", "B5_noMomT"])
    ap.add_argument("--data_dir", default=str(Path(__file__).resolve().parents[1] / "data/mnist"))
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--drift_epochs", type=int, default=150)
    ap.add_argument("--theta_max", type=float, default=180.0)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)   # 低 lr backbone
    ap.add_argument("--lr_proto", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.95, help="B5/A5 动量系数")
    ap.add_argument("--rho", type=float, default=1e-2)
    ap.add_argument("--freeze_backbone", action="store_true", help="冻结 backbone (曲率承载 setup)")
    ap.add_argument("--pretrain_epochs", type=int, default=0, help="0° 预训练 backbone epoch 数 (frozen-CL 协议)")
    ap.add_argument("--lam_K", type=float, default=1e-3)
    ap.add_argument("--init_K", type=float, default=1.0)
    ap.add_argument("--K_max", type=float, default=2.0)
    ap.add_argument("--max_tan_norm", type=float, default=1.5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/rotmnist")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
