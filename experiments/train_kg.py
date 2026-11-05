from pathlib import Path
"""
WN18RR KG link prediction — Lorentz distance model (README §6.4 / §1.8)。

模型: 实体 embed 到 Lorentz 双曲空间。关系族分组 = 11 个关系各一组独立 K_g。
score(h, r, t) = -d_{K_r}(h ⊕ r_translation, t)  (Lorentz 距离越小分越高)

简化 Lorentz translational: 每关系 r 有:
  - 独立曲率 K_r (关系族分组, B4/B5 可学习)
  - 关系平移向量 (切空间, Exp 到流形)
实体在共享的 Lorentz 表 (但打分时用该三元组关系的 K_r 度量)。

因不同关系 K_r 不同, 实体点需按关系跨曲率映射 Ψ 到该关系流形再打分。

训练: 负采样 (每正例 corrupt tail/head N 个负例), margin/softmax loss。
优化: B3(固定K) / B4(学K只映射点) / B5(学K+跨曲率动量), ρ, M1 每步投影。

seed=42, 500ep 预跑。
用法: python train_kg.py --method B5 --dim 20 --epochs 500 --rho 1e-2 --per_step_reproj
"""
import argparse, json, math, time, os
import numpy as np
import torch

import lorentz as L

SEED = 42


def load_triples(data_dir):
    ent2id, rel2id = {}, {}
    def gid(d, k):
        if k not in d:
            d[k] = len(d)
        return d[k]
    splits = {}
    for split in ["train", "valid", "test"]:
        trip = []
        for line in open(os.path.join(data_dir, f"{split}.txt")):
            h, r, t = line.strip().split("\t")
            trip.append((gid(ent2id, h), gid(rel2id, r), gid(ent2id, t)))
        splits[split] = trip
    return ent2id, rel2id, splits


class LorentzKG(torch.nn.Module):
    """Lorentz distance KG model, 关系族分组曲率。

    实体表: 欧氏参数 [n_ent, d], 通过 Exp_o 映射到流形 (切空间参数化, 保证在流形上)。
    关系: 每关系平移切向量 [n_rel, d] + 曲率坐标 c_r [n_rel]。
    """
    def __init__(self, n_ent, n_rel, d, K_min, init_K, learnable_K, device):
        super().__init__()
        self.n_ent, self.n_rel, self.d = n_ent, n_rel, d
        self.K_min = K_min
        self.device = device
        self.learnable_K = learnable_K
        # 实体切空间参数 (在 origin 切空间, v_0=0 隐含)
        self.ent_tan = torch.nn.Parameter(torch.randn(n_ent, d, device=device) * 1e-2)
        # 关系平移切向量
        self.rel_tan = torch.nn.Parameter(torch.randn(n_rel, d, device=device) * 1e-2)
        # 关系曲率坐标 (每关系一组)
        c0 = L.coord_from_curvature(torch.tensor(init_K), torch.tensor(K_min)).item()
        if learnable_K:
            self.rel_c = torch.nn.Parameter(torch.full((n_rel,), c0, device=device))
        else:
            self.register_buffer("rel_c", torch.full((n_rel,), c0, device=device))

    def K_r(self):
        return self.K_min + torch.nn.functional.softplus(self.rel_c)  # [n_rel]

    def ent_on_manifold(self, idx, K):
        """把实体切参数 Exp 到曲率 K 的流形。idx: [B], K: 标量 or [B,1] tensor。

        全向量化 (支持 per-sample K), 避免按关系循环 (237 关系时循环是瓶颈)。
        origin o=(1/sqrt(K),0..) 依赖 K, exp_map 也依赖 K → K 保 tensor 让曲率可学。
        """
        v = self.ent_tan[idx]                       # [B, d]
        B = v.shape[0]
        vfull = torch.cat([torch.zeros(B, 1, device=self.device), v], 1)  # [B, d+1]
        # K 广播成 [B,1] (若标量则扩展)
        if not torch.is_tensor(K) or K.dim() == 0:
            Kb = torch.as_tensor(K, device=self.device).reshape(1, 1).expand(B, 1)
        else:
            Kb = K.reshape(-1, 1)
        invsqrtK = 1.0 / torch.sqrt(Kb)             # [B,1]
        o = torch.zeros(B, self.d + 1, device=self.device)
        o = torch.cat([invsqrtK, torch.zeros(B, self.d, device=self.device)], 1)
        return L.exp_map(o, vfull, Kb)

    def _htrans(self, h_idx, r_idx):
        """计算 h ⊕ r_translation, 全向量化 (per-sample K)。返回 h_trans[B,d+1] + K[B,1]。"""
        Ks = self.K_r()                              # [n_rel]
        Kv = Ks[r_idx].reshape(-1, 1)                # [B,1] per-sample K (保 grad)
        hh = self.ent_on_manifold(h_idx, Kv)         # [B, d+1]
        B = h_idx.shape[0]
        rv = self.rel_tan[r_idx]                      # [B, d]
        rvfull = torch.cat([torch.zeros(B, 1, device=self.device), rv], 1)  # [B,d+1]
        rv_tan = L.project_to_tangent(hh, rvfull, Kv)
        h_trans = L.exp_map(hh, rv_tan, Kv)
        return h_trans, Kv

    def score(self, h_idx, r_idx, t_idx):
        """打分 = -d_{K_r}(h⊕r, t)。全向量化。"""
        h_trans, Kv = self._htrans(h_idx, r_idx)      # [B,d+1], [B,1]
        tt = self.ent_on_manifold(t_idx, Kv)          # [B,d+1]
        # dist 内部 lorentz_inner 返回 [B] (1D), K 需 [B] 匹配 (非 [B,1] 防广播成 [B,B])
        return -L.dist(h_trans, tt, Kv.squeeze(-1))   # [B]

    def score_neg_tails(self, h_idx, r_idx, neg_t):
        """一次算所有负 tail 的分数。neg_t: [B, N]。返回 [B, N]。全向量化。"""
        h_trans, Kv = self._htrans(h_idx, r_idx)      # [B,d+1], [B,1]
        B, N = neg_t.shape
        Kbn = Kv.expand(B, N).reshape(-1, 1)          # [B*N,1] (exp_map 用)
        tt = self.ent_on_manifold(neg_t.reshape(-1), Kbn).reshape(B, N, self.d + 1)
        ht_e = h_trans.unsqueeze(1).expand(B, N, self.d + 1)
        # dist 的 K 用 [B*N] (1D) 匹配 lorentz_inner 输出
        dneg = -L.dist(ht_e.reshape(-1, self.d + 1),
                       tt.reshape(-1, self.d + 1), Kbn.squeeze(-1)).reshape(B, N)
        return dneg


def kg_loss(model, batch_h, batch_r, batch_t, n_ent, n_neg, gen, device):
    """负采样 + softmax (self-adversarial 简化: 均匀负采样 + logsumexp)。"""
    pos = model.score(batch_h, batch_r, batch_t)     # [B]
    B = batch_h.shape[0]
    # corrupt tail (向量化, 一次算所有负例)
    neg_t = torch.randint(0, n_ent, (B, n_neg), generator=gen, device=device)
    neg = model.score_neg_tails(batch_h, batch_r, neg_t)   # [B, n_neg]
    # loss = -log( exp(pos) / (exp(pos)+sum exp(neg)) )
    logits = torch.cat([pos.unsqueeze(1), neg], 1)   # [B, 1+n_neg]
    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=device))
    return loss


@torch.no_grad()
def evaluate(model, triples, n_ent, device, max_eval=1000):
    """filtered-lite MRR/Hits@10 (采样 max_eval 个三元组, corrupt tail 全实体排名)。"""
    model.eval()
    idx = torch.randperm(len(triples))[:max_eval]
    ranks = []
    Ks = model.K_r()
    for i in idx:
        h, r, t = triples[i]
        K = Ks[r]
        hh = model.ent_on_manifold(torch.tensor([h], device=device), K)
        rv = model.rel_tan[r]
        rvfull = torch.cat([torch.zeros(1, 1, device=device), rv.unsqueeze(0)], 1)
        rv_tan = L.project_to_tangent(hh, rvfull, K.item())
        h_trans = L.exp_map(hh, rv_tan, K.item())
        # 所有实体作为 tail
        all_t = model.ent_on_manifold(torch.arange(n_ent, device=device), K)
        d_all = L.dist(h_trans.expand(n_ent, -1), all_t, K.item())  # [n_ent]
        rank = (d_all < d_all[t]).sum().item() + 1
        ranks.append(rank)
    model.train()
    ranks = np.array(ranks)
    return {"MRR": float((1.0 / ranks).mean()),
            "Hits@10": float((ranks <= 10).mean()),
            "Hits@1": float((ranks <= 1).mean()),
            "MeanRank": float(ranks.mean())}


def pick_gpu():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best, bu = 0, 1e18
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        if total - free < bu:
            bu, best = total - free, i
    return torch.device(f"cuda:{best}")


def run(args):
    torch.manual_seed(SEED)
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    device = torch.device(args.device) if args.device else pick_gpu()
    gen_dev = torch.Generator(device=device).manual_seed(SEED)

    data_dir = args.data_dir
    ent2id, rel2id, splits = load_triples(data_dir)
    n_ent, n_rel = len(ent2id), len(rel2id)
    train = torch.tensor(splits["train"], device=device)
    print(f"n_ent={n_ent} n_rel={n_rel} train={len(train)}", flush=True)

    learnable_K = args.method in ("B4", "B5")
    model = LorentzKG(n_ent, n_rel, args.dim, 0.1, args.init_K, learnable_K, device)

    # 优化: 实体/关系切参数用 Adam (欧氏), 关系曲率 c 用自定义慢速 (ρ) + M1 精神
    # 简化: 整体用一个 optimizer, 但曲率 lr=ρ, 其余 lr。
    euclid_params = [model.ent_tan, model.rel_tan]
    # KG 只坐实 G1(可学曲率 vs 固定)+G3(曲率分化)。G4(跨曲率动量) 只在 synthetic 测
    # (Director 裁决 C: KG 实体是共享 origin 切空间参数, 无每组独立流形点, B5 跨曲率
    #  动量 transport 在此无天然落点)。故 B3/B4 用 Adam (收敛快), 不做 SGD 动量对照。
    opt_e = torch.optim.Adam(euclid_params, lr=args.lr)
    if learnable_K:
        opt_c = torch.optim.SGD([model.rel_c], lr=args.rho)

    os.makedirs(args.out, exist_ok=True)
    log = []
    t0 = time.time()
    bs = args.batch_size
    n_batches = max(1, len(train) // bs)

    for epoch in range(args.epochs):
        perm = torch.randperm(len(train), generator=gen, device="cpu").to(device)
        ep_loss = 0.0
        for b in range(n_batches):
            bi = perm[b * bs:(b + 1) * bs]
            bh, br, bt = train[bi, 0], train[bi, 1], train[bi, 2]
            opt_e.zero_grad()
            if learnable_K:
                opt_c.zero_grad()
            loss = kg_loss(model, bh, br, bt, n_ent, args.n_neg, gen_dev, device)
            # 曲率正则
            if learnable_K and args.lam_K > 0:
                Kr = model.K_r()
                reg = args.lam_K * ((torch.log(Kr) - math.log(args.init_K)) ** 2).sum()
                (loss + reg).backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(euclid_params, args.grad_clip)
            if learnable_K:
                torch.nn.utils.clip_grad_norm_([model.rel_c], args.grad_clip)
            if learnable_K:
                K_before = model.K_r().detach().clone()   # Step5 log-curv clip 参考
            opt_e.step()
            if learnable_K:
                # NaN 守卫: 曲率梯度若 NaN 则跳过本步曲率更新 (防污染)
                if model.rel_c.grad is not None and torch.isfinite(model.rel_c.grad).all():
                    opt_c.step()
            # 实体/关系切向量范数裁剪 (防半径爆炸, M1 精神: 保参数在安全球内)
            with torch.no_grad():
                for p in (model.ent_tan, model.rel_tan):
                    nrm = p.norm(dim=-1, keepdim=True)
                    scale = (args.max_tan_norm / nrm).clamp(max=1.0)
                    p.mul_(scale)
                if learnable_K:
                    # (a) NaN 守卫: rel_c 若 NaN 回退到 clip 前
                    bad = ~torch.isfinite(model.rel_c)
                    if bad.any():
                        model.rel_c[bad] = L.coord_from_curvature(
                            torch.tensor(args.init_K), torch.tensor(0.1)).item()
                    # (b) README §3 Step5: 每步 log-curvature clipping |Δlog K| ≤ δ_K
                    K_now = model.K_r()
                    K_clip = L.clip_log_curvature(K_before, K_now, args.delta_K)
                    # (c) K 上界 K_max (防无界增长)
                    K_clip = torch.clamp(K_clip, min=0.1 + 1e-4, max=args.K_max)
                    # 反写 rel_c
                    model.rel_c.copy_(L.coord_from_curvature(K_clip, torch.tensor(0.1)))
            ep_loss += loss.item()
        ep_loss /= n_batches

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, splits["valid"], n_ent, device, max_eval=args.max_eval)
            Kr = model.K_r().detach().cpu().numpy()
            row = {"epoch": epoch, "loss": ep_loss, "wall": time.time() - t0,
                   "MRR": metrics["MRR"], "Hits@10": metrics["Hits@10"],
                   "Hits@1": metrics["Hits@1"], "MeanRank": metrics["MeanRank"],
                   "K_mean": float(Kr.mean()), "K_std": float(Kr.std()),
                   "K_per_rel": Kr.tolist()}
            log.append(row)
            print(f"[{args.method}] ep{epoch:3d} loss={ep_loss:.4f} MRR={metrics['MRR']:.4f} "
                  f"H@10={metrics['Hits@10']:.4f} K_std={Kr.std():.4f} "
                  f"K_range=[{Kr.min():.3f},{Kr.max():.3f}]", flush=True)

    json.dump({"method": args.method, "seed": SEED, "dim": args.dim,
               "rel2id": rel2id, "log": log,
               "final": log[-1]}, open(os.path.join(args.out, "kg_result.json"), "w"), indent=2)
    return log[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="B5", choices=["B3", "B4", "B5"])
    ap.add_argument("--data_dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--dim", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--rho", type=float, default=1e-2)
    ap.add_argument("--lam_K", type=float, default=1e-4)
    ap.add_argument("--init_K", type=float, default=1.0)
    ap.add_argument("--n_neg", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--max_eval", type=int, default=1000)
    ap.add_argument("--per_step_reproj", action="store_true")
    ap.add_argument("--max_tan_norm", type=float, default=1.5,
                    help="实体/关系切向量最大范数 (防半径爆炸)")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.9, help="B5 动量系数")
    ap.add_argument("--delta_K", type=float, default=1e-2, help="每步 |Δlog K| 上限 (§3 Step5)")
    ap.add_argument("--K_max", type=float, default=5.0, help="曲率 K 上界 (防无界增长)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/kg_run")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
