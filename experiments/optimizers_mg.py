"""
Multi-group optimizers (G>=2, 每组独立曲率 K_g)。检验 Corollary 2.1:
分组曲率是否严格优于单一全局曲率。

每组 g 维护: X_g ∈ H_{-K_g}^d, m_g ∈ T_{X_g}, c_g (曲率坐标)。
optimizer 接收 per-group 的 egrad 列表 + (learnable) per-group ∂loss/∂K_g。

B3_MG: 每组固定曲率 (所有组共享同一固定 K) + 平行移动动量。→ H0 主对比 baseline。
B4_MG: 每组独立可学习 K，无动量，曲率变化只映射点。
B5_MG: 每组独立可学习 K + 平行移动动量 + 点&动量同步跨曲率。→ 主算法。
B5_SHARED: A3 消融 — 单一共享可学习 K (所有组同一 K)，其余同 B5。

复用 lorentz.py 逐点算子 (对每组的 [n_g, d+1] 张量批量作用)。
"""
import math
import torch
import lorentz as L


def _rgrad(X, eg, K):
    return L.egrad_to_rgrad(X, eg, K)


class B3_MG:
    """每组固定曲率 (全局共享 K_fixed) + Riemannian momentum。

    geo_step (B3.5 ablation): 是否用曲率归一化测地步长 η/(1+γ√K‖m‖)。
      False (B3): 固定步长 lr。
      True  (B3.5): 固定曲率 + 曲率归一化步长但不学曲率 → 隔离归一化步长机制独立贡献。
    """
    def __init__(self, G, K_fixed, lr, beta, geo_step=False, gamma=1.0):
        self.G = G
        self.Ks = [K_fixed] * G
        self.lr = lr
        self.beta = beta
        self.geo_step = geo_step
        self.gamma = gamma
        self.m = [None] * G

    def K(self):
        return list(self.Ks)

    def step(self, Xs, egrads, curv_egrads=None):
        Xs_new, infos = [], []
        for g in range(self.G):
            K = self.Ks[g]
            gg = _rgrad(Xs[g], egrads[g], K)
            if self.m[g] is None:
                self.m[g] = torch.zeros_like(gg)
            m_tilde = self.beta * self.m[g] + (1 - self.beta) * gg
            m_tilde = L.project_to_tangent(Xs[g], m_tilde, K)
            if self.geo_step:
                mnorm = L.lorentz_norm(m_tilde, keepdim=True)
                chi = math.sqrt(K) * mnorm
                eta = self.lr / (1.0 + self.gamma * chi)
            else:
                eta = self.lr
            Xn = L.exp_map(Xs[g], -eta * m_tilde, K)
            self.m[g] = L.parallel_transport(Xs[g], Xn, m_tilde, K)
            Xs_new.append(Xn)
            infos.append({"rgrad_norm": L.lorentz_norm(gg).mean().item(),
                          "mom_norm": L.lorentz_norm(m_tilde).mean().item()})
        return Xs_new, infos


class B4_MG:
    """每组独立可学习 K，无动量，曲率变化只映射点。"""
    def __init__(self, G, K_min, c_init, lr, rho, delta_K, lam_K, K_ref):
        self.G = G
        self.K_min = torch.tensor(K_min)
        self.c = [torch.tensor(float(c_init)) for _ in range(G)]
        self.lr = lr; self.rho = rho; self.delta_K = delta_K
        self.lam_K = lam_K; self.K_ref = K_ref

    def K(self):
        return [(self.K_min + torch.nn.functional.softplus(c)).item() for c in self.c]

    def step(self, Xs, egrads, curv_egrads):
        Xs_new, infos = [], []
        for g in range(self.G):
            K = (self.K_min + torch.nn.functional.softplus(self.c[g])).item()
            gg = _rgrad(Xs[g], egrads[g], K)
            X_half = L.exp_map(Xs[g], -self.lr * gg, K)
            reg = 2 * self.lam_K * (math.log(K) - math.log(self.K_ref)) / K
            gc = curv_egrads[g] + reg
            dKdc = torch.sigmoid(self.c[g]).item()
            c_new = self.c[g] - self.rho * gc * dKdc
            K_new = (self.K_min + torch.nn.functional.softplus(c_new)).item()
            K_new = L.clip_log_curvature(torch.tensor(K), torch.tensor(K_new), self.delta_K).item()
            Xn = L.cross_curvature_point(X_half, K, K_new)
            self.c[g] = L.coord_from_curvature(torch.tensor(K_new), self.K_min)
            Xs_new.append(Xn)
            infos.append({"rgrad_norm": L.lorentz_norm(gg).mean().item(),
                          "curvature": K_new, "curv_grad": gc,
                          "log_curv_step": abs(math.log(K_new) - math.log(K))})
        return Xs_new, infos


class B5_MG:
    """主算法 multi-group: 每组独立可学习 K + PT 动量 + 点&动量同步跨曲率。

    mom_transport 控制 G4 命门 (Step6 动量处理):
      True  (B5 主算法): 曲率变时动量做跨曲率映射 C_{K→K'}(m_half) (§3 Step6)
      False (A5 消融):   曲率变时只映射点 Ψ, 动量不跨曲率映射 (直接重用 m_half)
    → B5 vs A5 干净隔离"跨曲率映射动量"的独立贡献 (G4 精确命题)。
    """
    def __init__(self, G, K_min, c_init, lr, beta, gamma, rho, delta_K, lam_K, K_ref,
                 shared_K=False, mom_transport=True):
        self.G = G
        self.K_min = torch.tensor(K_min)
        self.shared_K = shared_K
        self.mom_transport = mom_transport
        if shared_K:
            self.c = [torch.tensor(float(c_init))]        # 单一共享 (A3 消融)
        else:
            self.c = [torch.tensor(float(c_init)) for _ in range(G)]
        self.lr = lr; self.beta = beta; self.gamma = gamma
        self.rho = rho; self.delta_K = delta_K; self.lam_K = lam_K; self.K_ref = K_ref
        self.m = [None] * G

    def _c(self, g):
        return self.c[0] if self.shared_K else self.c[g]

    def _set_c(self, g, val):
        if self.shared_K:
            self.c[0] = val
        else:
            self.c[g] = val

    def K(self):
        if self.shared_K:
            k = (self.K_min + torch.nn.functional.softplus(self.c[0])).item()
            return [k] * self.G
        return [(self.K_min + torch.nn.functional.softplus(c)).item() for c in self.c]

    def step(self, Xs, egrads, curv_egrads):
        Xs_new, infos = [], []
        # shared_K: 曲率梯度对所有组求和 (共享参数)
        shared_gc = 0.0
        for g in range(self.G):
            K = (self.K_min + torch.nn.functional.softplus(self._c(g))).item()
            gg = _rgrad(Xs[g], egrads[g], K)                      # Step1
            if self.m[g] is None:
                self.m[g] = torch.zeros_like(gg)
            m_tilde = self.beta * self.m[g] + (1 - self.beta) * gg  # Step2
            m_tilde = L.project_to_tangent(Xs[g], m_tilde, K)
            mnorm = L.lorentz_norm(m_tilde, keepdim=True)          # Step3
            chi = math.sqrt(K) * mnorm
            eta_geo = self.lr / (1.0 + self.gamma * chi)
            X_half = L.exp_map(Xs[g], -eta_geo * m_tilde, K)       # Step4
            m_half = L.parallel_transport(Xs[g], X_half, m_tilde, K)
            reg = 2 * self.lam_K * (math.log(K) - math.log(self.K_ref)) / K  # Step5
            gc = curv_egrads[g] + reg
            if self.shared_K:
                shared_gc += gc
                # 延后更新, 先存中间量
                infos.append({"_pending": True, "K": K, "X_half": X_half, "m_half": m_half,
                              "gg": gg, "m_tilde": m_tilde, "eta_geo": eta_geo,
                              "mnorm": mnorm, "chi": chi})
                Xs_new.append(None)
                continue
            dKdc = torch.sigmoid(self._c(g)).item()
            c_new = self._c(g) - self.rho * gc * dKdc
            K_new = (self.K_min + torch.nn.functional.softplus(c_new)).item()
            K_new_clip = L.clip_log_curvature(torch.tensor(K), torch.tensor(K_new), self.delta_K).item()
            clipped = 1 if abs(K_new_clip - K_new) > 1e-12 else 0
            K_new = K_new_clip
            Xn = L.cross_curvature_point(X_half, K, K_new)          # Step6: 点跨曲率映射 Ψ
            if self.mom_transport:
                # B5 主算法: 动量跨曲率映射 C_{K→K'}
                m_new = L.cross_curvature_tangent(m_half, K, K_new)
            else:
                # A5 消融: 只映射点, 动量不做跨曲率映射 (直接重用)
                m_new = m_half
            m_new = L.project_to_tangent(Xn, m_new, K_new)
            self.m[g] = m_new
            self._set_c(g, L.coord_from_curvature(torch.tensor(K_new), self.K_min))
            Xs_new.append(Xn)
            infos.append({
                "rgrad_norm": L.lorentz_norm(gg).mean().item(),
                "mom_norm": L.lorentz_norm(m_tilde).mean().item(),
                "transported_mom_norm": L.lorentz_norm(m_half).mean().item(),
                "curvature": K_new, "curv_grad": gc,
                "log_curv_step": abs(math.log(K_new) - math.log(K)),
                "exp_arg": (math.sqrt(K) * (eta_geo * mnorm)).mean().item(),
                "curv_adj_lr": eta_geo.mean().item(), "clip_event": clipped,
                "tangent_residual": L.lorentz_inner(Xn, m_new).abs().max().item(),
            })
        if self.shared_K:
            # 用汇总曲率梯度更新共享 c, 再对所有组做 Step6
            K = (self.K_min + torch.nn.functional.softplus(self.c[0])).item()
            dKdc = torch.sigmoid(self.c[0]).item()
            c_new = self.c[0] - self.rho * shared_gc * dKdc
            K_new = (self.K_min + torch.nn.functional.softplus(c_new)).item()
            K_new = L.clip_log_curvature(torch.tensor(K), torch.tensor(K_new), self.delta_K).item()
            self.c[0] = L.coord_from_curvature(torch.tensor(K_new), self.K_min)
            new_infos = []
            for g in range(self.G):
                p = infos[g]
                Xn = L.cross_curvature_point(p["X_half"], p["K"], K_new)
                if self.mom_transport:
                    m_new = L.cross_curvature_tangent(p["m_half"], p["K"], K_new)
                else:
                    m_new = p["m_half"]
                m_new = L.project_to_tangent(Xn, m_new, K_new)
                self.m[g] = m_new
                Xs_new[g] = Xn
                new_infos.append({
                    "rgrad_norm": L.lorentz_norm(p["gg"]).mean().item(),
                    "mom_norm": L.lorentz_norm(p["m_tilde"]).mean().item(),
                    "transported_mom_norm": L.lorentz_norm(p["m_half"]).mean().item(),
                    "curvature": K_new, "curv_grad": shared_gc,
                    "log_curv_step": abs(math.log(K_new) - math.log(p["K"])),
                    "exp_arg": (math.sqrt(p["K"]) * (p["eta_geo"] * p["mnorm"])).mean().item(),
                    "curv_adj_lr": p["eta_geo"].mean().item(), "clip_event": 0,
                    "tangent_residual": L.lorentz_inner(Xn, m_new).abs().max().item(),
                })
            infos = new_infos
        return Xs_new, infos
