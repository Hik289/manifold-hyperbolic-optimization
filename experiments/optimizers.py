"""
5 个 optimizer (README §7):
  B1 Euclidean optimizer      : 欧氏 SGD-momentum (对比锚)
  B2 RSGD                     : Riemannian SGD, 固定曲率, 无动量
  B3 Riemannian momentum      : 平行移动动量, 固定曲率, 无跨曲率
  B4 Learnable-curvature RSGD : 可学习曲率, 无动量, 曲率变化时只映射点
  B5 HyperbolicMomentum       : 平行移动动量 + 分组可学习曲率 + 点&动量同步跨曲率 (README §3)

所有几何算子来自 lorentz.py, FP32。

实现约定 (embedding 任务): 参数是一组双曲点 X ∈ H_{-K}^d (shape [N, d+1]),
视为单一参数组 (G=1) 的 toy 情形。曲率坐标 c 是标量。
"""

import torch
import lorentz as L


# ----------------------------------------------------------------------------
# 公共: 从环境梯度构造 Riemannian gradient
# ----------------------------------------------------------------------------
def riemannian_grad(X, egrad, K):
    return L.egrad_to_rgrad(X, egrad, K)


class B1_Euclidean:
    """欧氏 SGD with momentum。参数不受流形约束 (纯欧氏空间对比锚)。
    注意: 为在同一 embedding 任务中可比, X 仍是 (d+1) 向量, 但用欧氏几何 +
    每步后重投影回流形 (欧氏更新 + retraction), 曲率固定。
    """
    def __init__(self, K, lr, beta):
        self._K = K
        self.lr = lr
        self.beta = beta
        self.m = None

    def K(self):
        return self._K

    def step(self, X, egrad):
        if self.m is None:
            self.m = torch.zeros_like(egrad)
        self.m = self.beta * self.m + (1 - self.beta) * egrad
        X_new = X - self.lr * self.m
        X_new = L.project_to_manifold(X_new, self._K)   # retraction 回流形
        return X_new, {}


class B2_RSGD:
    """Riemannian SGD, 固定曲率, 无动量。
    x_{t+1} = Exp_x(-lr * grad)
    """
    def __init__(self, K, lr, beta=None):
        self._K = K
        self.lr = lr

    def K(self):
        return self._K

    def step(self, X, egrad):
        g = riemannian_grad(X, egrad, self._K)
        X_new = L.exp_map(X, -self.lr * g, self._K)
        return X_new, {"rgrad_norm": L.lorentz_norm(g).mean().item()}


class B3_RiemannianMomentum:
    """固定曲率的平行移动动量 (无跨曲率映射, 无曲率归一化步长)。
    m_t = beta * P_{x_{t-1}->x_t}(m_{t-1}) + (1-beta) grad
    x_{t+1} = Exp_x(-lr * m_t)
    动量在更新后平行移动到新点。
    """
    def __init__(self, K, lr, beta):
        self._K = K
        self.lr = lr
        self.beta = beta
        self.m = None

    def K(self):
        return self._K

    def step(self, X, egrad):
        g = riemannian_grad(X, egrad, self._K)
        if self.m is None:
            self.m = torch.zeros_like(g)
        m_tilde = self.beta * self.m + (1 - self.beta) * g
        m_tilde = L.project_to_tangent(X, m_tilde, self._K)   # 保证切性
        X_new = L.exp_map(X, -self.lr * m_tilde, self._K)
        # 动量平行移动到新点
        self.m = L.parallel_transport(X, X_new, m_tilde, self._K)
        info = {
            "rgrad_norm": L.lorentz_norm(g).mean().item(),
            "mom_norm": L.lorentz_norm(m_tilde).mean().item(),
            "transported_mom_norm": L.lorentz_norm(self.m).mean().item(),
        }
        return X_new, info


class B4_LearnableCurvatureRSGD:
    """可学习曲率 RSGD, 无动量, 曲率变化时只映射点 (不映射动量, 因无动量)。
    对应 §7 B4: Curvature learned, cross-curvature = Point only。
    """
    def __init__(self, K_min, c_init, lr, rho, delta_K, lam_K, K_ref, beta=None):
        self.K_min = torch.tensor(K_min)
        self.c = torch.tensor(float(c_init))
        self.lr = lr
        self.rho = rho
        self.delta_K = delta_K
        self.lam_K = lam_K
        self.K_ref = K_ref

    def K(self):
        return (self.K_min + torch.nn.functional.softplus(self.c)).item()

    def step(self, X, egrad, curv_egrad):
        K = self.K()
        g = riemannian_grad(X, egrad, K)
        X_half = L.exp_map(X, -self.lr * g, K)
        # 曲率更新 (含正则梯度)
        reg_grad = 2 * self.lam_K * (torch.log(torch.tensor(K)) - torch.log(torch.tensor(self.K_ref))) / K
        gc = curv_egrad + reg_grad.item()
        # 链式: dK/dc = sigmoid(c)
        dKdc = torch.sigmoid(self.c).item()
        c_new = self.c - self.rho * gc * dKdc
        K_new = (self.K_min + torch.nn.functional.softplus(c_new)).item()
        K_new = L.clip_log_curvature(torch.tensor(K), torch.tensor(K_new), self.delta_K).item()
        # 只映射点
        X_new = L.cross_curvature_point(X_half, K, K_new)
        self.c = L.coord_from_curvature(torch.tensor(K_new), self.K_min)
        info = {
            "rgrad_norm": L.lorentz_norm(g).mean().item(),
            "curvature": K_new,
            "curv_grad": gc,
            "log_curv_step": abs(torch.log(torch.tensor(K_new)) - torch.log(torch.tensor(K))).item(),
        }
        return X_new, info


class B5_HyperbolicMomentum:
    """主算法 (README §3 六步更新)。单参数组 toy 情形。

    Step1 Riemannian gradient
    Step2 同点切空间动量 m_tilde = beta m + (1-beta) g
    Step3 曲率归一化步长 eta_geo = eta/(1+gamma*chi), chi=sqrt(K)||m_tilde||
    Step4 Exp 更新参数 + 平行移动动量到 x_half
    Step5 曲率更新 (softplus + log clip)
    Step6 点&动量同步跨曲率映射 + 切投影
    """
    def __init__(self, K_min, c_init, lr, beta, gamma, rho, delta_K, lam_K, K_ref):
        self.K_min = torch.tensor(K_min)
        self.c = torch.tensor(float(c_init))
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.delta_K = delta_K
        self.lam_K = lam_K
        self.K_ref = K_ref
        self.m = None

    def K(self):
        return (self.K_min + torch.nn.functional.softplus(self.c)).item()

    def step(self, X, egrad, curv_egrad):
        K = self.K()
        # Step 1
        g = riemannian_grad(X, egrad, K)
        if self.m is None:
            self.m = torch.zeros_like(g)
        # Step 2
        m_tilde = self.beta * self.m + (1 - self.beta) * g
        m_tilde = L.project_to_tangent(X, m_tilde, K)
        # Step 3: chi = sqrt(K) ||m_tilde|| (per-node norm, 用 mean 做标量步长诊断保持逐点)
        mnorm = L.lorentz_norm(m_tilde, keepdim=True)          # [N,1]
        import math
        chi = math.sqrt(K) * mnorm                              # [N,1]
        eta_geo = self.lr / (1.0 + self.gamma * chi)           # [N,1] 逐点
        # Step 4
        X_half = L.exp_map(X, -eta_geo * m_tilde, K)
        m_half = L.parallel_transport(X, X_half, m_tilde, K)
        # Step 5: 曲率更新
        reg_grad = 2 * self.lam_K * (torch.log(torch.tensor(K)) - torch.log(torch.tensor(self.K_ref))) / K
        gc = curv_egrad + reg_grad.item()
        dKdc = torch.sigmoid(self.c).item()
        c_new = self.c - self.rho * gc * dKdc
        K_new = (self.K_min + torch.nn.functional.softplus(c_new)).item()
        K_new_clip = L.clip_log_curvature(torch.tensor(K), torch.tensor(K_new), self.delta_K).item()
        clipped = 1 if abs(K_new_clip - K_new) > 1e-12 else 0
        K_new = K_new_clip
        # Step 6: 点 & 动量同步跨曲率映射
        X_new = L.cross_curvature_point(X_half, K, K_new)
        m_new = L.cross_curvature_tangent(m_half, K, K_new)
        m_new = L.project_to_tangent(X_new, m_new, K_new)      # 切投影
        self.m = m_new
        self.c = L.coord_from_curvature(torch.tensor(K_new), self.K_min)

        eta_adj = eta_geo.mean().item()
        info = {
            "rgrad_norm": L.lorentz_norm(g).mean().item(),
            "mom_norm": L.lorentz_norm(m_tilde).mean().item(),
            "transported_mom_norm": L.lorentz_norm(m_half).mean().item(),
            "curvature": K_new,
            "curv_grad": gc,
            "log_curv_step": abs(torch.log(torch.tensor(K_new)) - torch.log(torch.tensor(K))).item(),
            "exp_arg": (math.sqrt(K) * (eta_geo * mnorm)).mean().item(),
            "curv_adj_lr": eta_adj,
            "clip_event": clipped,
            "tangent_residual": L.lorentz_inner(X_new, m_new).abs().max().item(),
        }
        return X_new, info
