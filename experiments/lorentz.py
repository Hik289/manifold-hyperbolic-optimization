"""
Lorentz 双曲几何算子库 (HyperbolicMomentum)

严格逐字实现 README §1 / §2 的闭合公式。
所有几何运算 FP32 (硬约束)。

约定:
- 点 x, y 形状 (..., d+1), 第 0 维是时间分量 x_0 > 0。
- 曲率用 K = -kappa > 0 参数化, 流形 <x,x>_L = -1/K。
- 切向量 v 满足 <x,v>_L = 0。
"""

import torch


# ----------------------------------------------------------------------------
# 基础: Lorentz 内积与范数
# ----------------------------------------------------------------------------
def lorentz_inner(x, y, keepdim=False):
    """Lorentz 内积 <x,y>_L = -x_0 y_0 + sum_i x_i y_i  (README §1.1)"""
    prod = x * y
    # 时间分量取负, 空间分量取正
    inner = -prod[..., 0:1].sum(-1, keepdim=True) + prod[..., 1:].sum(-1, keepdim=True)
    if not keepdim:
        inner = inner.squeeze(-1)
    return inner


def lorentz_norm(v, keepdim=False, eps=1e-12):
    """切向量 Lorentz 范数 ||v||_x = sqrt(<v,v>_L)  (README §1.2)

    切向量在负曲率 Lorentz 模型中 <v,v>_L >= 0 (类空)。
    """
    sq = lorentz_inner(v, v, keepdim=keepdim)
    sq = torch.clamp(sq, min=0.0)
    return torch.sqrt(sq + eps)


# ----------------------------------------------------------------------------
# 原点与投影
# ----------------------------------------------------------------------------
def origin(d, K, dtype=torch.float32, device=None):
    """曲率 K 流形的原点 o_K = (1/sqrt(K), 0, ..., 0)  (README §1.1)"""
    o = torch.zeros(d + 1, dtype=dtype, device=device)
    o[0] = 1.0 / torch.sqrt(torch.as_tensor(K, dtype=dtype, device=device))
    return o


def project_to_manifold(x, K):
    """把 (d+1) 向量的空间分量固定, 重算时间分量使其落在流形上。

    <x,x>_L = -1/K  =>  x_0 = sqrt(1/K + ||x_{1:}||^2)
    """
    space = x[..., 1:]
    space_sq = (space * space).sum(-1, keepdim=True)
    inv_K = (1.0 / K)
    if not torch.is_tensor(inv_K):
        inv_K = torch.as_tensor(inv_K, dtype=x.dtype, device=x.device)
    x0 = torch.sqrt(inv_K + space_sq)
    return torch.cat([x0, space], dim=-1)


def project_to_tangent(x, u, K):
    """把环境向量 u 投影到 T_x 切空间  (README §3.2 Step6 / §11.2)

    Proj_{T_x}(u) = u + K <x,u>_L x
    验证: <x, Proj>_L = <x,u>_L + K<x,u>_L <x,x>_L = <x,u>_L + K<x,u>_L(-1/K) = 0
    """
    coef = K * lorentz_inner(x, u, keepdim=True)
    return u + coef * x


# ----------------------------------------------------------------------------
# 距离
# ----------------------------------------------------------------------------
def dist(x, y, K):
    """测地距离 d_K(x,y) = (1/sqrt K) arcosh(-K <x,y>_L)  (README §1.3)

    数值稳定实现: 直接算 alpha = -K<x,y>_L 在 x≈y 时 (alpha≈1) 会因 Lorentz
    内积 -x0y0+sum 的灾难性抵消损失精度 (FP32 尤甚)。利用恒等式
        <x-y, x-y>_L = <x,x>_L - 2<x,y>_L + <y,y>_L = -2/K - 2<x,y>_L
      => alpha - 1 = -K<x,y>_L - 1 = (K/2) <x-y, x-y>_L
    对流形上两点 <x-y,x-y>_L >= 0, 该式无抵消。再用
        arcosh(1+s) = log(1+s + sqrt(s^2 + 2s))    (s>=0, 稳定)
    """
    sqrtK = torch.sqrt(torch.as_tensor(K, dtype=x.dtype, device=x.device))
    diff = x - y
    s = 0.5 * K * lorentz_inner(diff, diff)          # = alpha - 1, 无抵消
    s = torch.clamp(s, min=0.0)
    # 数值卫生: s→0 (点重合) 时 d/ds sqrt(s^2+2s) → ∞ 导致对 K/输入的梯度 NaN。
    # 加 dtype-aware eps 地板使 sqrt 参数恒 >0, 梯度有限。FP64 用极小 eps (不污染
    # round-trip 精度), FP32 用稍大 eps。
    eps = 1e-15 if x.dtype == torch.float64 else 1e-12
    arcosh = torch.log1p(s + torch.sqrt(s * s + 2.0 * s + eps))
    return arcosh / sqrtK


# ----------------------------------------------------------------------------
# 指数 / 对数映射
# ----------------------------------------------------------------------------
def _sinhc(z, eps=1e-7):
    """sinh(z)/z, 在 z->0 用 Taylor 稳定: 1 + z^2/6 + ..."""
    small = z.abs() < eps
    z_safe = torch.where(small, torch.ones_like(z), z)
    out = torch.sinh(z_safe) / z_safe
    taylor = 1.0 + z * z / 6.0
    return torch.where(small, taylor, out)


def exp_map(x, v, K):
    """指数映射  (README §1.4)

    Exp_x^K(v) = cosh(sqrt K ||v||_x) x + sinh(sqrt K ||v||_x)/(sqrt K ||v||_x) v
    v=0 时返回 x。
    """
    sqrtK = torch.sqrt(torch.as_tensor(K, dtype=x.dtype, device=x.device))
    vnorm = lorentz_norm(v, keepdim=True)          # ||v||_x
    theta = sqrtK * vnorm                            # sqrt K ||v||_x
    cosh = torch.cosh(theta)
    # 需要系数 sinh(theta)/(sqrt K ||v||_x)。因 theta = sqrtK*vnorm,
    # sinhc(theta) = sinh(theta)/theta = sinh(theta)/(sqrt K ||v||_x) 正是所需系数。
    coef_v = _sinhc(theta)
    return cosh * x + coef_v * v


def log_map(x, y, K, eps=1e-12):
    """对数映射  (README §1.5)

    alpha = -K <x,y>_L
    Log_x^K(y) = arcosh(alpha)/sqrt(alpha^2 - 1) * (y - alpha x)

    数值稳定: 令 s = alpha - 1 = (K/2)<x-y,x-y>_L (无抵消, 见 dist)。
        arcosh(alpha) = log1p(s + sqrt(s^2+2s))
        sqrt(alpha^2-1) = sqrt(s(s+2)) = sqrt(s^2+2s)
        y - alpha x = (y - x) - s x
    coef = arcosh(alpha)/sqrt(alpha^2-1); s->0 时 coef->1 (Taylor)。
    """
    diff = x - y
    s = 0.5 * K * lorentz_inner(diff, diff, keepdim=True)   # alpha - 1 >= 0
    s = torch.clamp(s, min=0.0)
    sq = torch.sqrt(s * s + 2.0 * s + eps)                   # sqrt(alpha^2-1)
    arcosh = torch.log1p(s + sq)
    coef = arcosh / (sq + eps)                               # ->1 as s->0
    coef = torch.where(s < 1e-12, torch.ones_like(coef), coef)
    return coef * ((y - x) - s * x)


# ----------------------------------------------------------------------------
# Riemannian gradient
# ----------------------------------------------------------------------------
def egrad_to_rgrad(x, egrad, K):
    """欧氏(环境)梯度 -> Riemannian gradient  (README §1.6)

    J = diag(-1,1,...,1);  Lorentz gradient = J egrad
    grad = J egrad + K <x, J egrad>_L x
    """
    lgrad = egrad.clone()
    lgrad[..., 0] = -lgrad[..., 0]                   # J egrad
    coef = K * lorentz_inner(x, lgrad, keepdim=True)
    return lgrad + coef * x


# ----------------------------------------------------------------------------
# 平行移动
# ----------------------------------------------------------------------------
def parallel_transport(x, y, v, K, eps=1e-12):
    """沿 x->y 测地线的平行移动  (README §1.7)

    P_{x->y}^K(v) = v + K<y,v>_L / (1 - K<x,y>_L) * (x + y)
    """
    yv = lorentz_inner(y, v, keepdim=True)
    xy = lorentz_inner(x, y, keepdim=True)
    denom = 1.0 - K * xy
    coef = K * yv / (denom + eps)
    return v + coef * (x + y)


# ----------------------------------------------------------------------------
# 跨曲率映射
# ----------------------------------------------------------------------------
def cross_curvature_point(x, K, K_new):
    """跨曲率点映射 Psi_{K->K'}(x) = sqrt(K/K') x  (README §2.3)"""
    scale = torch.sqrt(torch.as_tensor(K / K_new, dtype=x.dtype, device=x.device))
    return scale * x


def cross_curvature_tangent(v, K, K_new):
    """跨曲率切映射 C_{K->K'}(v) = sqrt(K/K') v  (README §2.4)"""
    scale = torch.sqrt(torch.as_tensor(K / K_new, dtype=v.dtype, device=v.device))
    return scale * v


# ----------------------------------------------------------------------------
# 曲率参数化 (README §2.1 / §3.1)
# ----------------------------------------------------------------------------
def softplus(c):
    return torch.nn.functional.softplus(c)


def curvature_from_coord(c, K_min):
    """K = K_min + softplus(c)"""
    return K_min + softplus(c)


def coord_from_curvature(K, K_min, eps=1e-7):
    """inverse softplus: c = log(exp(s)-1), s = K - K_min"""
    s = torch.clamp(K - K_min, min=eps)
    # 稳定 inverse softplus: log(expm1(s)) ; 大 s 时 ≈ s
    return s + torch.log(-torch.expm1(-s) + eps)


def reproject_momentum(x, m, K, eps=1e-12):
    """动量切空间重投影 + 保范数  (数值卫生, 不改 §3 数学)

    FP32 长轨迹后 m 可能轻微偏离 T_x。先投回切空间, 再 rescale 使
    Lorentz 范数与投影前一致 (保持 §3 动量范数语义)。
    返回 (m_reproj, tangent_residual_before, norm_before)。
    """
    tan_res_before = lorentz_inner(x, m).abs().max().item()
    norm_before = lorentz_norm(m, keepdim=True)
    m_proj = project_to_tangent(x, m, K)
    norm_proj = lorentz_norm(m_proj, keepdim=True)
    # 保范数: 缩放回投影前的范数 (norm_proj=0 时保持 0)
    scale = torch.where(norm_proj > eps, norm_before / (norm_proj + eps),
                        torch.ones_like(norm_proj))
    m_reproj = m_proj * scale
    return m_reproj, tan_res_before, norm_before.mean().item()


def manifold_residual(x, K):
    """|<x,x>_L + 1/K| 的逐点最大值 (重投影前诊断)"""
    return (lorentz_inner(x, x) + 1.0 / K).abs().max().item()


def clip_log_curvature(K_old, K_new, delta_K):
    """限制单步 |log K_new - log K_old| <= delta_K  (README §3.2 Step5)"""
    log_old = torch.log(K_old)
    log_new = torch.log(K_new)
    log_new = torch.clamp(log_new, log_old - delta_K, log_old + delta_K)
    return torch.exp(log_new)
