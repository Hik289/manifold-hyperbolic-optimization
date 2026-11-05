# HyperbolicMomentum: Curvature-Adaptive Momentum for Hierarchical Neural Optimization

## 0. 项目定位

许多学习任务具有层级结构，例如：

- taxonomy reconstruction；
- tree decoding；
- hierarchical classification；
- knowledge graph embedding；
- mixture-of-experts routing；
- hierarchical clustering；
- graph representation learning。

欧氏空间中的球体体积随半径按多项式增长，而负曲率空间中的体积随半径近似指数增长。因此，双曲空间适合表示树和层级数据。

假设参数组位于曲率为：

$$
\kappa<0
$$

的双曲流形：

$$
\mathbb H_\kappa^d.
$$

普通 Riemannian stochastic gradient descent（RSGD）使用：

$$
x_{t+1}
=
\operatorname{Exp}_{x_t}
\left(
-\eta_t\operatorname{grad}f(x_t)
\right).
$$

但动量不能直接写成：

$$
m_t
=
\beta m_{t-1}
+
(1-\beta)\operatorname{grad}f(x_t),
$$

因为：

$$
m_{t-1}\in T_{x_{t-1}}\mathbb H_\kappa^d,
$$

而：

$$
\operatorname{grad}f(x_t)\in T_{x_t}\mathbb H_\kappa^d.
$$

两个向量位于不同切空间，直接相加没有几何意义。

本项目提出一个单一算法：

> **HyperbolicMomentum：在分组 Lorentz 双曲流形上维护平行移动动量，同时在线更新每组负曲率，并在曲率改变时对参数和动量进行一致的跨曲率映射。**

主算法固定包含三个新机制：

1. **切空间一致的双曲动量**
   $$
   m_t
   =
   \beta
   \mathcal P_{x_{t-1}\rightarrow x_t}^{\kappa_t}
   (m_{t-1})
   +
   (1-\beta)\operatorname{grad}_{x}F(x_t,\kappa_t).
   $$

2. **可学习的分组负曲率**
   $$
   \kappa_{g,t}
   =
   -\operatorname{softplus}(c_{g,t})
   -
   \kappa_{\min}.
   $$

3. **跨曲率状态 transport**
   当：
   $$
   \kappa_{g,t+1}\ne\kappa_{g,t},
   $$
   参数点和动量都通过同一 Lorentz 归一化映射进入新的双曲流形，而不是只改变一个曲率标量。

主算法的计算链为：

$$
\boxed{
\text{Riemannian gradient}
\rightarrow
\text{parallel transport}
\rightarrow
\text{tangent momentum}
\rightarrow
\text{curvature-aware step}
\rightarrow
\text{curvature update}
\rightarrow
\text{cross-curvature transport}
}
$$

核心科学问题是：

> **在层级结构的局部增长率随参数组和训练阶段变化时，固定曲率是否限制了优化；分组可学习曲率和一致的跨曲率动量 transport 是否可以同时改善表示失真、优化稳定性和收敛速度？**

所有正式实验只运行：

$$
\boxed{\textbf{1 seed}}
$$

固定：

$$
\boxed{\texttt{seed}=42}.
$$

所有长训练至少运行：

$$
\boxed{\textbf{1000 epochs}}.
$$

每个 epoch 的任务指标、损失、曲率、半径、动量范数、transport error、边界距离和 wall-clock 全部保存，用于绘制完整轨迹。

---

# 1. 双曲流形建模

## 1.1 Lorentz 模型

主算法统一使用 Lorentz 模型。

令：

$$
K=-\kappa>0.
$$

定义 Lorentz 内积：

$$
\langle x,y\rangle_{\mathcal L}
=
-x_0y_0
+
\sum_{i=1}^{d}x_iy_i.
$$

曲率为：

$$
\kappa=-K
$$

的双曲空间定义为：

$$
\boxed{
\mathbb H_{-K}^{d}
=
\left\{
x\in\mathbb R^{d+1}:
\langle x,x\rangle_{\mathcal L}
=
-\frac1K,
\quad
x_0>0
\right\}.
}
$$

原点为：

$$
o_K
=
\left(
\frac1{\sqrt K},
0,\ldots,0
\right).
$$

## 1.2 切空间

在：

$$
x\in\mathbb H_{-K}^d
$$

处，切空间为：

$$
\boxed{
T_x\mathbb H_{-K}^{d}
=
\{
v\in\mathbb R^{d+1}:
\langle x,v\rangle_{\mathcal L}=0
\}.
}
$$

切向量范数为：

$$
\|v\|_x
=
\sqrt{
\langle v,v\rangle_{\mathcal L}
}.
$$

## 1.3 距离

两点距离：

$$
\boxed{
d_K(x,y)
=
\frac1{\sqrt K}
\operatorname{arcosh}
\left(
-K\langle x,y\rangle_{\mathcal L}
\right).
}
$$

## 1.4 指数映射

对：

$$
v\in T_x\mathbb H_{-K}^{d},
$$

指数映射为：

$$
\boxed{
\operatorname{Exp}_x^K(v)
=
\cosh
\left(
\sqrt K\|v\|_x
\right)x
+
\frac{
\sinh(\sqrt K\|v\|_x)
}{
\sqrt K\|v\|_x
}
v.
}
$$

当：

$$
v=0
$$

时定义为 $x$。

## 1.5 对数映射

令：

$$
\alpha
=
-K\langle x,y\rangle_{\mathcal L}.
$$

则：

$$
\boxed{
\operatorname{Log}_x^K(y)
=
\frac{
\operatorname{arcosh}(\alpha)
}{
\sqrt{\alpha^2-1}
}
\left(
y-\alpha x
\right).
}
$$

该向量满足：

$$
\operatorname{Log}_x^K(y)
\in
T_x\mathbb H_{-K}^{d}.
$$

并且：

$$
\|
\operatorname{Log}_x^K(y)
\|_x
=
d_K(x,y).
$$

## 1.6 Riemannian gradient

给定环境欧氏梯度：

$$
\nabla_x^{\mathrm E}F,
$$

先使用 Lorentz metric inverse：

$$
J
=
\operatorname{diag}(-1,1,\ldots,1).
$$

环境 Lorentz gradient：

$$
\nabla_x^{\mathcal L}F
=
J\nabla_x^{\mathrm E}F.
$$

投影到切空间：

$$
\boxed{
\operatorname{grad}_xF
=
\nabla_x^{\mathcal L}F
+
K
\langle
x,\nabla_x^{\mathcal L}F
\rangle_{\mathcal L}
x.
}
$$

## 1.7 平行移动

沿 $x$ 到 $y$ 的唯一最短测地线，平行移动定义为：

$$
\boxed{
\mathcal P_{x\rightarrow y}^{K}(v)
=
v
+
\frac{
K\langle y,v\rangle_{\mathcal L}
}{
1-K\langle x,y\rangle_{\mathcal L}
}
(x+y).
}
$$

它满足：

$$
\mathcal P_{x\rightarrow y}^{K}(v)
\in
T_y\mathbb H_{-K}^{d},
$$

以及范数保持：

$$
\left\|
\mathcal P_{x\rightarrow y}^{K}(v)
\right\|_y
=
\|v\|_x.
$$

## 1.8 参数组

模型包含 $G$ 个双曲参数组：

$$
\mathcal X
=
\prod_{g=1}^{G}
\mathbb H_{-K_g}^{d_g}.
$$

每组对应：

- 一个层级分类头；
- 一个 embedding table；
- 一个路由器；
- 一个图表示层；
- 一个树 decoder 层；
- 一个知识图谱关系族。

乘积流形 metric 为：

$$
g_{\mathcal X}
=
\bigoplus_{g=1}^{G}
g_{K_g}.
$$

每组有独立曲率：

$$
K_g>0.
$$

---

# 2. 可学习曲率与跨曲率几何

## 2.1 曲率参数化

为保证负曲率，优化 unconstrained scalar：

$$
c_g\in\mathbb R.
$$

定义：

$$
\boxed{
K_g
=
K_{\min}
+
\operatorname{softplus}(c_g),
}
$$

所以：

$$
\kappa_g=-K_g<0.
$$

固定：

$$
K_{\min}>0.
$$

## 2.2 为什么不能只修改曲率标量

点：

$$
x\in\mathbb H_{-K}^{d}
$$

满足：

$$
\langle x,x\rangle_{\mathcal L}
=
-\frac1K.
$$

若直接把 $K$ 改为 $K'$ 而保持 $x$ 不变，则一般有：

$$
\langle x,x\rangle_{\mathcal L}
\ne
-\frac1{K'}.
$$

所以 $x$ 不再属于新流形。

同样，旧动量：

$$
m\in T_x\mathbb H_{-K}^{d}
$$

也不能被无条件解释为新流形上的切向量。

## 2.3 跨曲率点映射

定义曲率归一化映射：

$$
\boxed{
\Psi_{K\rightarrow K'}(x)
=
\sqrt{\frac K{K'}}x.
}
$$

因为：

$$
\left\langle
\Psi_{K\rightarrow K'}(x),
\Psi_{K\rightarrow K'}(x)
\right\rangle_{\mathcal L}
=
\frac K{K'}
\left(
-\frac1K
\right)
=
-\frac1{K'}.
$$

所以：

$$
\Psi_{K\rightarrow K'}(x)
\in
\mathbb H_{-K'}^{d}.
$$

## 2.4 跨曲率切向量映射

定义：

$$
\boxed{
\mathcal C_{K\rightarrow K'}(v)
=
\sqrt{\frac K{K'}}v.
}
$$

若：

$$
\langle x,v\rangle_{\mathcal L}=0,
$$

则：

$$
\left\langle
\Psi_{K\rightarrow K'}(x),
\mathcal C_{K\rightarrow K'}(v)
\right\rangle_{\mathcal L}
=
\frac K{K'}
\langle x,v\rangle_{\mathcal L}
=
0.
$$

因此：

$$
\mathcal C_{K\rightarrow K'}(v)
\in
T_{\Psi(x)}
\mathbb H_{-K'}^{d}.
$$

## 2.5 距离保持性质

对：

$$
x,y\in\mathbb H_{-K}^{d},
$$

有：

$$
-K'
\left\langle
\Psi(x),\Psi(y)
\right\rangle_{\mathcal L}
=
-K
\langle x,y\rangle_{\mathcal L}.
$$

所以：

$$
\boxed{
d_{K'}
(
\Psi_{K\rightarrow K'}(x),
\Psi_{K\rightarrow K'}(y)
)
=
\sqrt{\frac K{K'}}
d_K(x,y).
}
$$

跨曲率映射保持无量纲双曲角，但按曲率半径缩放真实距离。

## 2.6 曲率变化的几何代价

定义：

$$
\Delta\log K_t
=
\log K_{t+1}-\log K_t.
$$

小变化下：

$$
\sqrt{
\frac{K_t}{K_{t+1}}
}
=
1
-
\frac12\Delta\log K_t
+
O((\Delta\log K_t)^2).
$$

所以跨曲率点和动量扰动是一阶受控的。

## 2.7 曲率目标

主算法把曲率作为模型参数，通过任务损失和一个固定正则项更新：

$$
\boxed{
\mathcal J(x,K)
=
F(x,K)
+
\lambda_K
\sum_{g=1}^{G}
\left(
\log K_g-\log K_{\mathrm{ref}}
\right)^2.
}
$$

该正则避免曲率无界增大或退化到近欧氏空间。

曲率梯度：

$$
g_{c_g}
=
\frac{\partial\mathcal J}{\partial c_g}.
$$

---

# 3. 主算法：HyperbolicMomentum

本节只定义一个算法：

> **HyperbolicMomentum：在固定 Lorentz 模型中维护平行移动动量，并使用分组可学习曲率与解析跨曲率映射更新参数。**

主算法不包含多个可选 transport、不混入 Riemannian Adam、不使用不同双曲模型，也不加入额外候选步长规则。

## 3.1 优化状态

对每个参数组 $g$，维护：

$$
x_{g,t}
\in
\mathbb H_{-K_{g,t}}^{d_g},
$$

$$
m_{g,t}
\in
T_{x_{g,t}}
\mathbb H_{-K_{g,t}}^{d_g},
$$

以及曲率坐标：

$$
c_{g,t}\in\mathbb R.
$$

曲率定义为：

$$
K_{g,t}
=
K_{\min}
+
\operatorname{softplus}(c_{g,t}),
$$

因此实际截面曲率为：

$$
\kappa_{g,t}=-K_{g,t}<0.
$$

初始化：

$$
m_{g,0}=0.
$$

## 3.2 一次完整更新

每一步只执行以下六个步骤。

### Step 1：计算 Riemannian gradient

$$
g_{g,t}
=
\operatorname{grad}_{x_g}
\mathcal J(x_t,K_t).
$$

其中：

$$
g_{g,t}
\in
T_{x_{g,t}}
\mathbb H_{-K_{g,t}}^{d_g}.
$$

### Step 2：更新同点切空间动量

上一时刻动量已经在当前点切空间中完成 transport，因此：

$$
\widetilde m_{g,t}
=
\beta m_{g,t-1}
+
(1-\beta)g_{g,t}.
$$

### Step 3：使用曲率归一化步长

定义：

$$
\chi_{g,t}
=
\sqrt{K_{g,t}}
\|
\widetilde m_{g,t}
\|_{x_{g,t}}.
$$

主算法固定使用：

$$
\eta_{g,t}^{\mathrm{geo}}
=
\frac{
\eta_t
}{
1+\gamma\chi_{g,t}
}.
$$

该项控制指数映射参数：

$$
\sqrt{K_{g,t}}
\eta_{g,t}^{\mathrm{geo}}
\|
\widetilde m_{g,t}
\|.
$$

### Step 4：沿指数映射更新参数并平行移动动量

$$
x_{g,t+1/2}
=
\operatorname{Exp}_{x_{g,t}}^{K_{g,t}}
\left(
-\eta_{g,t}^{\mathrm{geo}}
\widetilde m_{g,t}
\right).
$$

然后：

$$
m_{g,t+1/2}
=
\mathcal P_{x_{g,t}\rightarrow x_{g,t+1/2}}^{K_{g,t}}
(
\widetilde m_{g,t}
).
$$

### Step 5：更新曲率

$$
c_{g,t+1}
=
c_{g,t}
-
\rho_t
\frac{
\partial\mathcal J
}{
\partial c_g
}.
$$

得到：

$$
K_{g,t+1}
=
K_{\min}
+
\operatorname{softplus}(c_{g,t+1}).
$$

限制单步变化：

$$
\left|
\log K_{g,t+1}
-
\log K_{g,t}
\right|
\le
\delta_K.
$$

### Step 6：跨曲率映射参数和动量

参数点：

$$
x_{g,t+1}
=
\sqrt{
\frac{
K_{g,t}
}{
K_{g,t+1}
}
}
x_{g,t+1/2}.
$$

动量：

$$
m_{g,t+1}
=
\operatorname{Proj}_{T_{x_{g,t+1}}}
\left[
\sqrt{
\frac{
K_{g,t}
}{
K_{g,t+1}
}
}
m_{g,t+1/2}
\right].
$$

## 3.3 完整算法公式

$$
\boxed{
\begin{aligned}
g_{g,t}
&=
\operatorname{grad}_{x_g}
\mathcal J(x_t,K_t),
\\
\widetilde m_{g,t}
&=
\beta m_{g,t-1}
+
(1-\beta)g_{g,t},
\\
\chi_{g,t}
&=
\sqrt{K_{g,t}}
\|
\widetilde m_{g,t}
\|,
\\
\eta_{g,t}^{\mathrm{geo}}
&=
\frac{\eta_t}{1+\gamma\chi_{g,t}},
\\
x_{g,t+1/2}
&=
\operatorname{Exp}_{x_{g,t}}^{K_{g,t}}
\left(
-\eta_{g,t}^{\mathrm{geo}}
\widetilde m_{g,t}
\right),
\\
m_{g,t+1/2}
&=
\mathcal P_{x_{g,t}\rightarrow x_{g,t+1/2}}^{K_{g,t}}
(
\widetilde m_{g,t}
),
\\
K_{g,t+1}
&=
K_{\min}
+
\operatorname{softplus}
\left(
c_{g,t}
-
\rho_t
\partial_{c_g}\mathcal J
\right),
\\
x_{g,t+1}
&=
\sqrt{
\frac{
K_{g,t}
}{
K_{g,t+1}
}
}
x_{g,t+1/2},
\\
m_{g,t+1}
&=
\operatorname{Proj}_{T_{x_{g,t+1}}}
\left[
\sqrt{
\frac{
K_{g,t}
}{
K_{g,t+1}
}
}
m_{g,t+1/2}
\right].
\end{aligned}
}
$$

## 3.4 伪代码

```text
Algorithm: HyperbolicMomentum

for each group g:
    initialize x_g on H_{-K_g}
    initialize m_g = 0
    initialize curvature coordinate c_g

for t = 0, ..., T - 1:

    compute task loss and curvature regularizer

    for each group g:

        grad_g = riemannian_gradient(loss, x_g, K_g)

        m_tilde = beta * m_g + (1 - beta) * grad_g

        chi = sqrt(K_g) * lorentz_norm(m_tilde)

        eta_geo = eta_t / (1 + gamma * chi)

        x_half = exp_map(x_g, -eta_geo * m_tilde, K_g)

        m_half = parallel_transport(
            x_g,
            x_half,
            m_tilde,
            K_g
        )

        c_new = c_g - rho_t * grad_curvature(loss, c_g)
        K_new = K_min + softplus(c_new)
        K_new = clip_log_curvature_change(K_g, K_new, delta_K)

        scale = sqrt(K_g / K_new)

        x_g = scale * x_half
        m_g = tangent_projection(
            x_g,
            scale * m_half,
            K_new
        )

        c_g = inverse_softplus(K_new - K_min)
        K_g = K_new
```

## 3.5 固定实现规则

主算法固定使用：

- Lorentz model；
- 精确指数映射；
- 精确平行移动；
- 每组独立负曲率；
- softplus 曲率参数化；
- log-curvature clipping；
- 参数与动量同步跨曲率映射；
- FP32 几何运算；
- 无 Riemannian Adam；
- 无多种 transport 候选；
- 无 Poincaré ball 作为主实现。

## 3.6 复杂度

每个参数组每步：

$$
O(d_g).
$$

总状态：

$$
O
\left(
\sum_g d_g+G
\right).
$$

---

# 4. 深入理论

本节不再把理论拆成很多彼此独立的小结论，而是围绕负曲率流形上的四个核心对象建立完整链条：

1. **Jacobi field 与指数映射畸变**；
2. **Hessian comparison 与曲率自适应步长**；
3. **可学习曲率下的两时间尺度动力学与非凸收敛**；
4. **曲率驱动的稳定性与 generalization**。

最终结果把：

$$
\text{sectional curvature}
\rightarrow
\text{Jacobi growth}
\rightarrow
\text{Exp-map conditioning}
\rightarrow
\text{momentum transport error}
\rightarrow
\text{optimization convergence}
\rightarrow
\text{generalization}
$$

连接成一个端到端理论。

---

## 4.1 几何预备：负曲率、Jacobi field 与测地线发散

考虑常负曲率：

$$
\operatorname{Sec}
\equiv
-K,
\qquad
K>0.
$$

令：

$$
\gamma:[0,\ell]\rightarrow\mathbb H_{-K}^{d}
$$

为单位速度测地线：

$$
\|\dot\gamma(t)\|=1.
$$

沿 $\gamma$ 的 Jacobi field：

$$
J(t)
$$

满足：

$$
\nabla_{\dot\gamma}^2J
+
R(J,\dot\gamma)\dot\gamma
=
0.
$$

在常曲率 $-K$ 下，若：

$$
J(t)\perp\dot\gamma(t),
$$

则曲率张量满足：

$$
R(J,\dot\gamma)\dot\gamma
=
-KJ.
$$

因此 Jacobi equation 变为：

$$
\boxed{
\nabla_{\dot\gamma}^2J
-
KJ
=
0.
}
$$

若：

$$
J(0)=0,
\qquad
\nabla_{\dot\gamma}J(0)=v_\perp,
$$

则：

$$
\boxed{
J(t)
=
\frac{
\sinh(\sqrt Kt)
}{
\sqrt K
}
\mathcal P_{0\rightarrow t}(v_\perp).
}
$$

这说明在负曲率空间中，相邻测地线的横向分离按：

$$
\sinh(\sqrt Kt)
$$

增长，而不是欧氏空间中的线性增长。

该增长率是整个算法理论的核心：它一方面解释双曲空间能够容纳树状结构，另一方面也说明大曲率和大步长会放大优化轨迹的敏感性。

---

## 4.2 主定理一：曲率控制的指数映射畸变与最优步长

定义：

$$
\operatorname{Exp}_x^K:
T_x\mathbb H_{-K}^d
\rightarrow
\mathbb H_{-K}^d.
$$

对：

$$
v\in T_x\mathbb H_{-K}^d,
\qquad
r=\|v\|,
$$

将任意扰动：

$$
\xi\in T_x\mathbb H_{-K}^d
$$

分解为：

$$
\xi
=
\xi_\parallel
+
\xi_\perp,
$$

其中：

$$
\xi_\parallel
\parallel v,
\qquad
\xi_\perp
\perp v.
$$

由 Jacobi field 公式可得：

$$
D\operatorname{Exp}_x^K(v)[\xi]
=
\mathcal P_{x\rightarrow \operatorname{Exp}_x(v)}
\left[
\xi_\parallel
+
\frac{
\sinh(\sqrt Kr)
}{
\sqrt Kr
}
\xi_\perp
\right].
$$

因此径向和横向方向的放大因子分别为：

$$
1,
\qquad
\frac{
\sinh(\sqrt Kr)
}{
\sqrt Kr
}.
$$

### Theorem 1：曲率—步长—条件数统一定理

设 HyperbolicMomentum 在某一步使用：

$$
v_t
=
-\eta_t^{\mathrm{geo}}m_t.
$$

定义无量纲曲率步长：

$$
\delta_t
=
\sqrt{K_t}
\eta_t^{\mathrm{geo}}
\|m_t\|.
$$

则指数映射微分的条件数精确满足：

$$
\boxed{
\kappa
\left(
D\operatorname{Exp}_{x_t}^{K_t}(v_t)
\right)
=
\frac{
\sinh(\delta_t)
}{
\delta_t
}.
}
$$

并且有以下三个结论。

### (i) 小步长局部近欧氏展开

当：

$$
\delta_t\ll1,
$$

有：

$$
\frac{\sinh\delta_t}{\delta_t}
=
1
+
\frac{\delta_t^2}{6}
+
O(\delta_t^4).
$$

所以曲率导致的局部畸变为二阶：

$$
O
\left(
K_t
(\eta_t^{\mathrm{geo}})^2
\|m_t\|^2
\right).
$$

### (ii) 大步长指数放大

当：

$$
\delta_t\gg1,
$$

有：

$$
\frac{\sinh\delta_t}{\delta_t}
\asymp
\frac{
e^{\delta_t}
}{
2\delta_t
}.
$$

因此若不控制：

$$
\sqrt K\,\eta\|m\|,
$$

轨迹扰动、随机梯度噪声和数值误差都会被指数放大。

### (iii) 曲率归一化步长的统一控制

若采用：

$$
\eta_t^{\mathrm{geo}}
=
\frac{
\eta_t
}{
1+\gamma\sqrt{K_t}\|m_t\|
},
$$

则：

$$
\delta_t
=
\frac{
\eta_t\sqrt{K_t}\|m_t\|
}{
1+\gamma\sqrt{K_t}\|m_t\|
}
\le
\frac{\eta_t}{\gamma}.
$$

因此：

$$
\boxed{
\kappa
\left(
D\operatorname{Exp}_{x_t}^{K_t}
\right)
\le
\frac{
\sinh(\eta_t/\gamma)
}{
\eta_t/\gamma
}.
}
$$

该上界与：

$$
K_t
$$

和：

$$
\|m_t\|
$$

无关。

这说明曲率归一化步长不是启发式缩放，而是对指数映射最坏条件数的统一控制。

### Corollary 1.1：随机梯度噪声传播

若切空间噪声为：

$$
\zeta_t,
$$

则参数空间中的一阶扰动满足：

$$
d_{K_t}
\left(
\operatorname{Exp}_{x_t}(v_t),
\operatorname{Exp}_{x_t}(v_t+\zeta_t)
\right)
\le
\frac{
\sinh(\delta_t+\sqrt{K_t}\|\zeta_t\|)
}{
\delta_t+\sqrt{K_t}\|\zeta_t\|
}
\|\zeta_t\|.
$$

因此曲率校正步长同时控制随机噪声传播。

---

## 4.3 Hessian comparison 与目标 landscape

设：

$$
f_y(x)
=
\frac12d_K^2(x,y).
$$

记：

$$
r=d_K(x,y).
$$

在常负曲率空间中，Hessian comparison 给出：

$$
\operatorname{Hess}f_y
=
dr\otimes dr
+
r\sqrt K
\coth(\sqrt Kr)
\left(
g-dr\otimes dr
\right).
$$

因此：

- 径向特征值为：
  $$
  1;
  $$
- 横向特征值为：
  $$
  \sqrt Kr\coth(\sqrt Kr).
  $$

当：

$$
r\rightarrow0,
$$

有：

$$
\sqrt Kr\coth(\sqrt Kr)
=
1
+
\frac{Kr^2}{3}
+
O(K^2r^4).
$$

当：

$$
r\rightarrow\infty,
$$

有：

$$
\sqrt Kr\coth(\sqrt Kr)
\sim
\sqrt Kr.
$$

因此负曲率不会破坏平方距离目标的凸性，但会使远距离横向方向变得更陡。

### Theorem 2：曲率驱动的 landscape 统一定理

考虑目标：

$$
F_K(x)
=
\mathbb E_y
\left[
\frac12d_K^2(x,y)
\right]
+
R(x),
$$

其中：

$$
R
$$

是 geodesically $\mu_R$-strongly convex 且 $L_R$-smooth 的正则项。

假设所有样本满足：

$$
d_K(x,y)\le R_{\max}
$$

于某 geodesic ball 内。

则：

### (i) Geodesic strong convexity

$$
\boxed{
\operatorname{Hess}F_K(x)
\succeq
(1+\mu_R)I.
}
$$

因此：

$$
F_K
$$

在该球内至少：

$$
1+\mu_R
$$

geodesically strongly convex。

### (ii) Smoothness

$$
\boxed{
\operatorname{Hess}F_K(x)
\preceq
\left[
\sqrt K R_{\max}
\coth(\sqrt K R_{\max})
+
L_R
\right]I.
}
$$

所以 smoothness 常数可取：

$$
L_K(R_{\max})
=
\sqrt K R_{\max}
\coth(\sqrt K R_{\max})
+
L_R.
$$

### (iii) 条件数

$$
\boxed{
\operatorname{cond}(F_K)
\le
\frac{
\sqrt K R_{\max}
\coth(\sqrt K R_{\max})
+
L_R
}{
1+\mu_R
}.
}
$$

这说明曲率既带来表示容量，也可能增大远距离横向 stiffness。

### (iv) 曲率学习的空间最优点

若层级数据的分支率为：

$$
b,
$$

平均层间径向增量为：

$$
\Delta r,
$$

则容量匹配曲率：

$$
K_{\mathrm{cap}}
=
\left[
\frac{
\log b
}{
(d-1)\Delta r
}
\right]^2.
$$

另一方面，优化条件数随：

$$
K
$$

增长。

因此正则化曲率目标：

$$
\mathcal R(K)
=
\operatorname{Distortion}(K)
+
\lambda_{\mathrm{opt}}
\operatorname{cond}(F_K)
$$

存在一个非退化折中点：

$$
K^\star.
$$

在：

$$
u=\log K
$$

坐标中，若：

$$
\mathcal R''(u)\ge\mu_K>0,
$$

则该最优点唯一。

### Corollary 2.1：分组曲率严格优于单曲率的条件

若组 $g$ 的：

$$
K_g^\star
$$

不同，且每组：

$$
\mathcal R_g(u_g)
$$

严格凸，则：

$$
\sum_g
\min_{u_g}
\mathcal R_g(u_g)
<
\min_u
\sum_g
\mathcal R_g(u)
$$

只要至少两个组的最优曲率不同。

因此分组曲率的优势不是“参数更多”，而是不同层级增长率对应不同几何最优点。

---

## 4.4 主定理三：可学习曲率下的两时间尺度收敛

令联合状态：

$$
z_t=(x_t,u_t,m_t),
$$

其中：

$$
u_t=\log K_t.
$$

定义联合目标：

$$
\mathcal J(x,u)
=
F_{e^u}(x)
+
\lambda_u(u-u_{\mathrm{ref}})^2.
$$

算法更新写为：

$$
m_t
=
\beta
\mathcal T_{t-1\rightarrow t}(m_{t-1})
+
(1-\beta)\widehat g_t,
$$

$$
x_{t+1/2}
=
\operatorname{Exp}_{x_t}^{K_t}
\left(
-\eta_t^{\mathrm{geo}}m_t
\right),
$$

$$
u_{t+1}
=
u_t
-
\rho_t\widehat h_t,
$$

其中：

$$
\widehat h_t
$$

是曲率梯度估计。

再通过：

$$
\Psi_{K_t\rightarrow K_{t+1}}
$$

把 $x_{t+1/2}$ 和 $m_t$ 同步映射到新曲率流形。

### 假设

**A1. 紧致安全区域**

$$
K_t\in[K_{\min},K_{\max}],
$$

且：

$$
d_{K_t}(x_t,o)\le R_{\max}.
$$

**A2. 空间梯度光滑**

存在：

$$
L_x
$$

使：

$$
\|
\operatorname{grad}_x\mathcal J(x,u)
-
\mathcal P_{y\rightarrow x}
\operatorname{grad}_x\mathcal J(y,u)
\|
\le
L_xd_K(x,y).
$$

**A3. 曲率梯度光滑**

$$
|
\partial_u\mathcal J(x,u)
-
\partial_u\mathcal J(y,v)
|
\le
L_u
\left[
d_K(x,y)+|u-v|
\right].
$$

**A4. 无偏随机梯度**

$$
\mathbb E_t[\widehat g_t]
=
\operatorname{grad}_x\mathcal J(x_t,u_t),
$$

$$
\mathbb E_t[\widehat h_t]
=
\partial_u\mathcal J(x_t,u_t).
$$

**A5. 方差有界**

$$
\mathbb E_t
\|
\widehat g_t-\operatorname{grad}_x\mathcal J
\|^2
\le
\sigma_x^2,
$$

$$
\mathbb E_t
|
\widehat h_t-\partial_u\mathcal J
|^2
\le
\sigma_u^2.
$$

**A6. 两时间尺度**

$$
\frac{\rho_t}{\eta_t}\rightarrow0.
$$

### Theorem 3：联合非凸收敛主定理

定义联合 stationarity measure：

$$
\mathcal G_t
=
\|
\operatorname{grad}_x\mathcal J(x_t,u_t)
\|^2
+
\lambda_c
|
\partial_u\mathcal J(x_t,u_t)
|^2.
$$

则存在常数：

$$
C_1,\ldots,C_7
$$

只依赖于：

$$
K_{\min},
K_{\max},
R_{\max},
L_x,
L_u,
\beta,
\gamma
$$

使：

$$
\boxed{
\begin{aligned}
\frac{
\sum_{t=0}^{T-1}
\eta_t^{\mathrm{geo}}
\mathbb E[\mathcal G_t]
}{
\sum_{t=0}^{T-1}
\eta_t^{\mathrm{geo}}
}
\le&
\frac{
C_1
[
\mathcal J(z_0)-\mathcal J_\star
]
}{
\sum_t\eta_t^{\mathrm{geo}}
}
\\
&
+
C_2
\frac{
\sum_t
(\eta_t^{\mathrm{geo}})^2
\sigma_x^2
}{
\sum_t\eta_t^{\mathrm{geo}}
}
\\
&
+
C_3
\frac{
\sum_t
\rho_t^2
\sigma_u^2
}{
\sum_t\eta_t^{\mathrm{geo}}
}
\\
&
+
C_4
\frac{
\sum_t
\eta_t^{\mathrm{geo}}
\mathbb E\|e_{m,t}\|^2
}{
\sum_t\eta_t^{\mathrm{geo}}
}
\\
&
+
C_5
\frac{
\sum_t
\eta_t^{\mathrm{geo}}
|\Delta u_t|^2
}{
\sum_t\eta_t^{\mathrm{geo}}
}
\\
&
+
C_6
\frac{
\sum_t
\eta_t^{\mathrm{geo}}
e_{\mathrm{geom},t}^2
}{
\sum_t\eta_t^{\mathrm{geo}}
}
\\
&
+
C_7
\frac{
\sum_t
\rho_t
\eta_t^{\mathrm{geo}}
}{
\sum_t\eta_t^{\mathrm{geo}}
}.
\end{aligned}
}
$$

其中：

$$
e_{\mathrm{geom},t}
$$

统一包含：

- 平行移动误差；
- 指数映射误差；
- 跨曲率映射误差；
- 切空间投影误差；
- 半径截断误差。

### 结论

若选择：

$$
\eta_t
=
\Theta(T^{-1/2}),
$$

$$
\rho_t
=
\Theta(T^{-3/4}),
$$

则：

$$
\rho_t/\eta_t
=
\Theta(T^{-1/4})
\rightarrow0.
$$

若：

$$
e_{\mathrm{geom},t}=o(1),
$$

则：

$$
\boxed{
\frac1T
\sum_{t=0}^{T-1}
\mathbb E[\mathcal G_t]
=
O(T^{-1/2}).
}
$$

因此算法同时收敛到：

$$
\operatorname{grad}_x\mathcal J=0
$$

和：

$$
\partial_u\mathcal J=0
$$

的联合 stationary point。

### Corollary 3.1：geodesic PL 下的 tracking contraction

若对每个固定 $u$：

$$
\frac12
\|
\operatorname{grad}_x\mathcal J(x,u)
\|^2
\ge
\mu_x
[
\mathcal J(x,u)-\mathcal J(x^\star(u),u)
],
$$

且 reduced curvature objective：

$$
\overline{\mathcal J}(u)
=
\mathcal J(x^\star(u),u)
$$

满足：

$$
\frac12
|
\overline{\mathcal J}'(u)
|^2
\ge
\mu_u
[
\overline{\mathcal J}(u)-\overline{\mathcal J}(u^\star)
],
$$

则联合误差：

$$
\mathcal E_t
=
\mathcal J(x_t,u_t)-\mathcal J(x^\star(u_t),u_t)
+
a
[
\overline{\mathcal J}(u_t)-\overline{\mathcal J}(u^\star)
]
$$

满足：

$$
\boxed{
\mathbb E[\mathcal E_{t+1}]
\le
\left(
1-c_x\eta_t-c_u\rho_t
\right)
\mathbb E[\mathcal E_t]
+
\mathcal N_t
+
\mathcal T_t.
}
$$

其中：

$$
\mathcal N_t
$$

是随机噪声项，

$$
\mathcal T_t
$$

是曲率漂移和 transport error。

---

## 4.5 主定理四：曲率驱动的 error amplification

考虑一次同曲率更新：

$$
x^+
=
\operatorname{Exp}_x^K(-\eta m).
$$

设实际使用：

$$
\widehat m=m+\delta m,
$$

实际曲率：

$$
\widehat K=K+\delta K,
$$

实际指数映射含数值误差：

$$
e_{\mathrm{num}}.
$$

### Theorem 4：一步误差传播定理

在：

$$
K\in[K_{\min},K_{\max}],
$$

$$
d_K(x,o)\le R_{\max},
$$

且：

$$
\sqrt K\eta\|m\|\le\delta
$$

时，有：

$$
\boxed{
\begin{aligned}
d_{\widehat K}
(
\widehat x^+,
\Psi_{K\rightarrow\widehat K}(x^+)
)
\le&
\underbrace{
\frac{
\sinh\delta
}{
\delta
}
\eta
\|\delta m\|
}_{\text{momentum perturbation}}
\\
&
+
\underbrace{
C_K
|\delta\log K|
\eta\|m\|
}_{\text{curvature perturbation}}
\\
&
+
\underbrace{
C_{\mathrm{PT}}
e_{\mathrm{PT}}
}_{\text{transport error}}
\\
&
+
\underbrace{
C_{\mathrm{Exp}}
e_{\mathrm{num}}
}_{\text{Exp-map error}}
\\
&
+
\underbrace{
C_{\mathrm{rad}}
e_{\mathrm{rad}}
}_{\text{radius error}}.
\end{aligned}
}
$$

关键放大因子为：

$$
\frac{\sinh\delta}{\delta}.
$$

如果没有曲率归一化步长，该项可随：

$$
\sqrt K\eta\|m\|
$$

指数增长。

如果使用主算法步长，则：

$$
\delta\le\eta/\gamma,
$$

从而误差放大得到统一上界。

### Corollary 4.1：跨曲率 transport 的一阶一致性

跨曲率映射：

$$
\Psi_{K\rightarrow K'}(x)
=
\sqrt{\frac K{K'}}x
$$

和：

$$
\mathcal C_{K\rightarrow K'}(v)
=
\sqrt{\frac K{K'}}v
$$

满足：

$$
\Psi_{K\rightarrow K'}\circ\operatorname{Exp}_x^K(v)
=
\operatorname{Exp}_{\Psi(x)}^{K'}
\left(
\mathcal C(v)
\right)
$$

当：

$$
K'=K
$$

时精确成立。

对小：

$$
\Delta u=\log K'-\log K,
$$

两边差异为：

$$
O
\left(
|\Delta u|
\sqrt K\|v\|^2
\right).
$$

因此曲率更新必须比参数更新更慢。

---

## 4.6 主定理五：曲率、稳定性与 generalization

令训练集：

$$
S,S'
$$

只差一个样本。

对应联合状态：

$$
z_t=(x_t,u_t,m_t),
$$

$$
z_t'=(x_t',u_t',m_t').
$$

定义耦合距离：

$$
\Delta_t
=
d_{K_t}(x_t,x_t')
+
a_u|u_t-u_t'|
+
a_m
\left\|
m_t
-
\mathcal P_{x_t'\rightarrow x_t}m_t'
\right\|.
$$

设单样本损失满足：

$$
\|
\operatorname{grad}\ell(x;z)
-
\mathcal P_{y\rightarrow x}
\operatorname{grad}\ell(y;z)
\|
\le
L_K(R)
d_K(x,y),
$$

其中：

$$
L_K(R)
=
L_0
+
c
\sqrt KR
\coth(\sqrt KR).
$$

这说明梯度 Lipschitz 常数本身由曲率和半径共同控制。

### Theorem 5：联合 uniform stability 主定理

存在常数：

$$
A_t,
B_t,
C_t
$$

使：

$$
\boxed{
\Delta_{t+1}
\le
\left[
1
+
A_t\eta_t^{\mathrm{geo}}
+
B_t\rho_t
\right]
\Delta_t
+
\frac{
C_t
(
\eta_t^{\mathrm{geo}}
+
\rho_t
)
}{
N
}
+
e_{\mathrm{geom},t}.
}
$$

其中：

$$
A_t
\le
C
\left[
L_0
+
\sqrt{K_t}R_t
\coth(\sqrt{K_t}R_t)
\right].
$$

递推展开得：

$$
\boxed{
\Delta_T
\le
\sum_{t=0}^{T-1}
\left[
\frac{
C_t
(
\eta_t^{\mathrm{geo}}
+
\rho_t
)
}{
N
}
+
e_{\mathrm{geom},t}
\right]
\prod_{j=t+1}^{T-1}
\left[
1
+
A_j\eta_j^{\mathrm{geo}}
+
B_j\rho_j
\right].
}
$$

若损失为：

$$
L_\ell
$$

-Lipschitz，则：

$$
\boxed{
\epsilon_{\mathrm{gen}}
\le
L_\ell\Delta_T.
}
$$

### 解释

该定理给出曲率对 generalization 的两条相反作用：

1. **表示收益**  
   合适的负曲率降低层级表示失真和训练偏差。

2. **稳定性代价**  
   过大的：
   $$
   \sqrt KR
   $$
   增大：
   $$
   L_K(R),
   $$
   放大轨迹对样本替换的敏感性。

因此可学习曲率的最优点不是“越负越好”，而是：

$$
\boxed{
\text{hierarchical representation gain}
\quad\text{vs}\quad
\text{curvature-induced instability}
}
$$

之间的平衡。

### Corollary 5.1：曲率正则的稳定性作用

若：

$$
u_t
$$

被限制在：

$$
[u_{\min},u_{\max}],
$$

且：

$$
R_t\le R_{\max},
$$

则：

$$
A_t
\le
A_{\max}
<\infty.
$$

此时：

$$
\epsilon_{\mathrm{gen}}
=
O
\left(
\frac{
\sum_t
(
\eta_t+\rho_t
)
}{
N
}
\exp
\left[
A_{\max}
\sum_t\eta_t
\right]
\right)
+
O
\left(
\sum_t e_{\mathrm{geom},t}
\right).
$$

---

## 4.7 最终统一大定理

### Theorem 6：Curvature-Adaptive Hyperbolic Momentum 总定理

设：

1. 每个参数组位于：
   $$
   \mathbb H_{-K_{g,t}}^{d_g};
   $$
2. 曲率满足：
   $$
   K_{\min}\le K_{g,t}\le K_{\max};
   $$
3. 参数位于安全半径：
   $$
   d_{K_{g,t}}(x_{g,t},o)\le R_{\max};
   $$
4. 目标满足 Riemannian smoothness；
5. 随机梯度与曲率梯度无偏且方差有界；
6. 曲率采用慢时间尺度：
   $$
   \rho_t/\eta_t\rightarrow0;
   $$
7. 曲率归一化步长保证：
   $$
   \sqrt{K_{g,t}}
   \eta_{g,t}^{\mathrm{geo}}
   \|m_{g,t}\|
   \le
   \delta;
   $$
8. 平行移动、指数映射和跨曲率 transport 误差有界。

则同时有：

### Optimization

$$
\boxed{
\begin{aligned}
\mathbb E
\left[
\|
\operatorname{grad}_x\mathcal J(z_\tau)
\|^2
+
\lambda_c
|
\partial_u\mathcal J(z_\tau)
|^2
\right]
\le&
O
\left(
\frac1{\sum_t\eta_t}
\right)
\\
&
+
O
\left(
\frac{
\sum_t\eta_t^2\sigma_x^2
}{
\sum_t\eta_t
}
\right)
\\
&
+
O
\left(
\frac{
\sum_t\rho_t^2\sigma_u^2
}{
\sum_t\eta_t
}
\right)
\\
&
+
O
\left(
\frac{
\sum_t\eta_t
[
e_{m,t}^2+e_{\mathrm{geom},t}^2
]
}{
\sum_t\eta_t
}
\right).
\end{aligned}
}
$$

### Landscape conditioning

$$
\boxed{
\operatorname{cond}
(
\operatorname{Hess}\mathcal J
)
\lesssim
\frac{
\sqrt{K_t}R_t
\coth(\sqrt{K_t}R_t)
+
L_R
}{
\mu_R
}.
}
$$

### Error amplification

$$
\boxed{
\operatorname{Amp}_t
\le
\frac{
\sinh\delta
}{
\delta
}.
}
$$

### Generalization

$$
\boxed{
\epsilon_{\mathrm{gen}}
\le
L_\ell
\sum_t
\left[
\frac{
C_t(\eta_t+\rho_t)
}{
N
}
+
e_{\mathrm{geom},t}
\right]
\prod_{j>t}
\left[
1+
A_j\eta_j+B_j\rho_j
\right].
}
$$

取：

$$
\eta_t=\Theta(T^{-1/2}),
$$

$$
\rho_t=\Theta(T^{-3/4}),
$$

若：

$$
e_{\mathrm{geom},t}=o(1),
$$

则：

$$
\boxed{
\mathbb E
\left[
\|
\operatorname{grad}_x\mathcal J
\|^2
+
|
\partial_u\mathcal J
|^2
\right]
=
O(T^{-1/2}).
}
$$

该定理明确说明 HyperbolicMomentum 的收益和风险都由同一个几何量控制：

$$
\boxed{
\sqrt K\,R
\quad\text{和}\quad
\sqrt K\,\eta\|m\|.
}
$$

前者控制：

- 双曲容量；
- Hessian smoothness；
- generalization stability。

后者控制：

- 指数映射畸变；
- 噪声放大；
- transport error；
- 单步优化稳定性。

---

# 5. 实验总协议

## 5.1 Seed

所有实验只运行：

$$
\boxed{\textbf{1 seed}}
$$

固定：

$$
\boxed{\texttt{seed}=42}.
$$

包括：

- 主结果；
- 所有任务；
- 所有模型；
- 所有消融；
- 所有超参数配置；
- 所有长训练。

不报告 mean、standard deviation 或 confidence interval。

## 5.2 长训练

所有正式训练：

$$
\boxed{\textbf{epochs}\ge1000}.
$$

固定：

- synthetic hierarchy：2000 epochs；
- 视觉分类：1000 epochs；
- 图学习：1000 epochs；
- 知识图谱：2000 epochs；
- 文本分类：1000 epochs；
- 推荐系统：1000 epochs；
- 小型语言表示任务：1000 epochs；
- 持续学习：1000 epochs。

不允许 early stopping 提前结束。

## 5.3 全量结果保留

每个 epoch 保存：

1. train loss；
2. validation metric；
3. test metric；
4. learning rate；
5. 每组曲率；
6. 曲率梯度；
7. log-curvature step；
8. geodesic radius；
9. maximum radius；
10. Riemannian gradient norm；
11. momentum norm；
12. transported momentum norm；
13. tangent residual；
14. manifold constraint residual；
15. parallel transport norm error；
16. Exp map argument；
17. curvature-adjusted learning rate；
18. clipping events；
19. wall-clock；
20. peak memory。

永久 checkpoint：

$$
0,1,2,5,10,20,50,100,200,500,1000
$$

以及之后每 100 epochs。

所有结果都保留用于画完整曲线。

---

# 6. 多任务实验

## 6.1 Synthetic hierarchy

### 设置

| 项目 | 设置 |
|---|---|
| Seed | 42 |
| Epochs | 2000 |
| Trees | binary, 4-ary, 8-ary, variable branching |
| Nodes | 1K, 5K, 10K |
| Dimensions | 2, 5, 10, 20 |
| Methods | Euclidean SGD, RSGD, fixed-curvature momentum, HyperbolicMomentum |

### 结果表

| Tree | Nodes | Dim. | Method | Final distortion | MAP | Mean rank | Final loss | Learned $K$ | Epoch to target | sec/epoch |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Binary | 10K | 10 | RSGD | TBD | TBD | TBD | TBD | Fixed | TBD | TBD |
| Binary | 10K | 10 | HyperbolicMomentum | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 6.2 视觉分类

### 设置

| 项目 | 设置 |
|---|---|
| Seed | 42 |
| Epochs | 1000 |
| Datasets | CIFAR-10, CIFAR-100, Tiny-ImageNet, ImageNet-100 |
| Models | ResNet-18, ResNet-50, ViT-S/16 |
| Hyperbolic component | classifier head or prototype layer |
| Methods | SGD-M, RSGD, fixed-curvature momentum, HyperbolicMomentum |

### 结果表

| Dataset | Model | Method | Final top-1 | Best top-1 | Hierarchical F1 | Final train loss | Learned $K$ | Epoch to 90% best | sec/epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | ResNet-18 | RSGD | TBD | TBD | TBD | TBD | Fixed | TBD | TBD |
| CIFAR-100 | ResNet-18 | HyperbolicMomentum | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 6.3 图学习

### 设置

| 项目 | 设置 |
|---|---|
| Seed | 42 |
| Epochs | 1000 |
| Datasets | Cora, Citeseer, PubMed, ogbn-arxiv |
| Models | GCN, GAT, GraphSAGE |
| Methods | AdamW, RSGD, fixed-curvature momentum, HyperbolicMomentum |

### 结果表

| Dataset | Model | Method | Final val | Final test | Best val | Epoch to best | Learned $K$ | Mean radius | Peak GB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ogbn-arxiv | GraphSAGE | RSGD | TBD | TBD | TBD | TBD | Fixed | TBD | TBD |
| ogbn-arxiv | GraphSAGE | HyperbolicMomentum | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 6.4 Knowledge graph embedding

### 设置

| 项目 | 设置 |
|---|---|
| Seed | 42 |
| Epochs | 2000 |
| Datasets | WN18RR, FB15k-237, YAGO3-10 |
| Models | Lorentz distance, hyperbolic translational |
| Methods | RSGD, fixed-curvature momentum, HyperbolicMomentum |

### 结果表

| Dataset | Model | Method | MRR | Hits@1 | Hits@3 | Hits@10 | Learned $K$ | Epoch to best |
|---|---|---|---:|---:|---:|---:|---:|---:|
| WN18RR | Lorentz | RSGD | TBD | TBD | TBD | TBD | Fixed | TBD |
| WN18RR | Lorentz | HyperbolicMomentum | TBD | TBD | TBD | TBD | TBD | TBD |

## 6.5 文本分类

### 设置

| 项目 | 设置 |
|---|---|
| Seed | 42 |
| Epochs | 1000 |
| Datasets | AG News, DBpedia, 20 Newsgroups, Reuters-21578 |
| Models | MLP, small Transformer encoder |
| Hyperbolic component | final representation and classifier |
| Methods | AdamW, RSGD, fixed-curvature momentum, HyperbolicMomentum |

### 结果表

| Dataset | Model | Method | Final acc. | Macro-F1 | Best val | Final loss | Learned $K$ | Epoch to best |
|---|---|---|---:|---:|---:|---:|---:|---:|
| AG News | Small Transformer | RSGD | TBD | TBD | TBD | TBD | Fixed | TBD |
| AG News | Small Transformer | HyperbolicMomentum | TBD | TBD | TBD | TBD | TBD | TBD |

## 6.6 推荐系统

### 设置

| 项目 | 设置 |
|---|---|
| Seed | 42 |
| Epochs | 1000 |
| Datasets | MovieLens-1M, Amazon Books subset, Yelp subset |
| Models | matrix factorization, graph recommender |
| Methods | AdamW, RSGD, fixed-curvature momentum, HyperbolicMomentum |

### 结果表

| Dataset | Model | Method | Recall@20 | NDCG@20 | Final loss | Learned $K$ | Epoch to target | Peak GB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| MovieLens-1M | MF | RSGD | TBD | TBD | TBD | Fixed | TBD | TBD |
| MovieLens-1M | MF | HyperbolicMomentum | TBD | TBD | TBD | TBD | TBD | TBD |

## 6.7 持续学习

### 设置

| 项目 | 设置 |
|---|---|
| Seed | 42 |
| Epochs | 1000 |
| Datasets | Split CIFAR-100, Rotated MNIST |
| Models | MLP, ResNet-18 |
| Methods | AdamW, RSGD, fixed-curvature momentum, HyperbolicMomentum |

### 结果表

| Dataset | Method | Final avg acc. | Forgetting | Recovery epochs | Learned $K$ drift | Mean radius | Final loss |
|---|---|---:|---:|---:|---:|---:|---:|
| Split CIFAR-100 | RSGD | TBD | TBD | TBD | Fixed | TBD | TBD |
| Split CIFAR-100 | HyperbolicMomentum | TBD | TBD | TBD | TBD | TBD | TBD |

## 6.8 任务筛选规则

主文只保留表现好的任务，但完整筛选矩阵必须保存。

进入主文需要至少满足：

1. 最终指标更好；
2. 达到目标更快；
3. 1000 epochs 后优势仍存在；
4. learned curvature 与数据层级结构相关；
5. fixed-curvature baseline 无法复现；
6. 曲率校正步长减少不稳定；
7. transport error 较低；
8. wall-clock 开销可接受。

---

# 7. 基线

| ID | Method | Momentum | Curvature | 跨曲率状态 |
|---|---|---|---|---|
| B1 | Euclidean optimizer | Euclidean | None | None |
| B2 | RSGD | None | Fixed | None |
| B3 | Riemannian momentum | Parallel transport | Fixed | None |
| B4 | Learnable-curvature RSGD | None | Learned | Point only |
| B5 | HyperbolicMomentum | Parallel transport | Group learned | Point and momentum |

---

# 8. 理论对应实验

## 8.1 空间最优性

| Task group | Branching proxy | Predicted $K^\star$ | Learned $K$ | Fixed-$K$ distortion | Learned-$K$ distortion |
|---|---:|---:|---:|---:|---:|
| Group 1 | TBD | TBD | TBD | TBD | TBD |

## 8.2 Landscape

| Task | $K$ | Radius | Hessian min eig. | Hessian max eig. | Smoothness proxy | Saddle escape epoch |
|---|---:|---:|---:|---:|---:|---:|
| Synthetic hierarchy | TBD | TBD | TBD | TBD | TBD | TBD |

## 8.3 Convergence

| Task | Method | Final grad norm | Min grad norm | Loss AUC | Final loss | Momentum error proxy |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-100 | HyperbolicMomentum | TBD | TBD | TBD | TBD | TBD |

## 8.4 Error bounds

| Task | Epoch | PT error | Exp error | Tangent residual | Radius error | Bound proxy |
|---|---:|---:|---:|---:|---:|---:|
| WN18RR | 100 | TBD | TBD | TBD | TBD | TBD |

## 8.5 Generalization

| Dataset | Method | Train metric | Test metric | Gap | Stability proxy | Smoothness proxy | Curvature spread |
|---|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | RSGD | TBD | TBD | TBD | TBD | TBD | 0 |
| CIFAR-100 | HyperbolicMomentum | TBD | TBD | TBD | TBD | TBD | TBD |

---

# 9. 必要消融

| ID | 消融 | 目的 |
|---|---|---|
| A1 | 无 momentum | 检验双曲动量 |
| A2 | 固定曲率 | 检验曲率学习 |
| A3 | 单一全局曲率 | 检验分组曲率 |
| A4 | 无曲率校正步长 | 检验 Exp 非线性控制 |
| A5 | 曲率变化时只映射点 | 检验动量一致性 |
| A6 | 不做切空间投影 | 检验数值切向性 |
| A7 | 同速更新 $x$ 和 $K$ | 检验两时间尺度 |
| A8 | 无 log-curvature clipping | 检验曲率稳定性 |
| A9 | 每 10 epochs 更新曲率 | 检验频率 |
| A10 | 每 100 epochs 更新曲率 | 检验频率 |
| A11 | 单组曲率 | 检验多组结构 |
| A12 | 不同初始曲率 | 检验初始化敏感性 |

所有消融只运行 seed 42，且至少 1000 epochs。

### 消融表

| Ablation | Final metric | Final loss | Loss AUC | Learned $K$ | PT error | Radius error | sec/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full HyperbolicMomentum | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

# 10. 超参数

| 参数 | 搜索范围 |
|---|---|
| Base learning rate | $\{10^{-5},3\cdot10^{-5},10^{-4},3\cdot10^{-4},10^{-3},3\cdot10^{-3},10^{-2}\}$ |
| Momentum $\beta$ | $\{0,0.8,0.9,0.95,0.99\}$ |
| Curvature learning rate | $\{10^{-6},3\cdot10^{-6},10^{-5},3\cdot10^{-5},10^{-4}\}$ |
| Initial $K$ | $\{0.01,0.1,0.5,1,2,5\}$ |
| Curvature regularization | $\{0,10^{-5},10^{-4},10^{-3},10^{-2}\}$ |
| Curvature correction $\gamma$ | $\{0,0.1,0.3,1,3\}$ |
| Log-curvature clip | $\{10^{-4},10^{-3},10^{-2},5\cdot10^{-2}\}$ |
| Number of groups | $\{1,2,4,8\}$ |

固定：

$$
\texttt{seed}=42.
$$

每个方法使用相同调参 run 数。

---

# 11. 数值测试

## 11.1 Manifold constraint

$$
\left|
\langle x,x\rangle_{\mathcal L}
+
\frac1K
\right|
<
10^{-6}.
$$

## 11.2 Tangent constraint

$$
|
\langle x,v\rangle_{\mathcal L}
|
<
10^{-6}.
$$

## 11.3 Exp–Log inverse

$$
d_K
\left(
\operatorname{Exp}_x(\operatorname{Log}_x(y)),
y
\right)
<
10^{-6}.
$$

## 11.4 Parallel transport isometry

$$
\left|
\|
\mathcal P_{x\rightarrow y}(v)
\|_y
-
\|v\|_x
\right|
<
10^{-6}.
$$

## 11.5 Cross-curvature constraint

$$
\left|
\left\langle
\Psi_{K\rightarrow K'}(x),
\Psi_{K\rightarrow K'}(x)
\right\rangle_{\mathcal L}
+
\frac1{K'}
\right|
<
10^{-6}.
$$

## 11.6 Cross-curvature tangency

$$
\left|
\left\langle
\Psi(x),
\mathcal C(v)
\right\rangle_{\mathcal L}
\right|
<
10^{-6}.
$$

## 11.7 Distance scaling

$$
\left|
d_{K'}(\Psi(x),\Psi(y))
-
\sqrt{
\frac K{K'}
}
d_K(x,y)
\right|
<
10^{-6}.
$$

## 11.8 Identity curvature map

当：

$$
K'=K,
$$

跨曲率映射退化为恒等。

## 11.9 Finite difference

Riemannian directional derivative 与 geodesic finite difference 一致。

## 11.10 Mixed precision

Lorentz inner product、Exp、Log、parallel transport 和曲率映射至少使用 FP32。

---

# 12. 成功与失败标准

## 12.1 成功标准

1. 至少两个任务族优于固定曲率 Riemannian momentum；
2. 至少一个 2000-epoch 任务保持优势；
3. 1000 epochs 后优势不消失；
4. learned curvature 与数据层级结构相关；
5. 曲率校正步长减少不稳定；
6. 跨曲率动量映射优于只映射参数点；
7. PT error 接近数值误差；
8. 至少两个任务达到目标更快；
9. generalization proxy 与测试 gap 趋势一致；
10. wall-clock 开销可接受。

## 12.2 失败标准

1. learned curvature 全部收敛到同一值；
2. 固定曲率完全复现结果；
3. momentum 不优于 RSGD；
4. 跨曲率映射无作用；
5. 曲率频繁触发 clipping；
6. 参数半径持续爆炸；
7. 1000 epochs 后优势消失；
8. 多数常见任务无改善；
9. 结果只在 synthetic hierarchy 上成立；
10. 单 seed 对超参数极度敏感。

---

# 13. 推荐论文叙述

## Observation

层级任务的局部分支率、深度和表示尺度不同，因此单一固定负曲率可能无法同时适配所有参数组。与此同时，双曲 momentum 必须跨不同点的切空间移动。

## Limitation

固定曲率 Riemannian momentum 只处理同一流形内的切空间变化。曲率可学习时，参数点、切空间和动量状态都会改变，直接更新曲率会破坏状态一致性。

## Method

HyperbolicMomentum 在 Lorentz 模型中维护精确平行移动动量，为不同参数组学习负曲率，并通过解析缩放同时映射参数点和动量。曲率校正步长使用：

$$
\sqrt K\|m\|
$$

控制指数映射非线性。

## Theory

- 给出层级分支率与最优曲率匹配；
- 证明异质层级下分组曲率不劣于单一全局曲率；
- 分析曲率对 Hessian、smoothness 和 strict-saddle 的影响；
- 给出固定曲率、动量和两时间尺度联合 convergence；
- 给出平行移动、Exp map、曲率更新和半径 error bounds；
- 给出 stability-based generalization bound；
- 给出端到端 stationarity 总公式。

## Experiments

所有实验只运行 seed 42。所有正式训练至少 1000 epochs。任务覆盖 synthetic hierarchy、视觉、图学习、知识图谱、文本分类、推荐系统和持续学习。每个 epoch 的完整结果永久保存。

---

# 14. 推荐标题

> **HyperbolicMomentum: Curvature-Adaptive Momentum for Hierarchical Neural Optimization**
