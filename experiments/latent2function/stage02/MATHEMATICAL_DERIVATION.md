# Stage 02 Mathematical Derivation

狀態：數學設計稿；尚未實作  
記號：所有向量預設為 column vector，實作分析使用 `float64`

---

## 1. 有限維 function representation

令 latent dimension 為 \(D=512\)，

\[
\mathcal W=\mathbb R^D,
\qquad
\mathcal H_D=\operatorname{span}\{P_0,\ldots,P_{D-1}\}.
\]

定義線性映射

\[
\Phi(w)(x)=\sum_{k=0}^{D-1}w_kP_k(x).
\]

因 \(\{P_k\}\) 線性獨立，\(\Phi\) 是 \(\mathbb R^D\) 與 \(\mathcal H_D\) 的線性同構。

### Proposition 1：線性 edit 等價

對任意 \(w,d\in\mathbb R^D\) 與 \(\alpha\in\mathbb R\)：

\[
\Phi^{-1}\left(\Phi(w)+\alpha\Phi(d)\right)=w+\alpha d.
\]

證明：由 \(\Phi\) 的線性，

\[
\Phi(w)+\alpha\Phi(d)=\Phi(w+\alpha d),
\]

再作用 \(\Phi^{-1}\) 即得。故 representation change 本身不會產生新的 semantic operation。

---

## 2. Function inner product 的 coefficient pullback

令 function inner product 的 Gram matrix為

\[
G_{ij}=\langle P_i,P_j\rangle_{\mathcal H}.
\]

則

\[
\langle\Phi(a),\Phi(b)\rangle_{\mathcal H}
=a^TGb,
\qquad
\|\Phi(a)\|_{\mathcal H}^2=a^TGa.
\]

對 standard Legendre basis 與 \(L^2[-1,1]\)：

\[
G_{ij}=\frac{2}{2i+1}\delta_{ij},
\]

所以

\[
G_{L^2}
=\operatorname{diag}\left(2,\frac23,\frac25,\ldots,\frac{2}{2D-1}\right).
\]

在 \(D=512\) 時，

\[
\kappa_2(G_{L^2})=2D-1=1023.
\]

若使用 normalized Legendre basis

\[
\widehat P_k=\sqrt{\frac{2k+1}{2}}P_k,
\]

則

\[
G_{\widehat P}=I.
\]

因此「Legendre representation」與「non-Euclidean geometry」不是同義詞；geometry 取決於 basis normalization與 latent coefficients 如何被識別。

---

## 3. 換座標與重新指派 basis degree

若只是做可逆座標變換

\[
w=T u,
\]

同一幾何在 \(u\)-coordinates 中必須寫成

\[
G_u=T^TG_wT.
\]

因此

\[
w^TG_ww=u^TG_uu.
\]

這是 coordinate invariance test。

但若將 latent coordinates permutation 後，仍重新套用固定對角矩陣

\[
G_{raw}=\operatorname{diag}\left(2,\frac23,\ldots\right),
\]

而沒有做 \(G\mapsto T^TGT\)，那不是同一幾何的換座標，而是把 polynomial degree重新指派給不同 latent axes。此操作應視為不同模型假設。

---

## 4. Semantic boundary 是 covector

對線性 semantic score

\[
s(w)=c^Tw+b,
\]

其一階變化為

\[
Ds(w)[\delta]=c^T\delta.
\]

所以 \(c\) 首先是 covector。若希望此 semantic score不變，constraint是

\[
c^T\delta=0.
\]

對 \(q\) 個 nuisance semantics：

\[
C=[c_1,\ldots,c_q]\in\mathbb R^{D\times q},
\qquad
C^T\delta=0.
\]

在 metric \(M\succ0\) 下，covector \(c\) 的 Riesz representative／normal vector為

\[
n_M=M^{-1}c,
\]

因為

\[
\langle n_M,\delta\rangle_M
=(M^{-1}c)^TM\delta=c^T\delta.
\]

---

## 5. 三種 projection 必須區分

### 5.1 Euclidean projection

給定原 target direction \(d\)，求

\[
\min_\delta\frac12\|\delta-d\|_2^2
\quad\text{s.t.}\quad C^T\delta=0.
\]

解為

\[
\boxed{
\delta_E=d-C(C^TC)^\dagger C^Td
}.
\]

### 5.2 Naive \(M\)-orthogonalization

若直接把 inner product 換成 \(M\)，但仍把 nuisance columns \(C\) 當成 vectors：

\[
\delta_{naive}
=d-C(C^TMC)^\dagger C^TMd.
\]

它滿足

\[
C^TM\delta_{naive}=0,
\]

通常不滿足 \(C^T\delta_{naive}=0\)。

#### 2-D counterexample

取

\[
M=\begin{bmatrix}2&0\\0&1\end{bmatrix},
\quad
c=\begin{bmatrix}1\\1\end{bmatrix},
\quad
d=\begin{bmatrix}1\\0\end{bmatrix}.
\]

則

\[
\delta_{naive}
=\begin{bmatrix}1/3\\-2/3\end{bmatrix},
\qquad
c^T\delta_{naive}=-1/3\ne0.
\]

### 5.3 Metric-closest semantic constrained projection

正確問題為

\[
\min_\delta
\frac12(\delta-d)^TM(\delta-d)
\quad\text{s.t.}\quad C^T\delta=0.
\]

Lagrangian：

\[
\mathcal L(\delta,\lambda)
=\frac12(\delta-d)^TM(\delta-d)+\lambda^TC^T\delta.
\]

stationarity給出

\[
M(\delta-d)+C\lambda=0,
\]

故

\[
\delta=d-M^{-1}C\lambda.
\]

代入 constraint：

\[
(C^TM^{-1}C)\lambda=C^Td.
\]

因此

\[
\boxed{
\delta_M
=d-M^{-1}C(C^TM^{-1}C)^\dagger C^Td
}.
\]

令

\[
P_M=I-M^{-1}C(C^TM^{-1}C)^\dagger C^T,
\]

則 \(\delta_M=P_Md\)。在 full-rank 與 exact arithmetic 下：

\[
C^TP_M=0,
\qquad
P_M^2=P_M,
\qquad
P_M^TM=MP_M.
\]

最後一式表示 \(P_M\) 是 \(M\)-self-adjoint projection，不必在 Euclidean sense對稱。

---

## 6. Target preservation 有兩種不同定義

### 6.1 保持接近原 semantic direction

先求 \(\delta_M=P_Md\)，再固定 boundary的一階 target effect：

\[
\widetilde\delta_M
=\delta_M\frac{d^Td}{d^T\delta_M}.
\]

此方法回答：在 metric \(M\) 下，哪個 nuisance-free direction最接近原方向 \(d\)？

### 6.2 指定 target change 下的最小 metric cost

另一個問題是

\[
\min_\delta\frac12\delta^TM\delta
\]

subject to

\[
d^T\delta=\kappa,
\qquad
C^T\delta=0.
\]

令

\[
A=[d,C],
\qquad
b=[\kappa,0,\ldots,0]^T.
\]

KKT conditions：

\[
M\delta-A\lambda=0,
\qquad
A^T\delta=b.
\]

得到

\[
\boxed{
\delta_{min-cost}
=M^{-1}A(A^TM^{-1}A)^\dagger b
}.
\]

這回答：要達成指定 target first-order change，同時完全壓制 nuisance，metric cost最小的 edit是什麼？

兩個 formulation一般不相同。Stage 02 primary protocol目前使用「closest-then-normalize」；\(\delta_{min-cost}\) 應列為預先指定的 secondary operator，不能在看到結果後任意替換。

---

## 7. 更一般的 soft-constrained operator

若 semantic predictors有噪音，hard constraint可能不穩。可定義

\[
\min_\delta
\frac12(\delta-d)^TM(\delta-d)
+\frac{\rho}{2}\|C^T\delta\|_2^2.
\]

閉式解：

\[
\boxed{
\delta_{soft}
=(M+\rho CC^T)^{-1}Md
}.
\]

當 \(\rho\to0\)，\(\delta_{soft}\to d\)；在適當 rank條件下，\(\rho\to\infty\) 趨近 hard-constrained solution。

此版本只能作 robustness analysis，不能與 hard constraint混成同一方法。

---

## 8. Candidate metrics

為使 regularization可比較，對任意 PSD matrix \(M_0\) 先做

\[
\overline M_0
=\frac{D}{\operatorname{tr}(M_0)}M_0,
\qquad
M_\varepsilon=\overline M_0+\varepsilon I.
\]

### 8.1 Euclidean

\[
M=I.
\]

### 8.2 Function-induced \(L^2\)

\[
M=G_{L^2}.
\]

### 8.3 Sobolev candidate

令 \(D_P\) 為 polynomial coefficient derivative operator，使

\[
\operatorname{coef}(f')=D_P\operatorname{coef}(f).
\]

則

\[
M_{H^1}=G_{L^2}+\beta D_P^TG_{L^2}D_P.
\]

它懲罰 function roughness，但只有在 latent coordinate order具有 degree意義時才可語意化。

### 8.4 Latent distribution metric

若 W covariance為 \(\Sigma_W\)：

\[
M_{Mahalanobis}=\Sigma_W^{-1}.
\]

### 8.5 SeFa/global generator metric

將 target layers 的 style-affine matrices縱向 concatenation成 \(A\)：

\[
M_{SeFa}=A^TA.
\]

### 8.6 Local pullback metric

對 generator或 feature map \(h=F\circ G\)：

\[
J_h(w)=\frac{\partial h(w)}{\partial w},
\qquad
M_h(w)=J_h(w)^TJ_h(w).
\]

因為

\[
\|h(w+\delta)-h(w)\|_2^2
\approx\delta^TM_h(w)\delta.
\]

若 \(F\) 是 identity，得到 pixel pullback；若 \(F\) 是 perceptual/identity embedding，得到 task-specific local metric。

---

## 9. Local semantic constraints 與 path ODE

global boundary \(c\) 可推廣為 image semantic score

\[
q_i(w)=s_i(G(w)).
\]

其 local covector為

\[
a_i(w)=\nabla_wq_i(w).
\]

令 target gradient為 \(a_t(w)\)，nuisance matrix為

\[
C(w)=[a_1(w),\ldots,a_q(w)].
\]

要求每單位 path time增加一單位 target score，且 nuisance瞬時不變：

\[
a_t(w)^T\dot w=1,
\qquad
C(w)^T\dot w=0.
\]

在 local metric \(M(w)\) 下，最小速度成本解為

\[
A(w)=[a_t(w),C(w)],
\qquad
b=[1,0,\ldots,0]^T,
\]

\[
\boxed{
\dot w
=M(w)^{-1}A(w)
\left[A(w)^TM(w)^{-1}A(w)\right]^\dagger b
}.
\]

以 Euler integration：

\[
w_{k+1}=w_k+\eta\dot w_k.
\]

較穩定版本可用 midpoint／RK4，但每個 substep都要重新計算 gradients與 metric。為避免 direction sign突然翻轉，可令

\[
\dot w_k^T\dot w_{k-1}\ge0,
\]

必要時翻轉 eigenvector／direction orientation。

---

## 10. Trust region 與 finite-edit error

一階 constraint只保證

\[
q_i(w+\delta)
=q_i(w)+a_i(w)^T\delta+O(\|\delta\|^2).
\]

即使 \(a_i(w)^T\delta=0\)，finite edit仍有

\[
\Delta q_i
\approx\frac12\delta^TH_i(w)\delta,
\]

其中 \(H_i=\nabla_w^2q_i\)。因此需要 trust region：

\[
\delta^TM(w)\delta\le r^2
\]

或以 path integration反覆 re-linearize。這解釋為何 global orthogonalization可能只在小 \(|\alpha|\) 有效。

---

## 11. Function-space operator 與 representability

若 operator只改 coefficients：

\[
T(\Phi(w))=\Phi(Lw+r),
\]

則

\[
T(\Phi(w))\in\mathcal H_D,
\qquad
\Phi^{-1}T\Phi(w)=Lw+r.
\]

若 operator產生 \(g\notin\mathcal H_D\)，需投影：

\[
\Pi_Dg=\sum_{k=0}^{D-1}c_kP_k.
\]

令

\[
b_i=\langle g,P_i\rangle,
\]

則 coefficients滿足

\[
Gc=b,
\qquad
c=G^\dagger b.
\]

並報 normalized projection residual：

\[
r_{proj}
=\frac{\|g-\Pi_Dg\|_{\mathcal H}}{\|g\|_{\mathcal H}}.
\]

exact coefficient pre-image不代表 \(w'\) 位於 W distribution高密度區域。

---

## 12. Numerical stability

對

\[
K=C^TM^{-1}C
\]

必須記錄：

- singular values \(\sigma_i(K)\)；
- effective rank；
- \(\kappa_2(K)\)；
- pseudoinverse threshold；
- ridge \(\varepsilon\)；
- constraint residual與KKT residual。

推薦不用顯式形成 \(M^{-1}\)。實作以 linear solve計算

\[
X=M^{-1}C
\quad\Longleftrightarrow\quad
MX=C,
\]

再解

\[
(C^TX)\lambda=C^Td,
\qquad
\delta=d-X\lambda.
\]

KKT residual：

\[
r_{stat}
=\frac{\|M(\delta-d)+C\lambda\|_2}
{\|Md\|_2+\|C\lambda\|_2+\epsilon},
\]

\[
r_{con}
=\frac{\|C^T\delta\|_2}
{\|C\|_2\|\delta\|_2+\epsilon}.
\]

---

## 13. 可直接轉成單元測試的 identities

對隨機 SPD \(M\)、full-rank \(C\) 與 direction \(d\)：

1. `C.T @ delta_M == 0`；
2. `P_M @ P_M == P_M`；
3. `P_M.T @ M == M @ P_M`；
4. `M = I` 時 \(\delta_M=\delta_E\)；
5. normalized Legendre時結果等於 Euclidean；
6. 正確 coordinate transform後，解滿足 \(\delta_w=T\delta_u\)；
7. \(C\) columns permute後結果不變；
8. \(C\) columns非零 rescale後 constraint subspace與結果不變；
9. Stage 2A function edit與latent edit相同；
10. 2-D counterexample中 naive residual非零、correct residual為零。

每項同時測 float64與 float32 tolerance。

---

## 14. Stage 02 的最小數學主張

若實驗只完成 global linear階段，最多可主張：

1. \(\Phi\) 提供 exact finite-dimensional representation；
2. 不同 inner products在 coefficient space誘導不同 SPD metrics；
3. semantic covector constraints下可得到 closed-form metric projection；
4. 某個 metric在 matched target change下是否降低 empirical leakage。

不能僅由 projection公式主張：

- semantic independence；
- manifold disentanglement；
- coordinate-invariant geometry；
- real-image editability；
- function representation天然優於 latent representation。

