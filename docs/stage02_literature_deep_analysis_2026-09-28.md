# Stage 02 文獻深度分析：從可逆表示到可辯護的語意幾何

日期：2026-09-28  
狀態：分析筆記，不是實驗結論；本輪沒有修改 Stage 02 程式  
文獻範圍：`/home/rz/raymond_project/literature_review_papers/` 下 86 個 PDF（含重複檔、掃描檔與損壞檔）以及目前 `f_sefa` 的 Stage 01 實作

---

## 1. 這份筆記回答什麼

Stage 01 已建立有限維線性對應

\[
\Phi:\mathbb R^{512}\to\mathcal H_D,
\qquad
\Phi(w)=\sum_{k=0}^{511}w_kP_k,
\]

並用投影／求積方式驗證在數值精度內可回到原係數。Stage 02 真正要問的已不是「能不能把向量寫成函數」，而是：

1. 對函數做的運算是否只是原 latent 運算的換座標版本？
2. 若結果不同，差異來自哪一個明確的幾何假設？
3. 這個幾何是否具有語意、生成器或資料分布上的依據？
4. 所謂 disentanglement 是「方向正交」、分類器分數不變、局部影像變化分離，還是統計獨立？
5. operator 做完後是否仍在 \(\operatorname{Range}(\Phi)\)，逆映射是否存在、唯一且穩定？

這些問題不能由單一指標回答。因此本文先整理數學層次、文獻支持與互相衝突之處，再提出可否證的實驗序列。現在不選定最終 metric，也不假定 raw Legendre \(L^2\) geometry 一定有效。

---

## 2. 文獻庫稽核結果

### 2.1 完整性與可讀性

- 共找到 86 個 `.pdf`，總量約 1.6 GB。
- 79 個 unique PDF 可正常抽取文字。
- 3 組為位元層級完全相同的重複檔：
  - `Functional_Space_A_Survey/Image2StyleGAN.pdf` = `GAN_Inversion/2019_Image2stylegan.pdf`
  - `Hilbert Space/张恭庆-泛函分析講義(上册).pdf` = `note_summerization/.../张恭庆-泛函分析講義(上册).pdf`
  - `Latent_Space_Entanglement/2020_Face Identity Disentanglement.pdf` = `Semantic_Entanglement/Face Identity Disentanglement via Latent Space Mapping.pdf`
- `Springer_Format_Reference/` 的 5 個檔案實際是 HTML 錯誤頁，不是有效 PDF；不能當作文獻證據。
- 張恭慶上下冊為掃描型 PDF，文字層極少。本次以頁面結構與其他可搜尋泛函分析教材交叉定位，沒有假裝做出逐字全文分析。

### 2.2 判讀原則

文獻被分成四種角色：

- **直接理論依據**：Hilbert geometry、Gram system、Riesz 表示、投影、條件數與 pre-image。
- **直接方法依據**：InterFaceGAN、SeFa、GANSpace、StyleSpace、StyleFlow、WarpedGANSpace、Jacobian/spectral、ReSeFa。
- **實驗設計依據**：attribute dependency、identity preservation、matched target strength、路徑相關、FID/LPIPS、區域變化。
- **背景或低直接相關**：一般 CV、渲染、硬體、影像處理教科書。它們可支援實作常識，但不能用來證明 Stage 02 的幾何主張。

---

## 3. 目前最重要的數學拆解

### 3.0 現有程式的語意邊界（先避免把兩種 Legendre 用法混在一起）

目前 repository 內有兩條不同路徑：

- `experiment_utils/function_space_mapping.py`：把單一 512-D W code 當成 512 個 basis coefficients，實作本筆記的 \(\Phi(w)=\sum_k w_kP_k\)。
- `explicit_sefa_mapping.py`：用 scalar parameter \(t\) 建立
  \[
  V(t)=w_0+\alpha_1P_1(t)n_1+\alpha_2P_2(t)n_2+\cdots,
  \]
  這是 latent trajectory parameterization，不是把 512 個 latent coordinates 映成一個 function 的 Stage 01 map。

兩者都使用 Legendre polynomial，但研究問題不同，Stage 02 不能混用其結論。

現有 Stage 01 runner 的命令說明稱為 `exact`，實際呼叫的是 `projected_round_trip()`。對 degree 0–511 的 Legendre/Chebyshev 展開，512-node 對應 Gauss quadrature 在精確算術下足以積分所需的 basis-product degree，因此理論上可 exact recovery；程式輸出仍是浮點 quadrature 結果，報告時應稱「quadrature projection，理論 exact、數值近似」。`exact_round_trip()` 另有實作但 runner 沒有使用。

runner 從 Z mapping 後取第一個 W code，再 broadcast 到所有 synthesis layers，所以當前驗證對象是單一 W，而不是自由的 W+。這點與後續 InterFaceGAN W boundaries 相容，但不能把結果直接宣稱適用於 W+。

### 3.1 純線性移動必然等價：Stage 2A 應先作為負結果／一致性基線

對任意已知方向 \(d\) 與強度 \(\alpha\)：

\[
\Phi^{-1}\bigl(\Phi(w)+\alpha\Phi(d)\bigr)
=w+\alpha d.
\]

只要 \(D=512\)、\(\Phi\) 是線性雙射，而且 operator 只有線性加法，函數空間不會產生新的語意行為。這不是失敗，而是必要的 control：後面若觀察到差異，才能歸因於 metric、constraint、非線性 operator 或 truncation，而不是 mapping 的數值誤差。

### 3.2 表示與幾何不是同一件事

若採 standard Legendre polynomials 與 \(L^2[-1,1]\) 內積：

\[
\langle P_i,P_j\rangle_{L^2}
=\frac{2}{2i+1}\delta_{ij},
\]

因此係數空間中的 Gram matrix 為

\[
G=\operatorname{diag}\left(2,\frac23,\frac25,\ldots,\frac{2}{1023}\right),
\qquad
\langle \Phi(a),\Phi(b)\rangle=a^TGb.
\]

這與 latent Euclidean geometry \(a^Tb\) 不同；但「不同」本身不等於「更有語意」。raw Legendre geometry 隱含了：latent 第 0 維最重要，越後面的維度按 polynomial degree 逐步降權。StyleGAN 的 512 個座標並沒有天然的 polynomial-order 語意，因此這個權重很可能只是人為座標編號的副作用。

更強的 sanity check 是：若改用正規化 Legendre basis

\[
\widehat P_k=\sqrt{\frac{2k+1}{2}}P_k,
\]

其 Gram matrix 即為 \(I\)。所以「Legendre 是正交函數」不能單獨推出新的 geometry；新 geometry 其實來自 basis normalization 與係數識別方式的選擇。

### 3.3 raw Gram 的數值與不變性風險

在 \(D=512\) 時：

\[
\kappa(G)=\frac{2}{2/1023}=1023.
\]

這不至於讓雙精度線性代數立即失效，但已足以造成方向權重強烈不均。更根本的問題是 coordinate permutation：把相同 latent 維度重新編號，就會把不同權重配到不同方向。以現有 InterFaceGAN W-space age/gender boundary 做診斷：

- Euclidean cosine 約為 0.0706。
- raw-\(G\) cosine 約為 0.3219。
- 隨機做 1000 次座標 permutation，raw-\(G\) cosine 約落在 -0.575 到 0.733，median 約 0.048。

因此，在沒有「座標順序 = 函數頻率／階數」的額外結構前，raw \(G\) 不是 coordinate-invariant 的語意結論。

### 3.4 semantic boundary 首先是 covector，不應直接混同為 metric vector

若 classifier boundary 為

\[
s(w)=c^Tw+b,
\]

則 \(c\) 首先是線性泛函（covector）。希望 nuisance score 在一階近似下不變，約束應是

\[
c^T\delta=0.
\]

多個 nuisance boundary 組成 \(C=[c_1,\ldots,c_m]\) 時，約束是 \(C^T\delta=0\)。

常見的 Euclidean conditional manipulation 是

\[
d_E=d-C(C^TC)^\dagger C^Td.
\]

若直接把內積改成 \(G\) 並寫成

\[
d_{\text{naive-}G}
=d-C(C^TGC)^\dagger C^TGd,
\]

得到的是 \(C^TGd_{\text{naive-}G}=0\)，而不是 classifier 所要求的 \(C^Td=0\)。在現有 age/gender boundary 上，這個 naive 版本仍留下約 -0.3094 的 gender linear-score leakage。

若真正的問題是「在 \(G\)-norm 下，找最接近原方向 \(d\) 且滿足 semantic constraints 的方向」，應解：

\[
\min_\delta (\delta-d)^TG(\delta-d)
\quad\text{s.t.}\quad C^T\delta=0,
\]

其閉式解為

\[
\boxed{
d_G=d-G^{-1}C(C^TG^{-1}C)^\dagger C^Td
}.
\]

這個版本保證 \(C^Td_G=0\)。此區分就是 Riesz map 的實際作用：給定 metric 後，covector \(c\) 對應的 normal vector 是 \(G^{-1}c\)，不是仍然把 \(c\) 原封不動當作 normal。

### 3.5 operator 的 range 與 pre-image

Preimage 文獻提醒：一般 feature-space operation 後的元素未必有 exact input-space pre-image。現在的有限維模型若所有操作都保持在

\[
\mathcal H_D=\operatorname{span}\{P_0,\ldots,P_{511}\}
\]

內，則 exact coefficient pre-image 仍存在。可是下列擴展會重新引入問題：

- 產生 \(P_{512}\) 以上的分量；
- 非線性函數運算後沒有再投影回 \(\mathcal H_D\)；
- 用 sampled function values 近似 inverse，設計矩陣病態；
- 用 learned/incomplete basis，target 不在 span 中；
- operator 進入 function space 的 closure，但不在有限 span 中。

因此每個 Stage 02 operator 都應回報：range membership、projection residual、inverse residual 與條件數，而不能只顯示生成影像。

還要再區分一件事：\(f'\in\mathcal H_D=\operatorname{Range}(\Phi)\) 只保證存在某個係數向量 \(w'\in\mathbb R^{512}\)，不保證 \(w'\) 落在 StyleGAN 實際 W distribution 的高密度區域。也就是：

\[
\text{exact pre-image}
\not\Rightarrow
\text{in-distribution latent}
\not\Rightarrow
\text{realistic／editable image}.
\]

因此 Stage 02 除了 function-space range，還需量 latent support deviation，例如到 sampled-W 的 Mahalanobis distance、density proxy、nearest-neighbor distance，或 generator/perceptual quality。這正是 inversion 與 nonlinear-path 文獻反覆出現的 editability 問題。

---

## 4. 理論書籍與 Hilbert-space 文獻能支持到哪裡

### 4.1 泛函分析教材的直接支援

Kreyszig、Conway 與 *Hilbert Spaces with Applications* 共同支援以下基礎：

- inner product 決定 norm、angle、orthogonality；
- closed subspace 上的 orthogonal projection；
- orthogonal complement 與 direct-sum decomposition；
- Riesz representation：continuous linear functional 與代表向量依賴 inner product；
- positive/self-adjoint operators、square root 與 spectral representation；
- approximation in Hilbert spaces、orthonormal systems 與 Gram-Schmidt。

它們能證明投影公式在指定 inner product 下成立，卻不能替我們選擇「哪個 inner product 對 GAN semantic 正確」。metric selection 是模型假設，需要資料或生成器結構支持。

### 4.2 Numerical Optimization 的用途

Nocedal & Wright 對 Stage 02 最有用的是：

- equality-constrained quadratic problem 的 KKT／null-space 觀點；
- projected direction 與 reduced Hessian；
- SPD matrix、preconditioning、condition number；
- pseudoinverse、ill-conditioning 與 regularization 的實務必要性。

這支援上面的 constrained metric projection，也提醒 \(C^TG^{-1}C\) 可能因 boundary 共線或極端 metric eigenvalues 而不穩定。實作時要記錄有效 rank、最小 singular value 與 regularization strength。

### 4.3 Function Encoders 的真正啟示

Function Encoders 明確區分：

\[
f\approx\sum_j c_jg_j,
\]

若 basis 非正交，係數不是單純的 \(\langle f,g_j\rangle\)，而是 Gram least squares：

\[
Gc=b,
\quad
G_{ij}=\langle g_i,g_j\rangle,
\quad
b_i=\langle f,g_i\rangle.
\]

文獻同時指出以 Monte Carlo 估 inner product、basis 趨近正交但未被強制時，會產生 coefficient error 與計算成本。這支持「顯式處理 Gram matrix」與「檢查 conditioning」，但不支持把任意 analytic basis 的 raw Gram 宣稱為 latent semantics。

### 4.4 learned Hilbert metric 文獻

*Learning an Invariant Hilbert Space for Domain Adaptation* 同時學習 source/target projection 與 latent Mahalanobis metric，目標是縮小同類跨域差異並分開異類。它帶來的原則是：如果聲稱 metric 有 task semantics，metric 應由 supervision、pair constraints 或 domain objective 學出，而非只由表示基底決定。

Semantic Hilbert Space（text/image retrieval）使用 complex representation、projective transformation 與 task loss，把幾何與 multimodal semantic objective 連結。它們是「Hilbert machinery 可承載 learned semantics」的例子，不是 Legendre-L2 metric 適合 StyleGAN 的證據。

### 4.5 DeepONet error decomposition 的邊界

DeepONet 文獻將總 approximation error 拆成 trunk projection error 與 branch/coefficient error，並進一步做 mode decomposition。對目前 \(D=512\) 的 exact linear map，它不是核心證據；若後續研究 truncation、learned basis 或 compression，它會很有價值，因為可以分辨：

- span 不夠造成的 projection error；
- coefficient predictor 造成的 branch error；
- 訓練 modes 雖在 output space 正交，但在 parameter space 梯度仍可能糾纏。

---

## 5. GAN latent manipulation／disentanglement 文獻的共同地圖

### 5.1 InterFaceGAN：固定線性 boundary 與條件投影

InterFaceGAN 將語意 score 寫成線性 boundary，並在 \(z\sim\mathcal N(0,I)\) 等假設下，把 normal 的正交性連到 semantic conditional independence。其 conditional manipulation 是把 target direction 對 nuisance normals 做投影消除。

可直接採用：

- supervised semantic boundaries 作為 Stage 02 的已知方向；
- target/nuisance 的 matched evaluation；
- 多 nuisance boundary 的 projection baseline。

不能無條件延伸：

- theorem 依賴 latent distribution 與線性 boundary；
- W space 不一定 isotropic；
- global orthogonality 不等同 image-level independence；
- 更換 metric 後，normal/covector 的 Riesz 對應也要一起更換。

### 5.2 SeFa：生成器權重誘導的 global quadratic form

SeFa 從 generator 首個 affine transformation \(A\) 出發，最大化

\[
\|An\|^2=n^TA^TAn,
\]

因此以 \(A^TA\) 的 eigenvectors 找 global semantics。它比 raw Legendre \(G\) 多一層重要依據：權重來自生成器對 latent perturbation 的響應。

對 Stage 02 的啟示不是「一定要用 SeFa」，而是可比較：

\[
M=I,
\quad M=G_{\text{Legendre}},
\quad M=A^TA.
\]

若 Hilbert metric 的增益只在 \(A^TA\) 或 learned metric 出現，而 raw \(G\) 沒有，就表示有效性來自 generator-awareness，不是 function representation 本身。

### 5.3 GANSpace：先建立有資料意義的座標，再談 basis ordering

GANSpace 對 W／feature samples 做 PCA，並搭配 layer-wise application。其座標依 explained variance 排序，至少讓 axis ordering 有統計意義。這提供一個值得測試的中介假說：先旋轉到 PCA coordinates，再把 component index 對應到 polynomial degree，是否比直接把 StyleGAN 原始維度編號當 degree 更合理？

仍要注意：variance 大不等於 semantic importance；PCA 是 global linear method；layer selection 本身也影響 locality。

### 5.4 StyleSpace 與 2023 disentanglement：方向發現不等於解耦評估

StyleSpace 逐 channel 分析 attribute control，並提出 Attribute Dependency（AD）衡量 target edit 對其他 attributes 的影響。2023 的 disentanglement 工作使用 path step 與 attribute predictor 輸出的 Pearson correlation，形成 correlation matrix；off-diagonal 越小，表示較少伴隨變化。

這兩者共同指出：

- 不應只量 direction cosine 或 projection residual；
- 應先把不同方法調到相同 target change，再比較 nuisance leakage；
- target 沒有實際改變時，低 nuisance change 沒有意義；
- classifier-based metric 要搭配 identity、quality 與人工抽查，避免 predictor blind spot。

### 5.5 Hessian Penalty：軸對齊是訓練 prior，不是後處理 metric

Hessian Penalty 最小化 generator 各輸出對 latent 的 Hessian off-diagonal energy：

\[
L_H=\sum_{i\ne j}H_{ij}^2,
\]

並用 Rademacher directional finite differences 做 stochastic estimate。其直覺是：\(z_j\) 不應改變 \(z_i\) 對輸出的作用。

有用之處：Stage 02 可量測 candidate coordinate system 下的 mixed second derivatives，檢查 operator 是否只在某點消除線性 leakage，卻保留強烈二階 interaction。

限制也很重要：弱 prior 不能保證完美解耦；可能把兩個獨立因素合到一維；fine-tuning 可能降低畫質；只在早期 layer regularize 可能出現降低 loss 卻不改善 disentanglement 的退化解。

### 5.6 Spectral regularizer：局部 pullback geometry 與曲線路徑

令 generator Jacobian 為 \(J_G(z)\)，文獻定義

\[
M_z(z)=J_G(z)^TJ_G(z).
\]

其 leading eigenvectors／right singular vectors 在該點最大化輸出的瞬時變化，且對應到彼此正交的 output changes。這只保證局部性；沿路反覆更新 eigenvector 才形成 curved path。論文也指出同一 eigenvector ordinal 在不同 latent 點未必對應同一 attribute，因而另以 regularizer 對齊 eigenvectors 與 coordinate axes。

這對 Stage 02 是核心對照：真正由 generator 定義的 metric 通常是位置相關的 \(M(w)\)，不是固定 \(G\)。若固定 Hilbert projection 只在起點有效，沿 \(\alpha\) 增大後 leakage 可能重新出現。

### 5.7 Latent Space Non-Linear Statistics：Riemannian pullback 與小 eigenvalue 風險

該工作同樣使用

\[
g(w)=J_f(w)^TJ_f(w)
\]

作為生成 manifold 的 pullback metric，進而定義 geodesic、Fréchet mean 與 principal geodesic analysis。它支持下列判斷：

- global straight direction 可能離開高密度／高品質區域；
- metric 應隨位置改變；
- inverse 或 geodesic 計算會對小 eigenvalues 敏感；
- 直接計算高維 Jacobian metric 很昂貴，可能需要近似器或 low-rank 方法。

### 5.8 WarpedGANSpace 與 StyleFlow：identity-conditioned、非線性路徑

WarpedGANSpace 用 RBF warp 的梯度產生依賴當前 latent 的方向場，linear path 是特殊情況。它用 100 個 latent、attribute predictors、step-attribute Pearson correlation、L1-normalized correlation matrix、attribute range，以及「路徑長／端點距離」衡量非線性。

StyleFlow 以 conditional continuous normalizing flow 做 attribute-conditioned editing。其分析顯示同一 edit 的 difference vectors 依起點改變，角度可有明顯差異；非線性 attribute path 與 W-space linear interpolation 也不一致。它還量測 identity、edit consistency、FID/LPIPS，並提醒 real-image editing 受 inversion 品質影響。

共同結論不是「線性方向無效」，而是 Stage 02 應先檢查：固定方向模型的有效半徑多大？當 \(|\alpha|\) 增加時，是否需要 local re-linearization 或 path integration？

### 5.9 Region-Based Semantic Factorization：metric 必須對應想改與不想改的輸出

ReSeFa 把影像分為 foreground/background，利用一階 Taylor expansion：

\[
\|\Delta x_f\|^2\approx\alpha^2n^TJ_f^TJ_fn,
\qquad
\|\Delta x_b\|^2\approx\alpha^2n^TJ_b^TJ_bn,
\]

並最大化 generalized Rayleigh quotient：

\[
\max_n\frac{n^TJ_f^TJ_fn}{n^TJ_b^TJ_bn},
\]

解 generalized eigenproblem

\[
J_f^TJ_fn=\lambda J_b^TJ_bn.
\]

這提供一個比「抽象正交」更可操作的設計原則：metric 或 quadratic form 應明確量到想保留／抑制的 output change。限制包括極小區域、重複或對稱物件，以及 global semantics 不易由局部 mask 發現。

### 5.10 Subspace Diffusion：disentanglement 對 metric 選錯非常敏感

該工作用 manifold、connection Laplacian 與 de Rham decomposition 討論 product factors，指出在有正確 manifold metric 與非平凡 holonomy 等條件下可作 unsupervised factorization。其實驗也顯示 arbitrary rescaling／linear transformation 會破壞結果。

它對 Stage 02 最重要的不是直接可移植算法，而是反證壓力：如果 metric 只因座標表示而產生、在 reparameterization 後不穩定，所得「解耦」很可能不是資料 manifold 的性質。

### 5.11 Kernel PCA、pre-image 與 symmetry

Generative Kernel PCA 在 feature space 找 direction，再透過 generator 產生 preimage；它展示 lighting 等 non-linear factor 的可能性，也承認 global aligned features 對 local feature 困難，並建議沿高密度區域走曲線。Latent Space Symmetry Discovery 與 GLO 類工作則提醒：latent structure 也可以由群作用、資料重建或生成 objective 形成，不必等同單一固定 orthogonal basis。

### 5.12 可解釋方向、因果介入與跨 latent 對齊

Voynov/Babenko 的 unsupervised direction discovery 同時學方向矩陣與 reconstructor：由前後影像差預測 direction index 和 shift magnitude，促使不同方向造成可辨識的 transformation。它能找到 rotation、zoom、background removal 等方向，但「reconstructor 易辨識」仍不是 target/nuisance independence；論文自己也承認 interpretability 很難直接量測。

GAN Dissection 更進一步區分 correlation 與 causation：先用 unit–concept segmentation agreement 找 interpretable units，再以 ablation/insertion 與 average causal effect 檢查它們是否真的控制輸出物件。對 Stage 02 的啟示是，attribute predictor correlation 只能算觀察性證據；若要聲稱 operator 控制某個機制，至少需要 intervention-style paired test。

*Comparing the Latent Space of Generative Models* 與 semantic-alignment 工作顯示，不同 encoder/generator latent 之間常可用 linear map 或近 orthogonal Procrustes map保留大量資訊。但它們也列出 conditioning、normalizing flow、Z→W 非線性 mapping 等例外。因此 Stage 02 可以把「換表示後 operation 是否可移植」當成實驗問題，不能先假設所有 latent spaces 都共享同一 metric。

---

## 6. GAN inversion 文獻對 Stage 02 實驗邊界的限制

### 6.1 先 synthetic，再 real image

e4e 明確分析 distortion–editability 與 distortion–perception trade-off。W+ 雖可降低 reconstruction distortion，卻可能離可編輯的原始分布更遠。PTI 與 HyperStyle 透過個別化／預測 generator weight offsets，改善 reconstruction 同時保留 editability，但引入新的變量：每張影像的 generator 已不再完全相同。

因此 Stage 02 的第一輪因果測試應使用原 generator 直接 sample 的 W codes。否則觀察到的 leakage 可能來自：

- inversion encoder；
- W、W+、Z+ 或 feature-latent space 的選擇；
- per-image generator tuning；
- out-of-domain real images；
- reconstruction/editability trade-off。

real-image 實驗應是獨立階段，並至少報 reconstruction LPIPS/MSE/identity，不能把 inversion error 算到 projection metric 頭上。

### 6.2 Robust inversion 的補充提醒

2024 robust real-image inversion 工作指出 unconstrained extended latent space 雖有更佳 reconstruction，edit 後可能出現 artifact；其 Z+/feature hybrid 企圖保留較受約束 latent 的 edit robustness。這再度說明「哪個 latent space」本身就是實驗因子，Stage 02 不應在 W、W+、Z+ 間混用同一套結論。

---

## 7. Wavelet 與 functional-space 文獻的可用與不可用之處

Continuous wavelet、wavelet transform 教材與 SWAGAN 都支持 multiresolution／frequency-localized image representation。SWAGAN 的 wavelet 是生成器中的 spatial multiresolution signal processing，與「把第 \(k\) 個 latent coordinate 指派給第 \(k\) 個 polynomial」不是同一件事。

可探討方向：

- 若 latent coordinates 已按 StyleGAN layer、spatial scale 或 GANSpace variance mode 排序，再引入多尺度 basis；
- 用 Sobolev inner product
  \[
  \langle f,g\rangle_{H^1}
  =\langle f,g\rangle_{L^2}+\lambda\langle f',g'\rangle_{L^2}
  \]
  控制函數的 roughness；
- 比較 polynomial、Fourier、wavelet basis 在相同 induced metric spectrum 下是否等價。

不可直接宣稱：wavelet 在影像域有 frequency meaning，所以任意 latent-to-function mapping 中的 basis frequency 也有 semantic meaning。這中間缺少 coordinate alignment 的證明。

---

## 8. 文獻間最值得深入探討的矛盾

### 8.1 固定 global direction vs. local vector field

- InterFaceGAN、SeFa、GANSpace 常產生 global directions。
- spectral Jacobian、WarpedGANSpace、StyleFlow 與 Riemannian work 顯示方向依位置改變。

值得問：Stage 02 的 Hilbert operator 是要改善小步長的 local edit，還是聲稱對整條 traversal 都有效？若是後者，固定投影很可能不足。

### 8.2 input-space orthogonality vs. output semantic independence

- \(d_i^Td_j=0\) 只描述 latent coordinates。
- \(d_i^TJ_G^TJ_Gd_j=0\) 描述局部 pixel-output differential orthogonality。
- \(\nabla s_i^T\delta=0\) 描述 attribute predictor 一階 score invariance。
- AD/correlation 描述 finite traversal 的 empirical semantic leakage。

四者不是同義詞。Stage 02 必須先指定要消除哪一種 entanglement，再選相應 operator 與 evaluation。

### 8.3 analytic metric vs. learned／generator-induced metric

- raw Legendre \(G\)：便宜、exact、但 coordinate-order arbitrary。
- covariance／Mahalanobis metric：反映 latent distribution，但不直接反映影像 semantics。
- \(A^TA\)：反映 generator 首層 global sensitivity。
- \(J_G(w)^TJ_G(w)\)：反映 local output sensitivity，但昂貴且可能 singular。
- feature-Jacobian metric \(J_{F\circ G}^TJ_{F\circ G}\)：可更貼近 perceptual/identity semantics，但依賴 feature extractor。
- learned metric：可綁定 task supervision，但會引入訓練偏差與 overfitting。

真正的研究問題不是「Hilbert 還是 Euclidean」，因為每個 SPD metric 都定義一個有限維 Hilbert geometry；問題是 metric 的來源與不變性。

### 8.4 basis choice vs. metric choice

如果兩個 basis 只是可逆換座標，而且 metric 做正確 pullback，幾何結果應一致。若換 Legendre／Chebyshev／orthonormalized Legendre 後結果大幅改變，表示實驗比較的其實是不同 metric，而不是不同 representation。這應成為必做的 reparameterization test。

---

## 9. 尚未下結論的研究假說

### H0：線性等價基線

對任意 \(w,d,\alpha\)，function-space addition 回到 latent 後應與 \(w+\alpha d\) 在數值精度內一致。若失敗，先修 mapping／quadrature，停止談 semantic gain。

### H1：raw Legendre metric 的效果主要來自任意 coordinate weighting

預測：對 latent dimensions 做 permutation 或改用 orthonormal Legendre basis，結果會顯著改變或退化為 Euclidean。若成立，raw \(G\) 不宜作主要方法，只能作受控 ablation。

### H2：若先建立有意義的 latent ordering，function metric 才可能有解釋力

候選 ordering：GANSpace PCA variance、StyleGAN layer/group、SeFa eigenvalue、attribute sensitivity。此時 degree-dependent weighting 才有可能對應 coarse-to-fine 或 high-to-low sensitivity，但仍須跨 ordering 與 seed 驗證。

### H3：generator-induced metric 比 basis-induced metric 更能降低可觀察 leakage

比較 \(I\)、raw \(G\)、\(A^TA\)、sample-averaged \(J^TJ\)、local \(J(w)^TJ(w)\)。預測 generator-aware metric 在 matched target strength 下，能更好控制 nuisance attribute 或 image region。

### H4：global constrained direction只在小步長有效

預測：\(|\alpha|\) 增大時，global projection 的 off-target leakage 上升；沿路重新估 metric／constraint 的 local integration 較穩定，但計算更高。

### H5：Sobolev／smoothness metric 只有在 coordinate-degree alignment 後才有意義

若直接套在原始 coordinate ordering，derivative penalty 只是更強烈的 index weighting。只有先證明 degree 對應多尺度或 semantic complexity，才能把 smoothness 解讀成有效 inductive bias。

### H6：最佳「解耦」可能是多目標 constrained optimization，而非 pairwise orthogonalization

例如：

\[
\min_\delta\|\delta-d\|_M^2
\quad\text{s.t.}\quad
C^T\delta=0,
\quad
\delta^TM_{bg}\delta\le\epsilon,
\quad
\|\delta\|_2\le r.
\]

這同時表達 semantic score、背景 preservation 與 trust region，比「對一條 nuisance direction 做一次 projection」更接近文獻中的實際目標。

---

## 10. 建議的實驗順序（設計稿，不代表已決定實作）

### Gate A：純代數與數值正確性

1. 驗證 \(\Phi^{-1}\Phi(w)\)、\(\Phi^{-1}\Phi(d)\)。
2. 驗證 Stage 2A 等價式。
3. 驗證 analytic Gram 與 numerical quadrature Gram。
4. 報 \(G\)、constraint matrix 與 normal equations 的 spectrum／condition number。
5. 檢查 operator 後的 range residual。

通過條件：誤差隨 dtype、quadrature order 符合理論；否則不要進影像評估。

### Gate B：不變性與反例測試

每個 candidate method 至少跑：

- latent coordinate permutation；
- orthonormal Legendre rescaling；
- Legendre vs. Chebyshev/Fourier 的等價 metric control；
- target/nuisance boundary scale change；
- nuisance columns 的順序與近共線情況；
- float32 vs. float64；
- pseudoinverse tolerance／ridge sensitivity。

若方法依賴任意命名、縮放或排列，必須把這種依賴列為假設，而不能稱為 intrinsic geometry。

### Gate C：只在 latent/boundary 層做診斷

候選方法：

1. Direct target direction。
2. Euclidean constrained projection。
3. naive raw-\(G\) projection（刻意保留作反例）。
4. 正確的 metric-closest semantic constrained projection。
5. SeFa \(A^TA\) metric。
6. latent covariance/Mahalanobis metric。
7. averaged/local Jacobian metric（資源允許時）。

記錄：target alignment、\(C^T\delta\)、metric norm、方向角、對 permutation 的分布，而不是只記一個 cosine。

### Gate D：synthetic W images

建議固定相同 latent samples，採 paired repeated-measures 設計。每個方法先校準到相同 target attribute change，再比較：

- nuisance attribute change／AD；
- path-step 與各 attribute predictor 的 Pearson/Spearman correlation；
- ArcFace identity similarity；
- pose/expression distance；
- LPIPS、FID/KID 或 generator quality proxy；
- region foreground/background MSE 或 perceptual change；
- path length、endpoint distance與非線性比；
- failure rate（face detector/classifier 無法輸出也要計入）。

用 bootstrap confidence intervals，sample 是推論單位；不要把同一張臉的多個 \(\alpha\) 當成獨立樣本。

### Gate E：local path

只在 Gate D 顯示 global method 的 leakage 隨 \(|\alpha|\) 增加後，再比較：

- 一次 global projection；
- 每步更新 constraint gradient；
- 每步更新 \(J^TJ\)；
- Warped/RBF 或 flow-like curved path。

這樣能判斷問題來自 metric 選錯，還是 linearization 半徑太小。

### Gate F：real-image inversion

最後才加入 e4e/pSp/PTI/HyperStyle 類 inversion。分開報：

- input → reconstruction error；
- reconstruction → edited image change；
- generator 是否 per-image tuned；
- W/W+/Z+/feature space；
- synthetic 與 real 結果差距。

---

## 11. 失敗時的反覆檢查流程

使用者特別要求「若失敗就重新反覆檢查現有資訊與文獻」。可把失敗分類為以下順序，避免看到差影像就盲目換 metric：

1. **Mapping failure**：round-trip、basis evaluation、quadrature、dtype。
2. **Algebra failure**：projection constraint 沒有真的為零；covector/vector 混用；矩陣 orientation 錯。
3. **Conditioning failure**：Gram 或 \(C^TM^{-1}C\) 病態；pseudoinverse threshold 不穩。
4. **Invariance failure**：permutation/rescaling 後結論改變；表示被誤當成 intrinsic geometry。
5. **Linearization failure**：小 \(\alpha\) 有效、大 \(\alpha\) 失效；改做 local path 而非再調 global coefficient。
6. **Semantic model failure**：boundary 或 attribute predictor 本身不可靠；用第二個 predictor、人工抽樣與 matched target check。
7. **Generator-space mismatch**：boundary 與 generator/checkpoint/W 定義不匹配。
8. **Inversion confound**：synthetic 成功、real 失敗；回到 inversion/editability 診斷。
9. **Hypothesis rejection**：若 raw Legendre metric 未通過 invariance 或 matched-target leakage，不應用更多視覺 cherry-picking 挽救；把它記為有資訊的 negative result。

---

## 12. 全文獻庫用途索引

下列清單覆蓋本次發現的所有 PDF；「低」不代表品質差，只表示對 Stage 02 核心問題的直接性較低。

### 12.1 CV_Foundation（13）

| 文獻 | Stage 02 用途 | 直接性 |
|---|---|---|
| Computer & Machine Vision | feature、vision pipeline 背景 | 低 |
| Computer Vision: Algorithms and Applications, 2e | geometry、feature、evaluation 背景 | 低 |
| Deep Learning for Vision Systems | CNN/GAN/metric 基礎 | 低 |
| Digital Image Processing, 4e | transform、sampling、影像距離基礎 | 中低 |
| Fundamental Digital Image Processing | filtering、transform、數值影像基礎 | 中低 |
| Fundamentals of Multimedia, 2e | media/vision 背景 | 低 |
| GANs in Action | GAN 入門與實務背景 | 低 |
| Handbook of Image & Video Processing | signal/image operators 參考 | 中低 |
| Introduction to Algorithms, 4e | 演算法與複雜度參考 | 低 |
| Multiple View Geometry, 2e | projective geometry；非 latent Hilbert 主證據 | 低 |
| Numerical Optimization, 2e | constrained quadratic、conditioning、stability | **高** |
| OpenCV with Python Blueprints | 實作背景 | 低 |
| Real-Time Rendering, 4e | rendering 背景 | 低 |

### 12.2 Functional_Space_A_Survey（7）

| 文獻 | Stage 02 用途 | 直接性 |
|---|---|---|
| Mathematical Problems in Image Processing | function-space/image PDE 與變分背景 | 中 |
| Continuous Wavelet Transform | multiscale basis；需避免誤類比 latent index | 中 |
| SWAGAN | image-domain wavelet generator；多尺度對照 | 中 |
| Wavelet Transforms Kith and Kin | basis/filter-bank 背景 | 中 |
| Image2StyleGAN++ | W+ manipulation/inversion | 中高 |
| Image2StyleGAN | latent inversion；與 GAN_Inversion 版本重複 | 中高 |
| WarpedGANSpace | 起點依賴的 nonlinear RBF path 與 correlation evaluation | **高** |

### 12.3 GAN_Architectures（2）

| 文獻 | Stage 02 用途 | 直接性 |
|---|---|---|
| PGGAN | Hessian Penalty 等方法的 generator 背景 | 中低 |
| BigGAN | latent discovery 跨架構背景 | 中低 |

### 12.4 GAN_Inversion（8）

| 文獻 | Stage 02 用途 | 直接性 |
|---|---|---|
| Image2StyleGAN | optimization inversion；重複檔 | 中高 |
| e4e | distortion–editability–perception trade-off | **高** |
| pSp | encoder-based W+ inversion | 高（real-image 階段） |
| StyleGAN survey | architecture/method map | 中 |
| EG3D | 3D-aware generator；泛化測試背景 | 中低 |
| GAN Inversion survey | latent spaces、inversion taxonomy、metrics | 高 |
| HyperStyle | hypernetwork weight offsets、editability/identity curves | 高 |
| PTI | per-image tuning、locality regularization、identity/editability | 高 |

### 12.5 Hilbert Space 主目錄（12）

| 文獻 | Stage 02 用途 | 直接性 |
|---|---|---|
| Kreyszig, Introductory Functional Analysis | projection、Riesz、adjoint、approximation | **高** |
| Hilbert Spaces with Applications | orthonormal systems、positive/projection operators、wavelets、optimization | **高** |
| Learning an Invariant Hilbert Space for DA | jointly learned projection + Mahalanobis metric | **高** |
| Latent Complete Lattice Structure of Hilbert Space Projectors | projector algebra；偏理論延伸 | 中 |
| Semantic Hilbert Space for Text | task-learned complex semantic geometry | 中 |
| Semantic Hilbert Space for Interactive Image Retrieval | multimodal projection與 learned semantic metric | 中高 |
| 2026 Latent Space survey | 廣泛脈絡；屬二手整理，主張應回查 primary source | 中 |
| Conway, A Course in Functional Analysis | geometry、adjoints、projections、spectral theory | **高** |
| 張恭慶上冊 | 中文泛函分析參考；掃描、與 note 版本重複 | 高但本次全文可讀性低 |
| 張恭慶下冊 | 中文泛函分析參考；掃描 | 高但本次全文可讀性低 |
| 從幾何向量空間到函數空間 | 教學銜接 | 中低 |
| 無限維向量空間的基底 | basis 概念背景 | 中低 |

### 12.6 Hilbert Space/5_heros（9）

| 文獻 | Stage 02 用途 | 直接性 |
|---|---|---|
| Preimage problem in kernel-based ML | existence、uniqueness、stability、approximate preimage | **高** |
| Invariant Hilbert Space for DA | 主目錄重複版本 | 高 |
| Semantic Hilbert Space for Text | 主目錄重複版本 | 中 |
| Semantic Hilbert Space for Image Retrieval | 主目錄重複版本 | 中高 |
| Function Encoders | Gram least squares、非正交 learned basis、Monte Carlo inner product | **高** |
| DeepONet Error Decomposition | truncation/learned basis 時拆 projection 與 coefficient error | 中高 |
| Riemannian Optimization | manifold constraints、SPD/Stiefel optimization | 高（擴展） |
| note.pdf | 現有研究筆記／架構輔助，須回查 primary sources | 中 |
| reference_architecture.pdf | 研究架構輔助，不作主要理論證據 | 中 |

### 12.7 Latent_Space_Entanglement（6）

| 文獻 | Stage 02 用途 | 直接性 |
|---|---|---|
| Face Identity Disentanglement | ArcFace identity、pose/expression separation；另一路徑重複 | **高** |
| InterFaceGAN | linear boundaries、conditional manipulation、理論假設 | **高** |
| Latent Space Translation via Semantic Alignment | Procrustes/orthogonal alignment；檢查跨表示近等距假說 | 高 |
| Clustering by Directly Disentangling Latent Space | clustering-oriented discrete/continuous split；與本題較遠 | 中低 |
| Disentangling Latent Space of GANs for Semantic Face Editing | path-attribute correlation、disentanglement evaluation | **高** |
| Revisiting GAN Inversion for Robust Real Image Editing | constrained latent 與 edit robustness | 高 |

### 12.8 Latent_Space_Manipulation（10）

| 文獻 | Stage 02 用途 | 直接性 |
|---|---|---|
| GAN Dissection | unit/region semantics 與 causal intervention | 高 |
| GANSpace | PCA coordinates、layer-wise global controls | **高** |
| Hessian Penalty | mixed second derivatives、axis-aligned training prior、failure modes | **高** |
| Unsupervised Discovery of Interpretable Directions | learned directions 與 recognition network | 高 |
| SeFa | \(A^TA\) closed-form generator-aware geometry | **高** |
| StyleSpace | channel semantics、Attribute Dependency | **高** |
| StyleFlow | conditional nonlinear path、identity/edit consistency | **高** |
| GAN Inversion survey | 與 GAN_Inversion 版本重複 | 高 |
| InferFaceGAN | inference/disentangled face editing 對照 | 中高 |
| ReStyle | iterative encoder inversion；real-image 階段 | 中高 |

### 12.9 Semantic_Entanglement（9）

| 文獻 | Stage 02 用途 | 直接性 |
|---|---|---|
| Spectral Regularizer for Unsupervised Disentanglement | local \(J^TJ\)、eigenpaths、axis alignment | **高** |
| Comparing the Latent Space of Generative Models | 跨模型 latent geometry 評估 | 高 |
| Disentangling by Subspace Diffusion | manifold metric、de Rham factors、metric sensitivity | **高** |
| Face Identity Disentanglement via Latent Space Mapping | 重複檔 | 高 |
| Latent Space Exploration Using Generative Kernel PCA | nonlinear feature directions與 preimage | 高 |
| Latent Space Non-Linear Statistics | pullback metric、geodesic、Fréchet/PGA | **高** |
| Latent Space Symmetry Discovery | group/symmetry structure；超越線性方向 | 高（擴展） |
| Optimizing the Latent Space of Generative Networks (GLO) | learned latent structure/optimization 背景 | 中 |
| Region-Based Semantic Factorization in GANs | foreground/background Jacobian generalized eigenproblem | **高** |

### 12.10 Others（4）

| 文獻 | Stage 02 用途 | 直接性 |
|---|---|---|
| Deep Image Harmonization | local/global appearance editing 背景 | 低 |
| NVIDIA LYRA | 與核心問題直接性低 | 低 |
| TPU/GPU/CPU Benchmark | 未來 Jacobian/Hessian 計算資源規劃 | 低 |
| Vector Space | 線性代數教學背景 | 中低 |

### 12.11 Springer_Format_Reference（5）

以下五個副檔名為 `.pdf`，但檔案內容是 HTML error page，無法當作文獻：

1. `01_Bayoudh_Deep_Multimodal_Learning_Survey.pdf`
2. `02_Challa_CNN_BiLSTM_Human_Activity_Recognition.pdf`
3. `03_Chen_YOLO_Face.pdf`
4. `04_Gupta_SIFT_SURF_Face_Recognition.pdf`
5. `05_Chen_MFFN_Image_Super_Resolution.pdf`

### 12.12 note_summerization（1 PDF + 2 DOCX）

- PDF 是張恭慶上冊的完全重複檔。
- 兩個 DOCX 是既有中英文學習指南，屬輔助筆記，不應替代原始教材引用。本次使用者要求的「文獻書籍等」以 PDF 全庫稽核為主；DOCX 可在後續若要統整論文章節時另做文件級 review。

---

## 13. 目前最審慎的暫時判斷

1. **已確定**：只把 W vector 寫成 Legendre coefficients，再做線性加法，不會創造新 operation。
2. **已確定**：standard Legendre \(L^2\) 會誘導非 Euclidean diagonal Gram matrix；但它依賴 arbitrary latent coordinate ordering 與 basis normalization。
3. **已確定**：classifier boundary 是 covector。若目標是 nuisance score invariance，naive \(G\)-orthogonal projection 解錯了 constraint；應使用 metric-closest constrained formulation。
4. **文獻一致支持**：有意義的 disentanglement geometry 通常來自 latent distribution、generator derivatives、regions/features、attribute supervision 或 learned metric，而不是單純來自換一組函數 basis。
5. **尚未確定**：function-space regularity／Sobolev metric 在經過有意義 coordinate alignment 後，能否成為有效 inductive bias。這值得研究，但要先通過 permutation、rescaling 與 matched-target tests。
6. **尚未確定**：global fixed metric 是否足夠。大量文獻指向 local/nonlinear path，因此它較可能是小步長 baseline，而不是完整答案。
7. **實驗優先級**：先做 algebra/invariance/latent diagnostic，再做 synthetic W images；real-image inversion 最後獨立處理。

最值得保留的 Stage 02 核心，不是「Legendre projection 一定比 Euclidean 好」，而是建立一個可檢驗的框架：

\[
\boxed{
\text{representation}
\;\neq\;
\text{geometry}
\;\neq\;
\text{semantic constraint}
\;\neq\;
\text{finite-edit validity}
}
\]

只有把這四層分開，任何正結果或負結果才具有論文價值。

---

## 14. 優先回讀來源

- [InterFaceGAN](/home/rz/raymond_project/literature_review_papers/Latent_Space_Entanglement/2020_Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%20Face%20Editing.pdf)
- [SeFa](/home/rz/raymond_project/literature_review_papers/Latent_Space_Manipulation/2021_SeFa_Close-Form%20Factorization_Of_Semantics_In_GANs.pdf)
- [Function Encoders](/home/rz/raymond_project/literature_review_papers/Hilbert%20Space/5_heros/2025_Function%20Encoders_A%20Principled%20Approach%20to%20Transfer%20Learning%20in%20Hilbert%20Spaces.pdf)
- [Preimage Problem](/home/rz/raymond_project/literature_review_papers/Hilbert%20Space/5_heros/2011_Preimage%20problem%20in%20kernel-based%20machine%20learning.pdf)
- [Spectral Regularizer](/home/rz/raymond_project/literature_review_papers/Semantic_Entanglement/A%20spectral%20regularizer%20for%20unsupervised%20disentanglement.pdf)
- [Latent Space Non-Linear Statistics](/home/rz/raymond_project/literature_review_papers/Semantic_Entanglement/Latent%20Space%20Non-Linear%20Statistics.pdf)
- [Subspace Diffusion](/home/rz/raymond_project/literature_review_papers/Semantic_Entanglement/Disentangling%20by%20Subspace%20Diffusion.pdf)
- [Region-Based Semantic Factorization](/home/rz/raymond_project/literature_review_papers/Semantic_Entanglement/Region-Based%20Semantic%20Factorization%20in%20GANs.pdf)
- [WarpedGANSpace](/home/rz/raymond_project/literature_review_papers/Functional_Space_A_Survey/WarpedGANSpace-Finding_Non_Linear_RBF_paths_in_GAN_Latent_Space.pdf)
- [StyleFlow](/home/rz/raymond_project/literature_review_papers/Latent_Space_Manipulation/2021_Styleflow.pdf)
- [StyleSpace](/home/rz/raymond_project/literature_review_papers/Latent_Space_Manipulation/2021_StyleSpace_Analysis.pdf)
- [e4e](/home/rz/raymond_project/literature_review_papers/GAN_Inversion/2021_e4e.pdf)
- [Numerical Optimization](/home/rz/raymond_project/literature_review_papers/CV_Foundation/Numerical_Optimization_v2.pdf)

