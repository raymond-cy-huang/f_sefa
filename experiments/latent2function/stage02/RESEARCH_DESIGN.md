# Stage 02 Research Design

狀態：研究與 evaluation protocol；尚未實作  
依據：`docs/stage02_literature_deep_analysis_2026-09-28.md` 與本地文獻庫  
完整公式：`experiments/latent2function/stage02/MATHEMATICAL_DERIVATION.md`  
研究對象：StyleGAN FFHQ、W space、已知 InterFaceGAN semantic boundaries

---

## 1. 研究目的

Stage 02 不再驗證「latent vector 能否表示成 function」，而是檢驗 function-induced geometry 是否產生可重現、可解釋、優於 Euclidean baseline 的 semantic operation。

### RQ1：Operation consistency

純線性 function-space movement 是否與 latent movement 完全一致？

\[
\Phi^{-1}(\Phi(w)+\alpha\Phi(d))=w+\alpha d.
\]

這是 negative/control result，不預期有 semantic gain。

### RQ2：Geometry validity

使用 Legendre Gram matrix 改變 projection 後，差異是否：

1. 通過代數 constraint；
2. 不依賴任意 coordinate ordering／normalization；
3. 比 Euclidean projection 更能降低可觀察 semantic leakage？

### RQ3：Global vs. local validity

固定 global direction 的改善是否只存在於小步長？若 leakage 隨 edit strength 增加，local generator-aware metric 是否更穩定？

---

## 2. 預先指定的假說

### Confirmatory hypotheses

- **H0-equivalence**：direct latent edit 與 function-space linear edit的 latent/image output只差數值誤差。
- **H1-primary**：在 matched target change 下，raw-Legendre metric constrained projection 的 nuisance leakage AUC 低於 Euclidean constrained projection。
- **H1-validity condition**：H1 只有在 raw-Legendre 方法同時優於 isospectral random/permuted metrics，且通過 coordinate sensitivity test 時，才能解讀成 basis/function geometry 的支持。
- **H2-constraint**：metric-constrained projection滿足 \(C^T\delta\approx0\)；naive \(G\)-orthogonalization通常不滿足此 semantic constraint。

### Exploratory hypotheses

- generator-aware \(A^TA\) 或 \(J^TJ\) metric 比 basis-only metric更能降低 leakage；
- global direction 的 leakage 隨 target change 增加；
- local metric/path integration 能改善大幅度 editing，但增加計算與失敗率。

不在看完 test set 後改寫 primary hypothesis。

---

## 3. 固定數學定義

令 target boundary 為 \(d\)，nuisance covectors 為

\[
C=[c_1,\ldots,c_q].
\]

語意一階不變條件固定為

\[
C^T\delta=0.
\]

### Euclidean constrained baseline

\[
\delta_E=d-C(C^TC)^\dagger C^Td.
\]

### Naive metric projection（diagnostic negative control）

\[
\delta_{N(M)}
=d-C(C^TMC)^\dagger C^TMd.
\]

它只保證 \(C^TM\delta=0\)，不能當作 semantic constraint 已滿足。

### Metric-closest semantic constrained direction

\[
\delta_M
=d-M^{-1}C(C^TM^{-1}C)^\dagger C^Td.
\]

所有方向先做 target-effect normalization：

\[
\widetilde\delta
=\delta\frac{d^Td}{d^T\delta},
\]

使 \(d^T\widetilde\delta=d^Td\)。若 \(|d^T\delta|\) 小於預先設定 tolerance，記為 method failure，不以極端 rescaling 補救。

---

## 4. 方法組

### Core confirmatory set

| ID | 方法 | 角色 |
|---|---|---|
| M0 | Direct target direction | manipulation baseline |
| M1 | Euclidean constrained projection | primary baseline／InterFaceGAN form |
| M2 | Naive raw-Legendre \(G\) projection | diagnostic negative control |
| M3 | Raw-Legendre metric-closest constrained projection | primary candidate |
| M4 | Orthonormal Legendre，\(G=I\) | representation-only control；應等價 M1 |
| M5 | 20 個 coordinate-permuted/ispectral \(G\) | arbitrary-order control distribution |

Raw Legendre：

\[
G=\operatorname{diag}\left(2,\frac23,\ldots,\frac{2}{1023}\right).
\]

### Exploratory set

| ID | Metric | 依據 |
|---|---|---|
| M6 | W covariance/Mahalanobis | latent distribution |
| M7 | SeFa \(A^TA+\lambda I\) | generator affine weights |
| M8 | averaged \(J_G^TJ_G+\lambda I\) | global generator sensitivity |
| M9 | local \(J_G(w)^TJ_G(w)+\lambda I\) | local pullback geometry |
| M10 | feature-Jacobian \(J_{F\circ G}^TJ_{F\circ G}\) | perceptual/identity geometry |

SeFa 的 \(A\) 必須由 StyleGAN 各 target synthesis layers 的 style-affine weights縱向 concatenation；不得任選一層後仍稱完整 SeFa metric。所有 PSD metrics 用 trace normalization後加 ridge，並報告 ridge sensitivity。

---

## 5. Semantic tasks

使用本機 `stylegan_ffhq_*_w_boundary.npy`：age、gender、eyeglasses、pose、smile。

### Primary entangled family

1. age，condition on gender + eyeglasses；
2. eyeglasses，condition on age + gender；
3. gender-presentation，condition on age + eyeglasses。

### Negative-control family

4. pose，condition on smile；
5. smile，condition on pose。

本機 boundary audit：

| Pair | Euclidean cosine | Raw-\(G\) cosine |
|---|---:|---:|
| age–gender | 0.0706 | 0.3219 |
| age–eyeglasses | 0.1008 | 0.4878 |
| gender–eyeglasses | 0.0026 | 0.3480 |
| pose–smile | 0.0045 | 0.0017 |

這與 InterFaceGAN 論文報告的關聯量級不一致，所以進入主實驗前必須重現 direct edits 與 classifier-score correlation。`gender` 僅解讀成 evaluator 對外觀的 perceived gender-presentation score，不當作個人真實性別。

---

## 6. Experimental gates

### Gate 0：資產與 checkpoint 相容性

必須記錄：

- generator checkpoint SHA-256；
- boundary SHA-256、shape、dtype、norm；
- model definition/version；
- truncation與 noise設定；
- evaluator checkpoint與訓練資料來源。

本機 `f_sefa` 與 InterfaceGAN 的 FFHQ `.pth` 檔案大小不同，不能只憑名稱假設相同。至少要用同一批 Z/W 比較兩套 wrapper 的 mapping statistics與 synthesis，或確認兩者都忠實轉換自同一官方 checkpoint。

### Gate 1：代數與數值驗證

通過條件：

- Stage 2A latent relative error \(<10^{-10}\)（float64 analysis）；
- M1 與 M4 direction cosine \(>1-10^{-10}\)；
- M1/M3/M4 的 \(\|C^T\delta\|_2/\|\delta\|_2<10^{-10}\)；
- M2 預期可失敗，需如實記錄；
- 所有 solve 回報 rank、condition number、ridge、pseudoinverse tolerance；
- float32 另報實際 pipeline error，不用 float64 threshold 假裝通過。

### Gate 2：reparameterization與 arbitrary-order tests

分成兩種，不可混淆：

1. **正確換座標**：方向與 metric一起 pullback；輸出應 equivariant，用來抓 implementation bug。
2. **重新把 polynomial degree 指派給 permuted latent axes**：測 raw Legendre index dependence；用 M5 形成 null distribution。

若 M3 表現落在 M5 隨機分布中央，不能宣稱 Legendre-specific effect。

### Gate 3：boundary/evaluator reproduction

- direct direction對 target score必須有正確符號與足夠 monotonicity；
- age/gender/eyeglasses image-score correlation需重新估計；
- evaluator failure、face detection failure不可丟棄；
- evaluator不得與 boundary training使用完全相同的模型作唯一證據。

Gate 0–3 未通過，不進 confirmatory image experiment。

### Gate 4：global synthetic-W experiment

主實驗只用 generator sample，不混入 inversion。

### Gate 5：local-path experiment

只有 Gate 4 顯示 leakage 隨 edit strength上升後才啟動 M8–M10。

### Gate 6：real-image extension

另立 protocol；不納入 Stage 02 primary claim。

---

## 7. Sampling與資料切分

### Generator setting

- primary：`stylegan_ffhq_2019`，W，`trunc_psi=0.7`、`trunc_layers=8`；
- fixed synthesis noise；
- robustness：`trunc_psi=1.0`；
- 同一 latent seed在所有方法重複使用。

### Three-stage sample plan

| 階段 | Seeds | 用途 |
|---|---:|---|
| smoke | 8 | shape、方向、圖像與輸出格式 |
| pilot/development | 100 | evaluator calibration、effect variance、threshold |
| locked confirmatory | 初始 500 | 主比較；不得調參 |

pilot 與 confirmatory seeds 不重疊。pilot 後以 paired difference variance計算最終樣本量：

\[
n=\frac{(z_{0.975}+z_{0.8})^2\sigma_\Delta^2}{\Delta_{\min}^2},
\]

再依 face/evaluator failure rate膨脹。\(\Delta_{\min}\) 是預先定義的實務最小 leakage-AUC 改善，不以 pilot observed mean 取代。

另用 10,000 個獨立 generated images估 evaluator logit的 mean/std與 W covariance；這批資料不進 paired hypothesis test。

---

## 8. Edit protocol：兩層公平比較

### 8.1 Fixed first-order target effect

所有 \(\delta\) 先依第 3 節 normalization，使 latent boundary的一階 target effect一致。用共同

\[
\alpha\in\{-3,-2,-1,-0.5,0,0.5,1,2,3\}
\]

產生全路徑。這保留 path cost與遠距失效資訊。

### 8.2 Matched image-space target change（primary）

定義 evaluator standardized logit：

\[
z_a(x)=\frac{l_a(x)-\mu_a}{\sigma_a}.
\]

對每張圖、每方法與正負方向，在安全 trust radius內以 interpolation/root search尋找達到

\[
\tau\in\{0.25,0.5,0.75,1.0\}
\]

的 signed target change：

\[
\operatorname{sign}\cdot[z_t(x')-z_t(x)]=\tau.
\]

未達到者記為 reach failure。主結果同時報：

- reach rate；
- conditional-on-reached leakage；
- intention-to-treat結果；
- 達成相同 target change 所需的 \(\|\Delta w\|_2\)。

不能只刪除失敗樣本後比較成功案例。

---

## 9. Evaluation metrics

### 9.1 Primary endpoint：matched-target nuisance leakage AUC

對 nuisance attribute \(i\)：

\[
L_i(\tau)=|z_i(x_{m,\tau})-z_i(x_0)|.
\]

\[
\operatorname{MeanLeak}(\tau)
=\frac1q\sum_{i=1}^{q}L_i(\tau),
\qquad
\operatorname{MaxLeak}(\tau)=\max_iL_i(\tau).
\]

主 endpoint：

\[
\operatorname{LeakAUC}
=\frac1{\tau_{max}}
\int_0^{\tau_{max}}\operatorname{MeanLeak}(\tau)d\tau,
\]

以離散 trapezoid rule計算。MeanLeak相當於 StyleSpace mean-AD 的 matched-target curve版本；MaxLeak是防止平均值掩蓋單一嚴重 nuisance。

### 9.2 Target efficacy

- reach rate at each \(\tau\)；
- target monotonicity：Spearman \(\rho(\alpha,z_t)\)；
- target range；
- edit cost：達成 \(\tau\) 的 \(\|\Delta w\|_2\)；
- sign failure／non-monotonic root failure。

低 leakage 但 target 無法改變，視為失敗而非 disentanglement。

### 9.3 Identity與非目標保存

- ArcFace identity drift：
  \[
  1-\cos(e(x_0),e(x')).
  \]
- pose drift：yaw/pitch/roll差；
- expression／landmark drift；
- background/non-face masked LPIPS；
- region experiment另報 foreground/background masked MSE或 LPIPS。

Identity evaluator應與任何 boundary-learning model獨立。

### 9.4 Image realism與支持域

- face detection/evaluator failure rate；
- artifact rate（blinded human audit或獨立 artifact model）；
- W-distribution Mahalanobis/kNN deviation；
- KID 作 distribution-level secondary metric；
- FID 只在每方法至少有足夠大樣本（建議 10K）時報告，不用 500 張 FID支持主結論。

原圖與 edited image 的全圖 LPIPS不是 realism metric，因為成功 semantic edit本來就應改變影像。

### 9.5 Path metrics

- attribute correlation matrix：step index與每個 attribute score的 Pearson及 Spearman；
- L1-normalized off-diagonal correlation；
- path nonlinearity：path length／endpoint distance；
- feature acceleration：相鄰 feature displacement的二階差分；
- local path才報 forward/backward cycle consistency。固定線性 direction 的 latent cycle必然 exact，不作主要成果。

### 9.6 Algebraic diagnostic metrics

- semantic constraint residual \(\|C^T\delta\|\)；
- naive metric residual \(\|C^TM\delta\|\)；
- target alignment \(d^T\delta\)；
- Euclidean與metric norm；
- matrix spectrum、effective rank、condition number；
- coordinate permutation sensitivity percentile。

這些只能解釋機制，不能取代 image-space semantic metrics。

---

## 10. Evaluator suite

最低組合：

- ArcFace：identity；
- Hopenet或同級 head-pose estimator：pose；
- FairFace類模型：age與gender-presentation；
- 獨立 CelebA multi-attribute classifier：smile、eyeglasses及其他 nuisance；
- face detector與landmarks。

要求：

1. 保存 raw logits，不只保存 threshold label；
2. 在10K reference set上標準化；
3. 用第二 evaluator或人工子集檢查主要結論；
4. 按初始 attribute score、pose、age等 strata檢查 subgroup failure；
5. evaluator disagreement另列，不把缺失當零 leakage。

---

## 11. Statistical analysis

### Unit與配對

統計單位是 latent seed。相同 seed、task、sign、\(\tau\) 下的方法結果配對。多個 path steps不是獨立樣本。

### Primary contrast

\[
\Delta_j
=\operatorname{LeakAUC}_{M3,j}
-\operatorname{LeakAUC}_{M1,j}.
\]

報 paired mean/median difference、standardized effect與95% cluster-bootstrap CI（按 seed resample）。CI不跨0且達到 \(\Delta_{min}\) 才視為實務支持。

### Multiple comparisons

- primary family只有 M3 vs M1；
-三個 primary tasks用 Holm correction；
- M5 permutation distribution用 empirical percentile/p-value；
- M6–M10全部標為 exploratory，不與 primary混報。

### Missingness

face detection、target reach與evaluator failures均按 method報率。主分析同時提供：

- complete-pair estimate；
- intention-to-treat sensitivity；
- failure原因分布。

不允許 method-specific silent filtering。

---

## 12. Success／rejection criteria

### 支持 raw-Legendre geometry 至少需要

1. Gate 0–3全通過；
2. M3在 locked set的 primary LeakAUC優於 M1；
3. target reach、identity與artifact沒有實務上不可接受的惡化；
4. M3優於大多數 M5 isospectral permutations，而非只是任意 weighting幸運結果；
5. 正負 edit signs與至少兩個 primary tasks方向一致；
6. 第二 evaluator／人工 audit不推翻結論。

### 應拒絕或降級為 negative result 若

- M3只優於 M1，卻與 random/permuted metrics無差；
- 結果只在單一 attribute、單一方向或調參 seed成立；
- matched target後優勢消失；
- leakage下降是因 target reach下降；
- checkpoint/evaluator reproduction失敗；
- raw basis normalization改變後結論完全翻轉且無 coordinate semantics依據。

Negative result仍有價值：它證明 function representation 本身不會自動帶來 semantic geometry。

---

## 13. 預定輸出結構

```text
experiments/latent2function/stage02/
  RESEARCH_DESIGN.md
  configs/                 # 之後實作
  run_algebra.py           # Gate 1–2
  run_boundary_audit.py    # Gate 0、3
  run_global_edit.py       # Gate 4
  run_local_path.py        # Gate 5，若觸發
  evaluate.py
  summarize.py

results/latent2function/stage02/
  manifest.json
  algebra/
  boundary_audit/
  pilot/
  confirmatory/
  figures/
```

每次 run 保存 config、git commit、dirty status、checkpoint/boundary/evaluator hashes、seed list與環境版本。開發與 confirmatory output分目錄，避免覆寫或偷看後調參。

---

## 14. 文獻到 protocol 的對應

| Protocol 元件 | 主要文獻依據 |
|---|---|
| linear boundary、conditional projection | InterFaceGAN |
| matched target logit、mean/max AD | StyleSpace |
| path correlation、range、nonlinearity | WarpedGANSpace；2023 disentanglement |
| identity、pose、expression、FID | Face Identity Disentanglement |
| generator \(A^TA\) | SeFa |
| local \(J^TJ\)、eigenpath | Spectral Regularizer；Latent Non-Linear Statistics |
| foreground/background metric | Region-Based Semantic Factorization |
| metric sensitivity／coordinate distortion | Subspace Diffusion |
| reconstruction–editability分離 | e4e、PTI、HyperStyle |
| Gram system與conditioning | Function Encoders；Numerical Optimization |
| pre-image/range檢查 | Preimage Problem in Kernel-based ML |
| intervention而非只有 correlation | GAN Dissection |

---

## 15. 實作前仍需解決的項目

1. 確認 FFHQ generator與 boundaries的 checkpoint lineage。
2. 選定且下載獨立 evaluator checkpoints。
3. 用 pilot決定 trust radius、\(\Delta_{min}\) 與最終樣本量。
4. 明確定義 covariance與Jacobian metric的 regularization。
5. 確認 1024px synthesis與 evaluator batch size的 GPU 預算。

這些項目未完成前，不應鎖定 confirmatory run。

