# Task: feature-aware 打分 A 层——S_coarse 进 union 仲裁

## 背景

FreeZe 的 feature-aware 打分（论文 Eq. 6）：S_coarse = 对应点集 C 上融合描述子
余弦相似度的均值，用于 RANSAC 假设选择与跨掩码最终排序；v2.2 把它加进 RANSAC
fitness。我们的现状：最终仲裁用 fit×s_feat_1（26 规则消融冠军），RANSAC 内部
仍是几何 inlier 计数。本任务只做 A 层（Selector/仲裁层），B 层（GPU RANSAC
内部 fitness）等 A 层出信号后另开任务。

昨天的三份候选 dump 在 gedi/ycbv_local_data/union_scoring_20260716/
（union2/union3 cands CSV，列：scene_id,im_id,obj_id,cand,w,s_icp,s_feat_1,
metric_fit,score,R,t）。

## 工作分块

1. **判定**：读 popoe 打分代码，写清 s_feat_1 的精确定义（在哪个阶段、对哪个
   点集、什么空间算的），与论文 S_coarse 的差异逐条列出（pre/post-ICP、
   对应集构造、inlier 过滤与否）。结论写进本文件的"判定结论"节。若二者
   实质等价 → 停下报告，A 层没有增量空间，不要硬做。
2. **实现**（仅当有差异）：popoe 里加 S_coarse 计算（pre-ICP、假设生成所用
   对应集上的均值余弦，规范空间），作为 PoseScorer 可选组件 + bop_eval
   cand-csv 新列 `s_coarse`。老配置输出逐字节不变（不开开关时）。
3. **重放工具**：examples/ 下加 rule replay 脚本——从 cand CSV 重放任意
   仲裁规则组合（含 s_coarse 列存在时的新规则族），纯 pandas 本地零 GPU。
   若 s_coarse 无法从现有 dump 重建（大概率，需要特征缓存），脚本要明确
   报"此规则需要带 --cand-csv 重跑"而不是静默算错。
4. **单测 + 固定流程**：每块 codex review → 处置 findings → uv run pytest
   全绿 → commit（main）。

## 判定结论（Block 1 — 决定性）

**结论：s_feat_1 与论文 S_coarse 不等价。A 层有增量空间，继续做 Block 2。**

### s_feat_1 的精确定义（现状）

仲裁冠军规则 `ChampionScorer`（`src/popoe/scoring.py`）算：
`score = s_icp * max(s_feat_1, 0) * (metric_fit if size_aware else 1)`，其中
`s_feat_1 = feature_aware_score(pose.R, pose.t, query.pts, target.pts,
feats_w1_q, feats_w1_t, τ)`（scoring.py:50-55）。

- **阶段/位姿**：`pose.R/pose.t` 是 **ICP 之后（post-ICP，refined）** 的位姿——
  eval 循环是 `solve → refiner.refine → scorer.score`（bop_eval.py），
  `ICPRefiner` 返回 refined 位姿（adapters.py:167）。
- **点集**：稀疏 `query.pts / target.pts`（配准用的采样云），公制（米）。
- **特征空间**：`feats_w1`（视觉权重=1 的规范融合特征，从 `meta["feats_w1"]`），
  与权重扫描解耦。
- **对应集构造**：在 `feature_aware_score`（pose_estimator.py:72）内部重建——
  对每个 target 点，在 (refined 位姿) 变换后的 query 云里取 k=1 最近邻，
  再按 `dist < τ` 过滤 inlier；`τ = 0.03 * query_extent`（公制，≈规范空间 0.03）。
- **取值**：inlier 对应集上逐点余弦的**均值**。

→ 即：s_feat_1 是**在 refined 位姿上重算的特征分**，对应论文的 **S_fine**，不是 S_coarse。

### 论文 S_coarse（Eq. 5/6）

同一条 `feature_aware_score` 公式，但喂入 **coarse（pre-ICP）位姿**。
popoe **已经算了它**：`Open3DFeatureRansacSolver.solve` 对每个候选算
`s_coarse = feature_aware_score(R_coarse, t_coarse, …)` 存进
`breakdown["s_coarse"]`（open3d_ransac.py:106-110）。但——
- `ChampionScorer` 的最终分**不用** s_coarse（只用 post-ICP 的 s_feat_1）；
- `bop_eval` 的 cand-csv **不导出** s_coarse（只有 s_icp/s_feat_1/metric_fit/score）；
- 且 solver 那份 s_coarse 用的是**扫描权重 w 的特征**（w≠1 非规范），
  而非 w=1 规范空间。

### 逐条差异（s_feat_1 vs S_coarse）

| 轴 | s_feat_1（现仲裁） | S_coarse（论文/待接入） | 差异 |
|----|----|----|----|
| 位姿阶段 | post-ICP（refined） | pre-ICP（coarse RANSAC） | **实质不同**（主差异）|
| 对应集构造 | target→变换后 query 的 k=1 NN + `dist<τ` | 同公式 | 方法同，但位姿不同→inlier 集不同 |
| inlier 过滤 | `dist < 0.03·extent` | 同阈值 | 相同 |
| 特征/权重 | w=1 规范 | 待接入取 w=1；现 solver 那份为扫描权重 w | 需在 w=1 规范空间算 |
| 归一 | inlier 均值 | inlier 均值 | 相同（均相对论文 1/|P_T| 用 inlier 计数，同）|

### 为何不等价、A 层为何有增量

s_feat_1（post-ICP 特征）+ s_icp（post-ICP 几何）+ metric_fit（post-ICP 公制几何）
全是**精修后**信号。S_coarse 是**精修前**的特征一致性：ICP 可能把一个特征不合理
的粗假设"吸附"到高几何 fitness（thin/近对称几何上尤甚，正是已知 -20.8pt 缺口所在），
此时 post-ICP 的 s_icp/s_feat_1 都会偏高而 pre-ICP 的 S_coarse 偏低——这是互补的判别
信号。故把 S_coarse 引入仲裁有增量空间。现有 dump 未含 s_coarse 列，无法离线复算
（需带特征缓存重跑），因此 Block 2 先把规范 w=1 的 S_coarse 落成可选 PoseScorer 组件
+ cand-csv 新列，供监督者重扫比较新规则族。

## 约束

- 不开 pod（重跑扫描由监督者另行安排）；不改 O3D RANSAC 内部（那是 B 层）。
- 26 规则消融的结论是既有事实：规则不跨数据集迁移。新规则族设计时按
  "逐数据集重扫"预设，不要写死单一规则。
- ARCHITECTURE.md 若加了新 Protocol 组件要同步。
