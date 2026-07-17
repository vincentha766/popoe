# Task: B 层——feature-aware fitness 进 RANSAC（GPU solver）

## 背景与目标

A 层（S_coarse 进仲裁）已测：YCB-V +2.5 / LM-O −1.9（遮挡使绝对特征分不可靠）。
B 层把特征相似度放进 **RANSAC 假设评估内部**（FreeZe v2.2 的自述增量）：改变
"哪些假设存活"，而非重排幸存者。LM-O 侧的假说：同掩码内假设间的**相对**特征
排序可能仍有效，即使绝对值被遮挡压低。O3D 的 C++ RANSAC 插不进自定义
fitness → 落点是把 gedi 的 GPU RANSAC 移植进 popoe 并加 feature fitness。

## 工作分块

1. **移植 GPU RANSAC 为 popoe PoseSolver**：源头在 ~/work/gedi 的
   freezev2_*.py（向量化三元组采样 + batched SVD，~54ms/10k 假设）。
   ⚠️ 以 gedi 仓库里 **freezev2_ 前缀**文件为准（bare 名副本是 pod 遗留，
   不可信）。落成 `solvers/gpu_ransac.py` 的 `GPURansacSolver`，行为对齐
   现有 Open3DFeatureRansacSolver 的接口/返回（含 breakdown 键）。
   CPU 上可运行（torch device 自动选择），小规模单测走 CPU。
2. **feature-aware fitness**：假设评估分改为论文 Eq.5 语义：
   `Σ_{inliers} cos(f_q, f_t) / |P_T|`——分母是**固定稀疏点数 |P_T|**，
   不是 inlier 数（ch3 复现税第 2 案：normalize-by-inlier-count 曾致 −31pt，
   千万别重演）。特征用 **w=1 规范空间**（A 层教训）。做成 solver 参数
   `fitness="geometric"|"feature"`（默认 geometric，保证纯移植行为可单独验证）。
3. **接线**：recipes/bop_eval 加 `--solver o3d|gpu|gpu-feat`（默认 o3d 不动，
   正式 mainline 不受影响）；cand-csv 记录 solver 标识。
4. **测试**：
   - 数值单测：合成点云已知位姿，两种 fitness 都能恢复；
   - **对抗性相似度结构**测试（ch3 教训：干净合成特征测不出 Eq.5 类 bug）——
     构造少数高相似度伪对应 vs 大量普通真对应，验证 feature fitness 不被劫持；
   - 与 O3D solver 在同一 fixture 上的位姿一致性（容差内）。

## 固定流程（不可跳过）

每块：codex review（`codex exec --sandbox read-only ...`）→ 处置 findings →
`uv run pytest` 全绿 → commit main。GPU 扫描由监督者另行安排（本机无 GPU，
实现阶段全部 CPU 小规模验证）。

## 约束

- O3D mainline 的行为与数字**零扰动**（默认路径不变）。
- ARCHITECTURE.md solver 一节同步。
- B 层结果将作为**独立 solver 配置**报告，不并入现有基线叙事。
