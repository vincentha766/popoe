# Task: NIDS-Net 分割源接入 + 三路集成（CNOS + SAM-6D ISM + NIDS-Net）

## 背景（一段话）

popoe 是从 gedi 复现工程抽出的 BOP 向开源 6-DoF 位姿框架。FreeZe v2.1（BOP 2024
冠军）用四路分割集成（CNOS / SAM-6D / NIDS / MUSE，各源出带置信度候选掩码 →
top-M 并集不过滤 → 每掩码独立走位姿流水线 → feature-aware scoring 统一排序，
打分与掩码置信度解耦）。MUSE 掩码不可得；其余三路开源。本任务把 NIDS-Net 作为
第三个检测源接入，并把分割抽象成可插拔 backend。卖点：公开可复现的最大子集。

## 已就位的数据

`data/detections/nids/nids_wa_sappe_{ycbv,lmo}.json` — NIDS-Net 官方发布的
WA_Sappe 变体 BOP 预测（来源：UT Dallas Box，仓库 IRVLUTD/NIDS-Net README 的
"Inference on BOP datasets" 一节）。已验证：YCB-V 12 场景 900 图 / LM-O 200 图
全覆盖，object id 集合正确，每图 13.4 / 35.9 个候选，score 0.18–0.90。

**已知格式坑（必须处理）**：
1. 所有字段值都是字符串（`"scene_id": "48"`、`"score": "0.742..."`、bbox 是字符串化列表）——加载时转型。
2. `segmentation` 是**未压缩 RLE**（`{'counts': [int,...], 'size': [h,w]}`），
   而现有 loader 走的是 COCO 压缩 RLE 路径——用 pycocotools `frPyObjects` 转换，
   或在 mask 解码处兼容两种。注意 counts 里的坑：它是 dict 不是 JSON 字符串时
   pycocotools 的行为差异，写测试钉住。

## 工作分块（每块一个 commit，完成一块再进下一块）

1. **NIDS 加载适配**：让 `segmentor_detections.py`（或其新子类/新 loader 函数）
   能吃 NIDS JSON——字段转型 + RLE 兼容。用真实文件写单测（抽几条真实记录做
   fixture，不要把 43MB 全塞进测试）。
2. **可插拔 segmentation backend 抽象**：现有 CNOS / SAM-6D / NIDS 三种文件式
   来源统一到一个 backend 接口（参考 `interfaces.py` 现有风格），配置里按名字
   选源、可组合多源。保持向后兼容：现有 recipe/测试不许挂。
3. **三路 top-M union**：把现有双路并集逻辑（gedi 里 LM-O 的 CNOS+SAM-6D 并集
   是模板，popoe 里对应 `segmentor_chain` / `select_instances` 一带）推广到 N 路，
   per-source top-M、不过滤、保留来源标注（后续分析要用）。默认 M=2 与现配置一致。
4. **端到端冒烟**：无 GPU（本机没有），所以到"检测加载→掩码解码→实例选择"为止，
   用真实 NIDS JSON + 本地已有的 CNOS/SAM-6D 检测文件（如 popoe 里没有，从
   `~/work/gedi/ycbv_local_data/` 找，找不到就只跑 NIDS 单路冒烟并在 PR 说明里注明）
   跑通三路 union 的形状/数量/来源分布检查，输出一份简短统计到 stdout。

## 每块完成后的固定流程（不可跳过）

1. `git add -A && git diff HEAD --stat` 确认改动范围符合本块预期；
2. **codex review**：`codex exec --sandbox read-only "Review the uncommitted changes
   (git diff HEAD) in this repo for correctness bugs, silent failure modes, and
   API-compatibility breaks. Be specific: file:line + failure scenario."`
   逐条处置 findings（修掉或写明为何不改）；
3. **验证**：`uv run pytest tests/ -x -q` 全绿 + 本块相关的真实数据冒烟
   （如 loader 块：真实 JSON 加载 + 一条 RLE 解码 → mask 面积与 bbox 一致性）；
4. commit（消息说清楚这一块做了什么、codex findings 处置了几条）。

## 约束

- 不开 GPU pod、不装 NIDS-Net 推理环境（只消费其发布的 JSON）。
- 遵循仓库现有代码风格与测试习惯；公共 API 变化要更新 ARCHITECTURE.md 对应段落。
- 有疑问先查 `ARCHITECTURE.md`、`ISSUES.md`；不确定的设计决策记进 ISSUES.md 而不是拍脑袋。
- 完成全部四块后：更新 README 的 detections 一节（三路来源、下载出处、格式说明），
  最后把本文件里"工作分块"改为完成状态记录。
