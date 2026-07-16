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

## 工作分块（完成状态记录）

全部四块已完成，各自一个 commit（均经 codex review、处置 findings、`uv run
pytest` 全绿后提交），在分支 `nids-integration` 上：

1. ✅ **NIDS 加载适配** — `409bed4`。`segmentor_detections.py` 新增
   `load_bop_detections`（字段转型，非整数 id 报错不静默截断）+
   `decode_detection_mask`（压缩/未压缩 RLE 兼容）。真实 fixture
   `tests/fixtures/nids_lmo_sample.json`（5 条真实 LM-O 记录）+ 9 项单测。
   codex 3 findings 全修（压缩 RLE 首字符可为 `[`、`_to_int` 截断、丢弃 `time`）。
   *实测：交付的 NIDS 文件本身已是数值化 + 未压缩 RLE，现有解码路径即可读；
   适配主要是为文档所述的全字符串 Box 变体做加固。详见 ISSUES.md。*
2. ✅ **可插拔 backend 抽象** — `346c2de`。`DetectionSource(name, path)` +
   `BOPDetectionsSegmentor(sources=…)`，按名字选源（dict / 元组 / `name=path`）、
   可组合多源，每源 top-M、`Detection.source` 保留来源。单文件形式向后兼容
   （统一 `bop-detections` 标签）。ARCHITECTURE.md 新增一节。codex 3 findings 全修。
3. ✅ **N 路 top-M union** — `dc73e93`。`iou_dedupe` 改为 per-source 作用域：
   跨源不过滤（FreeZe「top-M 并集不过滤」），源内仍去重；单源行为逐字节不变
   （v5 基线不受影响）。默认 M=2。codex 无 findings。
4. ✅ **端到端冒烟（无 GPU）** — `473f6ba`。`examples/union_smoke.py`：
   加载→RLE 解码→N 路 union→实例选择，输出来源分布统计。本地实跑
   CNOS+NIDS 两路（YCB-V 900 图 / LM-O 200 图）。codex 4 findings 全修。
   **SAM-6D ISM 本地无产出文件**（需 pod 跑 ISM），故三路降级为两路子集并注明；
   N=3 路径由合成源单测覆盖，`--source sam6d=<file>` 可接真实第三源。

收尾：README「Detections」一节（三路来源 + 下载出处 + 格式说明）、ISSUES.md
设计决策记录、本节完成状态。约束遵守：未开 GPU pod、未装 NIDS/SAM-6D 推理环境，
仅消费已发布 JSON。

CNOS 默认检测文件由督导补充（`data/detections/cnos/cnos-fastsam_{ycbv,lmo}-test.json`，
来源 HF bop-benchmark/bop_extra）。

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
