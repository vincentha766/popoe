# AGENTS.md — popoe 仓库 AI 协作规则

popoe 是从毕业论文项目中开源出来的模块化 6-DoF 位姿框架（Apache-2.0）。
项目全局规则见 `../gedi/AGENTS.md`（硬规则：数字双线纪律、RunPod 铁律、
多会话 lane、删除先确认），GPU 操作手册见 `../gedi/RUNPOD.md`。

本仓库补充规则：

1. **边界**：popoe 只做 BOP 向的位姿估计（数据集、指标、配方）。应用层
   （抓取、HTTP 服务、机器人）属于 `../rams-grasp` / `../lab-sim`，不要往
   这里加。
2. **REPRODUCTION.md 是对账清单**：论文要引用的每个 popoe 数字都必须在
   清单里有行、有 commit hash、有产物路径。跑出新数字先登记再引用。
3. **pod 上只准 fresh git clone 本仓库**（记录 commit），禁止 scp 单文件。
4. **改框架层（src/popoe）须跑 `pytest tests/`**；contracts/fusion 有测试
   覆盖，红了不许合。
5. 默认**不要 git commit/push**——留给 Vincent review 后自己提交，除非他
   明确让你提交。
