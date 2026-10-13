# Cathead 拆解产物记录

- 输入：`example/cathead.3mf`
- 输出：`example/cathead_split_parts.3mf`
- 流程：`codex/sync-installed-skill-20260922` 分支上的 painted-3MF 拆件流程；Stage 03 区域语义和 Stage 04 接口计划均经用户确认。
- 结果：导出 13 个零件，包含 12 个接口。
- 校验：Stage 05 状态为 `complete_parts_exported`；3MF 回读校验通过，13 个零件均有效、单连通、watertight、绕序一致；12 个接口均满足配置的 0.20 mm 间隙。阻断质量门未触发。
- 源文件未被改写。完整阶段诊断保存在本地 `example/cathead_split_parts_stages/` 与 `example/cathead_split_parts_final_stages/` 目录，这些诊断目录不纳入版本控制。
