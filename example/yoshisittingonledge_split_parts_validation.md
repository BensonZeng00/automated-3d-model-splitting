# Yoshi 拆解产物记录

- 输入：`example/yoshisittingonledge.3mf`
- 输出：`example/yoshisittingonledge_split_parts.3mf`
- 审核：Stage 03 的六个区域和 Stage 04 的五个接口已经用户确认。按用户纠正重建 Stage 04：I001 为 P01 绿色头身主体（榫）→P03 左鞋（卯），I002 为 P01 绿色头身主体（榫）→P02 右鞋（卯）。其余接口关系不变。
- 结果：Stage 05 导出 6 个零件、5 个接口；3MF 回读有效，质量门未阻断。六个零件均 watertight 且绕序一致，接口采用 0.20 mm 装配间隙。
- 容差：源外壳与冻结边界的最大匹配偏差为 0.982 mm，低于 1 mm 上限。P05 输出保留源网格原有的 1 个退化面，没有新增退化面。
- 外观诊断：I001 的 P01/P03 侧分别报告 27/14 处过渡带自交；I003 的 P05 侧报告 46 处过渡带自交和 9,192 个高度小于 0.01 mm 的桥接面。文件通过结构与回读检查，接口外观仍需目视复核。
- 文件 SHA-256：`5f200841b310413849a4d1b51a23f8d2bda191fbccaa35b708a6f73a07229401`。
- 完整阶段诊断保存在本地 `example/yoshisittingonledge_split_parts_stages/` 与 `example/yoshisittingonledge_split_parts_final_stages/`，不纳入版本控制。
