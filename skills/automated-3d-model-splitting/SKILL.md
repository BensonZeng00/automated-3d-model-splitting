---
name: automated-3d-model-splitting
description: Split externally painted 3MF source regions into an assembled multi-part 3MF by planning shared seams and interfaces without repairing or merging source defects.
---

# 自动化 3D 模型拆件

Current release: `2.1.0`, positioned as automated painted-3MF model-part splitting.

## 职责边界

拆件只负责识别 source 区域、确定零件关系、规划双方共用的交接线、生成交接面与装配结构，并导出一个分组 3MF。输入模型已有的小碎片、噪声、小孔、翻转面、退化面、非流形边和断开壳体均属于 source：默认原样保留，不自动合并、删除、补面、焊接或翻面，也不因面积、跨度、面数或比例阻断拆件。

只有交接线、交接面或本工具新增的几何可以被调整。source 异常可写入 `source_diagnostics`，但必须标记为非阻断；只有无法形成一致交接线、交接面或必要生成结构时，当前接口才可失败。

只读取用户指定的 `.3mf`。拒绝 STL、OBJ 或改扩展名的输入。使用 `scripts/split_painted_3mf.py`，不要使用 Blender 或切片器 UI。

## 100/1000 面语义审核

面数只筛选审核候选，不决定几何处置：

- `<=100` 面：`noise_candidate`；
- `101–999` 面：`small_region_candidate`；
- `>=1000` 面：不因面数审核；
- 长细条候选：不受面数限制，必须审核。

所有候选都必须由用户分类为 `noise`、`part` 或 `uncertain`。分类只改变报告标签；三类都保持完整 source 顶点、面、颜色、方向和组件身份，并进入相同的正常拆件流程。确认成 noise 也不得合并、删除、改色、修复或跳过交接面规划。

运行识别：

```bash
python scripts/split_painted_3mf.py --input model.3mf --recognize-only
```

若需要审核，命令以退出码 `4` 生成六视图和 `user_decisions.json`。检查图片、填写每项 `semantic_label`、`visual_confidence` 与 `classification`，设置 `user_confirmed=true`，然后运行：

```bash
python scripts/split_painted_3mf.py \
  --input model.3mf \
  --output model_split_parts.3mf \
  --region-review-json review/user_decisions.json
```

审核阈值 CLI：

```text
--noise-review-max-faces 100
--small-region-review-max-faces 999
--region-review-json PATH
--region-review-dir PATH
--region-review-resolution 320
```

旧版 `--min-faces`、`--tiny-component-policy`、`--tiny-component-auto-noise-max-faces` 和 `--tiny-component-review-*` 不兼容且不再接受。不得增加兼容别名或把旧参数静默映射到新参数。

## Source 保持约束

- 不运行按 `<=2 mm` 区域归并或小口封闭。
- 不按 `<1%`、`<=1 mm²`、长度、体积或面数自动修复 source。
- 不对 source 执行全局 winding 修复；生成面的朝向由父子关系和共享接口决定。
- source 小孔远离接口时忽略；碰到接口时重新规划接口，不修补整个 source。
- 孤立 source shell 原样输出，不虚构连接结构。
- `--merge-body-parts` 仅在用户明确指定时允许合并。

## 交接线与交接面

识别完成后，每个非主体 source 区域均采用 inward 流程。父件和子件必须消费同一个不可变共享边界定义。允许处理范围限于接口局部和新生成几何；不得以生成可打印实体为由改写无关 source。

交接验证至少包括：共享边界一致、生成面非退化、生成面方向正确、接口不明显穿出、必要装配间隙、材料槽保持，以及输出 3MF 可重新读取。source 自身的开边、翻面或非流形只记录。

边界统一使用 `--boundary-shape smooth`。阅读 [boundary-smoothing.md](references/boundary-smoothing.md)、[assembly-algorithm.md](references/assembly-algorithm.md) 和 [architecture.md](references/architecture.md) 后再修改接口算法。

## CLI 与执行

安装 `requirements.txt` 后先运行：

```bash
python scripts/split_painted_3mf.py --input model.3mf --preflight-only
```

CLI 只负责参数解析、输入校验、调用可测试服务、输出报告和退出码。不要把审核、分类或几何逻辑复制到 CLI。退出码 `4` 仅表示等待 source 区域语义确认；退出码 `3` 表示交接面、生成结构或包输出失败，不能用它表示无关 source 缺陷。

最终默认只生成一个分组多零件 `.3mf`。保留原始颜色槽与逐面材料。诊断报告必须区分：

```text
source_diagnostics: non_blocking
interface_validation: blocking_when_invalid
```

## 开发验证

修改后至少运行区域审核、CLI、最终整理和端到端测试。关键不变量：对同一候选分别选择 `noise`、`part`、`uncertain` 时，source 顶点、面、颜色和组件成员必须一致；只有语义 metadata 可以不同。
