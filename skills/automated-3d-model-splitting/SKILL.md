---
name: automated-3d-model-splitting
description: Split externally painted 3MF source regions into an assembled multi-part 3MF by planning shared seams and interfaces without repairing or merging source defects.
---

# 自动化 3D 模型拆件

Current release: `2.1.0`, positioned as automated painted-3MF model-part splitting.

## 职责边界

拆件只负责识别 source 区域、确定零件关系、规划双方共用的交接线、生成交接面与装配结构，并导出一个分组 3MF。识别时自动将总表面积小于 `1 mm²` 的连通涂色区域排除为噪声，不作为独立零件候选；源网格、面和涂色数据不被改写。其他小碎片、小孔、翻转面、退化面、非流形边和断开壳体默认原样保留，不自动补面、焊接或翻面。

只有交接线、交接面或本工具新增的几何可以被调整。source 异常可写入 `source_diagnostics`，但必须标记为非阻断；只有无法形成一致交接线、交接面或必要生成结构时，当前接口才可失败。

只读取用户指定的 `.3mf`。拒绝 STL、OBJ 或改扩展名的输入。使用 `scripts/split_painted_3mf.py`，不要使用 Blender 或切片器 UI。

## 100/1000 面语义审核

先按区域总表面积过滤噪声：

- `<1 mm²`：自动排除为噪声，不进入零件区域、边界和语义审核清单；源网格和涂色数据保持不变。
- `>=1 mm²`：再按面数与长细形状筛选审核候选。

面数只筛选剩余审核候选，不决定几何处置：

- `<=100` 面：`noise_candidate`；
- `101–999` 面：`small_region_candidate`；
- `>=1000` 面：不因面数审核；
- 长细条候选：不受面数限制，必须审核。

所有剩余候选都必须由用户分类为 `noise`、`part` 或 `uncertain`。分类只改变报告标签；三类都保持完整 source 顶点、面、颜色、方向和组件身份，并进入相同的正常拆件流程。确认成 noise 也不得合并、删除、改色、修复或跳过交接面规划。

阶段 03 完成后，完整拆件流程还必须经过识别结果确认。查看 `03_recognition_boundary_review.md` 与预览；如果识别有误，可在 `03_recognition_review.json` 的 `actions` 中明确写 `delete` 或 `merge`，再通过 `--recognition-review-json PATH` 重跑。应用修改后必须复核新报告，并将新 JSON 中 `user_confirmed` 设为 `true`；只有匹配当前结果指纹的确认才会进入装配。这里的删除/合并是独立的用户明确操作，不由 `noise`/`part`/`uncertain` 分类自动触发。

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
- 除识别阶段 `<1 mm²` 连通涂色区域过滤规则外，不按 `<1%`、长度、体积或面数自动修复 source。
- 不对 source 执行全局 winding 修复；接口关系独立记录两侧的内向方向。
- source 小孔远离接口时忽略；碰到接口时重新规划接口，不修补整个 source。
- 孤立 source shell 原样输出，不虚构连接结构。

## 交接线与交接面

识别完成后，第 04 阶段以冻结的简化边界为准，逐对规划接触零件的榫件、卯件和方向，不选择主体，也不建立父子树。接触两侧引用同一条识别边界。允许处理范围限于接口局部和新生成几何；不得以生成可打印实体为由改写无关 source。

交接验证至少包括：共享边界一致、生成面非退化、生成面方向正确、接口不明显穿出、必要装配间隙、材料槽保持，以及输出 3MF 可重新读取。source 自身的开边、翻面或非流形只记录。

边界统一使用 `--boundary-shape smooth`。阅读 [boundary-smoothing.md](references/boundary-smoothing.md)、[assembly-algorithm.md](references/assembly-algorithm.md) 和 [architecture.md](references/architecture.md) 后再修改接口算法。

拆件流程阶段、输入输出和单步调试工件见 [application-stages.md](references/application-stages.md)。开发或诊断时可使用 `--stop-after-stage` 和 `--stage-artifacts-dir` 查看每个阶段的 JSON/NPZ 结果。

## CLI 与执行

安装 `requirements.txt` 后先运行：

```bash
python scripts/split_painted_3mf.py --input model.3mf --preflight-only
```

CLI 只负责参数解析、输入校验、调用可测试服务、输出报告和退出码。不要把审核、分类或几何逻辑复制到 CLI。退出码 `4` 仅表示等待 source 区域语义确认；退出码 `3` 表示交接面、生成结构或包输出失败，不能用它表示无关 source 缺陷。

当前迁移目标是可审阅的第 04 阶段规划工件，运行时使用 `--stop-after-stage assembly`。第 05 阶段之后仍待接入榫卯关系清单。最终发布恢复后，保留原始颜色槽与逐面材料。诊断报告必须区分：

```text
source_diagnostics: non_blocking
interface_validation: blocking_when_invalid
```

若本次识别生成 `03_recognition_boundaries.png`，最终回复必须内嵌显示该图片（`![边界预览](绝对路径)`），不能只给文件链接。回复前先确认图片文件存在且可读取；报告仍可另附链接。

## 开发验证

修改后至少运行区域审核、CLI、最终整理和端到端测试。关键不变量：对同一候选分别选择 `noise`、`part`、`uncertain` 时，source 顶点、面、颜色和组件成员必须一致；只有语义 metadata 可以不同。
