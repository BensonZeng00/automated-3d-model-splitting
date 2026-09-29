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

当符合审核条件的候选超过 `--max-region-review-candidates`（默认 10）时，按完整源表面积从大到小只审核前 N 个，其余候选自动分类为噪声并从独立零件识别中排除。该分类不删除或改写源面；面积相同时按最小源面索引稳定排序。

进入审核的前 N 个候选由用户分类为 `noise`、`part` 或 `uncertain`。`part` 必须作为独立零件进入识别，即使少于默认面数门槛；`noise` 不生成独立零件，源面由完整归属阶段保留到宿主零件。`uncertain` 继续按普通识别门槛处理。超过上限的候选自动分类为 `noise`。原始 3MF 的顶点、面与涂色数据不改写；输出时，归并到宿主的面使用宿主零件颜色，已识别零件的源面保持自身颜色。归属和输出颜色须在导出回读中核对。

首次识别不要求预先提供 `--visual-semantics-json`：未命名区域以零置信度的“待确认区域 Pxx”写入报告，一次运行即可生成边界图和审核文件，不能将占位标签当作已确认部件名称。阶段 03 完成后，完整拆件流程还必须经过识别结果确认。查看 `03_recognition_boundary_review.md` 与预览；如果识别有误，可在 `03_recognition_review.json` 的 `actions` 中明确写 `delete` 或 `merge`，再通过 `--recognition-review-json PATH` 重跑。应用修改后必须复核新报告，并将新 JSON 中 `user_confirmed` 设为 `true`；只有匹配当前结果指纹的确认才会进入装配。这里的删除/合并是独立的用户明确操作，不由 `noise`/`part`/`uncertain` 分类自动触发。

阶段 04 完成后同样必须停下供用户审核 `04_interface_table.md` 和冻结边界。`04_assembly_review.json` 绑定源文件、边界和当前榫卯规划；只在用户明确确认后设置 `user_confirmed=true`，用 `--assembly-review-json PATH` 进入阶段 05。发现错配时保持 `false`，在 `correction_requests` 逐项记录接口 ID、问题及用户要求；当要求更改榫方选择策略时，可明确设置 `direction_policy` 为 `smaller_area_first`。按用户指令修正并重做阶段 04，再让用户确认新规划。未确认、存在待改项或规划指纹失效时，全流程和阶段工件续跑均不得进入阶段 05。

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
--max-region-review-candidates 10
--region-review-json PATH
--region-review-dir PATH
--region-review-resolution 320
```

旧版 `--min-faces`、`--tiny-component-policy`、`--tiny-component-auto-noise-max-faces` 和 `--tiny-component-review-*` 不兼容且不再接受。不得增加兼容别名或把旧参数静默映射到新参数。

## 图像语义审核硬门槛

小区域分类审核前，Codex 必须逐一查看前 N 个候选的多视图图像，提出具体部位名称、图像依据和置信度，再让用户决定 `noise`、`part` 或 `uncertain`。`F001` 等编号仅用于定位，不是语义标签。阶段 03 识别表对每个 P 区域同样必须给出部位判断、图像依据和置信度；“待确认区域 Pxx”不得作为已完成的语义识别。若阶段 03 已有 JSON 和预览图，使用 `scripts/render_semantic_review.py` 从工件生成审核表，不再读取原 3MF。缺少上述语义判断时，不得进入阶段 04 或阶段 05。

## Source 保持约束

- 不运行按 `<=2 mm` 区域归并或小口封闭。
- 除识别阶段 `<1 mm²` 连通涂色区域过滤规则外，不按 `<1%`、长度、体积或面数自动修复 source。
- 不对 source 执行全局 winding 修复；接口关系独立记录两侧的内向方向。
- source 小孔远离接口时忽略；碰到接口时重新规划接口，不修补整个 source。
- 孤立 source shell 原样输出，不虚构连接结构。

## 交接线与交接面

独立零件间的边界配对使用统一的双向覆盖判定：计算一侧采样点到另一侧边界线段的最近距离，并反向计算；闭合轮廓对闭合轮廓，碎片化宿主对宿主局部边界线段集合。两方向覆盖率均须至少为 80%，距离容差沿用当前轮廓采样尺度规则。记录容差、两方向覆盖率以及 P50、P90、最大距离。宿主候选仍须通过唯一性判定；有多个相近候选时不得猜测。未通过的边界保留在审核诊断中，不得把少数离群点解释为接口完全不存在。

识别完成后，第 04 阶段以冻结的简化边界为准，逐对规划接触零件的榫件、卯件和方向，不选择主体，也不建立父子树。接触两侧引用同一条识别边界。允许处理范围限于接口局部和新生成几何；不得以生成可打印实体为由改写无关 source。

交接审计至少记录：共享边界、生成面退化与方向、接口穿出、装配间隙、材料槽保持，以及输出 3MF 回读状态。接口外观诊断保留在报告中；无法闭合、绕序错误、装配余量错误、新增退化面或导出回读失败时不得把零件标记为完成。缺少第 04 阶段关系或必需方向等导致无法构造的输入仍作为执行错误；不得猜测补榫。source 自身的缺陷只在远离本次交接面的范围内记录，不自动修复。

简化边界由阶段 03 冻结。修改接口算法前阅读 [boundary-smoothing.md](references/boundary-smoothing.md)、[assembly-algorithm.md](references/assembly-algorithm.md) 和 [application-stages.md](references/application-stages.md)。

拆件流程阶段、输入输出和单步调试工件见 [application-stages.md](references/application-stages.md)。开发或诊断时可使用 `--stop-after-stage` 和 `--stage-artifacts-dir` 查看每个阶段的 JSON/NPZ 结果。

阶段 03 和 04 同时输出表格型 JSON（`03_recognition_table.json`、`04_interface_table.json`）及用户可读的 Markdown 表格（同名 `.md`），终端也打印表格；不输出 CSV。已有阶段 JSON 可通过 `scripts/render_stage_tables.py --run-dir RUN_DIR` 补生成表格，无需重读源模型。

已存在 02/03/04 阶段工件时，若只需重跑 05，使用 `scripts/run_stage05_from_artifacts.py --source-run-dir RUN_DIR --output-root OUTPUT_DIR`；该入口直接校验并读取既有 NPZ/JSON，不重读 3MF，也不重建识别边界。

若阶段 03 已完成且用户确认了识别结果，之后才补齐图像语义标签，使用 `scripts/run_stage04_from_artifacts.py --source-run-dir RUN_DIR --input SOURCE.3mf --visual-semantics-json SEMANTICS.json --recognition-review-json RUN_DIR/03_recognition_review.json` 从已保存网格、区域和冻结边界生成阶段 04；不得为补标签重读源 3MF。该入口核验工件清单、源 SHA-256、识别确认指纹和区域分类，并拒绝覆盖已有阶段 04 计划。

正式输出也可从主 CLI 续跑：

```bash
python scripts/split_painted_3mf.py --input model.3mf \
  --resume-stage05-from PREVIOUS_RUN_DIR --output model_split_parts.3mf \
  --stage-artifacts-dir NEW_ARTIFACT_ROOT
```

该入口核对源文件路径与 Stage 02 记录的 SHA-256，并验证 02/03/04 工件的校验和与冻结边界指纹。新生成的 Stage 04 工件还记录上游参数和实现指纹；续跑时若上游代码已变化则拒绝复用。旧工件无此记录时仍可按工件校验和与边界指纹复用，但不具备代码版本校验。入口只重新构造并验证 Stage 05；修改识别、边界或 Stage 04 算法时必须重新运行相应上游阶段。

## CLI 与执行

安装 `requirements.txt` 后先运行：

```bash
python scripts/split_painted_3mf.py --input model.3mf --preflight-only
```

CLI 只负责参数解析、输入校验、调用可测试服务、输出报告和退出码。不要把审核、分类或几何逻辑复制到 CLI。退出码 `4` 仅表示等待 source 区域语义确认；退出码 `3` 表示交接面、生成结构或包输出失败，不能用它表示无关 source 缺陷。

当前后续构建阶段直接消费第 04 阶段的榫方、卯方、方向和冻结共享边界；不得恢复主体推断或父子树递归。原 05/06/07 合并为 `interface-assembly`，早停参数为 `--stop-after-stage interface-assembly`。05 阶段以冻结边界的面积法向为统一构造轴，04 的榫卯方向只决定轴的朝向。按默认 0.50 比例生成椭圆内轮廓，只沿榫卯轴尝试一次三角剖分；不进行其他方向探索或该椭圆候选的投影/三维自相交扫描。闭合、绕序、退化面和卯侧厚度仍须通过。若候选因其他阻断检查失败，只允许一个回退：采用第 03 阶段简化后的完整边界轮廓（100%），沿插入轴平移 1 mm，以侧壁连接。不得在该回退中使用向中心汇聚的扇形端盖；端面必须用不增加中心顶点的三角剖分，无法满足时拒绝候选。回退仍须通过厚度和完整三维互穿检查。04 外圈在完整零件中只作为连接边界；保留源外观并将其局部接入交接面，不做布尔并集。最终发布保留原始颜色槽与逐面材料，要求完整零件及 3MF 回读通过。诊断报告必须区分：

```text
source_diagnostics: non_blocking
interface_validation: diagnostic_only
```

若本次识别生成 `03_recognition_boundaries.png`，最终回复必须内嵌显示该图片（`![边界预览](绝对路径)`），不能只给文件链接。回复前先确认图片文件存在且可读取；报告仍可另附链接。

关键不变量：对同一候选分别选择 `noise`、`part`、`uncertain` 时，source 顶点、面、颜色和组件成员必须一致；只有语义 metadata 可以不同。
