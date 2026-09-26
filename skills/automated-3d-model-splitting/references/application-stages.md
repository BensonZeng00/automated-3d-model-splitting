# Application 拆件阶段与工件

`split3mf.pipeline.SplitPipeline` 当前承载生产拆件用例。阶段工件由 `split3mf.application.StageArtifactStore` 原子写入；算法模块不负责 CLI 输出目录，也不直接发布阶段文件。

## 阶段顺序与输入输出

| 阶段 | 输入 | 主要输出 | 工件 |
|---|---|---|---|
| 01 预检 | 输入路径、运行环境 | 格式、依赖和参数预检记录 | `01_preflight.json` |
| 02 读取项目 | 通过预检的 3MF | 毫米坐标顶点/三角面、逐面 paint token、项目设置 | `02_loaded_project.npz`、`02_loaded_project_summary.json` |
| 03 识别区域 | 已读取网格、外表面颜色、语义审核与边界简化策略 | 区域来源面索引、简化后的冻结边界环、彩色边界预览与判断摘要 | `03_recognition_regions.npz`、`03_recognized_boundaries.npz`、`03_recognition_summary.json`、`03_recognition_boundaries.png`、`03_recognition_boundary_review.md` |
| 04 装配规划 | 识别区域、冻结的简化边界、零件内部方向证据 | 每对接触零件的榫件、卯件、插入方向和判定证据 | `04_assembly_plan.json` |
| 05 接口规划与预检 | 待迁移：消费第 04 阶段榫卯关系 | 尚未迁移 | `05_interface_plan.json` |
| 06 递归构建 | 接口决策、源网格、递归执行计划 | 最终活动零件网格及递归记录 | `06_recursive_build_meshes.npz`、`06_recursive_build_summary.json` |
| 07 装配适配 | 已构建零件、缩放/就位策略 | 适配后的零件网格、交叠/可见性/多视角检查 | `07_fitted_assembly_meshes.npz`、`07_fitted_assembly_summary.json` |
| 08 验证发布 | 适配结果、源项目元数据 | 回读验证结果和最终 3MF 路径 | `08_validated_publish.json` |

NPZ 使用 `numpy.load(path, allow_pickle=False)` 检查。每个 NPZ 旁有同前缀的 manifest JSON，记录数组名、形状、类型和 SHA-256。阶段摘要 JSON 使用 UTF-8，可直接查看。

识别完成后，application 为本次运行创建不可变的 `RecognizedBoundaries`。候选环在识别阶段按约 5% 等距弧长抽样简化，并按区域保存有序源网格顶点 ID 和坐标。第 04 阶段以简化环决定哪些边界参与规划；为精确计数接触边，只把简化环相邻采样点之间映射回对应的源边链，不再另行推断父子关系。阶段摘要记录同一边界快照的指纹。NPZ 中 `component_loop_offsets` 索引简化环，`boundary_loop_offsets` 索引每个环的顶点范围，`boundary_vertex_ids` 和 `boundary_points` 一一对应；`source_*` 数组保存源拓扑环。识别阶段同时输出每环独立颜色的 PNG 和判断摘要。

连通分组后先按三角面总面积过滤小于 `1 mm²` 的区域，将其作为噪声排除在有效零件、边界与语义审核清单之外；原始网格与涂色数组不改写，排除区域的面数、面积、颜色和原因写入 `03_recognition_summary.json`。剩余区域再进入面数/长细形状审核与边界提取。识别会在正式生成区域清单前删除所有边界环都被过滤的候选区域，并在识别摘要中保留剔除原因。绘图只接收最终有效区域和边界，不改变零件集合。生成识别报告前，`--visual-semantics-json` 必须为每个有效 P 区域提供非空视觉语义标签；标签缺失时明确列出缺项并停止输出报告，禁止静默写成“未标注”。报告将语义标签与具体颜色并列展示。正常完整流程在阶段 03 后必须经过一次识别确认。首次运行会在工件目录生成 `03_recognition_review.json`，并以退出码 4 暂停装配。用户可在 `actions` 中填写 `{"action":"delete","parts":["P06"]}` 或 `{"action":"merge","parts":["P02","P03"],"keep":"P02"}`，然后用 `--recognition-review-json` 重跑。删除/合并应用后会生成修订报告和新的复核 JSON，再次确认该文件中的 `user_confirmed=true` 才继续；结果指纹必须与复核文件一致。没有修改时将 `actions` 保持为空，确认后重跑即可继续。预览图例和区域表使用同一组有效 P 编号。

## 单步调试

完整运行会在输出 3MF 同目录创建 `<输出名>_stages/<run-id>/`。每次运行有唯一目录，早停时不会把上次运行的后续阶段文件混进来。也可以指定工件根目录：

```powershell
python scripts/split_painted_3mf.py --input model.3mf --stage-artifacts-dir .\debug\model
```

用 `--stop-after-stage` 在某一步写完工件后正常退出；再次运行会重新执行此前阶段，并写入新的 run-id 目录：

```text
preflight | load | recognize | assembly | interfaces | build | fit
```

例如先用 `--stop-after-stage recognize` 检查区域面索引和审核结果，再用 `assembly` 检查逐接口榫卯关系。当前只迁移了第 04 阶段；命令必须以 `--stop-after-stage assembly` 结束。第 05 阶段及之后仍待接入新的关系契约。

## 模块边界

- CLI 构造配置、执行预检、选择用例并映射退出码。
- Application 管理阶段顺序、早停点和阶段工件。
- `domain/` 中每个文件定义一个跨阶段数据类型。
- `project.py`、`recognition.py`、`assembly.py` 和 `part_geometry.py` 提供领域/几何服务。
- `package_io.py` 与 `validation.py` 负责最终 3MF 序列化和回读验证。
- `stage_cache.py` 管理内容寻址的递归计算缓存；application 阶段工件是面向人工检查的运行记录，两者用途不同。

每个 application 类单独放在一个文件中。几何服务按一个明确职责拆文件；一个阶段可以调用多个服务，但不能把算法复制到编排器中。
