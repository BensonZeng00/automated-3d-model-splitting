# Application 拆件阶段与工件

`split3mf.pipeline.SplitPipeline` 当前承载生产拆件用例。阶段工件由 `split3mf.application.StageArtifactStore` 原子写入；算法模块不负责 CLI 输出目录，也不直接发布阶段文件。

## 阶段顺序与输入输出

| 阶段 | 输入 | 主要输出 | 工件 |
|---|---|---|---|
| 01 预检 | 输入路径、运行环境 | 格式、依赖和参数预检记录 | `01_preflight.json` |
| 02 读取项目 | 通过预检的 3MF | 毫米坐标顶点/三角面、逐面 paint token、项目设置 | `02_loaded_project.npz`、`02_loaded_project_summary.json` |
| 03 识别区域 | 已读取网格、外表面颜色、语义审核与边界简化策略 | 区域来源面索引、简化后的冻结边界环、彩色边界预览与判断摘要 | `03_recognition_regions.npz`、`03_recognized_boundaries.npz`、`03_recognition_summary.json`、`03_recognition_boundaries.png`、`03_recognition_boundary_review.md` |
| 04 装配规划 | 识别区域、冻结的简化边界、零件内部方向证据 | 每对接触零件的榫件、卯件、插入方向和判定证据 | `04_assembly_plan.json` |
| 05 接口构建与装配适配 | 第 04 阶段逐对接触关系（含已判定榫方、卯方及方向）、冻结边界、打印间隙配置 | 通过预检的成对交接面、配合间隙和适配网格 | `05_interface_assembly_meshes.npz`、`05_interface_assembly_summary.json` |
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
preflight | load | recognize | assembly | interface-assembly
```

例如先用 `--stop-after-stage recognize` 检查区域面索引和审核结果，再用 `assembly` 检查逐接口关系。旧 05、06、07 阶段合并为一个 `interface-assembly` 阶段，消费第 04 阶段逐对关系并依序完成接口预检、构建和适配；不重新判定榫方、卯方或接口方向。

已有一次完整的 02/03/04 运行工件时，可只续跑 05，不重新读取源 3MF 或识别边界：

```powershell
python scripts/run_stage05_from_artifacts.py `
  --source-run-dir .\artifacts\model_stages\<run-id> `
  --output-root .\artifacts\model_stage05_resume `
  --interface-scale-ratio 0.50 `
  --interface-clearance-mm 0.20
```

该命令校验 02/03 NPZ manifest 与校验和、04 关系和冻结边界指纹；新 05 工件写入独立 run-id 目录。

合并阶段的几何契约：

- 以第 04 阶段的每条接触关系、榫方/卯方判定、插入方向/卯方内向方向和冻结的共同边界作为唯一接口依据；下游不得覆盖或重新推断这些关系，也不从接触图擅自推断唯一主体或强行将多父关系改成树。方向向量按对应 interface ID 传递，禁止退化成每个零件一个全局轴。
- 每条第 04 阶段关系各自生成一组成对交接结构：冻结简化共享边界投影到接口平面作为外圈，以面积质心按默认 `0.50` 比例缩小得到内圈。内圈沿第 04 阶段卯方内向方向探测，最大 10 mm；命中或穿过卯方外壳时深度减半，最低 0.2 mm。达到最小深度仍碰撞时继续采用该深度并记录碰撞诊断。翻折、面积差、自交、退化面、闭合/绕序和余量偏差均只写入诊断，不阻断 05 阶段或接口批次。
- 交接轮廓的真实自交判断使用毫米世界坐标中的 XYZ 线段最短距离；同时分别记录冻结 3D 边界和投影后接口轮廓的结果。平面环带面积、方向与覆盖检查仍针对交接面所在的二维平面域计算，不将其当作原始 3D 边界相交结论。
- 内圈缩放中心使用投影轮廓的面积质心；点序保持 04 提供的顺序。探测方向采用关系中的 `socket_inward_direction`；接口平面和榫方方向采用 `insertion_direction`，不可用模型中心或零件全局轴替代。
- 先按冻结轮廓生成榫并记录尺寸、位置和探测深度；再据此生成对应卯的负形，卯侧向轮廓和底部深度均额外增加完整配置余量。榫卯内端面分别尝试封闭，外圈冻结边界保持原坐标。对余量后的卯结构执行卯壳碰撞/穿越探测并记录结果，不作阻断判断。记录榫深、卯深、侧向余量和底部余量。
- 合并阶段在内存中完成预检、构建和适配，并写出 05 工件；接口质量审计作为诊断数据保留，不作为阶段拦截门槛。阶段 08 对最终输出执行发布检查；当前阶段不运行旧的递归装配视图评估。

## 模块边界

- CLI 构造配置、执行预检、选择用例并映射退出码。
- Application 管理阶段顺序、早停点和阶段工件。
- `domain/` 中每个文件定义一个跨阶段数据类型。
- `recognition.py`、`contact_interface_planner.py` 和 `interface_assembly.py` 分别负责识别、04 关系规划和 05 接口构建。
- `package_io.py` 与 `validation.py` 负责最终 3MF 序列化和回读验证。
- 阶段摘要中的输入哈希由 `stage_cache.sha256_file` 计算；当前流程不启用旧递归缓存。

每个 application 类单独放在一个文件中。几何服务按一个明确职责拆文件；一个阶段可以调用多个服务，但不能把算法复制到编排器中。
