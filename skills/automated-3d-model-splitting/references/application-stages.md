# Application 拆件阶段与工件

`split3mf.pipeline.SplitPipeline` 当前承载生产拆件用例。阶段工件由 `split3mf.application.StageArtifactStore` 原子写入；算法模块不负责 CLI 输出目录，也不直接发布阶段文件。

## 阶段顺序与输入输出

| 阶段 | 输入 | 主要输出 | 工件 |
|---|---|---|---|
| 01 预检 | 输入路径、运行环境 | 格式、依赖和参数预检记录 | `01_preflight.json` |
| 02 读取项目 | 通过预检的 3MF | 毫米坐标顶点/三角面、逐面 paint token、项目设置 | `02_loaded_project.npz`、`02_loaded_project_summary.json` |
| 03 识别区域 | 已读取网格、识别策略、可选用户确认 | 区域来源面索引、颜色与语义审核记录、冻结后的源边界环 | `03_recognition_regions.npz`、`03_recognized_boundaries.npz`、`03_recognition_summary.json` |
| 04 装配规划 | 识别区域、冻结边界、主体策略、连接证据 | 主体索引、父子关系、递归层 | `04_assembly_plan.json` |
| 05 接口规划与预检 | 装配树、共享边界策略 | 接口预检记录、递归层数、接口策略 | `05_interface_plan.json` |
| 06 递归构建 | 接口决策、源网格、递归执行计划 | 最终活动零件网格及递归记录 | `06_recursive_build_meshes.npz`、`06_recursive_build_summary.json` |
| 07 装配适配 | 已构建零件、缩放/就位策略 | 适配后的零件网格、交叠/可见性/多视角检查 | `07_fitted_assembly_meshes.npz`、`07_fitted_assembly_summary.json` |
| 08 验证发布 | 适配结果、源项目元数据 | 回读验证结果和最终 3MF 路径 | `08_validated_publish.json` |

NPZ 使用 `numpy.load(path, allow_pickle=False)` 检查。每个 NPZ 旁有同前缀的 manifest JSON，记录数组名、形状、类型和 SHA-256。阶段摘要 JSON 使用 UTF-8，可直接查看。

识别完成后，application 为本次运行创建不可变的 `RecognizedBoundaries`。它按组件保存有序边界环、源网格顶点 ID 和对应坐标；后续装配规划及父子边界关系校正读取同一快照，阶段摘要记录其指纹。NPZ 中 `component_loop_offsets` 索引组件拥有的环，`boundary_loop_offsets` 索引每个环的顶点范围，`boundary_vertex_ids` 和 `boundary_points` 一一对应。后续几何构建可以产生构建网格自己的拓扑边界，但不能改写这份识别快照。

## 单步调试

完整运行会在输出 3MF 同目录创建 `<输出名>_stages/<run-id>/`。每次运行有唯一目录，早停时不会把上次运行的后续阶段文件混进来。也可以指定工件根目录：

```powershell
python scripts/split_painted_3mf.py --input model.3mf --stage-artifacts-dir .\debug\model
```

用 `--stop-after-stage` 在某一步写完工件后正常退出；再次运行会重新执行此前阶段，并写入新的 run-id 目录：

```text
preflight | load | recognize | assembly | interfaces | build | fit
```

例如先用 `--stop-after-stage recognize` 检查区域面索引和审核结果，再用 `assembly` 检查父子树，最后用 `interfaces` 审核整树预检报告。`build` 会在递归构建后停止，`fit` 会在缩放、就位和多视角检查后停止。最终发布阶段不支持早停，因为它是完整命令的提交点。

## 模块边界

- CLI 构造配置、执行预检、选择用例并映射退出码。
- Application 管理阶段顺序、早停点和阶段工件。
- `domain/` 中每个文件定义一个跨阶段数据类型。
- `project.py`、`recognition.py`、`assembly.py` 和 `part_geometry.py` 提供领域/几何服务。
- `package_io.py` 与 `validation.py` 负责最终 3MF 序列化和回读验证。
- `stage_cache.py` 管理内容寻址的递归计算缓存；application 阶段工件是面向人工检查的运行记录，两者用途不同。

每个 application 类单独放在一个文件中。几何服务按一个明确职责拆文件；一个阶段可以调用多个服务，但不能把算法复制到编排器中。
