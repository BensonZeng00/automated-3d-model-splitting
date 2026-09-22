# 大型实模拆件基准与优化建议

## 结论

2026-09-22 在同一容器内对 `example/Kamesennin+AMS.3mf` 和
`example/yoshisittingonledge.3mf` 执行了实际 CLI 预检和拆件识别流程。
两个输入都通过依赖和格式预检，但本次运行**没有产出可宣称为可打印的最终拆件
3MF**：两个模型都被正确阻断在边界语义确认。初始 Yoshi 运行在完成 paint 展开后
长时间停留在边界分析并被运行环境终止；边界优化后，同一检查可在 96.712 秒内生成
完整审核候选并以预期的 exit 4 结束。不能把预览、候选或中间网格冒充最终结果。

这次实测同时确认：拓扑来源版 paint 展开不再是两个模型的主要瓶颈。下一轮通用
优化应优先处理边界图分析、候选生成和预览，而不是放松 T-junction 审计或跳过人工
语义确认。

## 方法

预检命令：

```bash
python skills/automated-3d-model-splitting/scripts/split_painted_3mf.py \
  --input INPUT.3mf --preflight-only
```

实际拆件识别命令：

```bash
python skills/automated-3d-model-splitting/scripts/split_painted_3mf.py \
  --input INPUT.3mf --recognize-only
```

外层使用 Bash `time` 记录 wall/user/sys 时间；内部使用 `runtime_step` JSON 记录阶段
时间和 paint 一致化审计。计时是单次冷运行，不应解释为跨硬件的稳定基准。

## 实测结果

| 指标 | Kamesennin+AMS | Yoshi sitting on ledge |
| --- | ---: | ---: |
| 输入大小 | 7,684,304 bytes | 7,001,724 bytes |
| 预检 wall time | 1.361 s | 1.320 s |
| 原始 source faces | 379,648 | 391,844 |
| 唯一 paint tokens | 2,242 | 6,871 |
| token 解码 | 0.244 s | 2.665 s |
| paint 展开及一致化 | 14.361 s | 21.558 s |
| 展开后 faces | 458,624 | 728,002 |
| 展开后 vertices | 237,705 | 396,986 |
| 审计细分原始边 | 5,293 | 17,002 |
| 审计细分边段 | 19,968 | 69,083 |
| 共享边一致率 | 100% | 100% |
| source 归属率 | 100% | 100% |
| 读取及标准化完成 | 30.323 s | 37.280 s |
| 初始端到端状态 | 229.258 s 后要求边界确认（exit 4） | 标准化后边界阶段运行超过 12 分钟，约 1.88 GiB RSS 时被环境终止 |
| 边界优化后状态 | 未重复实模计时 | 96.712 s 后生成 3 个候选和 2 张审核图，要求边界确认（exit 4） |
| 最终可打印输出 | 无 | 无 |

## 结果分析

### Kamesennin

- paint 一致化只检查 5,293 条有内部细分点的原始边，并成功审计 19,968 个边段；
  该阶段 14.361 秒，占本次 229.258 秒 wall time 的约 6.3%。
- 边界检查发现 3 个候选并生成预览。报告包含 25,758 个不确定面；多个材料对仍有
  大量未解释端点，候选也标记为 `needs_review`，因此程序以 exit 4 阻断是正确行为。
- 预览表明主要材料区域符合角色外观，但腰部、眼镜/面部和裤脚附近存在窄带与断续
  边界。仅凭自动分数选择候选会把材料归属判断误当成纯几何问题。
- 没有生成最终件，因此不能声称装配间隙、插入方向、底面厚度或可打印性已通过。

### Yoshi

- paint 一致化处理 391,844 个 source faces，生成 728,002 个一致面，并对 69,083
  个细分边段全部通过阻断式审计；21.558 秒远低于旧全局投影实现的量级。
- 最大单 source triangle 展开为 2,769 个 leaf faces，说明后续算法必须按局部区域、
  稀疏边界和流式数据设计，不能对全部 leaf faces 构造稠密候选关系。
- 读取完成后长时间无新的 machine-readable checkpoint，进程接近 1.88 GiB RSS 后被
  终止。当前日志粒度只能把瓶颈定位到读取之后、边界报告之前，尚不足以区分边界图
  构建、候选搜索、外部可见性或预览渲染各自的耗时。
- 优化复测已生成边界报告，但仍没有经用户确认的最终 3MF，因此不能对 Yoshi 的拼装
  便利性或打印有效性作肯定结论。

### Yoshi 边界阶段优化复测

优化后的 `--boundary-check-only` 冷运行在 96.712 秒完成，阶段记录如下：

| 子阶段 | 耗时 |
| --- | ---: |
| 稀疏面邻接图（1,008,513 条邻接边） | 0.960 s |
| 共享边交叉清理 | 0.975 s |
| 边界拓扑评估（121,205 个不确定面） | 0.348 s |
| 3 个候选的局部归属搜索 | 13.589 s |
| 2 张有界审核图渲染 | 32.661 s |

主要改动是：交叉清理只从材料间共享边构造环，不再为每种材料反复扫描全部三角面；
超出精确二次相交预算的复杂 seam 不做自动改写，而是保持原归属进入全量拓扑评估和
人工审核；候选投票改用稀疏矩阵批量计算；预览仅使用确定性、有界的显示采样，而全部
拓扑审计仍使用完整网格。这里限制的是可选自动清理和显示资源，不是正确性审计范围。

## 通用优化方案

### P0：可观测性和可恢复执行

1. 给外部可见性、材料邻接图、边界连通分量、候选搜索、预览渲染分别增加 start/done
   checkpoint，并记录 faces、edges、components、候选数、wall time 和峰值 RSS。
2. 将 paint 标准化结果和边界图做内容寻址缓存；用户确认候选后不应重复解码 token、
   展开 70 万面或重建不变的邻接关系。复用现有 `--resume`/stage cache 基础设施，避免
   新建平行缓存系统。
3. 将长阶段写成可恢复的确定性阶段；只缓存完整且通过指纹校验的结果，禁止把超时或
   半成品当成成功。

### P1：拆件速度和内存

1. 边界分析以“异材料共享边”的稀疏 edge index 为输入，不要在所有 leaf face 对之间
   搜索。按材料对和连通分量分桶后独立处理，释放已完成分桶的临时数据。
2. 用 NumPy 整型数组/CSR 邻接替代 Python `set`、嵌套字典和大量短数组；顶点对先规范
   化后排序归并，一次得到使用次数和材料对。
3. 候选搜索限定在不确定边界 band 的 k-ring 内，但限制必须来自拓扑范围而不是超时、
   随机抽样或“最多 N 个候选”；范围外面保持原 source 归属。
4. 预览延迟到结构候选完成后，并只上传边界附近的子网格；同一相机的深度/投影结果
   在候选间复用。预览不能参与几何正确性判定。
5. 对最大 leaf expansion、边界边数和预计内存做前置估算；超出预算时选择分块算法，
   而不是降低精度或跳过审计。

### P1：后续拼装便利性

1. 在候选评分中加入装配方向可达性、单轴插入路径、支撑面面积、防转约束和拆装顺序，
   但与材料归属置信度分开报告，不能用“更好装”覆盖错误边界。
2. 父子件共享同一个不可变 seam definition；公母结构从该定义派生，并在输出 metadata
   中保存 part id、parent id、插入方向、建议间隙和装配顺序。
3. 接插件尺寸使用喷嘴、层高、材料和机器相关的显式 profile；保留 CLI 参数和报告值，
   不把某台打印机的经验间隙硬编码成通用常量。
4. 对细长或大面积接口优先生成多个有防呆方向的小连接结构，而不是单个难以对准的大
   榫；同时检查连接器与可见表面、薄壁及相邻子件的距离。

### P1：准确性与可打印性

1. 继续保留当前细分边段实际使用次数和 source ownership 的阻断式审计；不得为提速
   关闭、抽样或改成警告。
2. 最终输出必须重新读取，并独立检查每件的退化生成面、生成接口开边/过共享边、法向、
   最小壁厚、局部自交、父子穿透和装配间隙。source 自带缺陷仍应单独标为非阻断。
3. 对接口做切片尺度检查：最小特征应同时满足喷嘴宽度与层高约束；薄于阈值时重新规划
   接口，而不是全局修复 source。
4. 自动候选只有在拓扑闭合、材料归属明确且所有生成结构审计通过时才能继续；像本次
   Kamesennin 的语义歧义必须保留人工确认门。

## 下一轮验收标准

- 在固定硬件、冷/热缓存各至少 3 次运行，报告中位数和峰值 RSS。
- 两个实模都必须产生可重新读取的最终 3MF，且记录每个阶段的 machine-readable 时间。
- 细分边段一致率和 source 归属率必须保持 100%。
- 每个生成件必须通过接口局部拓扑、最小壁厚、装配干涉和材料槽保持检查。
- Kamesennin 的边界候选必须由用户确认；Yoshi 不得再在无 checkpoint 的阶段运行数分钟。

## Cathead 回归实测

2026-09-22 使用 `example/cathead.3mf` 对优化后的边界路径和完整拆件入口进行了额外
冷运行测试。

| 测试 | wall time | 结果 |
| --- | ---: | --- |
| `--preflight-only` | 1.305 s | 通过 |
| `--boundary-check-only` | 25.820 s | 通过，边界清晰 |
| `--recognize-only` | 36.174 s | 通过，识别 13 个组件，无语义审核候选 |
| 完整拆件 | 47.054 s | 在首个大布尔前被 full-tree preflight 阻断（exit 3） |

该模型包含 238,156 个 source faces、119,080 个顶点和 4 个唯一 paint tokens，且没有
递归 paint triangle。paint 标准化约 7.36 秒；共享边一致率和 source 归属率均为 100%。
边界图含 357,234 条面邻接边，构建约 0.21–0.27 秒；共享边交叉清理约 4.90–5.55 秒；
完整边界评估约 0.10–0.16 秒，结果没有不确定面。说明本次大模型边界优化没有破坏
Cathead 的 clear fast path。

首次 strict 完整拆件没有生成输出文件。阻断发生在 P01/P05 的可见 planar-arc surface band 质量
审计：生成带没有退化面、反向面或 edge flip，但受影响面积比例约 0.3841，超过 0.01
限制，并测得最大边拉伸约 2.7248。该结果触发了下面针对“覆盖范围”和“实际物理偏移”
的分级复核；它不是 paint 一致化、边界识别或资源耗尽问题。

### Cathead 可见影响复核与分级策略

`affected_area_ratio=0.3841` 表示求解器触达的三角形面积范围，不表示 38.41% 的表面产生
了肉眼可见的 3D 偏差；高密度网格上很小的平滑位移也可能传播到大量相邻面。
`maximum_edge_stretch_ratio=2.7248` 也低于 `print-balanced` 原有的 16 倍上限，并且此次
没有退化面、反向面、edge flip 或新增拓扑错误。因此，这两个值本身不足以证明打印失败。

优化后保留两级策略：

1. 默认 `strict` 仍使用 1% 面积和顶点覆盖预算，适合无人审核的批处理。
2. 用户确认肉眼效果后可使用 `--surface-band-validation advisory`。此模式只有在最大位移
   和 P95 位移都不超过 smoothing profile 的实际毫米预算时，才把面积/顶点覆盖超限降为
   visual advisory；本轮测试前 `print-balanced` 的两项位移上限都是 0.5 mm。退化面、拓扑错误、边界
   不匹配、实质性布尔失败和装配失败仍然阻断。

Cathead 使用 advisory 复测后，P01/P05 不再因为 38.41% 的“触达面积”被阻断，证明旧
面积指标确实过度保守。流程继续到 P01/P09，并因目标边界需要移动 5.975 mm、远超
0.5 mm 的真实位移限制而阻断。这个后续失败属于肉眼明显且可能改变外形/配合的实质
问题，不能降级为警告。也就是说，新策略放行的是“大范围但亚毫米”的平滑，不放行
毫米级外形漂移。

### Cathead 10 mm 阈值续测

按要求新增 `--maximum-boundary-displacement-mm`，默认值为 10 mm，并同时控制最大与 P95
目标边界位移。参数必须为正数；它独立于 3 mm transition band，避免用扩大 band 的方式
间接改变边界目标。

使用默认 10 mm、`--surface-band-validation advisory` 重新执行 Cathead 后：

1. P01/P05 的广域平滑继续通过；
2. P01/P09 原先需要 5.975 mm、超过旧 0.5 mm 限制的目标也通过；
3. full-tree preflight 继续完成 P01 的后续厚度测量、P07/P08 连接器规划和 P09 大边界处理；
4. 正式 step 0 再次计算 P01 的大 surface band，`surface_band_relaxation_done` 在约
   284.2 秒完成，但随后 boundary-ear repair 超过 10 分钟仍没有完成或输出新 checkpoint；
   人工结束测试时没有生成最终 3MF。

本轮暴露的首要问题已不再是位移阈值，而是同一大型接口在 full-tree preflight 和正式执行
之间重复计算，以及 `_repair_flipped_boundary_ears` 对大 affected band 的 Python 级候选扫描。
建议按以下顺序修复：

1. 将 full-tree preflight 已验证的 surface-band 顶点结果、修复后 faces 和质量报告按
   source/interface/config 指纹缓存，正式执行直接消费，避免重复求解。
2. 将 boundary-ear 初始 suspect 检测向量化，只对 active band 及其一环邻面建立工作队列，
   不扫描整个 55,163-face 子件。
3. 缓存 face 邻接、edge-to-face、source normal 和 shape quality；每次 edge flip 只增量更新
   两个替换面及其邻面。
4. 增加 `surface_band_boundary_repair_start` 和周期性 progress checkpoint，记录候选面、
   suspect 数、已修复数及阶段 RSS，防止再次出现数分钟无日志。
5. 继续保留退化面、反向面、拓扑、布尔和装配阻断；10 mm 只扩大目标搜索范围，不能把
   未完成修复或未审计的几何直接输出。

### 为什么先做 boundary-ear repair、此时还没有分割面

当前 inward 流程的实际顺序是：

1. 从 source component 提取共享边界环；
2. 将该环调整为双方唯一共享的 planar-arc target；
3. 在原可见表面上扩散边界位移；
4. 检查移动后的 source triangles，必要时对局部四边形换对角线，即 boundary-ear repair；
5. 只有可见边界及其邻域通过审计后，才从最终 top ring 向内生成 side wall、lead-in、
   bottom ring 和 cap；
6. 再使用生成的封闭实体执行父件布尔切割。

因此日志停在 boundary-ear repair 时“没有形成分割面”是当前事务顺序的直接结果，不是
cap 生成函数漏调。side wall 和 cap 必须消费最终审核后的共享边界；如果先建 cap、之后
ear repair 又替换 source face 对角线或调整边界，父子两侧可能不再使用同一条 seam，造成
裂缝、重叠或无法装配。

boundary-ear repair 之所以被触发，是因为把边界最多移动 10 mm、却只在默认 3 mm
surface band 内衰减，会使原三角面局部折叠或相对邻面翻转。它只修改两个相邻 source
triangle 的公共对角线，不创建封口面，也不能替代 cap。

更合理的优化不是跳过 repair，而是根据位移规模选择不同几何路径：

1. **小位移路径**：当目标偏移可被现有 surface band 平滑吸收时，继续使用 source-band
   deformation，并将 ear repair 改成向量化、增量工作队列。
2. **大位移路径**：当偏移明显大于可见 band 宽度时，保持原 visible source boundary
   不动，在模型内部生成 target ring；用生成的过渡 annulus 连接 source ring 和 target
   ring，再从 target ring 生成 side wall 和 cap。这样 10 mm 位移落在隐藏生成几何上，
   不强迫 3 mm 宽的可见 source 网格承担形变。
3. **共享定义**：父件 socket 和子件 insert 必须消费同一个 immutable source ring、target
   ring 和 annulus triangulation；不能各自重新投影或重新采样。
4. **先局部闭合再布尔**：生成 annulus、wall、cap 后立即做局部 watertight、winding、
   退化面、厚度和自交审计；通过后才进入 expensive Boolean。
5. **复用预检结果**：full-tree preflight 生成并通过审核的 interface patch 应按指纹传给
   正式执行，避免再次运行 surface relaxation 和 ear repair。

这一双路径方案可以同时解决两个问题：小位移仍保持平顺可见表面；大位移可以尽早形成
真正的分割实体，而不会为了迁就 10 mm 目标去大面积改写原可见 source mesh。

### 双路径实现与 Cathead 完整复测

现已在边界目标进入 visible surface-band 求解前实施上述分流：目标最大位移不超过
`retopology_band_mm` 时沿用可见表面平滑；超过 band 时保留原 source ring，后续 inward
builder 从这个不可变共享环生成隐藏侧壁和 cap。诊断记录同时保存请求位移、band 宽度、
是否保留 source ring 和选中的策略，避免把“没有移动可见边界”误判为求解器静默失败。

使用 Cathead、默认 10 mm 位移阈值和 advisory 验证进行完整复测：

- 命令成功退出并生成 13-object assembly 3MF；
- 总墙钟时间 446.114 秒，输出文件 4,577,383 bytes；
- 最终文件从磁盘回读复检通过，结构/颜色/耗材槽错误数为 0；
- 32 视角验证覆盖率约 92.54%，生成表面侵入率和正面材质不匹配率均为 0；
- 首个大型正式 surface band 的 relaxation 约 1.20 秒、boundary-ear repair 约
  16.97 秒并完成 267 次局部换边，不再出现此前超过 10 分钟无 checkpoint 的停滞；
- 峰值观测 RSS 约 1.78 GiB，说明正确性路径已经打通，但 Boolean 和最终多部件验证仍是
  后续最需要降低内存的阶段。

本轮实现刻意没有增加第二套 cap API：它只在既有 retopology 入口选择 top ring，继续复用
现有 side-wall、cap、Boolean 和最终审计流程。这样父子件仍消费同一环，避免并行几何实现
产生不同采样、不同 winding 或不同容差。
