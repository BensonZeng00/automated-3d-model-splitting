# 1.3.5 最终测试结果

- 测试日期：2026-07-30
- 系统：Microsoft Windows NT 10.0.26200.0
- Python：3.12.13
- Skill：`automated-3d-model-splitting`
- 版本：1.3.5
- Python 3.10：由 GitHub Actions 矩阵验证，本地未运行
- Linux/macOS：本次本地发布检查未运行

## 结果摘要

| 检查 | 结果 |
|---|---:|
| 无输入依赖预检 | 通过 |
| 单元与回归测试 | 54/54 通过 |
| 发布版本、Schema 和仓库卫生校验 | 通过 |
| 合成涂色 3MF 端到端拆分 | 通过 |

端到端输出结构：

- 网格零件：2
- 装配对象：1
- 装配组件：2
- build item：1，且引用装配对象
- 合成源三角面：768
- 临时输出 SHA-256：`70660657ecdc34fe0a824fca9f7a1fd031ecf7993514b473e6129a4f76238c19`

## 执行命令

```powershell
python -X utf8 -B .\skills\automated-3d-model-splitting\scripts\split_painted_3mf.py --preflight-only
python -X utf8 -B -m unittest discover -s .\skills\automated-3d-model-splitting\tests -v
python -X utf8 -B .\tools\validate_release.py
python -X utf8 -B .\tools\run_synthetic_e2e.py
```

合成输入和输出都位于系统临时目录，并在测试结束后自动清除。仓库没有保存或上传用户模型、厂商模型、切片器截图或测试 3MF。
