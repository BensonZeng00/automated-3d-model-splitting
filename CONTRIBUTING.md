# 参与贡献

感谢你改进“自动化3d模型拆件”。提交修改前，请确认改动属于通用涂色 3MF 拆件能力，不包含任何未经授权的模型、截图、用户目录或厂商私有数据。

## 开发环境

- Windows 10/11
- Python 3.10、3.11 或 3.12
- Git

在仓库根目录安装依赖：

```powershell
python -m pip install -r .\skills\automated-3d-model-splitting\requirements.txt
```

## 提交前检查

```powershell
Push-Location .\skills\automated-3d-model-splitting
python -X utf8 -B -m unittest discover -s tests -v
Pop-Location
python -X utf8 -B -m unittest discover -s tools/tests -v
python -X utf8 -B .\tools\validate_release.py
python -X utf8 -B .\tools\run_synthetic_e2e.py
```

所有命令都应从仓库根目录执行。合成端到端测试只使用运行时生成的几何体，不需要提交 3MF 测试夹具。

发布检查覆盖 Git 索引和未忽略的新文件；`outputs/`、缓存和未列入允许清单的模型不得进入发布包。已有公开示例和维护中的回归夹具按精确路径列入 `tools/release_files.py`。检查结果中的 `skill_sha256` 标识本次 Skill 内容，请在 `TEST_RESULTS.md` 记录对应指纹、运行环境和实际验证范围。

### macOS 安装与验证

使用已配置的兼容 Python 环境；没有时可创建虚拟环境。下面从仓库根目录执行，`python3` 应指向 Python 3.10–3.12：

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r skills/automated-3d-model-splitting/requirements.txt
python -m pip check
(cd skills/automated-3d-model-splitting && python -X utf8 -B -m unittest discover -s tests -v)
python -X utf8 -B -m unittest discover -s tools/tests -v
python -X utf8 -B tools/validate_release.py
python -X utf8 -B tools/run_synthetic_e2e.py
```

CI 覆盖 Windows、macOS ARM64（`macos-15`）和 Intel（`macos-15-intel`），各运行 Python 3.10、3.11、3.12。[GitHub runner 平台说明](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)。

## Pull Request 要求

- 说明要解决的通用问题、预期行为和验证结果。
- 对算法修复增加最小化的自动回归测试。
- 修改版本号时，同步更新 `SKILL.md`、CLI、参考文档和 JSON Schema 标识。
- 不降低装配结构、颜色覆盖、封闭性、法线或 Bambu 项目元数据校验。
- 如果仍需切片器人工确认，请让测试者提供截图；不要用 Computer Use 控制切片器。

安全漏洞请按 [SECURITY.md](SECURITY.md) 私下报告，不要先创建公开 Issue。
