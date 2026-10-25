# 安装 / Installation

## 一句话让 AI 安装 / Ask your assistant

把下面这句话发给支持本地文件和命令执行的 AI 助手即可启动安装流程：
Send this request to an assistant that can access local files and run commands:

```text
请从 https://github.com/BensonZeng00/automated-3d-model-splitting 安装 automated-3d-model-splitting Skill 和所需依赖。
```

助手负责获取仓库、选择环境、调用脚本及验证安装。用户不需要提供脚本路径、系统盘符或命令参数；仍需在安装前回答套餐情况并明确接受消耗。下面的命令供助手执行或手动安装使用。

The assistant obtains the repository, selects the environment, runs the installer, and verifies installation. The user need not supply script paths or command arguments, but must still answer the plan question and explicitly accept usage before installation. The commands below are for the assistant or a manual installation.

## 命令行安装 / Command-line installation

首次安装时，先准备 Git 和 Python 3.10–3.12，在终端运行以下命令。无需把风险提示写进命令，安装器会显示提示并询问。
For a first installation, have Git and Python 3.10–3.12 available and run these commands. The installer displays the usage notice and asks for consent.

```bash
git clone https://github.com/BensonZeng00/automated-3d-model-splitting.git
cd automated-3d-model-splitting
python tools/install_skill.py
```

已有仓库时直接进入仓库并运行最后一行。也可从 GitHub 下载 ZIP、解压后进入仓库，省去 Git。若系统的 Python 命令是 `python3`，将最后一行的 `python` 换为 `python3`；Windows 也可使用 `py -3.12`。所选解释器须满足版本要求，不依赖某台电脑上的固定路径。

For an existing checkout, run only the final command from its root. Downloading and extracting the GitHub ZIP is also supported without Git. Use `python3` if that is your system's Python command, or `py -3.12` on Windows. The selected interpreter must meet the version requirement; no machine-specific path is needed.

使用已有的 Python 3.10–3.12 环境；`python` 应替换为你选择的解释器。默认使用启动脚本的解释器安装依赖，不创建额外环境、不自动升级 pip。可用 `--python` 明确指定另一个解释器。安装目标默认是 `$CODEX_HOME/skills/automated-3d-model-splitting`；未设置 `CODEX_HOME` 时使用用户目录内 `.codex/skills/automated-3d-model-splitting`。其他助手可通过 `--target` 指定自己的技能目录。

Use an existing Python 3.10–3.12 environment. The launching interpreter is used for dependencies unless `--python` selects another one. Use `--target` to override the destination Skill directory and `--source` to select a complete local Skill payload. Run `--help` for all options.

## 安装确认 / Consent

脚本先说明：拆件可能消耗较多 token，复杂模型拆件失败的概率较高；即使失败也可能消耗较多 token 和额度。实际额度与费用以所用服务为准，Coding Plan 不代表无限或免费使用。

然后分别收集：

1. 是否有 Coding Plan（编程助手订阅或额度套餐）：`有 / 没有 / 不确定`，或 `yes / no / unsure`。
2. 是否了解消耗并继续安装：输入 `同意` 或 `yes`；其他输入取消。

没有套餐也可同意安装。首次安装请求、仅回答套餐情况、空输入和 `ok` 都不构成完整确认。确认前不复制技能、不执行 pip、不运行示例。

The script explains that splitting can consume substantial tokens and allowance, complex models have a relatively high failure risk, and failed runs still consume usage. Actual limits and charges depend on the provider; a plan does not imply unlimited or free usage. It separately collects a plan answer and explicit acceptance. A plan is not required to proceed.

## AI 助手调用 / Assistant invocation

收到普通安装请求时，按以下步骤完成，不要求用户重新发送带脚本名的长指令：

1. 读取仓库 `AGENTS.md` 与本说明；通过可用的 Git 或 ZIP 获取完整仓库。已有对应仓库时复用，避免覆盖用户改动。用户未指定版本时使用所获取仓库的版本。
2. 只读检查可用 Python，遵守用户已有环境配置，选择 3.10–3.12；自动判断当前助手的技能目标目录。Codex 使用脚本默认值，其他助手通过 `--target` 指定已知目录。仅在缺少必要环境或无法确定目录时询问用户。
3. 展示上述安装消耗说明并获得两项确认；同一会话已有完整确认则复用。
4. 使用所选解释器和安装脚本的绝对路径执行，参数由助手根据实际回答组装。普通后台执行通常没有交互终端，应先在对话中问用户，再使用 `--plan` 与 `--accept-usage`，不要启动一个等待终端输入的安装进程。
5. 检查退出码。成功后，从仓库以外的工作目录运行安装目标内 `scripts/split_painted_3mf.py` 的 `--version` 和 `--help`；均成功后才报告安装目录、解释器及验证结果。失败时按错误定位原因，未完整确认前不重试安装；不得把下载成功或单独复制文件称为安装成功。
6. 安装完成后询问是否试用 `cathead.3mf`，等用户选择后再运行模型。

For a plain installation request, obtain the complete repository, read its instructions, select an available Python 3.10–3.12 and the current assistant's Skill destination, and collect consent. Use absolute paths and pass the actual consent answers for non-interactive execution. Check the exit code, then run the installed CLI's `--version` and `--help` from outside the checkout before reporting success. Ask separately before running the example. Do not require the user to restate the request with implementation details.

没有交互终端时，助手须在对话中展示上述说明，获得套餐回答和明确同意，再用 `--plan` 传入实际回答，并用 `--accept-usage` 表示本次会话已取得的确认。缺少任何一项时脚本以退出码 2 停止。同一安装会话无需重复询问；不得替用户猜测套餐或同意，也不得将确认永久保存供以后安装复用。

Without an interactive terminal, the assistant must present the notice and obtain both answers in chat first. Only then may it pass the actual answer with `--plan` and attest to the user's explicit consent with `--accept-usage`. These flags convey the current session's consent; they do not authorize an assistant to invent it. No persistent consent is stored.

## 安装行为与失败处理 / Installation and failures

- 校验完整源目录与 Python 版本；源与目标不能重叠，目标不能是链接或非技能目录。
- 完整复制技能文件（包含许可、工具和参考资料），排除缓存与日志；在临时目录安装并验证依赖与 CLI 版本入口。
- 验证通过后完整删除目标中的旧 Skill，再安装新版，避免旧脚本与新版本混用；不保留备份或恢复旧版。安装期间请关闭正在使用该技能的任务。删除失败会停止，不在残留文件上覆盖安装；删除后若新版写入失败，需要重新安装。
- pip 或验证失败时保留原技能目录。**Python 环境中的依赖改动无法自动回滚**；脚本会报告错误，需检查环境后再重试。共享环境应先确认依赖兼容；存在明确冲突时选择独立环境。
- 不运行拆件示例；安装完成后另行选择是否运行 `cathead.3mf`。

The installer validates the source and Python version, stages the complete payload, installs requirements with the selected interpreter, and verifies imports and the CLI version. After validation, it deletes the old Skill completely and installs the new version without retaining a backup or restoring the old version. Removal failure stops installation; failure to place the new version after deletion requires reinstallation. Dependency or validation failure before deletion leaves the old Skill files in place. **Changes made by pip cannot be automatically rolled back.** No sample model is processed. Close tasks using the Skill before updating it.

退出码 / Exit codes: `0` 成功 / success；`1` 安装失败 / install failure；`2` 参数或确认不完整 / missing input or consent；`130` 用户中断 / interrupted。
