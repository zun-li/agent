# Agent

## 项目简介

`Agent` 是在终端里使用的编码助手。你可以问它“这个项目从哪里启动”，也可以让它修改文件、运行测试，再根据失败输出继续排查。对话保存在本地工作区，下次打开时可以接着做。

项目使用 Python 标准库实现，默认通过 Anthropic Messages 接口调用 DeepSeek。分层设计参考了 Hucci 写代码的《Agent 就是个循环，然后呢？10 分钟学会设计 Agent 架构》，工具使用和多 Agent 协作参考了 [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)。

## 项目结构

```text
.
├── main.py           # 命令行入口，读取配置并组装模型、工具和存储
├── domain.py         # 消息、工具及模型和存储接口
├── app/
│   ├── service.py    # 对话循环、工具审核和会话恢复
│   ├── context.py    # 上下文压缩、项目技能和记忆
│   ├── work.py       # 任务依赖、后台命令和定时调度
│   ├── teams.py      # 队友协作与 Git worktree
│   └── workflows.py  # 支持中断恢复的代码审查工作流
├── adapters/
│   ├── cli.py        # 终端对话和审批提示
│   ├── models.py     # 模型服务接入
│   ├── tools.py      # 文件读写和 Shell 命令
│   └── store.py      # 本地会话记录
├── tests/            # 离线测试
└── pyproject.toml    # Python 版本要求和 agent 命令入口
```

## 功能描述

- **读懂和修改代码。** 可以问“这个项目在哪里读取 API 密钥”，也可以接着说“给密钥缺失的情况补一个测试”。助手能查找文件、读取源码并直接编辑工作区。内置文件编辑工具在模型审核通过后即可写入。
- **运行命令，验证结果。** 让助手运行一条测试、复现一个错误，或者在后台执行命令。终端会先显示待执行的 Shell 命令。输入 `y` 批准，直接回车拒绝。用 `/goal 修复失败用例，直到测试全部通过` 可以设定完成条件，助手会根据测试输出等记录检查进度。
- **接着上次的工作。** 重新打开会话就能沿用已有的对话和工具结果。对话变长时，助手会将较早的内容压缩成摘要。新会话也能检索之前保存的项目约定。团队若有固定的测试或发布步骤，可以写在工作区的 `skills/<名称>/SKILL.md` 里，供助手按需读取。
- **把大任务分给队友。** 可以让助手拆出有依赖关系的任务，安排一个队友排查失败原因，另一个检查相关用例。在 Git 仓库中，也可以让队友在单独的 worktree 里修改代码。
- **审查一份改动。** 把一段 diff 交给助手，请它使用内置的 `review-changes` 工作流。它会分别检查正确性、安全性和可能的回归，复核发现的问题，再汇总结果。

## 快速开始

准备 Python 3.11 或更高版本、uv 和一个 DeepSeek API 密钥。[uv 安装指南](https://docs.astral.sh/uv/getting-started/installation/)

**1. 配置密钥。** 进入本仓库根目录，首次使用时从模板创建 `.env`。

```sh
test -f .env || cp .env.example .env
```

打开 `.env`，在 `DEEPSEEK_API_KEY=` 后填入自己的密钥。

**2. 启动助手。**

```sh
uv run agent
```

**3. 试一次完整的任务。** 看到 `你 ›` 后，输入下面这句话。

```text
读一下这个项目的 README，找出测试命令，然后运行测试，告诉我有没有失败的用例。
```

本项目的测试命令是 `uv run python -m unittest discover -v`。看到审批提示后可以先核对命令，再决定是否执行。输入 `/help` 查看终端命令，`/exit` 退出。

## 用户配置

### 选择模型

只填写 `DEEPSEEK_API_KEY` 就会使用默认模型 `deepseek-flash`。要切换服务，在 `.env` 中设置 `AGENT_PROVIDER`，并填写对应的密钥。需要指定模型时，设置 `AGENT_MODEL`。

| `AGENT_PROVIDER` | 对应的密钥变量 |
| --- | --- |
| `deepseek`（默认） | `DEEPSEEK_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `google` | `GEMINI_API_KEY` |

审核工具调用时默认使用主模型，也可以通过 `REVIEWER_PROVIDER` 和 `REVIEWER_MODEL` 单独指定审核模型。密钥优先取启动目录 `.env` 中的非空值，留空时读取同名环境变量。其余模型设置优先取环境变量。修改配置后重新启动即可生效。

### 指定工作区与恢复会话

默认工作区是当前目录。要操作另一个本地项目，从本仓库启动时传入目标路径即可。`.env` 仍从启动目录读取，会话和任务进度保存在目标工作区的 `.agent/` 下。

```sh
uv run agent --workspace ../my-project
```

在对话里输入 `/sessions` 可查看会话 ID，`/use ID` 可切换会话。下次启动时，用实际 ID 替换下面的 `SESSION_ID`，即可打开原对话。如果上一轮任务中断，进入对话后输入 `/resume` 继续。

```sh
uv run agent --workspace ../my-project --resume SESSION_ID
```
