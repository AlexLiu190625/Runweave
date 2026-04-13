# Runweave

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![smolagents](https://img.shields.io/badge/built_on-smolagents_1.24-orange.svg)](https://github.com/huggingface/smolagents)

一个构建在 [smolagents](https://github.com/huggingface/smolagents) 之上的轻量运行时，为 agent 添加持久化工作区和线程，使其能够跨会话恢复长期任务。

A thin runtime built on [smolagents](https://github.com/huggingface/smolagents), adding persistent workspaces and threads for long-lived agent tasks.

---

## 为什么需要 Runweave / Why Runweave

smolagents 提供了完善的 agent 循环：代码执行、工具调度、LLM 调用——但它是无状态的。每次 `agent.run()` 结束，所有运行时上下文随之消失。如果你需要一个 agent 在同一个目录里连续工作数十次，记住之前做过什么，并且不会因为上下文窗口爆满而崩溃，smolagents 本身不提供这些能力。

smolagents provides a solid agent loop — code execution, tool dispatch, LLM calls — but it's stateless. When `agent.run()` finishes, everything is gone. If you need an agent to work in the same directory across dozens of sessions, remember what it did before, and not crash from context overflow, smolagents doesn't cover that.

Runweave 补上了这个缺口。它不重新实现 smolagents 的任何功能——agent 循环、代码解析、工具调度全部由 smolagents 处理。Runweave 只负责 smolagents 不做的事：线程、持久化、上下文压缩和摘要。

Runweave fills that gap. It doesn't reimplement anything smolagents already does — agent loop, code parsing, tool dispatch all stay in smolagents. Runweave only handles what smolagents doesn't: threads, persistence, context compression, and summaries.

## 安装 / Install

```bash
# 从源码安装（尚未发布到 PyPI）
# From source (package not yet on PyPI)
git clone https://github.com/AlexLiu190625/Runweave.git
cd Runweave
pip install -e .
```

需要 Python 3.12+。

Requires Python 3.12+.

## 配置 / Configuration

### 1. 模型配置 / Model Configuration

Runweave 通过 smolagents 调用 LLM。项目根目录有一个 `.env.example` 文件，复制并填入你的配置：

Runweave calls LLMs through smolagents. Copy the `.env.example` file in the project root and fill in your credentials:

```bash
cp .env.example .env
```

`.env` 文件内容 / `.env` file contents:

```bash
# OpenAI（大部分示例使用）/ OpenAI (used by most examples)
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# 如果使用第三方代理，改为代理地址
# If using a third-party proxy, change to your proxy URL
# OPENAI_BASE_URL=https://api.your-proxy.com/v1

# Anthropic（参见 examples/08）/ Anthropic (see examples/08)
# ANTHROPIC_API_KEY=your_anthropic_key_here
```

所有示例都会通过 `python-dotenv` 自动加载 `.env`，无需手动 export。也可以在代码中直接传参：

All examples auto-load `.env` via `python-dotenv`, no manual export needed. You can also pass credentials directly in code:

```python
from smolagents import OpenAIServerModel

model = OpenAIServerModel(
    model_id="gpt-5.3",
    api_key="sk-...",
    api_base="https://api.your-proxy.com/v1",
)
```

### 2. 上下文窗口配置（可选）/ Context Window Configuration (optional)

一般不需要配置。`Runtime` 会从模型名自动查表确定 context window 大小，并按默认比例分配 token 预算。只有在需要微调时才手动传入 `ContextBudget`：

Usually no configuration needed. `Runtime` automatically looks up the context window size from the model name and allocates token budgets with sensible defaults. Only pass a custom `ContextBudget` if you need to tune:

```python
from smolagents import OpenAIServerModel
from runweave import Runtime
from runweave.context import ContextBudget

model = OpenAIServerModel(model_id="gpt-5.3")

budget = ContextBudget(
    model_id=model.model_id,  # 保持和 model 一致 / keep in sync with model
    buffer_tokens=8192,       # 预留给输出的安全余量，默认 4096 / output margin, default 4096
    instruction_ratio=0.40,   # 指令占可用 token 的比例，默认 0.25 / instruction share, default 0.25
)

rt = Runtime(model=model, context_budget=budget)
```

三个参数的含义 / What each parameter does:

- **`buffer_tokens`** — 从 context window 中扣除的安全余量，预留给模型输出和系统开销。默认 4096。/ Tokens reserved from the context window for model output and overhead. Default 4096.
- **`instruction_ratio`** — 剩余 token 中分给跨 run 指令（历史 + 摘要 + 技能目录）的比例，其余留给 run 内步骤历史。默认 0.25。比例越大，注入的历史越详细，但留给 agent 思考的空间越小。/ Share of remaining tokens for cross-run instructions (history + summary + skill catalog); the rest goes to intra-run step history. Default 0.25. Higher ratio = more history detail, less room for agent reasoning.

详见 [examples/05_context_budget.py](examples/05_context_budget.py)。

See [examples/05_context_budget.py](examples/05_context_budget.py) for a working example.

## 快速上手 / Quick Start

```python
from smolagents import OpenAIServerModel
from runweave import Runtime

model = OpenAIServerModel(model_id="gpt-5.3")
rt = Runtime(model=model)

# Run 1: create a script
result = rt.run("Create a Python script that generates Fibonacci numbers.")
print(result.thread_id)           # a3f7c2b1
print(result.output)              # "Created fibonacci.py..."

# Run 2 (days later): continue the work
result = rt.run(
    "Add error handling and tests.",
    thread_id="a3f7c2b1",
)
# The agent already knows what was done in Run 1,
# without receiving the full memory — just a summary.
```

### 刚才发生了什么？/ What just happened?

在 Run 1 和 Run 2 之间，Runweave 做了这些事：

Between Run 1 and Run 2, Runweave:

1. 把 Run 1 的完整记忆保存到 `~/.runweave/threads/a3f7c2b1/memory.json`
2. 通过 LLM 生成了一段约 200 字的摘要："Created fibonacci.py with a recursive implementation..."
3. 把摘要写入 `summary.txt`
4. 在 Run 2 启动时，把这段摘要注入 agent 的指令中
5. Agent 看到摘要后，在工作区里找到了 `fibonacci.py`，读取它，添加了错误处理和测试——整个过程中它没有看到 Run 1 的原始记忆

1. Saved Run 1's full memory to `~/.runweave/threads/a3f7c2b1/memory.json`
2. Generated a ~200-word summary via LLM: "Created fibonacci.py with a recursive implementation..."
3. Persisted the summary to `summary.txt`
4. In Run 2, injected that summary into the agent's instructions
5. The agent saw the summary, found `fibonacci.py` in the workspace, read it, and added tests — without ever seeing Run 1's raw memory

## 核心概念 / Core Concepts

### 线程 / Thread

线程是 Runweave 的基本工作单元。一个线程拥有自己的工作区目录、内存归档和运行历史。线程之间相互隔离。

A thread is Runweave's unit of work. Each thread owns its own workspace directory, memory archive, and run history. Threads are isolated from each other.

### 运行 / Run

一次 `runtime.run(task, thread_id)` 调用就是一次 run。同一个线程可以有很多次 run。每次 run 结束后，Runweave 会归档 agent 的记忆、记录执行步骤、生成运行摘要。

One call to `runtime.run(task, thread_id)` is a run. A thread can have many runs. After each run, Runweave archives the agent's memory, records execution steps, and generates a summary.

### 恢复 / Resume

当你在已有线程上开始新 run 时，agent 不会收到之前所有 run 的完整记忆——那会撑爆上下文。它收到的是一段 200-300 字的压缩摘要，由 LLM 在上一次 run 结束时生成。

When you start a new run on an existing thread, the agent doesn't receive full memory of all prior runs — that would blow up the context window. Instead it receives a compressed summary of everything done so far, a 200-300 word narrative generated by the LLM after the previous run.

这是一个有意的取舍：agent 不能完美回忆过去说过的每一个字，但线程可以跑上百次而不会溢出。

This is a deliberate trade-off: the agent can't perfectly recall every word from past runs, but threads can survive hundreds of runs without overflow.

## 架构 / Architecture

Runweave 有三层，依赖方向严格向下。

Runweave has three layers with strict downward dependency.

```
Layer 3: Runweave Runtime Shell
         Runtime, Thread, ThreadStore, MemoryIO,
         SummaryGenerator, HistoryWriter, SkillLoader, ToolLoader,
         ContextBudget, InstructionCompressor, StepCompressor
              |
              v
Layer 2: Runweave Executor Extension
         WorkspaceExecutor (subclasses LocalPythonExecutor, ~30 lines)
              |
              v
Layer 1: smolagents (imported, never modified)
         CodeAgent, Tool, LocalPythonExecutor, AgentMemory, ...
```

第一层是 smolagents，原封不动地导入使用。第二层只有一个类 `WorkspaceExecutor`，在执行代码前切换到线程的工作目录。第三层是 Runweave 自身的代码，全是 smolagents 没有对应功能的东西。

Layer 1 is smolagents, imported as-is. Layer 2 is a single class `WorkspaceExecutor` that chdirs into the thread's workspace before executing code. Layer 3 is Runweave's own code — everything that smolagents doesn't provide.

## `runtime.run()` 的完整流程 / Full Flow

```
task, thread_id
    |
    v
[1]  Load or create thread from ThreadStore
[2]  Build WorkspaceExecutor for thread's workspace
[3]  Collect instructions: user prompt + skill catalog + run history + thread summary
[4]  Compress instructions within token budget (InstructionCompressor)
[5]  Merge tools: user tools + custom tools (ToolLoader) + skill tools (SkillLoader)
[6]  Build smolagents CodeAgent with step compression callback
[7]  agent.run(task) — smolagents handles the loop
[8]  Extract which skills were used during this run
[9]  Save agent memory to disk (for inspection, not replay)
[10] Write run record (run-NNN.json/md), regenerate HISTORY.md
[11] Generate/update thread summary via LLM
[12] Return RunResult
```

其中第 7 步以外的所有步骤都是 Runweave 的工作。第 7 步完全由 smolagents 处理——agent 循环、代码执行、工具调度、LLM 调用、最终答案检测，Runweave 不碰这些。

Everything except step 7 is Runweave's job. Step 7 is entirely smolagents — agent loop, code execution, tool dispatch, LLM calls, final answer detection. Runweave doesn't touch any of that.

## 上下文窗口管理 / Context Window Management

Runweave 有三个组件处理上下文管理，各管一层：

Runweave has three components for context management, each handling a different layer:

- **ContextBudget** — 配置 token 预算和分配比例，其他两个组件都读它 / Configures token budget and allocation ratios; the other two components read from it.
- **InstructionCompressor** — 压缩**跨 run** 的指令（历史 + 摘要 + 技能目录）/ Compresses **cross-run** instructions (history + summary + skill catalog).
- **StepCompressor** — 压缩**单次 run 内**的步骤历史（通过 `step_callbacks`）/ Compresses **intra-run** step history (via `step_callbacks`).

长时间运行的 agent 有一个现实问题：上下文窗口会满。smolagents 没有处理这个问题——它每步都把完整记忆发给 LLM，直到 API 报错。

Long-running agents have a practical problem: the context window fills up. smolagents doesn't handle this — it sends the full memory to the LLM every step until the API errors out.

Runweave 在两个层面解决这个问题：

Runweave addresses this at two levels:

**跨 run 压缩 (InstructionCompressor)**：注入给 agent 的指令（用户提示 + 技能目录 + 运行历史 + 线程摘要）会被压缩到 token 预算内。历史记录超长时逐级压缩——先删除步骤详情，再只保留运行日志表，最后截断日志行数。用户指令和摘要永不裁剪。

**Cross-run compression (InstructionCompressor)**: Instructions injected into the agent (user prompt + skill catalog + run history + thread summary) are compressed within a token budget. When history gets too long, it's progressively compressed — first strip step details, then keep only the run log table, then truncate log rows. User instructions and summaries are never trimmed.

**单次 run 内压缩 (StepCompressor)**：通过 smolagents 的 `step_callbacks` 扩展点，每步结束后检查实际 token 使用量。超过阈值时，对旧步骤逐级压缩——截断输出、清除推理过程、最后完全清除代码和输出。最近的 3 步始终保持完整。

**Intra-run compression (StepCompressor)**: Via smolagents' `step_callbacks` extension point, actual token usage is checked after each step. When above threshold, old steps are progressively compressed — truncate output, clear reasoning, finally clear code and output entirely. The most recent 3 steps always remain intact.

## 技能系统 / Skill System

技能是可复用的指令文档，带有可选的脚本和参考文件。agent 在运行时按需加载技能。

Skills are reusable instruction documents with optional scripts and reference files. The agent loads skills on demand during a run.

```
skills/
  deploy/
    SKILL.md            # frontmatter + instructions
    scripts/
      check_status.sh
    references/
      runbook.md
```

`SKILL.md` 格式 / `SKILL.md` format:

```markdown
---
name: "deploy"
description: "Production deployment procedures"
---

# Deploy

Step-by-step instructions for deployment...
```

Runweave 自动为 agent 注册三个工具：`load_skill`（加载指令）、`read_skill_resource`（读取参考文件）、`run_skill_script`（执行脚本）。agent 看到技能目录后自行决定何时加载哪个技能。

Runweave automatically registers three tools for the agent: `load_skill` (load instructions), `read_skill_resource` (read reference files), `run_skill_script` (execute scripts). The agent sees the skill catalog and decides when to load which skill.

## 自定义工具 / Custom Tools

在指定目录下放置 `.py` 文件，Runweave 的 `ToolLoader` 会自动发现并加载 smolagents `Tool` 实例。

Place `.py` files in a specified directory and Runweave's `ToolLoader` will automatically discover and load smolagents `Tool` instances.

```python
# tools/search.py
from smolagents import Tool

class SearchTool(Tool):
    name = "search"
    description = "Search the web"
    inputs = {"query": {"type": "string", "description": "Search query"}}
    output_type = "string"

    def forward(self, query: str) -> str:
        ...

search = SearchTool()  # must be instantiated at module level
```

```python
rt = Runtime(
    model=model,
    tools_dir=Path("./tools"),
)
```

## 磁盘布局 / Disk Layout

```
~/.runweave/threads/<thread-id>/
    workspace/          # agent 的工作目录 / agent's working directory
    memory.json         # 归档记忆（仅供检查）/ archived memory (inspection only)
    summary.txt         # 线程摘要 / thread summary
    HISTORY.md          # 运行历史索引 / run history index
    runs/
        run-001.json    # 运行记录 / run record
        run-001.md      # 可读报告 / readable report
        ...
    meta.json           # {id, created_at}
```

`memory.json` 保存了完整的 agent 记忆，但仅供检查——不会在下一次 run 时注入回 LLM 上下文。

`memory.json` stores the full agent memory, but only for inspection — it is not injected back into the LLM context on the next run.

## API

### `Runtime`

```python
Runtime(
    model: Model,                                     # smolagents model instance
    tools: list[Tool] | None = None,                  # tools passed directly
    instructions: str | None = None,                  # system instructions
    base_dir: Path | None = None,                     # data dir, default ~/.runweave
    additional_authorized_imports: list[str] = None,   # extra imports for executor
    skills_dir: Path | None = None,                   # skills directory
    tools_dir: Path | None = None,                    # tools directory
    context_budget: ContextBudget | None = None,      # token budget config
)
```

### `Runtime.run()`

```python
result = rt.run(
    task: str,                            # task description
    thread_id: str | None = None,         # thread ID, None to auto-create
    tool_names: list[str] | None = None,  # select specific tools from tools_dir
)
```

### `RunResult`

```python
@dataclass
class RunResult:
    output: Any              # agent's final output
    thread_id: str           # thread ID
    state: str               # "success" | "error" | ...
    step_count: int          # number of steps executed
    token_usage: dict | None # token usage stats
    timing: dict | None      # timing stats
    summary: str             # thread summary after this run
    skills_used: list[str]   # skills loaded during this run
```

## 依赖 / Dependencies

运行时依赖只有一个：`smolagents[openai]==1.24.0`。开发依赖：`pytest`、`python-dotenv`。

The only runtime dependency is `smolagents[openai]==1.24.0`. Dev dependencies: `pytest`, `python-dotenv`.

## 许可 / License

Apache License 2.0. See [LICENSE](LICENSE).
