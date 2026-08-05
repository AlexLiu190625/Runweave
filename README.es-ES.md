

# Runweave

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![smolagents](https://img.shields.io/badge/built_on-smolagents_1.24-orange.svg)](https://github.com/huggingface/smolagents)

Un runtime ligero construido sobre [smolagents](https://github.com/huggingface/smolagents) que añade espacios de trabajo persistentes e hilos para tareas de agente a largo plazo, permitiéndoles recuperarse entre sesiones.

---

## ¿Por qué Runweave?

`smolagents` proporciona un ciclo de agente sólido: ejecución de código, despacho de herramientas y llamadas a LLM, pero es sin estado. Cuando `agent.run()` termina, todo desaparece. Si necesitas que un agente trabaje en el mismo directorio a lo largo de decenas de sesiones, recuerde lo que hizo anteriormente y no se bloque por desbordamiento de contexto, `smolagents` por sí solo no cubre esto.

Runweave llena ese vacío. No reimplementa nada que `smolagents` ya haga: el ciclo del agente, el análisis de código y el despacho de herramientas permanecen en `smolagents`. Runweave solo se encarga de lo que `smolagents` no hace: hilos, persistencia, compresión de contexto y resúmenes.

## Instalación

```bash
# 从源码安装（尚未发布到 PyPI）
# From source (package not yet on PyPI)
git clone https://github.com/AlexLiu190625/Runweave.git
cd Runweave
pip install -e .
```

Requiere Python 3.12+.

## Configuración

### 1. Configuración del modelo

Runweave invoca a los LLM a través de `smolagents`. En la raíz del proyecto hay un archivo `.env.example`. Cópialo y rellena tus credenciales:

```bash
cp .env.example .env
```

Contenido del archivo `.env`:

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

Todos los ejemplos cargan automáticamente `.env` mediante `python-dotenv`, sin necesidad de exportar manualmente. También puedes pasar las credenciales directamente en el código:

```python
from smolagents import OpenAIServerModel

model = OpenAIServerModel(
    model_id="gpt-5.3",
    api_key="sk-...",
    api_base="https://api.your-proxy.com/v1",
)
```

### 2. Configuración de la ventana de contexto (opcional)

Por lo general, no se requiere configuración. `Runtime` consulta automáticamente el tamaño de la ventana de contexto según el nombre del modelo y asigna presupuestos de tokens con valores predeterminados sensatos. Solo pasa un `ContextBudget` personalizado si necesitas ajustarlo:

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

Significado de los tres parámetros:

- **`buffer_tokens`** — Margen de seguridad restado de la ventana de contexto, reservado para la salida del modelo y sobrecarga del sistema. Por defecto 4096.
- **`instruction_ratio`** — Proporción de los tokens restantes asignados a instrucciones entre ejecuciones (historial + resumen + catálogo de habilidades); el resto se destina al historial de pasos dentro de una ejecución. Por defecto 0.25. Una proporción mayor significa más detalle histórico, pero menos espacio para el razonamiento del agente.

Consulta [examples/05_context_budget.py](examples/05_context_budget.py) para un ejemplo funcional.

## Inicio rápido

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

### ¿Qué acaba de ocurrir?

Entre la Ejecución 1 y la Ejecución 2, Runweave realizó lo siguiente:

1. Guardó la memoria completa de la Ejecución 1 en `~/.runweave/threads/a3f7c2b1/memory.json`
2. Generó un resumen de ~200 palabras mediante LLM: "Created fibonacci.py with a recursive implementation..."
3. Persistió el resumen en `summary.txt`
4. Al iniciar la Ejecución 2, inyectó ese resumen en las instrucciones del agente
5. El agente vio el resumen, encontró `fibonacci.py` en el espacio de trabajo, lo leyó y añadió pruebas, sin ver nunca la memoria cruda de la Ejecución 1

## Conceptos clave

### Hilo (Thread)

Un hilo es la unidad básica de trabajo de Runweave. Cada hilo posee su propio directorio de espacio de trabajo, archivo de memoria e historial de ejecuciones. Los hilos están aislados entre sí.

### Ejecución (Run)

Una llamada a `runtime.run(task, thread_id)` es una ejecución. Un hilo puede tener muchas ejecuciones. Después de cada ejecución, Runweave archiva la memoria del agente, registra los pasos de ejecución y genera un resumen.

### Reanudación (Resume)

Cuando inicias una nueva ejecución en un hilo existente, el agente no recibe la memoria completa de todas las ejecuciones anteriores, lo que desbordaría la ventana de contexto. En su lugar, recibe un resumen comprimido de 200-300 palabras generado por el LLM al finalizar la ejecución anterior.

Este es un compromiso intencional: el agente no puede recordar perfectamente cada palabra de ejecuciones pasadas, pero los hilos pueden sobrevivir a cientos de ejecuciones sin desbordarse.

## Arquitectura

Runweave tiene tres capas con una dependencia estricta hacia abajo.

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

La Capa 1 es `smolagents`, importada tal cual. La Capa 2 es una sola clase `WorkspaceExecutor` que cambia al directorio de trabajo del hilo antes de ejecutar código. La Capa 3 es el código propio de Runweave: todo lo que `smolagents` no proporciona.

## Flujo completo de `runtime.run()`

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

Todo excepto el paso 7 es trabajo de Runweave. El paso 7 es enteramente `smolagents`: ciclo del agente, ejecución de código, despacho de herramientas, llamadas a LLM y detección de respuesta final. Runweave no toca nada de eso.

## Gestión de la ventana de contexto

Runweave tiene tres componentes para la gestión del contexto, cada uno manejando una capa diferente:

- **ContextBudget** — Configura el presupuesto de tokens y las proporciones de asignación; los otros dos componentes leen de él.
- **InstructionCompressor** — Comprime instrucciones **entre ejecuciones** (historial + resumen + catálogo de habilidades).
- **StepCompressor** — Comprime el historial de pasos **dentro de una ejecución** (mediante `step_callbacks`).

Los agentes de larga duración tienen un problema práctico: la ventana de contexto se llena. `smolagents` no maneja esto: envía la memoria completa al LLM en cada paso hasta que la API devuelve un error.

Runweave aborda esto en dos niveles:

**Compresión entre ejecuciones (InstructionCompressor)**: Las instrucciones inyectadas al agente (prompt del usuario + catálogo de habilidades + `key_facts` + resumen del hilo + historial de ejecuciones) se comprimen dentro del presupuesto de tokens. Cuando el historial es demasiado largo, se comprime mediante un **decaimiento en forma de U**: las ejecuciones del inicio y del final siempre se renderizan en modo COMPLETO; las ejecuciones intermedias se agrupan progresivamente según su distancia del final (TAKEAWAY → TITLE → LOG_LINE). La presión del presupuesto solo comprime la parte intermedia; el inicio y el final son fijos. Las instrucciones del usuario, `key_facts` y el resumen nunca se recortan.

```
N=10 history decay (head_count=2, tail_count=3):

  Run:    1     2     3      4      5      6        7        8     9     10
  Level:  FULL  FULL  LOG    TITLE  TITLE  TAKEAWAY TAKEAWAY FULL  FULL  FULL
          └──head──┘  └────────── middle ──────────────────┘ └─────tail──────┘
```

`ContextBudget(head_count=N, tail_count=M)` ajusta el tamaño de la ventana de anclaje. Para hilos largos (N > 20) que usan `key_facts` + resumen, `head_count=0` es razonable: la señal de dirección temprana ya está capturada por esas pistas.

**Compresión dentro de una ejecución (StepCompressor)**: Mediante el punto de extensión `step_callbacks` de `smolagents`, se verifica el uso real de tokens después de cada paso. Cuando supera el umbral, los pasos antiguos se comprimen progresivamente: truncar salida, limpiar razonamiento y, finalmente, limpiar código y salida por completo. Los 3 pasos más recientes siempre permanecen intactos.

## Sistema de habilidades

Las habilidades son documentos de instrucciones reutilizables con scripts y archivos de referencia opcionales. El agente carga las habilidades bajo demanda durante una ejecución.

```
skills/
  deploy/
    SKILL.md            # frontmatter + instructions
    scripts/
      check_status.sh
    references/
      runbook.md
```

Formato de `SKILL.md`:

```markdown
---
name: "deploy"
description: "Production deployment procedures"
---

# Deploy

Step-by-step instructions for deployment...
```

Runweave registra automáticamente tres herramientas para el agente: `load_skill` (cargar instrucciones), `read_skill_resource` (leer archivos de referencia) y `run_skill_script` (ejecutar scripts). El agente ve el catálogo de habilidades y decide cuándo cargar cada una.

## Herramientas personalizadas

Coloca archivos `.py` en un directorio especificado y `ToolLoader` de Runweave descubrirá y cargará automáticamente instancias de `Tool` de `smolagents`.

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

## Estructura en disco

```
~/.runweave/threads/<thread-id>/
    workspace/          # agent 的工作目录 / agent's working directory
    memory.json         # 归档记忆（仅供检查）/ archived memory (inspection only)
    summary.txt         # 线程摘要（叙事）/ thread summary (narrative)
    key_facts.md        # 关键事实（锚点）/ curated anchor facts
    HISTORY.md          # 运行历史索引 / run history index
    runs/
        run-001.json    # 运行记录 / run record
        run-001.md      # 可读报告 / readable report
        ...
    meta.json           # {id, created_at}
```

`memory.json` almacena la memoria completa del agente, pero solo para inspección; no se inyecta de vuelta en el contexto del LLM en la siguiente ejecución.

### Dos pistas de memoria paralelas

Después de cada ejecución, Runweave dispara dos llamadas independientes al LLM en paralelo, produciendo dos artefactos complementarios:

- **`summary.txt` — resumen narrativo**: describe qué ocurrió en esta ejecución y el estado actual. Crece a lo largo de las ejecuciones y luego se condensa cuando supera un umbral de palabras.
- **`key_facts.md` — hechos clave curados**: inspirado en MemReader, realiza un triaje LLM independiente. Conserva solo objetivos, restricciones estrictas, decisiones y artefactos producidos, cada uno etiquetado con `[run N]`. Un hecho nuevo que reemplaza a uno existente lo sustituye en lugar de añadirse. Este es el anclaje del hilo: resiste la dilución por actividad reciente.

En la siguiente ejecución, ambos se inyectan en las instrucciones del agente (`key_facts` primero) y tienen prioridad sobre el historial. Cuando el presupuesto de instrucciones es ajustado, el historial se recorta antes que cualquiera de los dos.

### Migración para hilos existentes

- Los hilos creados antes de la v0.2 no tienen `key_facts.md`. En la primera reanudación después de la actualización, el extractor produce un archivo inicial a partir de la tarea/salida de esa ejecución; las ejecuciones posteriores lo evolucionan normalmente.
- No se requiere migración manual ni eliminación de hilos existentes. Las ejecuciones históricas no se rellenan retrospectivamente (el extractor solo ve la ejecución actual + el archivo `key_facts.md` existente).
- Para curar o restablecer manualmente los hechos clave de un hilo, edita o elimina `~/.runweave/threads/<id>/key_facts.md`; la siguiente ejecución continuará desde el estado actual del archivo.

## API

### `Runtime`

```python
Runtime(
    model: Model,                                     # smolagents model instance
    tools: list[Tool] | None = None,                  # tools passed directly
    instructions: str | None = None,                  # additional instructions appended to smolagents' built-in system prompt
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

### `PlanningRuntime` (v0.3+)

`Runtime` ejecuta un solo modelo de extremo a extremo. Para tareas multipaso donde los pasos difieren ampliamente en dificultad, `PlanningRuntime` es la mejor opción: un **LLM planificador produce un plan una vez**, el **Router selecciona el modelo más adecuado por paso según sus metadatos**, y **cada paso se ejecuta mediante `smolagents.CodeAgent`**.

```python
from runweave import ModelProfile, PlanningRuntime, Router
from smolagents import OpenAIServerModel

haiku  = OpenAIServerModel(model_id="claude-haiku-4-5-20251001")
sonnet = OpenAIServerModel(model_id="claude-sonnet-4-6")
opus   = OpenAIServerModel(model_id="claude-opus-4-7")

models = [
    ModelProfile(model=haiku,  context_window=200_000, supports_tools=True,
                 supports_structured_output=True, coding_score=0.7,
                 long_context_score=0.65, latency="low",    cost_tier="low"),
    ModelProfile(model=sonnet, context_window=1_000_000, supports_tools=True,
                 supports_structured_output=True, coding_score=0.9,
                 long_context_score=0.85, latency="medium", cost_tier="medium"),
    ModelProfile(model=opus,   context_window=1_000_000, supports_tools=True,
                 supports_structured_output=True, coding_score=0.95,
                 long_context_score=0.9,  latency="high",   cost_tier="high"),
]

rt = PlanningRuntime(planner_model=opus, models=models, router=Router())
result = rt.run("Build a dataclass + tests + README")
```

`PlanningRuntime` es un **orquestador, no un agente**: nunca invoca a un LLM para decidir "qué sigue". El siguiente paso proviene topológicamente de `plan.json`; los replaneamientos se activan solo ante condiciones de fallo deterministas (paso fallido / tiempo agotado / salida esperada faltante), limitados a `max_replans=3` por defecto.

Cada `PlanningRuntime.run()` escribe un `plan.json` activo bajo el hilo, archivado en `plans/plan-NNN.json` al completarse. El registro guarda `selected_model_id`, estado, salida y motivo de fallo para cada paso: un registro completamente trazable.

Ejemplo completo: [`examples/11_planning_runtime.py`](examples/11_planning_runtime.py).

#### Parámetros opcionales

```python
PlanningRuntime(
    planner_model=opus,                 # 出计划 + 默认跑 summary/key_facts
    models=[...],
    summary_model=haiku,                # 可选：用便宜模型跑 summary/key_facts 省钱
    router=Router(),
    step_timeout_seconds=600,           # 单 step 超时
    max_step_iterations=30,             # step 执行总次数上限
    max_replans=3,                      # 跨-step 重规划次数上限
)
```

`summary_model` cae en retroceso a `planner_model`. Apuntarlo a un modelo más barato puede reducir significativamente el costo: `summary` + `key_facts` suelen representar el 5-15% de los tokens totales, y usar Haiku en lugar de Opus ahorra más del 70% en esa porción.

`RunResult.token_usage` agraga las cinco categorías de llamadas LLM: planificador, replanificación, CodeAgent por paso, resumen y hechos clave. Totalmente auditable.

#### Limitaciones conocidas (v0.3)

- `expected_outputs` solo verifica la **existencia** de archivos, no las **modificaciones**. Si un paso anterior ya creó el archivo y el paso actual lo declaró como salida pero no hizo nada, la verificación pasa. Solución temporal: el planificador no debe duplicar `expected_outputs` entre pasos. La v0.4 añadirá seguimiento de mtime.
- Actualmente, todos los pasos se ejecutan secuencialmente incluso cuando son independientes. La ejecución en paralelo se pospone a la v0.4.
- No existe `PlanningRuntime.run_stream()` aún. La semántica de flujo multipaso se pospone a la v0.4.

## Dependencias

La única dependencia de tiempo de ejecución es `smolagents[openai]==1.24.0`. Dependencias de desarrollo: `pytest`, `python-dotenv`.

## Licencia

Apache License 2.0. Consulta [LICENSE](LICENSE).
