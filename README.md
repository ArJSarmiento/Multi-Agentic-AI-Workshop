# The Startup’s Guide to PydanticAI + MCP

### A hands-on workshop (cloud-agnostic, no vendor lock-in)

This is the workshop handout. Each section includes a short goal, commands, and **copy-pasteable code** with file paths. If someone falls behind, you can checkpoint the repo by committing at the end of each section.

---
## Repo layout & branches you’ll create

```
00-boot
01-agent-basics
02-typed-output
03-tools-fundamentals
04-mcp-stdio
05-mcp-http
06-usage-limits-retries
07-pattern-router
08-pattern-pipeline
09-pattern-critic-editor
10-tests-and-evals
```

> Models in the examples use `gemini-2.5-flash`. Swap to any provider/model your team uses by changing the model string and setting the corresponding API key.

---

# Getting Your Gemini API Key

**Goal:** Set up a free Gemini API key to power your AI agents.

### Step 1: Create Your API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API key"** <img width="1482" height="790" alt="image" src="https://github.com/user-attachments/assets/a9de98dc-f7b3-4307-bb51-16dfe5125c44" />
4. Choose **"Create API key in new project"** <img width="1151" height="740" alt="image" src="https://github.com/user-attachments/assets/3039b1ba-1e0a-4939-8897-c9dd76eb7822" />
5. Copy the generated API key (starts with `AI...`)

⚠️ **Keep this key secure!** Don't commit it to version control or share it publicly.

### Step 2: Set Environment Variable

**Create a .env file with content:**
```bash
GEMINI_API_KEY=YOUR_API_KEY_HERE
```

### Security Best Practices

- **Never commit API keys** to Git repositories
- **Use server-side calls** for production applications  
- **Consider API key restrictions** in Google Cloud Console to limit usage
- **Rotate keys periodically** if they might be compromised

For more details, see the [official Gemini API documentation](https://ai.google.dev/gemini-api/docs/api-key).

---

# 00-boot — Project scaffolding & smoke test 

(If you feel lost go to the finished section of this at `git checkout 00-boot` and run `uv sync --all-groups --all-extras`)

**Goal:** Create a clean Python 3.12 project, install PydanticAI (with MCP), and run a minimal agent.

### Commands

```bash
# Create and activate a virtualenv with uv
uv sync
source .venv/bin/activate

# Add deps
uv add "pydantic-ai-slim[mcp]" "httpx>=0.28.1" "pydantic>=2.11.7" tenacity nest-asyncio fastmcp
uv add "google-generativeai>=0.8.5" "pydantic-ai-slim[google]"
uv add --dev pytest "python-dotenv==1.1.1" ruff
```

### `pyproject.toml` (ensure these exist)

```toml
[project]
name = "multi-agentic-ai-workshop"
version = "0.1.0"
requires-python = ">=3.12"
```

### `src/boot_smoke.py`

```python
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()

def main() -> None:
    agent = Agent("gemini-2.5-flash", instructions="Be concise.")
    res = agent.run_sync("Say 'hello workshop' exactly.")
    print(res.output)

if __name__ == "__main__":
    main()
```

### Run

```bash
uv run src/boot_smoke.py
```

---

# 01-agent-basics — Minimal agent (sync/async) + streaming

(If you feel lost go to the finished section of this at `git checkout 01-agent-basics` and run `uv sync --all-groups --all-extras`)

**Goal:** Use `run_sync`, `run`, and stream final text with `run_stream`.

### `src/agent_basics.py`

```python
import asyncio
import nest_asyncio
from pydantic_ai import Agent
from dotenv import load_dotenv
import sys

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

load_dotenv()

agent = Agent("gemini-2.5-flash", instructions="Answer briefly.")


def run_sync_demo() -> None:
    res = agent.run_sync("Name three prime numbers.")
    print("SYNC:", res.output)


async def run_async_demo() -> None:
    res = await agent.run("One sentence on Fibonacci numbers.")
    print("ASYNC:", res.output)


async def run_stream_demo() -> None:
    async with agent.run_stream(
        "Stream a short paragraph on solar eclipses."
    ) as stream:
        async for text_chunk in stream.stream_text():
            print(text_chunk, end="", flush=True)

        print("\n---")
        final = await stream.get_output()
        print("FINAL:", final)


async def run_all_async_demos() -> None:
    """Run all async demos in a single event loop"""
    await run_async_demo()
    await run_stream_demo()


if __name__ == "__main__":
    # Set event loop policy for better compatibility
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    run_sync_demo()

    # Run all async demos in a single event loop to avoid conflicts
    asyncio.run(run_all_async_demos())

```

Run:

```bash
uv run src/agent_basics.py
```


---

# 02-typed-output — Pydantic models & unions

(If you feel lost go to the finished section of this at `git checkout 02-typed-output` and run `uv sync --all-groups --all-extras`)

**Goal:** Enforce structure using `output_type` with Pydantic models. Use a union for graceful fallback.

### `src/models/answer_schema.py`

```python
from typing import Literal
from pydantic import BaseModel


class Answer(BaseModel):
    kind: Literal["fact"]
    text: str


class Fallback(BaseModel):
    kind: Literal["fallback"]
    message: str


Typed = Answer | Fallback
```

### `src/typed_output.py`

```python
from dotenv import load_dotenv
from pydantic_ai import Agent
from models.answer_schema import Typed, Answer, Fallback

load_dotenv()

agent = Agent[None, Typed](
    "gemini-2.5-flash",
    output_type=Answer | Fallback,  # type: ignore[valid-type]
    instructions="Return a factual Answer model; if unsure, return Fallback.",
)


def main() -> None:
    result1 = agent.run_sync("What is the capital of France?").output
    print(result1.model_dump_json(indent=2))
    
    result2 = agent.run_sync("Gibberish 123??").output
    print(result2.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
```

Run:

```bash
uv run src/typed_output.py
```


---

# 03-tools-fundamentals — `@agent.tool` and `RunContext`

(If you feel lost go to the finished section of this at `git checkout 03-tools-fundamentals` and run `uv sync --all-groups --all-extras`)

**Goal:** Add function tools the model can call. Access conversation context via `RunContext`.

### `src/tools_fundamentals.py`

```python
from dotenv import load_dotenv
from datetime import datetime, UTC
from pydantic_ai import Agent, RunContext

load_dotenv()

agent = Agent("gemini-2.5-flash", instructions="""
You can call `now()` for the current ISO timestamp.
Call it before answering time-sensitive questions.
""")

@agent.tool
def now(ctx: RunContext) -> str:
    """
    Returns the current ISO timestamp.
    """
    return datetime.now(UTC).isoformat()

@agent.tool
def echo_with_ctx(ctx: RunContext, msg: str) -> str:
    """
    Returns the echo of the message.
    """
    message_count = len(ctx.messages) if hasattr(ctx, 'messages') else 0
    return f"echo={msg} (message_count={message_count})"

def main() -> None:
    print(agent.run_sync("What's the time? Use tools.").output)
    print(agent.run_sync("Call echo_with_ctx with 'hi'.").output)

if __name__ == "__main__":
    main()
```

Run:

```bash
uv run src/tools_fundamentals.py
```


---

# 04-mcp-stdio — Use a local MCP server as a toolset (subprocess)

(If you feel lost go to the finished section of this at `git checkout 04-mcp-stdio` and run `uv sync --all-groups --all-extras`)

**Goal:** Create a pure-Python MCP server with useful tools and connect via stdio transport.

Install FastMCP:

```bash
uv add fastmcp
```

### `src/servers/calc_server.py`

```python
from fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("Calculator")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.""" 
    return a * b

@mcp.tool()
def days_between(start_date: str, end_date: str) -> int:
    """Calculate days between two dates (YYYY-MM-DD format)."""
    from datetime import datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return (end - start).days

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### `src/mcp_stdio_client.py`

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

calc_server = MCPServerStdio("uv", args=["run", "src/servers/calc_server.py"], timeout=30)

agent = Agent(
    "gemini-2.5-flash",
    toolsets=[calc_server],
    instructions="Use tools when math calculations or date operations help."
)

async def main() -> None:
    async with agent:
        res = await agent.run("How many days between 2000-01-01 and 2025-03-18?")
        print(res.output)
        
        res2 = await agent.run("What is 17 times 23?")
        print(res2.output)

if __name__ == "__main__":
    asyncio.run(main())
```

Run:

```bash
uv run src/mcp_stdio_client.py
```


---

# 05-mcp-http — Connect to an MCP server over Streamable HTTP

(If you feel lost go to the finished section of this at `git checkout 05-mcp-http` and run `uv sync --all-groups --all-extras`)

**Goal:** Run the same server over HTTP instead of stdio for network access.

### `src/servers/calc_http_server.py`

```python
from fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("Calculator-HTTP")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.""" 
    return a * b

@mcp.tool()
def days_between(start_date: str, end_date: str) -> int:
    """Calculate days between two dates (YYYY-MM-DD format)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return (end - start).days

@mcp.tool()
def factorial(n: int) -> int:
    """Calculate factorial of a number."""
    if n < 0:
        return 0
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

if __name__ == "__main__":
    mcp.run(transport="streamable-http")  # http://localhost:8000/mcp
```

### `src/mcp_http_client.py`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from dotenv import load_dotenv
load_dotenv()

server = MCPServerStreamableHTTP("http://localhost:8000/mcp")
agent = Agent("gemini-2.5-flash", toolsets=[server])

async def main() -> None:
    async with agent:
        res = await agent.run("What is 7 plus 5? Use the tool.")
        print(res.output)
        
        res2 = await agent.run("Calculate the factorial of 6.")
        print(res2.output)

if __name__ == "__main__":
    asyncio.run(main())
```

Run:

```bash
# terminal 1
uv run src/servers/calc_http_server.py
# terminal 2
uv run src/mcp_http_client.py
```


---

# 06-usage-limits-retries — Guardrails and resilient calls

(If you feel lost go to the finished section of this at `git checkout 06-usage-limits-retries` and run `uv sync --all-groups --all-extras`)

**Goal:** Bound work with `UsageLimits`, and show retry/backoff using Tenacity in a tool (no external network required).

### `src/limits_retries.py`

```python
import random
from time import sleep
from tenacity import retry, wait_exponential, stop_after_attempt
from pydantic_ai import Agent, UsageLimits, RunContext

agent = Agent("gemini-2.5-flash", instructions="Be brief; avoid verbosity.")

# Simulate a flaky HTTP call via a tool; Tenacity handles retries/backoff
@agent.tool
@retry(wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
       stop=stop_after_attempt(5))
def flaky_fetch(ctx: RunContext, q: str) -> str:
    if random.random() < 0.6:
        # Simulate transient failure
        sleep(0.05)
        raise RuntimeError("Transient network error")

    return f"data-for:{q}"

def main() -> None:
    limits = UsageLimits(request_limit=4, tool_calls_limit=3, total_tokens_limit=1500)
    res = agent.run_sync("Call flaky_fetch with 'hello' then summarize result.", usage_limits=limits)
    print(res.output)

if __name__ == "__main__":
    main()
```

Run:

```bash
uv run src/limits_retries.py
```


---

# 07-pattern-router — Router/Delegator with typed outcomes

(If you feel lost go to the finished section of this at `git checkout 07-pattern-router` and run `uv sync --all-groups --all-extras`)

**Goal:** Route to specialist agents using output functions, with a typed failure fallback.

### `src/pattern_router.py`

```python
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

# Specialist agents
math_agent = Agent("gemini-2.5-flash", instructions="Compute or reason step-by-step; output the final number.")
qa_agent   = Agent("gemini-2.5-flash", instructions="Answer factual questions concisely.")

# Router output choices (functions are selectable outputs)
async def hand_off_to_math(ctx: RunContext, query: str) -> str:
    messages = ctx.messages[:-1]  # drop the output-function call msg
    res = await math_agent.run(query, message_history=messages)
    return res.output

async def hand_off_to_qa(ctx: RunContext, query: str) -> str:
    messages = ctx.messages[:-1]
    res = await qa_agent.run(query, message_history=messages)
    return res.output

class RouterFailure(BaseModel):
    explanation: str

RouterOut = str | RouterFailure

router = Agent[None, RouterOut](
    "gemini-2.5-flash",
    output_type=[hand_off_to_math, hand_off_to_qa, RouterFailure],
    instructions=(
        "If the query is numeric/math/code-like, use hand_off_to_math. "
        "If general knowledge, use hand_off_to_qa. "
        "Otherwise, return RouterFailure with an explanation."
    ),
)

def demo() -> None:
    print("MATH:", router.run_sync("What is 17 * 23?").output)
    print("QA:", router.run_sync("Who wrote The Hobbit?").output)
    print("FALLBACK:", router.run_sync("Tell me something only you know.").output)

if __name__ == "__main__":
    demo()
```

Run:

```bash
uv run src/pattern_router.py
```


---

# 08-pattern-pipeline — Deterministic stages & idempotent steps

(If you feel lost go to the finished section of this at `git checkout 08-pattern-pipeline` and run `uv sync --all-groups --all-extras`)

**Goal:** Chain multiple agents programmatically for a predictable, testable flow.

### `src/pattern_pipeline.py`

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()


class Requirements(BaseModel):
    topic: str
    audience: str
    length_words: int = Field(ge=50, le=800)


extractor = Agent[None, Requirements](
    "gemini-2.5-flash",
    output_type=Requirements,
    instructions="Extract topic, audience, and a reasonable length (50-800).",
)


class Outline(BaseModel):
    headings: list[str]


outliner = Agent[None, Outline](
    "gemini-2.5-flash",
    output_type=Outline,
    instructions="Produce 3-6 descriptive headings.",
)

drafter = Agent(
    "gemini-2.5-flash", instructions="Write a crisp draft under the provided headings."
)


def pipeline_run(user_brief: str) -> str:
    req = extractor.run_sync(user_brief).output
    outline = outliner.run_sync(
        f"Topic: {req.topic}\nAudience: {req.audience}\nLength: {req.length_words}"
    ).output
    draft = drafter.run_sync(
        "Write with these headings:\n"
        + "\n".join(f"- {h}" for h in outline.headings)
        + f"\nTarget length ~{req.length_words} words."
    ).output
    return draft


if __name__ == "__main__":
    print(
        pipeline_run(
            "I need a short blog about zero-copy networking for backend engineers."
        )
    )

```

Run:

```bash
uv run src/pattern_pipeline.py
```


---

# 09-pattern-critic-editor — Two-role refinement loop

(If you feel lost go to the finished section of this at `git checkout 09-pattern-critic-editor` and run `uv sync --all-groups --all-extras`)

**Goal:** Improve draft quality with a bounded Editor↔Critic loop.

### `src/pattern_critic_editor.py`

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from dotenv import load_dotenv

load_dotenv()


class Review(BaseModel):
    score: int = Field(ge=1, le=10)
    suggestions: list[str]


editor = Agent("gemini-2.5-flash", instructions="Draft clearly. Avoid fluff.")
critic = Agent[None, Review](
    "gemini-2.5-flash",
    output_type=Review,
    instructions="Score 1-10; include concrete revision suggestions.",
)


def improve(prompt: str, rounds: int = 2, target: int = 8) -> str:
    """
    Improve the draft by the critic's suggestions.
    """
    draft = editor.run_sync(prompt).output
    for _ in range(rounds):
        review = critic.run_sync(
            f"Rate this draft and suggest edits:\n\n{draft}"
        ).output
        if review.score >= target:
            break
        revision_instructions = "Apply these improvements:\n- " + "\n- ".join(
            review.suggestions
        )
        draft = editor.run_sync(
            f"{prompt}\n\n{revision_instructions}\n\nRevise:"
        ).output

    return draft


if __name__ == "__main__":
    print(improve("Write a 120-word product blurb for a privacy-first notes app."))

```

Run:

```bash
uv run src/pattern_critic_editor.py
```


---

# 10-tests-and-evals — Fast, no-network CI

(If you feel lost go to the finished section of this at `git checkout 10-tests-and-evals` and run `uv sync --all-groups --all-extras`)

**Goal:** Prevent accidental live calls and test patterns by overriding models.

```bash
mkdir -p tests
```

### `tests/conftest.py`

```python
import os
# Block accidental live model calls in tests (local only)
os.environ.setdefault("ALLOW_MODEL_REQUESTS", "false")
```

### `tests/test_pipeline.py`

```python
from pydantic_ai.models.test import TestModel
from src.pattern_pipeline import extractor, outliner, drafter, pipeline_run

def test_pipeline_without_network():
    # Override models with TestModel to avoid network calls
    with extractor.override(model=TestModel()), \
         outliner.override(model=TestModel()), \
         drafter.override(model=TestModel()):
        draft = pipeline_run("Write about unit testing for data pipelines.")
        assert isinstance(draft, str)
        assert len(draft) > 0
```

### `tests/test_router.py`

```python
from pydantic_ai.models.test import TestModel
from src.pattern_router import router

def test_router_runs_locally():
    with router.override(model=TestModel()):
        r = router.run_sync("What is 2+2?")
        assert r.output is not None
```

Run:

```bash
uv run -m pytest -q
```


---

## Troubleshooting & tips

* **Model strings:** Replace `"gemini-2.5-flash"` with your provider/model (e.g., `"openai:gpt-4o-mini"`, `"anthropic:claude-3-5-sonnet-latest"`) and set the correct API key environment variable.
* **Streaming:** `run_stream` yields final text chunks. If you need full event-by-event control, use the async `.run()` API and inspect messages/events.
* **Unions:** When using unions or output functions, parameterize `Agent[DepsT, OutputT]` and use `# type: ignore[valid-type]` if your type checker complains on `output_type=`.
* **MCP:** FastMCP provides pure-Python MCP servers. Use stdio for local subprocess servers; use Streamable HTTP for network servers. Add `tool_prefix` if multiple MCP servers expose identically named tools.
* **Guardrails:** `UsageLimits` prevents runaway loops and caps tokens/tool calls. For resiliency, add Tenacity retries to your own tools or HTTP calls.
* **Repro:** Commit your `uv.lock` to pin dependency versions for the workshop.

You now have a compact, production-shaped toolkit: typed agents, practical MCP integrations, and three multi-agent patterns (Router, Pipeline, Critic–Editor) that scale from MVP to real-world workloads—without rewriting your stack.
