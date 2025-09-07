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
08-pattern-router
11-pattern-pipeline
13-pattern-critic-editor
tests-and-evals
```

> Models in the examples use `gemini-2.5-flash`. Swap to any provider/model your team uses by changing the model string and setting the corresponding API key.

---

# Getting Your Gemini API Key

**Goal:** Set up a free Gemini API key to power your AI agents.

### Step 1: Create Your API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API key"** 
4. Choose **"Create API key in new project"** (recommended for new users)
5. Copy the generated API key (starts with `AI...`)

⚠️ **Keep this key secure!** Don't commit it to version control or share it publicly.

### Step 2: Set Environment Variable

**For this dev container (Linux):**
```bash
# Add to ~/.bashrc or ~/.zshrc
export GEMINI_API_KEY=YOUR_API_KEY_HERE
source ~/.bashrc  # or source ~/.zshrc
```

### Step 3: Verify Setup

Test your key works:
```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$GEMINI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"contents":[{"parts":[{"text":"Hello"}]}]}'
```

You should see a JSON response with generated text.

### Security Best Practices

- **Never commit API keys** to Git repositories
- **Use server-side calls** for production applications  
- **Consider API key restrictions** in Google Cloud Console to limit usage
- **Rotate keys periodically** if they might be compromised

For more details, see the [official Gemini API documentation](https://ai.google.dev/gemini-api/docs/api-key).

---

# 00-boot — Project scaffolding & smoke test

**Goal:** Create a clean Python 3.12 project, install PydanticAI (with MCP), and run a minimal agent.

### Commands

```bash
git init pydanticai-mcp-workshop && cd pydanticai-mcp-workshop
git checkout -b 00-boot

# Create and activate a virtualenv with uv
uv sync
source .venv/bin/activate

# Initialize a project and add deps
uv init --package
uv add "pydantic-ai-slim[mcp]" "httpx>=0.27" "pydantic>=2.7" tenacity
uv add "google-generativeai>=0.8.0"

# Optional: tests
uv add -d pytest
```

### `pyproject.toml` (ensure these exist)

```toml
[project]
name = "pydanticai-mcp-workshop"
version = "0.1.0"
requires-python = ">=3.12"
```

### `src/boot_smoke.py`

```python
from pydantic_ai import Agent

def main() -> None:
    agent = Agent("gemini-2.5-flash", instructions="Be concise.")
    res = agent.run_sync("Say 'hello workshop' exactly.")
    print(res.output)

if __name__ == "__main__":
    main()
```

### Run

```bash
export GEMINI_API_KEY=AI...   # or your provider's env var
uv run src/boot_smoke.py
```

Commit:

```bash
git add -A && git commit -m "00-boot"
```

---

# 01-agent-basics — Minimal agent (sync/async) + streaming

**Goal:** Use `run_sync`, `run`, and stream final text with `run_stream`.

```bash
git checkout -b 01-agent-basics
```

### `src/agent_basics.py`

```python
import asyncio
from pydantic_ai import Agent

agent = Agent("gemini-2.5-flash", instructions="Answer briefly.")

def run_sync_demo() -> None:
    res = agent.run_sync("Name three prime numbers.")
    print("SYNC:", res.output)

async def run_async_demo() -> None:
    res = await agent.run("One sentence on Fibonacci numbers.")
    print("ASYNC:", res.output)

async def run_stream_demo() -> None:
    async with agent.run_stream("Stream a short paragraph on solar eclipses.") as stream:
        async for text_chunk in stream.stream_text():
            print(text_chunk, end="", flush=True)
        print("\n---")
        final = await stream.get_output()
        print("FINAL:", final)

if __name__ == "__main__":
    run_sync_demo()
    asyncio.run(run_async_demo())
    asyncio.run(run_stream_demo())
```

Run:

```bash
uv run src/agent_basics.py
```

Commit:

```bash
git add -A && git commit -m "01-agent-basics"
```

---

# 02-typed-output — Pydantic models & unions

**Goal:** Enforce structure using `output_type` with Pydantic models. Use a union for graceful fallback.

```bash
git checkout -b 02-typed-output
```

### `src/typed_output.py`

```python
from typing import Literal
from pydantic import BaseModel
from pydantic_ai import Agent

class Answer(BaseModel):
    kind: Literal["fact"]
    text: str

class Fallback(BaseModel):
    kind: Literal["fallback"]
    message: str

Typed = Answer | Fallback

agent = Agent[None, Typed](
    "gemini-2.5-flash",
    output_type=Answer | Fallback,  # type: ignore[valid-type]
    instructions="Return a factual Answer model; if unsure, return Fallback."
)

def main() -> None:
    print(agent.run_sync("What is the capital of France?").output)
    print(agent.run_sync("Gibberish 123??").output)

if __name__ == "__main__":
    main()
```

Run:

```bash
uv run src/typed_output.py
```

Commit:

```bash
git add -A && git commit -m "02-typed-output"
```

---

# 03-tools-fundamentals — `@agent.tool` and `RunContext`

**Goal:** Add function tools the model can call. Access conversation context via `RunContext`.

```bash
git checkout -b 03-tools-fundamentals
```

### `src/tools_fundamentals.py`

```python
from datetime import datetime
from pydantic_ai import Agent, RunContext

agent = Agent("gemini-2.5-flash", instructions="""
You can call `now()` for the current ISO timestamp.
Call it before answering time-sensitive questions.
""")

@agent.tool
def now() -> str:
    return datetime.utcnow().isoformat() + "Z"

@agent.tool
def echo_with_ctx(ctx: RunContext, msg: str) -> str:
    user_text = ctx.messages[-1].content_text or ""
    return f"echo={msg} (user_prompt_len={len(user_text)})"

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

Commit:

```bash
git add -A && git commit -m "03-tools-fundamentals"
```

---

# 04-mcp-stdio — Use a local MCP server as a toolset (subprocess)

**Goal:** Launch an MCP server as a subprocess and expose its tools to your agent.

```bash
git checkout -b 04-mcp-stdio
```

Install a sample MCP server (on demand):

```bash
uvx mcp-run-python --help   # optional: sanity check
```

### `src/mcp_stdio_client.py`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

runpy = MCPServerStdio("uv", args=["run", "mcp-run-python", "stdio"], timeout=10)

agent = Agent(
    "gemini-2.5-flash",
    toolsets=[runpy],
    instructions="Use tools when code execution or math helps."
)

async def main() -> None:
    async with agent:
        res = await agent.run("How many days between 2000-01-01 and 2025-03-18?")
        print(res.output)

if __name__ == "__main__":
    asyncio.run(main())
```

Run:

```bash
uv run src/mcp_stdio_client.py
```

Commit:

```bash
git add -A && git commit -m "04-mcp-stdio"
```

---

# 05-mcp-http — Connect to an MCP server over Streamable HTTP

**Goal:** Stand up a tiny MCP server and connect via HTTP.

```bash
git checkout -b 05-mcp-http
uv add fastmcp
```

### `src/servers/add_server.py`

```python
from fastmcp import FastMCP

app = FastMCP("Adder")

@app.tool()
def add(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    app.run(transport="streamable-http")  # http://localhost:8000/mcp
```

### `src/mcp_http_client.py`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

server = MCPServerStreamableHTTP("http://localhost:8000/mcp")
agent = Agent("gemini-2.5-flash", toolsets=[server])

async def main() -> None:
    async with agent:
        res = await agent.run("What is 7 plus 5? Use the tool.")
        print(res.output)

if __name__ == "__main__":
    asyncio.run(main())
```

Run:

```bash
# terminal 1
uv run src/servers/add_server.py
# terminal 2
uv run src/mcp_http_client.py
```

Commit:

```bash
git add -A && git commit -m "05-mcp-http"
```

---

# 06-usage-limits-retries — Guardrails and resilient calls

**Goal:** Bound work with `UsageLimits`, and show retry/backoff using Tenacity in a tool (no external network required).

```bash
git checkout -b 06-usage-limits-retries
```

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

Commit:

```bash
git add -A && git commit -m "06-usage-limits-retries"
```

---

# 08-pattern-router — Router/Delegator with typed outcomes

**Goal:** Route to specialist agents using output functions, with a typed failure fallback.

```bash
git checkout -b 08-pattern-router
```

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

Commit:

```bash
git add -A && git commit -m "08-pattern-router"
```

---

# 11-pattern-pipeline — Deterministic stages & idempotent steps

**Goal:** Chain multiple agents programmatically for a predictable, testable flow.

```bash
git checkout -b 11-pattern-pipeline
```

### `src/pattern_pipeline.py`

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent

class Requirements(BaseModel):
    topic: str
    audience: str
    length_words: int = Field(ge=50, le=800)

extractor = Agent[None, Requirements](
    "gemini-2.5-flash",
    output_type=Requirements,
    instructions="Extract topic, audience, and a reasonable length (50-800)."
)

class Outline(BaseModel):
    headings: list[str]

outliner = Agent[None, Outline](
    "gemini-2.5-flash",
    output_type=Outline,
    instructions="Produce 3-6 descriptive headings."
)

drafter = Agent("gemini-2.5-flash", instructions="Write a crisp draft under the provided headings.")

def pipeline_run(user_brief: str) -> str:
    req = extractor.run_sync(user_brief).output
    outline = outliner.run_sync(
        f"Topic: {req.topic}\nAudience: {req.audience}\nLength: {req.length_words}"
    ).output
    draft = drafter.run_sync(
        "Write with these headings:\n" + "\n".join(f"- {h}" for h in outline.headings) +
        f"\nTarget length ~{req.length_words} words."
    ).output
    return draft

if __name__ == "__main__":
    print(pipeline_run("I need a short blog about zero-copy networking for backend engineers."))
```

Run:

```bash
uv run src/pattern_pipeline.py
```

Commit:

```bash
git add -A && git commit -m "11-pattern-pipeline"
```

---

# 13-pattern-critic-editor — Two-role refinement loop

**Goal:** Improve draft quality with a bounded Editor↔Critic loop.

```bash
git checkout -b 13-pattern-critic-editor
```

### `src/pattern_critic_editor.py`

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent

class Review(BaseModel):
    score: int = Field(ge=1, le=10)
    suggestions: list[str]

editor = Agent("gemini-2.5-flash", instructions="Draft clearly. Avoid fluff.")
critic = Agent[None, Review](
    "gemini-2.5-flash",
    output_type=Review,
    instructions="Score 1-10; include concrete revision suggestions."
)

def improve(prompt: str, rounds: int = 2, target: int = 8) -> str:
    draft = editor.run_sync(prompt).output
    for _ in range(rounds):
        review = critic.run_sync(f"Rate this draft and suggest edits:\n\n{draft}").output
        if review.score >= target:
            break
        revision_instructions = "Apply these improvements:\n- " + "\n- ".join(review.suggestions)
        draft = editor.run_sync(f"{prompt}\n\n{revision_instructions}\n\nRevise:").output
    return draft

if __name__ == "__main__":
    print(improve("Write a 120-word product blurb for a privacy-first notes app."))
```

Run:

```bash
uv run src/pattern_critic_editor.py
```

Commit:

```bash
git add -A && git commit -m "13-pattern-critic-editor"
```

---

# Tests & evals — Fast, no-network CI

**Goal:** Prevent accidental live calls and test patterns by overriding models.

```bash
git checkout -b tests-and-evals
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

Commit:

```bash
git add -A && git commit -m "tests-and-evals"
```

---

## Troubleshooting & tips

* **Model strings:** Replace `"gemini-2.5-flash"` with your provider/model (e.g., `"openai:gpt-4o-mini"`, `"anthropic:claude-3-5-sonnet-latest"`) and set the correct API key environment variable.
* **Streaming:** `run_stream` yields final text chunks. If you need full event-by-event control, use the async `.run()` API and inspect messages/events.
* **Unions:** When using unions or output functions, parameterize `Agent[DepsT, OutputT]` and use `# type: ignore[valid-type]` if your type checker complains on `output_type=`.
* **MCP:** Use stdio for local subprocess servers; use Streamable HTTP for network servers. Add `tool_prefix` if multiple MCP servers expose identically named tools.
* **Guardrails:** `UsageLimits` prevents runaway loops and caps tokens/tool calls. For resiliency, add Tenacity retries to your own tools or HTTP calls.
* **Repro:** Commit your `uv.lock` to pin dependency versions for the workshop.

You now have a compact, production-shaped toolkit: typed agents, practical MCP integrations, and three multi-agent patterns (Router, Pipeline, Critic–Editor) that scale from MVP to real-world workloads—without rewriting your stack.
