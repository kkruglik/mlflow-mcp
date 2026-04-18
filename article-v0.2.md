# I Let Claude Talk to My MLflow Server — Here's What It Actually Took

*building an MCP server for ML experiment tracking: tools, prompts, bugs, and tradeoffs*

---

I built [mlflow-mcp](https://github.com/kkruglik/mlflow-mcp) because I was tired of copy-pasting run IDs. The idea was simple: wrap MLflow's tracking API in an MCP server so Claude could query my experiments directly. Ask "what's the best run by recall?" and get an answer in context, without leaving the editor.

It worked. Barely. It was read-only, it had 19 tools, and it was good enough for me to use personally. Then I stopped touching it.

A few months later I came back to actually finish it — and ended up doing a lot more than I planned. This post covers how the server works, what decisions went into it, and the stuff that broke along the way.

---

## What is MCP and why does it matter

[Model Context Protocol](https://modelcontextprotocol.io) is an open standard from Anthropic for connecting LLMs to external tools and data sources. Think of it as USB-C for AI — one standard plug, any device.

Before MCP, every AI assistant had its own plugin system. Claude had one format, Cursor had another, Continue had a third. If you built a tool integration you had to rewrite it for each platform. MCP standardizes the interface: you write a server once, and any MCP-compatible client — Claude Desktop, Claude Code, Cursor, Continue, or your own app — can use it without changes.

An MCP server exposes three primitives:

- **Tools** — functions the LLM actively calls, like API endpoints. `get_runs()`, `register_model()`, `set_model_alias()` — all tools. The LLM decides when to call them based on the conversation.
- **Prompts** — pre-built message templates that surface as slash commands in the client. You define the structure, the user fills in the parameters.
- **Resources** — data exposed as URIs that the client can read passively, without the LLM making an explicit call. Think `mlflow://experiments/active` as ambient context that's always available. More like a file system than an API.
- **Resource Templates** — parameterized resource URIs, like `mlflow://experiments/{id}/runs`. A pattern that expands into many resources on demand.

mlflow-mcp uses tools and prompts. Resources and templates are both empty — intentionally. Resources make sense for persistent ambient context: something you want always loaded, without the LLM having to ask for it. Templates make sense when you have resources with variable parts. Neither applies here. For a stateless query-and-act workflow, the LLM just calls `get_experiments()` once and moves on. Every call is independent, there's no session state to expose. Resources shine in long multi-turn agent workflows where re-fetching context each turn would be wasteful. For mlflow-mcp, the added complexity wasn't justified.

---

## Testing with MCP Inspector

Before plugging an MCP server into Claude, test it directly. The easiest way is [MCP Inspector](https://github.com/modelcontextprotocol/inspector) — an official browser-based debugging tool from Anthropic.

```bash
npx @modelcontextprotocol/inspector uvx mlflow-mcp
```

This opens a UI where you can browse all available tools, call them with custom inputs, and see the raw responses. No LLM involved — you're talking directly to the server. It's the fastest way to catch serialization bugs or verify a new tool before it touches Claude.

For the actual Claude integration:

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "mlflow": {
      "command": "uvx",
      "args": ["mlflow-mcp"],
      "env": { "MLFLOW_TRACKING_URI": "http://127.0.0.1:5000" }
    }
  }
}
```

**Claude Code** — drop a `.mcp.json` in your project root with the same format. After that, Claude sees all the tools automatically.

---

## First thing I checked: did Databricks ship this already?

Before writing a line of code I wanted to know if the repo had become pointless. Databricks has been very active with MCP — they shipped an official MLflow MCP server. Turns out it only covers GenAI Tracing. Not experiment tracking, not the model registry, not runs or metrics. So the gap this repo fills is still real.

---

## What the server covers

The original version was read-only — list experiments, query runs, browse metrics and artifacts. Useful, but limited. You could ask "what's the best model?" but couldn't do anything about it.

The current version adds:

**MLflow 3 LoggedModel support.** MLflow 3 introduced `LoggedModel` as a first-class entity, separate from runs and the registry. When you call `mlflow.sklearn.log_model()`, you get a `LoggedModel` with its own ID, metrics, params, and status. `search_logged_models` and `get_logged_model` cover this.

**Write and action tools.** `register_model`, `set_model_alias`, `copy_model_version`, `set_run_tag`, `set_experiment_tag`, `set_registered_model_tag`, `update_model_version`, `transition_model_version_stage`. The full lifecycle, not just reading.

**Extended registry tools.** `get_registered_model` with aliases, `get_model_version_by_alias`, `get_latest_versions` — the pieces needed to actually navigate a registry that uses the alias pattern.

**MCP Prompts for guided workflows.** `compare_runs_by_ids`, `find_best_run`, `promote_best_model`.

34 tools total.

---

## MLflow 3 LoggedModel: the serialization bug

The first thing I hit when adding LoggedModel support was a serialization error. MLflow's `LoggedModel.to_dictionary()` has two problems: it converts `status` to an int instead of a string, and it drops `model_uri` entirely. On top of that, `to_dictionary()` leaves `Metric` objects unserialized in the `metrics` field — which the MCP transport can't handle.

```
Unable to serialize unknown type: Metric
```

Fix lives in `helpers.py`:

```python
def serialize_logged_model(model) -> dict:
    d = model.to_dictionary()
    d["status"] = str(model.status)
    d["metrics"] = [m.to_dictionary() for m in (model.metrics or [])]
    return d
```

Using `to_dictionary()` as the base is intentional — it'll pick up new fields automatically as MLflow adds them. We just patch the two broken fields on top.

---

## Write tools: what to return

When I added the setter tools — `set_run_tag`, `set_experiment_tag`, `set_registered_model_tag`, `set_model_alias` — I initially had them return `None`. They call the SDK, no error, done.

The problem: the LLM has no way to confirm success. It can only assume. When you're chaining multiple write operations in a workflow, that assumption failure is silent and hard to debug.

The same question came up when I added delete tools — `delete_run`, `delete_experiment`, `delete_model_alias`, `delete_model_version`, `delete_registered_model`. `{"run_id": run_id}` felt wrong too, like just echoing input back with no signal that anything actually happened.

The fix: every write tool that doesn't return a rich object returns a confirmation dict with `"success": True` as the first field:

```python
return {"success": True, "run_id": run_id, "key": key, "value": value}
return {"success": True, "name": name, "alias": alias}
```

It's a small thing. But now the LLM can say "I deleted alias `champion` from `fraud-classifier`" with actual evidence, and any client can check the field programmatically without parsing a message string.

---

## Tool annotations

MCP 2025 defines hint fields for tools: `readOnlyHint`, `destructiveHint`, `idempotentHint`. These don't enforce anything — they're metadata for clients so they can show confirmation prompts for destructive actions and skip them for read-only ones.

FastMCP supports them via `ToolAnnotations`:

```python
from mcp.types import ToolAnnotations

@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_runs(...):
    ...

@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def copy_model_version(...):
    ...
```

All 34 tools are annotated. Read-only tools, write/idempotent tools, and the two destructive promotion tools are all classified correctly.

---

## The `register_model` 500 error

When I first tested registering a LoggedModel I got a 500. I was using `runs:/run_id/model` as the URI — the classic pattern for run artifacts. But the run had no artifacts at that path because the model was logged as a `LoggedModel`, not a run artifact.

The correct URI for a LoggedModel is `models:/m-{model_id}`. MLflow's `mlflow.register_model()` handles both patterns transparently. `client.create_model_version()` with a `runs:/` URI does not. Switched to `mlflow.register_model()` and it worked immediately.

---

## Prompts: don't script the LLM

The first version of the MCP prompts was bad. I had them spelling out exact tool call sequences with interpolated IDs — basically trying to script what the LLM should do step by step.

```
# bad: too prescriptive
Call search_logged_models with experiment_ids=["{experiment_id}"]...
Then call register_model with model_uri=...
```

It felt wrong and it was. The whole point of an LLM is that it can figure out the steps. The prompts were rewritten as high-level natural language:

> "Find the best logged model by {metric}. Register it as "{model_name}" with a selection_metric tag. Add relevant model-level tags. Assign the 'champion' alias. Ask the user if they want to copy it to a separate production model entry."

That's it. The LLM picks the right tools, in the right order, with the right arguments. Much better.

---

## A few things that tripped me up

**Port 5000 on macOS.** `localhost:5000` doesn't always mean what you think. On macOS, `localhost` can resolve to `::1` (IPv6), which hits Apple's AirPlay Receiver on port 5000. I was getting connection refused until I switched to `http://127.0.0.1:5000`. If your MLflow server seems unreachable on Mac, try this first.

**MCP running the installed package, not source.** The `.mcp.json` originally had `uv run mlflow-mcp` with a `--project` flag pointing to the local repo. Fine for development, non-portable for everyone else. Switched to `uvx mlflow-mcp` which always runs the published package.

**`transition_model_version_stage` is deprecated.** Deprecated since MLflow 2.9. The modern pattern is aliases + `copy_model_version`. I kept the tool anyway because plenty of people still run older MLflow in production, and removing it would silently break things. Added a deprecation note in the docstring pointing to the alternatives.

---

## What it looks like in practice

The full promotion flow is now a single prompt:

> "Find the best logged model in experiment 'fraud-detection' by test/recall. Register it as 'fraud-classifier', tag it with the framework and problem type, and set it as champion. Ask me before copying to prod."

Claude runs `search_logged_models`, picks the winner, calls `register_model`, calls `set_registered_model_tag` a few times, calls `set_model_alias` with `champion`, then asks if you want `copy_model_version`. About 30 seconds end-to-end.

---

## What's next

The server is stateless — no concept of "current experiment" that persists across tool calls. Every call is independent. Fine for most use cases, but for longer workflows Claude has to re-fetch context repeatedly. I'm thinking about whether MCP Resources are the right primitive for this, or whether it's just a prompt engineering problem. [WHY — add your motivation here]

---

**Links:**
- [GitHub](https://github.com/kkruglik/mlflow-mcp)
- [PyPI](https://pypi.org/project/mlflow-mcp/)
- [MLflow Model Registry tutorial](https://mlflow.org/docs/latest/ml/model-registry/tutorial/)
