# How and Why I Built an MCP Server for MLflow

*From copy-pasting run IDs in an algo trading project to letting Claude query experiments directly — and what I learned building it.*

At my last job we worked on an algo trading project. We started simple — algorithmic models based on technical indicators, not too complex metric calculations, looking for entry signals. It worked and gave decent results. But this approach had a ceiling, and at some point we decided we needed to experiment with ML.

So MLflow entered the picture. Nothing fancy in our setup, really.

Each experiment had a config that listed models, data, training and test split parameters, features and their configs. From all that we'd build a Cartesian product and train everything. Sometimes for many hours depending on how much data there was and what resolution — from 1h candles all the way down to 1m.

```yaml
# experiment config
name: "btc_trading_experiment_catboost"

data_config: "configs/data/training_config.yaml"
labels_config: "configs/ml/labels/target_stop_loss.yaml"
features_config: "configs/ml/features/improved_config.yaml"
preprocessing_config: "configs/ml/preprocessing/catboost_lgbm.yaml"
models_config: "configs/ml/models/catboost_lgbm.yaml"
training_config: "configs/ml/training/config.yaml"
```

```yaml
# models config — each entry is one run
models:
  - name: xgboost
    params:
      n_estimators: 200
      max_depth: 6
      learning_rate: 0.05
      ...
  - name: catboost
    params:
      iterations: 200
      depth: 6
      learning_rate: 0.05
      ...
```

Then from that array we'd pick the most promising runs. Several approaches:

- Naive: sort by metric (precision, f1, etc), take top 5
- Same but also split by model type so you don't end up with 5 XGBoost runs
- Filter by confusion matrix: tp > fp
- Trading-specific metrics — how profitable was the model in actual simulation
- And so on

After a few rounds of this, a notebook would accumulate a bunch of MLflow queries with hardcoded experiment IDs and magic thresholds. And at some point I caught myself copy-pasting run IDs from the UI into the console — and thought, there's probably a better way to get this data somewhere more useful.

That's how the idea for a simple MCP server came up. Instead of all that manual work, just write to the model:

> Look at the runs in experiment_1. Pick 10 models with the most promising results. By "promising" I mean models suited for different trading styles — both aggressive and conservative. Define those criteria in terms of metrics, give them aliases, find the models, and explain why we should pick these for the next experiments.

And the model makes a bunch of MLflow requests and replies with a table. Not one query — four, each targeting a different angle:

```
get_experiment_metrics("btc_trading_v1")
→ ["test/precision", "test/recall", "test/tp", "test/fp", "val/threshold_0.65/net_profit", ...]

# conservative: sort by precision
query_runs(..., "", order_by="test/precision DESC", limit=5)

# trading-aware: runs that are actually profitable on validation
query_runs(..., "metrics.`val/threshold_0.65/net_profit` > 0", order_by="val/threshold_0.65/net_profit DESC", limit=5)

# aggressive: enough signal volume for backtesting
query_runs(..., "metrics.`test/tp` > 50", order_by="test/recall DESC", limit=5)
```

Here's what came out:

| Run ID | Model | Precision | TP / FP | Net profit (val) | Style |
|---|---|---|---|---|---|
| `85640cf8` | decision_tree_simple | 0.912 | 7 / 0 | — | Conservative — zero false positives, every signal is certain |
| `04e38a58` | catboost | 0.780 | 1 / 0 | +11 at threshold 0.7 | Conservative/trading — profitable on validation with tight threshold |
| `aeecd943` | lightgbm (gbdt) | 0.626 | 10 / 9 | +15 at threshold 0.5 | Balanced — best net profit in the experiment, tp ≈ fp |
| `21aa0a86` | lightgbm (dart) | 0.650 | 111 / 287 | — | Aggressive — best tp/fp ratio among high-volume models |
| `4cfc4f07` | lightgbm (dart) | 0.787 | 90 / 554 | — | Aggressive/recall — highest recall, different dataset slice |

Done.

At some point I wanted to go further and have the model add extended analysis directly into Jupyter notebooks via MCP. That didn't work well. So markdown tables were enough.

Since it was useful for me, I put the repo up publicly. Personally I only needed read tools — querying runs, metrics, artifacts. Promoting models I could do myself. But since we're delegating new powers to the model, why not give it the full toolkit: promote, tag, delete. Everything you can do in the MLflow UI, the model can do too. The server grew to 39 tools.

---

I'm not going to write another "what is MCP" post — there are plenty of those on Medium. Just a few interesting challenges.

## The model was greedy

```python
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_runs(
    experiment_id: str,
    limit: int = 3,
    offset: int = 0,
    order_by: list[str] | None = None,
) -> list[dict]:
```

Left to its own devices, the model would just grab everything. And a single MLflow run has a lot of data — metrics, params, tags, all logged steps. Ten runs can fill up a model's context pretty fast.

MLflow's `search_runs` does have pagination via `page_token`, but it's token-based — you get a cursor for the next page, not an offset. That's fine for sequential browsing but awkward when you want to say "skip the first 20". So the server implements offset manually: fetch `offset + limit` results, then slice. Not elegant, but it works and keeps the interface simple for the model.

Small default limits help too. And the docstring warns the model directly:

```python
"""Get runs for a specific experiment with full details.

Runs can be large. Use wise limits to avoid flooding context.

Args:
    limit: Maximum number of runs to return.
    offset: Number of runs to skip
    order_by: List of sort clauses, e.g. ['metrics.rmse DESC', 'params.lr ASC'].
              Use backticks for special characters: ['metrics.`trading/loss` DESC']

Examples:
    get_runs("1", limit=5)
    get_runs("1", order_by=["metrics.accuracy DESC"])
"""
```

## Marking destructive tools

When you add tools that can delete data, you need to be explicit about it — both in the docstring and in the tool metadata.

MCP 2025 defines hint fields for tools: `readOnlyHint`, `destructiveHint`, `idempotentHint`. These don't enforce anything — they're metadata that clients can use to show confirmation prompts for destructive actions and skip them for read-only ones. FastMCP supports them via `ToolAnnotations`:

```python
# read-only — safe to call freely
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_runs(experiment_id: str, ...) -> list[dict]:
    ...

# write but idempotent — calling twice has the same result
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True))
def set_model_alias(name: str, alias: str, version: str) -> dict:
    ...

# destructive — data can be lost
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def delete_model_version(name: str, version: str) -> dict:
    """Delete a specific model version from the registry.
    Irreversible — the version and its metadata cannot be recovered.
    """
    ...

@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def delete_registered_model(name: str) -> dict:
    """Delete an entire registered model and all its versions.
    Irreversible — all versions, aliases, and tags are permanently removed.
    """
    ...
```

All 39 tools are annotated. Read-only, write/idempotent, and destructive — each classified correctly.

## Confirming that something actually happened

When a write tool returns `None`, the model has no way to verify success. It can only assume. In a chained workflow with multiple write operations, that silent assumption is hard to debug.

The fix: every write tool that doesn't return a rich object returns a confirmation dict with `"success": True` as the first field:

```python
# tagging a run
return {"success": True, "run_id": run_id, "key": key, "value": value}

# setting an alias
return {"success": True, "name": name, "alias": alias, "version": version}

# deleting an alias
return {"success": True, "name": name, "alias": alias}
```

Small thing. But now the model can say "I removed alias `champion` from `fraud-classifier`" with actual evidence rather than just hoping the call didn't error out.

---

Overall, I was pretty skeptical about MCP when I first heard about it. Felt like another wheel being reinvented. But the idea that you can plug any tool into a model through the same interface is genuinely attractive. The problem is most MCPs are situational — nobody's going to keep dozens of them running. But when you need one, it's nice to know someone's probably already built it. If you're using MLflow and want to talk to it through Claude — mine's on GitHub.

[mlflow-mcp on GitHub](https://github.com/kkruglik/mlflow-mcp)
