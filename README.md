# MLflow MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that enables LLMs to interact with [MLflow](https://mlflow.org) tracking servers. Query experiments, analyze runs, compare metrics, manage the model registry, and promote models to production — all through natural language.

## Features

- **Experiment Management**: List, search, and filter experiments
- **Run Analysis**: Query runs, compare metrics, find best performing models
- **Metrics & Parameters**: Get metric histories, compare parameters across runs
- **Artifacts**: Browse and download run artifacts
- **LoggedModel Support**: Search and retrieve MLflow 3 LoggedModel entities
- **Model Registry**: Full registry management — register, tag, alias, stage, and promote models
- **Write & Delete Actions**: Tag, alias, register, promote, and delete runs/experiments/models
- **MCP Prompts**: Built-in guided workflows for common tasks
- **Pagination**: Offset-based pagination for browsing large result sets

## Installation

### Using uvx (Recommended)

```bash
# Run directly without installation
uvx mlflow-mcp

# Or install globally
pip install mlflow-mcp
```

### From Source

```bash
git clone https://github.com/kkruglik/mlflow-mcp.git
cd mlflow-mcp
uv sync
uv run mlflow-mcp
```

## Configuration

### Claude Desktop

Add to your Claude Desktop config file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mlflow": {
      "command": "uvx",
      "args": ["mlflow-mcp"],
      "env": {
        "MLFLOW_TRACKING_URI": "http://localhost:5000"
      }
    }
  }
}
```

### Claude Code (project-scoped)

Add `.mcp.json` to your project root:

```json
{
  "mcpServers": {
    "mlflow": {
      "command": "uvx",
      "args": ["mlflow-mcp"],
      "env": {
        "MLFLOW_TRACKING_URI": "http://localhost:5000"
      }
    }
  }
}
```

### Authenticated Server

For MLflow servers with authentication, add credentials to the `env` block:

```json
{
  "mcpServers": {
    "mlflow": {
      "command": "uvx",
      "args": ["mlflow-mcp"],
      "env": {
        "MLFLOW_TRACKING_URI": "https://mlflow.company.com",
        "MLFLOW_TRACKING_USERNAME": "your-username",
        "MLFLOW_TRACKING_PASSWORD": "your-password"
      }
    }
  }
}
```

For Databricks or token-based auth, use `MLFLOW_TRACKING_TOKEN` instead:

```json
{
  "mcpServers": {
    "mlflow": {
      "command": "uvx",
      "args": ["mlflow-mcp"],
      "env": {
        "MLFLOW_TRACKING_URI": "https://mlflow.company.com",
        "MLFLOW_TRACKING_TOKEN": "your-token"
      }
    }
  }
}
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | Yes | MLflow tracking server URL, e.g. `http://127.0.0.1:5000` |
| `MLFLOW_TRACKING_USERNAME` | No | HTTP Basic Auth username (MLflow built-in auth) |
| `MLFLOW_TRACKING_PASSWORD` | No | HTTP Basic Auth password (MLflow built-in auth) |
| `MLFLOW_TRACKING_TOKEN` | No | Bearer token (Databricks or token-based setups) |

## Tools

### Experiments

| Tool | Description |
|---|---|
| `get_experiments()` | List all experiments |
| `search_experiments(filter_string, order_by, max_results)` | Filter and sort experiments |
| `get_experiment_by_name(name)` | Get experiment by name |
| `get_experiment_metrics(experiment_id)` | Discover all unique metric keys |
| `get_experiment_params(experiment_id)` | Discover all unique parameter keys |
| `get_experiment_tags(experiment_id)` | Discover all unique tag keys used across runs |
| `set_experiment_tag(experiment_id, key, value)` | Tag an experiment |
| `delete_experiment(experiment_id)` | Delete an experiment (moves to deleted stage) |

### Runs

| Tool | Description |
|---|---|
| `get_runs(experiment_id, limit, offset, order_by)` | List runs with full details, sorting and pagination |
| `get_run(run_id)` | Get detailed run information including metrics, params, tags, artifact URI, and dataset inputs |
| `get_parent_run(run_id)` | Get parent run for nested runs |
| `query_runs(experiment_id, query, limit, offset, order_by)` | Filter runs, e.g. `"metrics.accuracy > 0.9"` |
| `search_runs_by_tags(experiment_id, tags, limit, offset)` | Find runs by tag key/value |
| `set_run_tag(run_id, key, value)` | Tag a run |
| `delete_run(run_id)` | Delete a run (moves to deleted stage) |

### Metrics & Parameters

| Tool | Description |
|---|---|
| `get_run_metrics(run_id)` | Get all metrics for a run |
| `get_run_metric(run_id, metric_name)` | Get full metric history with steps |

### Artifacts

| Tool | Description |
|---|---|
| `get_run_artifacts(run_id, path)` | List artifacts, supports browsing subdirectories |
| `get_run_artifact(run_id, artifact_path)` | Download an artifact file |
| `get_artifact_content(run_id, artifact_path)` | Read artifact content as text/JSON |

### Analysis & Comparison

| Tool | Description |
|---|---|
| `get_best_run(experiment_id, metric, ascending)` | Find best run by metric |
| `compare_runs(experiment_id, run_ids)` | Side-by-side run comparison |

### Logged Models (MLflow 3)

| Tool | Description |
|---|---|
| `search_logged_models(experiment_ids, filter_string, order_by, max_results)` | Search logged models by metrics/params/tags |
| `get_logged_model(model_id)` | Get full details of a logged model |

### Model Registry

| Tool | Description |
|---|---|
| `get_registered_models()` | List all registered models |
| `get_registered_model(name)` | Full model details including versions and aliases |
| `get_model_versions(model_name)` | Get all versions of a model |
| `get_model_version(model_name, version)` | Get version details with metrics |
| `get_model_version_by_alias(name, alias)` | Get version by alias, e.g. `"champion"` |
| `get_latest_versions(name, stages)` | Get latest versions per stage |
| `register_model(model_name, model_uri, tags)` | Register a model into the registry |
| `update_model_version(name, version, description)` | Update version description |
| `set_registered_model_tag(name, key, value)` | Tag a registered model |
| `set_model_alias(name, alias, version)` | Assign an alias to a model version |
| `delete_model_alias(name, alias)` | Remove an alias from a model |
| `copy_model_version(src_model_name, src_version, dst_model_name)` | Promote version to another registered model |
| `transition_model_version_stage(name, version, stage)` | Transition to Staging/Production/Archived *(deprecated since MLflow 2.9, use aliases instead)* |
| `delete_model_version(name, version)` | Delete a model version |
| `delete_registered_model(name)` | Delete a registered model and all its versions |

### Health

| Tool | Description |
|---|---|
| `health()` | Check server connectivity |

## Prompts

Built-in guided workflows available as slash commands in Claude:

| Prompt | Description |
|---|---|
| `compare_runs_by_ids` | Compare specific runs side-by-side |
| `find_best_run` | Find and analyze the best run in an experiment by metric |
| `promote_best_model` | End-to-end: find best model → register → tag → alias → promote |
| `audit_mlflow_setup` | Audit the MLflow setup against industry best practices — scores 7 categories 1–10 and produces a prioritized improvement roadmap |

## Usage Examples

### Explore experiments and runs

> "Show me all experiments. Which ones were updated recently?"

> "What metrics and parameters are tracked in experiment 'fraud-detection'?"

> "Get the top 10 runs in 'fraud-detection' sorted by test/f1. Show me the params that differ most between the top 3."

> "Find all runs tagged with model_type=lightgbm and compare their recall scores."

### Analyze a training run

> "Show me the full details of run abc123 — metrics, params, and artifacts."

> "Plot the training loss curve for run abc123." *(Claude fetches metric history and renders a chart)*

> "This run has a parent — show me the parent run and compare their metrics."

### Find and register the best model

> "Find the best logged model in experiment 'fraud-detection' by test/recall. Register it as 'fraud-classifier' with a selection_metric tag."

> "Which logged model in experiments 1 and 2 has the highest F1 score on the validation set?"

> "Register the model from run abc123 artifact path 'model/' as 'my-classifier'."

### Manage the model registry

> "Show me all versions of 'fraud-classifier' with their aliases and stages."

> "Set the champion alias on version 3 of fraud-classifier."

> "Update the description of fraud-classifier v3 to explain what dataset it was trained on."

> "Copy fraud-classifier v3 to a separate 'fraud-classifier-prod' model as the production entry."

### Audit your MLflow setup

> "Audit my MLflow setup"

*(Triggers the `audit_mlflow_setup` built-in prompt — Claude explores experiments, runs, artifacts, and the model registry, then scores each area against Google/Databricks best practices)*

<details>
<summary>Example output</summary>

```
| Category             | Score  | Top Issue                                      |
|----------------------|--------|------------------------------------------------|
| Experiment Org       |  5/10  | Flat namespace, no dot-notation hierarchy      |
| Parameter Logging    |  7/10  | No parent-child nesting for tuning sweeps      |
| Metric Logging       |  6/10  | Only final values logged, no training curves   |
| Tagging Strategy     |  5/10  | Params duplicated as tags; stale test_tag      |
| Artifact Management  |  2/10  | No log_model(); artifacts on local disk        |
| Model Registry       |  3/10  | Duplicate prod models instead of aliases       |
| Reproducibility      |  3/10  | No git SHA; no mlflow.log_input() datasets     |
| Mean Score           |  4.4/10|                                                |

Top 3 improvements:
1. Call log_model() and move artifact store to S3/GCS
2. Add git SHA tag + mlflow.log_input() for dataset tracking
3. Consolidate registry to one model entry with @champion alias
```

</details>

### End-to-end promotion workflow

> "Find the best model in 'fraud-detection' by test/recall, register it as 'fraud-classifier', tag it with the framework and problem type, and set it as champion. Ask me before copying to prod."

*(This maps directly to the `promote_best_model` built-in prompt)*

## Debugging

Use [MCP Inspector](https://github.com/modelcontextprotocol/inspector) to browse tools, call them with custom inputs, and inspect raw responses — without involving an LLM.

**Published package:**
```bash
npx @modelcontextprotocol/inspector uvx mlflow-mcp
```

**Local source:**
```bash
npx @modelcontextprotocol/inspector uv run --project /path/to/mlflow-mcp mlflow-mcp
```

Set `MLFLOW_TRACKING_URI` in the Inspector's environment panel, or pass it inline:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 npx @modelcontextprotocol/inspector uvx mlflow-mcp
```

## Requirements

- Python >=3.10
- MLflow >=3.4.0
- Access to an MLflow tracking server

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Links

- [PyPI Package](https://pypi.org/project/mlflow-mcp/)
- [GitHub Repository](https://github.com/kkruglik/mlflow-mcp)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Model Context Protocol](https://modelcontextprotocol.io)
