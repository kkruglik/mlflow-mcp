# MLflow MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that enables LLMs to interact with [MLflow](https://mlflow.org) tracking servers. Query experiments, analyze runs, compare metrics, manage the model registry, and promote models to production — all through natural language.

## Features

- **Experiment Management**: List, search, and filter experiments
- **Run Analysis**: Query runs, compare metrics, find best performing models
- **Metrics & Parameters**: Get metric histories, compare parameters across runs
- **Artifacts**: Browse and download run artifacts
- **LoggedModel Support**: Search and retrieve MLflow 3 LoggedModel entities
- **Model Registry**: Full registry management — register, tag, alias, stage, and promote models
- **Write Actions**: Tag runs/experiments, assign aliases, register and promote models
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

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/claude/claude_desktop_config.json`

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

### Environment Variables

- **`MLFLOW_TRACKING_URI`** (required): Your MLflow tracking server URL
  - Examples: `http://127.0.0.1:5000`, `https://mlflow.company.com`

## Available Tools

### Experiments

- **`get_experiments()`** - List all experiments
- **`search_experiments(filter_string, order_by, max_results)`** - Filter and sort experiments
- **`get_experiment_by_name(name)`** - Get experiment by name
- **`get_experiment_metrics(experiment_id)`** - Discover all unique metrics
- **`get_experiment_params(experiment_id)`** - Discover all unique parameters
- **`set_experiment_tag(experiment_id, key, value)`** - Tag an experiment

### Runs

- **`get_runs(experiment_id, limit=3, offset=0, order_by=None)`** - Get runs with full details
- **`get_run(run_id)`** - Get detailed run information
- **`get_parent_run(run_id)`** - Get parent run for nested runs
- **`query_runs(experiment_id, query, limit=3, offset=0, order_by=None)`** - Filter and sort runs
- **`search_runs_by_tags(experiment_id, tags, limit=3, offset=0)`** - Find runs by tags
- **`set_run_tag(run_id, key, value)`** - Tag a run

### Metrics & Parameters

- **`get_run_metrics(run_id)`** - Get all metrics for a run
- **`get_run_metric(run_id, metric_name)`** - Get full metric history with steps

### Artifacts

- **`get_run_artifacts(run_id, path="")`** - List artifacts (supports browsing directories)
- **`get_run_artifact(run_id, artifact_path)`** - Download artifact
- **`get_artifact_content(run_id, artifact_path)`** - Read artifact content (text/json)

### Analysis & Comparison

- **`get_best_run(experiment_id, metric, ascending=False)`** - Find best run by metric
- **`compare_runs(experiment_id, run_ids)`** - Side-by-side run comparison

### Logged Models (MLflow 3)

- **`search_logged_models(experiment_ids, filter_string, order_by, max_results)`** - Search logged models by metrics/params/tags
- **`get_logged_model(model_id)`** - Get full details of a logged model

### Model Registry

- **`get_registered_models()`** - List all registered models
- **`get_registered_model(name)`** - Full model details including versions and aliases
- **`get_model_versions(model_name)`** - Get all versions of a model
- **`get_model_version(model_name, version)`** - Get version details with metrics
- **`get_model_version_by_alias(name, alias)`** - Get version by alias (e.g. "champion")
- **`get_latest_versions(name, stages)`** - Get latest versions per stage
- **`register_model(model_name, model_uri, tags)`** - Register a model into the registry
- **`update_model_version(name, version, description)`** - Update version description
- **`set_registered_model_tag(name, key, value)`** - Tag a registered model
- **`set_model_alias(name, alias, version)`** - Assign an alias to a model version
- **`transition_model_version_stage(name, version, stage)`** - Transition to Staging/Production/Archived
- **`copy_model_version(src_model_name, src_version, dst_model_name)`** - Promote version to another registered model

### Health

- **`health()`** - Check server connectivity

## Prompts

Built-in guided workflows available as slash commands:

- **`compare_runs_by_ids`** - Compare specific runs side-by-side
- **`find_best_run`** - Find and analyze the best run in an experiment by metric
- **`promote_best_model`** - End-to-end: find best model → register → tag → alias → promote

## Usage Examples

### Ask Claude

> "Show me all experiments in MLflow"

> "What are the top 5 runs by accuracy in experiment 'my-experiment'?"

> "Compare runs abc123 and def456"

> "Find the best model by test/recall and register it as 'my-classifier'"

> "Which logged model has the highest F1 score?"

> "Set the champion alias on version 3 of my-model"

> "Promote my-classifier v2 to production"

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
