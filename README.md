# MLflow MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that enables LLMs to interact with [MLflow](https://mlflow.org) tracking servers. Query experiments, analyze runs, compare metrics, and explore the model registry - all through natural language.

## Features

- **Experiment Management**: List and search experiments, discover available metrics and parameters
- **Run Analysis**: Retrieve run details, query runs with filters, find best performing models
- **Metrics & Parameters**: Get metric histories, compare parameters across runs
- **Artifacts**: Browse and download run artifacts
- **Model Registry**: Access registered models, versions, and deployment stages
- **Comparison Tools**: Side-by-side run comparisons, best run selection
- **Tag-based Search**: Filter runs by custom tags
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

### Environment Variables

- **`MLFLOW_TRACKING_URI`** (required): Your MLflow tracking server URL
  - Examples: `http://localhost:5000`, `https://mlflow.company.com`

## Available Tools

### Experiments

- **`get_experiments()`** - List all experiments
- **`get_experiment_by_name(name)`** - Get experiment by name
- **`get_experiment_metrics(experiment_id)`** - Discover all unique metrics
- **`get_experiment_params(experiment_id)`** - Discover all unique parameters

### Runs

- **`get_runs(experiment_id, limit=3, offset=0, order_by=None)`** - Get runs with full details. Supports sorting and pagination
- **`get_run(run_id)`** - Get detailed run information for a specific run
- **`query_runs(experiment_id, query, limit=3, offset=0, order_by=None)`** - Filter and sort runs (e.g., `"metrics.accuracy > 0.9"`, order_by=`"metrics.accuracy DESC"`)
- **`search_runs_by_tags(experiment_id, tags, limit=3, offset=0)`** - Find runs by tags with pagination

### Metrics & Parameters

- **`get_run_metrics(run_id)`** - Get all metrics for a run
- **`get_run_metric(run_id, metric_name)`** - Get full metric history with steps

### Artifacts

- **`get_run_artifacts(run_id, path="")`** - List artifacts (supports browsing directories)
- **`get_run_artifact(run_id, artifact_path)`** - Download artifact
- **`get_artifact_content(run_id, artifact_path)`** - Read artifact content (text/json)

### Analysis & Comparison

- **`get_best_run(experiment_id, metric, ascending=False)`** - Find best run by metric (supports special characters)
- **`compare_runs(experiment_id, run_ids)`** - Side-by-side comparison with full data

### Model Registry

- **`get_registered_models()`** - List all registered models
- **`get_model_versions(model_name)`** - Get all versions of a model
- **`get_model_version(model_name, version)`** - Get version details with metrics

### Health

- **`health()`** - Check server connectivity

## Usage Examples

### Ask Claude

> "Show me all experiments in MLflow"

> "What are the top 5 runs by accuracy in experiment 'my-experiment'?"

> "Compare runs abc123 and def456"

> "Which model has the highest F1 score?"

> "Show me the training loss curve for run xyz789"

> "List all production models in the registry"

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
