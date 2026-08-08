# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-08-08

### Changed
- Pinned `mcp` dependency to `>=1.16.0,<2` — MCP Python SDK v2 (stable since 2026-07-27) removes the `mcp.server.fastmcp.FastMCP` import path this server relies on, so an unpinned install could resolve to a breaking major version

## [0.4.0] - 2026-04-22

### Added
- `get_experiment_tags(experiment_id)` — discover all unique tag keys used across runs in an experiment (completes the metrics/params/tags symmetrical trio)
- `audit_mlflow_setup` prompt — evaluates an MLflow deployment against Google/Databricks best practices across 7 categories (experiment organization, parameter logging, metric logging, tagging strategy, artifact management, model registry, reproducibility); scores each 1–10 and produces a prioritized improvement roadmap with a mean score

### Changed
- `get_run()` now returns an `inputs` field containing dataset inputs logged via `mlflow.log_input()` (MLflow 3 dataset tracking)

## [0.3.0] - 2026-04-18

### Added
- Delete tools: `delete_run`, `delete_experiment`, `delete_model_alias`, `delete_model_version`, `delete_registered_model`
- Tool annotations on all 39 tools (`readOnlyHint`, `idempotentHint`, `destructiveHint`)
- Server `instructions` metadata for MCP clients

### Changed
- All write tools now return `{"success": True, ...}` confirmation dicts instead of `None`
- README rewritten with tables for tools, prompts, and environment variables; added Debugging section
- Docstrings warn about large run output on tools that return run data
- `.mcp.json` includes dev entry for running from source

## [0.2.1] - 2026-04-17

### Changed
- Add authentication documentation (Basic Auth and token-based) to README
- Expand usage examples with real-world flows and scenarios

## [0.2.0] - 2026-04-17

### Added

**MLflow 3 LoggedModel support**
- `search_logged_models` — search logged models across experiments by metrics/params/tags with ordering
- `get_logged_model` — fetch full details of a logged model by ID

**Extended experiment & run tools**
- `search_experiments` — filter and sort experiments by name, tags, timestamps
- `get_parent_run` — navigate nested run hierarchies

**Extended model registry tools**
- `get_registered_model` — full model details including all versions and aliases
- `get_model_version_by_alias` — retrieve a version by alias (e.g. "champion")
- `get_latest_versions` — get latest versions per stage

**Write / action tools**
- `register_model` — register a logged model or run artifact into the registry
- `set_registered_model_tag` — add tags to a registered model
- `set_model_alias` — assign an alias to a model version
- `set_run_tag` — annotate runs with tags
- `set_experiment_tag` — annotate experiments with tags
- `update_model_version` — update model version description
- `transition_model_version_stage` — transition version to Staging/Production/Archived
- `copy_model_version` — promote a model version to another registered model (MLflow 3 pattern)

**MCP Prompts**
- `compare_runs_by_ids` — compare specific runs side-by-side
- `find_best_run` — find and analyze the best run in an experiment by metric
- `promote_best_model` — end-to-end workflow: find best model → register → tag → alias → promote

**Project-scoped MCP config**
- Added `.mcp.json` for running the server directly from the repo in Claude Code

## [0.1.7] - 2025-01-11

### Added
- Pagination support with `offset` parameter for `get_runs()`, `query_runs()`, and `search_runs_by_tags()`
- Sorting support with `order_by` parameter for `get_runs()` and `query_runs()`

### Changed
- Simplified API by removing summary/lightweight response modes
- All tools now return full data by default with lower default limits (3 instead of 5-10)
- Reduced default limits to avoid MCP token limit issues

### Removed
- `get_runs_sorted()` function (replaced by `query_runs()` with `order_by` parameter)
- `include_details`, `include_all_data` flags (always return full data now)
- Summary response modes with metric/param key previews

## [0.1.0] - 2025-01-10

### Added
- Initial release of MLflow MCP Server
- Experiment management tools (list, search by name, discover metrics/params)
- Run analysis tools (get, query, search by tags)
- Metrics and parameters tools (get all metrics, metric history)
- Artifact management (list, download, read content)
- Model registry support (list models, versions, version details)
- Comparison tools (compare runs, find best run)
- Health check endpoint
- Comprehensive logging with proper error handling
- Support for Python 3.10+
- PyPI package distribution via uvx/pip

### Features
- 19 MCP tools for complete MLflow interaction
- Environment variable configuration (MLFLOW_TRACKING_URI)
- Directory browsing for artifacts
- Tag-based run filtering
- Best run selection by metric
- Side-by-side run comparison

[0.1.0]: https://github.com/kirillkruglikov/mlflow-mcp/releases/tag/v0.1.0
