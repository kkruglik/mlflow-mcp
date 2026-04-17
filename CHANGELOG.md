# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
