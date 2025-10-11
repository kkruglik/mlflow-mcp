# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
