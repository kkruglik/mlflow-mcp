# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
