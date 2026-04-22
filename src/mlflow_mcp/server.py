import logging
import os
from typing import Any

import mlflow
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from mlflow.tracking import MlflowClient

from mlflow_mcp.helpers import serialize_logged_model

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

if not MLFLOW_TRACKING_URI:
    logger.error("MLFLOW_TRACKING_URI is not set")
    exit(1)


mcp = FastMCP(
    "mlflow",
    instructions="MLflow MCP server — experiment tracking, model registry, and promotion workflows",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logger.info(f"MLflow MCP server initialized with tracking URI: {MLFLOW_TRACKING_URI}")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_experiments() -> list[dict]:
    """Get all experiments"""
    logger.info("Fetching all experiments")
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        logger.info(f"Found {len(experiments)} experiments")
        return [{"name": e.name, "id": e.experiment_id} for e in experiments]
    except Exception as e:
        logger.error(f"Error fetching experiments: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def search_experiments(
    filter_string: str | None = None,
    order_by: list[str] | None = None,
    max_results: int = 100,
) -> list[dict]:
    """Search experiments with optional filtering and sorting.

    Args:
        filter_string: Filter query, e.g. "name LIKE 'btc%'" or "tags.team = 'ml'".
                       Supports name, creation_time, last_update_time, tags.<key>.
        order_by: List of sort clauses, e.g. ["last_update_time DESC", "name ASC"].
        max_results: Maximum number of experiments to return (default 100).

    Examples:
        search_experiments(filter_string="name LIKE 'btc%'")
        search_experiments(order_by=["last_update_time DESC"])
    """
    logger.info(
        f"Searching experiments (filter={filter_string!r}, order_by={order_by})"
    )
    try:
        client = MlflowClient()
        experiments = client.search_experiments(
            filter_string=filter_string,
            order_by=order_by,
            max_results=max_results,
        )
        logger.info(f"Found {len(experiments)} experiments")
        return [
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "artifact_location": e.artifact_location,
                "lifecycle_stage": e.lifecycle_stage,
                "creation_time": e.creation_time,
                "last_update_time": e.last_update_time,
                "tags": e.tags,
            }
            for e in experiments
        ]
    except Exception as e:
        logger.error(f"Error searching experiments: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_experiment_by_name(name: str) -> dict:
    """Get experiment details by name (more convenient than ID)"""
    logger.info(f"Fetching experiment by name: {name}")
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(name)

        if experiment is None:
            logger.warning(f"Experiment with name '{name}' not found")
            raise ValueError(f"Experiment with name '{name}' not found")

        logger.info(f"Found experiment: {experiment.experiment_id}")
        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "artifact_location": experiment.artifact_location,
            "lifecycle_stage": experiment.lifecycle_stage,
            "tags": experiment.tags,
        }
    except Exception as e:
        logger.error(f"Error fetching experiment by name '{name}': {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_experiment_metrics(experiment_id: str) -> list[str]:
    """Get all unique metric names used across all runs in an experiment"""
    logger.info(f"Fetching metrics for experiment: {experiment_id}")
    try:
        client = MlflowClient()
        runs = client.search_runs(experiment_ids=[experiment_id], max_results=1000)

        metric_names = set()
        for run in runs:
            metric_names.update(run.data.metrics.keys())

        logger.info(f"Found {len(metric_names)} unique metrics across {len(runs)} runs")
        return sorted(list(metric_names))
    except Exception as e:
        logger.error(f"Error fetching metrics for experiment {experiment_id}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_experiment_params(experiment_id: str) -> list[str]:
    """Get all unique parameter names used across all runs in an experiment"""
    logger.info(f"Fetching params for experiment: {experiment_id}")
    try:
        client = MlflowClient()
        runs = client.search_runs(experiment_ids=[experiment_id], max_results=1000)

        param_names = set()
        for run in runs:
            param_names.update(run.data.params.keys())

        logger.info(f"Found {len(param_names)} unique params across {len(runs)} runs")
        return sorted(list(param_names))
    except Exception as e:
        logger.error(f"Error fetching params for experiment {experiment_id}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_experiment_tags(experiment_id: str) -> list[str]:
    """Get all unique tag keys used across all runs in an experiment"""
    logger.info(f"Fetching tags for experiment: {experiment_id}")
    try:
        client = MlflowClient()
        runs = client.search_runs(experiment_ids=[experiment_id], max_results=1000)

        tag_keys = set()
        for run in runs:
            tag_keys.update(run.data.tags.keys())

        logger.info(f"Found {len(tag_keys)} unique tag keys across {len(runs)} runs")
        return sorted(list(tag_keys))
    except Exception as e:
        logger.error(f"Error fetching tags for experiment {experiment_id}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_runs(
    experiment_id: str,
    limit: int = 3,
    offset: int = 0,
    order_by: list[str] | None = None,
) -> list[dict]:
    """Get runs for a specific experiment with full details.

    Each run contains full metrics, params, and tags — keep limit small (3-10)
    to avoid flooding context. Use offset to paginate.

    Args:
        experiment_id: The experiment ID
        limit: Maximum number of runs to return. Keep small — each run is large.
        offset: Number of runs to skip
        order_by: List of sort clauses, e.g. ['metrics.rmse DESC', 'params.lr ASC'].
                  Use backticks for special characters: ['metrics.`trading/loss` DESC']

    Examples:
        get_runs("1", limit=5)
        get_runs("1", order_by=["metrics.accuracy DESC"])
    """
    logger.info(
        f"Fetching runs for experiment {experiment_id} (limit={limit}, offset={offset}, order_by={order_by})"
    )
    try:
        client = MlflowClient()
        # Fetch offset + limit results, then slice
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=order_by,
            max_results=offset + limit,
        )
        # Apply offset
        runs = runs[offset:]
        logger.info(f"Returning {len(runs)} runs (after offset={offset})")

        return [
            {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }
            for run in runs
        ]
    except Exception as e:
        logger.error(f"Error fetching runs for experiment {experiment_id}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_run(run_id: str) -> dict:
    """Get detailed information about a specific run. Run data can be large — avoid fetching many runs at once."""
    logger.info(f"Fetching run details: {run_id}")
    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        logger.info(f"Retrieved run {run_id} with status {run.info.status}")
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "artifact_uri": run.info.artifact_uri,
            "lifecycle_stage": run.info.lifecycle_stage,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
            "inputs": [
                {
                    "name": di.dataset.name,
                    "digest": di.dataset.digest,
                    "source_type": di.dataset.source_type,
                    "tags": {t.key: t.value for t in di.tags},
                }
                for di in (run.inputs.dataset_inputs or [])
            ],
        }
    except Exception as e:
        logger.error(f"Error fetching run {run_id}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_parent_run(run_id: str) -> dict | None:
    """Get the parent run of a nested run. Returns None if the run has no parent.

    Args:
        run_id: The child run ID to find the parent for.
    """
    logger.info(f"Fetching parent run for: {run_id}")
    try:
        client = MlflowClient()
        parent = client.get_parent_run(run_id)
        if parent is None:
            logger.info(f"Run {run_id} has no parent")
            return None
        logger.info(f"Found parent run: {parent.info.run_id}")
        return {
            "run_id": parent.info.run_id,
            "experiment_id": parent.info.experiment_id,
            "status": parent.info.status,
            "start_time": parent.info.start_time,
            "end_time": parent.info.end_time,
            "metrics": parent.data.metrics,
            "params": parent.data.params,
            "tags": parent.data.tags,
        }
    except Exception as e:
        logger.error(f"Error fetching parent run for {run_id}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def query_runs(
    experiment_id: str,
    query: str,
    limit: int = 3,
    offset: int = 0,
    order_by: list[str] | None = None,
) -> list[dict]:
    """Query runs using MLflow's filter syntax with optional sorting.

    Runs can be large. Use wise limits to avoid flooding context.

    Args:
        experiment_id: The experiment ID
        query: MLflow filter string (e.g., 'metrics.accuracy > 0.9')
        limit: Maximum number of runs to return
        offset: Number of runs to skip
        order_by: List of sort clauses, e.g. ['metrics.rmse DESC', 'params.lr ASC'].
                  Use backticks for special characters: ['metrics.`trading/loss` DESC']

    Examples:
        query_runs("1", "metrics.accuracy > 0", order_by=["metrics.accuracy DESC"])
        query_runs("1", "", order_by=["metrics.`f1/score` DESC"])
    """
    logger.info(
        f"Querying runs in experiment {experiment_id} with filter: {query} "
        f"(limit={limit}, offset={offset}, order_by={order_by})"
    )
    try:
        client = MlflowClient()

        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=query,
            order_by=order_by,
            max_results=offset + limit,
        )

        # Apply offset
        runs = runs[offset:]
        logger.info(f"Query returned {len(runs)} runs (after offset={offset})")

        return [
            {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "metrics": run.data.metrics,
                "params": run.data.params,
            }
            for run in runs
        ]
    except Exception as e:
        logger.error(f"Error querying runs with filter '{query}': {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_run_artifacts(run_id: str, path: str = "") -> list[dict]:
    """List artifacts for a specific run. Use 'path' to browse into directories (e.g., 'configs')"""
    logger.info(f"Listing artifacts for run: {run_id} (path: '{path}')")
    try:
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id, path=path)
        logger.info(
            f"Found {len(artifacts)} artifacts for run {run_id} at path '{path}'"
        )
        return [
            {
                "run_id": run_id,
                "path": artifact.path,
                "is_dir": artifact.is_dir,
                "file_size": artifact.file_size,
            }
            for artifact in artifacts
        ]
    except Exception as e:
        logger.error(f"Error listing artifacts for run {run_id} at path '{path}': {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_run_artifact(run_id: str, artifact_path: str) -> dict:
    """Download and return the local path to a specific artifact"""
    logger.info(f"Downloading artifact {artifact_path} from run {run_id}")
    try:
        client = MlflowClient()
        local_path = client.download_artifacts(run_id, artifact_path)
        size_bytes = os.path.getsize(local_path)
        logger.info(f"Artifact downloaded to: {local_path} ({size_bytes} bytes)")
        return {
            "local_path": local_path,
            "run_id": run_id,
            "artifact_path": artifact_path,
            "size_bytes": size_bytes,
        }
    except Exception as e:
        logger.error(
            f"Error downloading artifact {artifact_path} from run {run_id}: {e}"
        )
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_run_metrics(run_id: str) -> dict:
    """Get all metrics for a specific run with their latest values"""
    logger.info(f"Fetching metrics for run: {run_id}")
    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        logger.info(f"Retrieved {len(run.data.metrics)} metrics for run {run_id}")
        return run.data.metrics
    except Exception as e:
        logger.error(f"Error fetching metrics for run {run_id}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_run_metric(run_id: str, metric_name: str) -> list[dict]:
    """Get the full history of a specific metric for a run"""
    logger.info(f"Fetching metric history for {metric_name} in run {run_id}")
    try:
        client = MlflowClient()
        metric_history = client.get_metric_history(run_id, metric_name)
        logger.info(
            f"Retrieved {len(metric_history)} data points for metric {metric_name}"
        )
        return [
            {
                "step": metric.step,
                "timestamp": metric.timestamp,
                "value": metric.value,
            }
            for metric in metric_history
        ]
    except Exception as e:
        logger.error(
            f"Error fetching metric history for {metric_name} in run {run_id}: {e}"
        )
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_best_run(experiment_id: str, metric: str, ascending: bool = False) -> dict:
    """Get the best run by a specific metric (e.g., highest accuracy, lowest loss). Works with metrics containing special characters like '/' (e.g., 'trading/total_profit')"""
    direction = "lowest" if ascending else "highest"
    logger.info(
        f"Finding best run by {metric} ({direction}) in experiment {experiment_id}"
    )
    try:
        client = MlflowClient()
        # Use backticks to escape metric names with special characters (/, -, etc.)
        order_by = f"metrics.`{metric}` {'ASC' if ascending else 'DESC'}"
        runs = client.search_runs(
            experiment_ids=[experiment_id], order_by=[order_by], max_results=1
        )

        if not runs:
            logger.warning(
                f"No runs found with metric {metric} in experiment {experiment_id}"
            )
            raise ValueError(
                f"No runs found in experiment {experiment_id} with metric {metric}"
            )

        best_run = runs[0]
        best_value = best_run.data.metrics.get(metric)
        logger.info(f"Best run: {best_run.info.run_id} with {metric}={best_value}")

        return {
            "run_id": best_run.info.run_id,
            "experiment_id": best_run.info.experiment_id,
            "status": best_run.info.status,
            "metrics": best_run.data.metrics,
            "params": best_run.data.params,
            "tags": best_run.data.tags,
            "best_metric_value": best_value,
        }
    except Exception as e:
        logger.error(
            f"Error finding best run by {metric} in experiment {experiment_id}: {e}"
        )
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def compare_runs(experiment_id: str, run_ids: list[str]) -> dict:
    """Compare runs side-by-side with full metrics and params. Runs can be large — keep the list short."""
    logger.info(f"Comparing {len(run_ids)} runs in experiment {experiment_id}")
    try:
        client = MlflowClient()

        runs_data = []
        all_metrics = set()
        all_params = set()

        for run_id in run_ids:
            run = client.get_run(run_id)
            runs_data.append(run)
            all_metrics.update(run.data.metrics.keys())
            all_params.update(run.data.params.keys())

        comparison = {
            "runs": [
                {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                }
                for run in runs_data
            ],
            "all_metrics": sorted(list(all_metrics)),
            "all_params": sorted(list(all_params)),
        }

        logger.info(
            f"Comparison complete: {len(all_metrics)} metrics, {len(all_params)} params"
        )
        return comparison
    except Exception as e:
        logger.error(f"Error comparing runs: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_registered_models() -> list[dict]:
    """List all registered models in the model registry"""
    logger.info("Fetching all registered models")
    try:
        client = MlflowClient()
        models = client.search_registered_models()
        logger.info(f"Found {len(models)} registered models")

        return [
            {
                "name": model.name,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "description": model.description,
                "tags": model.tags,
            }
            for model in models
        ]
    except Exception as e:
        logger.error(f"Error fetching registered models: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_model_versions(model_name: str) -> list[dict]:
    """Get all versions of a registered model"""
    logger.info(f"Fetching versions for model: {model_name}")
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        logger.info(f"Found {len(versions)} versions for model {model_name}")

        return [
            {
                "name": version.name,
                "version": version.version,
                "creation_timestamp": version.creation_timestamp,
                "last_updated_timestamp": version.last_updated_timestamp,
                "current_stage": version.current_stage,
                "description": version.description,
                "run_id": version.run_id,
                "status": version.status,
                "tags": version.tags,
            }
            for version in versions
        ]
    except Exception as e:
        logger.error(f"Error fetching versions for model {model_name}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_model_version(model_name: str, version: str) -> dict:
    """Get specific model version details (metrics, stage, run_id)"""
    logger.info(f"Fetching model version: {model_name} v{version}")
    try:
        client = MlflowClient()
        model_version = client.get_model_version(model_name, version)

        run_id = model_version.run_id

        if not run_id:
            error_message = f"Model {model_name} v{version} has no associated run"
            logger.error(error_message)
            raise ValueError(error_message)

        run = client.get_run(run_id)
        logger.info(
            f"Retrieved model {model_name} v{version} (stage: {model_version.current_stage})"
        )

        return {
            "name": model_version.name,
            "version": model_version.version,
            "creation_timestamp": model_version.creation_timestamp,
            "last_updated_timestamp": model_version.last_updated_timestamp,
            "current_stage": model_version.current_stage,
            "description": model_version.description,
            "run_id": model_version.run_id,
            "status": model_version.status,
            "tags": model_version.tags,
            "source": model_version.source,
            "run_metrics": run.data.metrics,
            "run_params": run.data.params,
        }
    except Exception as e:
        logger.error(f"Error fetching model version {model_name} v{version}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_registered_model(name: str) -> dict:
    """Get full details of a registered model including all versions and aliases. Can be large for models with many versions.

    Args:
        name: Name of the registered model.
    """
    logger.info(f"Fetching registered model: {name}")
    try:
        client = MlflowClient()
        model = client.get_registered_model(name)
        logger.info(f"Retrieved registered model '{name}'")
        return {
            "name": model.name,
            "creation_timestamp": model.creation_timestamp,
            "last_updated_timestamp": model.last_updated_timestamp,
            "description": model.description,
            "tags": model.tags,
            "aliases": {a.alias: a.version for a in (model.aliases or [])},
            "latest_versions": [
                {
                    "version": v.version,
                    "current_stage": v.current_stage,
                    "status": v.status,
                    "run_id": v.run_id,
                    "description": v.description,
                }
                for v in (model.latest_versions or [])
            ],
        }
    except Exception as e:
        logger.error(f"Error fetching registered model '{name}': {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_model_version_by_alias(name: str, alias: str) -> dict:
    """Get a model version by its alias (e.g. 'champion', 'production').

    Args:
        name: Name of the registered model.
        alias: The alias assigned to the version, e.g. 'champion'.
    """
    logger.info(f"Fetching model version by alias: {name}@{alias}")
    try:
        client = MlflowClient()
        version = client.get_model_version_by_alias(name, alias)
        logger.info(f"Found version {version.version} for alias '{alias}'")
        return {
            "name": version.name,
            "version": version.version,
            "alias": alias,
            "current_stage": version.current_stage,
            "status": version.status,
            "run_id": version.run_id,
            "creation_timestamp": version.creation_timestamp,
            "last_updated_timestamp": version.last_updated_timestamp,
            "description": version.description,
            "tags": version.tags,
            "source": version.source,
        }
    except Exception as e:
        logger.error(f"Error fetching model version by alias {name}@{alias}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_latest_versions(name: str, stages: list[str] | None = None) -> list[dict]:
    """Get latest model versions for each stage (e.g. 'Staging', 'Production').

    Args:
        name: Name of the registered model.
        stages: List of stages to filter by, e.g. ['Production', 'Staging'].
                If None, returns latest version for all stages.
    """
    logger.info(f"Fetching latest versions for model '{name}' (stages={stages})")
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(name, stages=stages)
        logger.info(f"Found {len(versions)} latest versions for '{name}'")
        return [
            {
                "name": v.name,
                "version": v.version,
                "current_stage": v.current_stage,
                "status": v.status,
                "run_id": v.run_id,
                "creation_timestamp": v.creation_timestamp,
                "last_updated_timestamp": v.last_updated_timestamp,
                "description": v.description,
                "tags": v.tags,
            }
            for v in versions
        ]
    except Exception as e:
        logger.error(f"Error fetching latest versions for model '{name}': {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def search_runs_by_tags(
    experiment_id: str, tags: dict, limit: int = 3, offset: int = 0
) -> list[dict]:
    """Find runs with specific tags (e.g., {'team': 'nlp', 'production': 'true'}). Runs can be large — use wise limits."""
    logger.info(
        f"Searching runs by tags in experiment {experiment_id}: {tags} (limit={limit}, offset={offset})"
    )
    try:
        client = MlflowClient()

        # Use backticks to escape special characters (/, -, etc.) or reserved words (case, name, etc.)
        filter_parts = [f"tags.`{key}` = '{value}'" for key, value in tags.items()]
        filter_string = " and ".join(filter_parts)

        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=offset + limit,
        )

        # Apply offset
        runs = runs[offset:]
        logger.info(
            f"Found {len(runs)} runs matching tag filters (after offset={offset})"
        )

        return [
            {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }
            for run in runs
        ]
    except Exception as e:
        logger.error(f"Error searching runs by tags {tags}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_artifact_content(run_id: str, artifact_path: str) -> str:
    """Read and return artifact content (for text/json files)"""
    logger.info(f"Reading artifact content: {artifact_path} from run {run_id}")
    try:
        client = MlflowClient()
        local_path = client.download_artifacts(run_id, artifact_path)

        with open(local_path, "r") as f:
            content = f.read()

        logger.info(f"Read {len(content)} bytes from artifact {artifact_path}")
        return content
    except Exception as e:
        logger.error(
            f"Error reading artifact content {artifact_path} from run {run_id}: {e}"
        )
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def search_logged_models(
    experiment_ids: list[str],
    filter_string: str | None = None,
    max_results: int = 5,
    datasets: list[dict[str, Any]] | None = None,
    order_by: list[dict[str, Any]] | None = None,
) -> list[dict]:
    """Search for logged models across one or more experiments. Results can be large — use wise limits.

    Args:
        experiment_ids: List of experiment IDs to search in (at least one required).
        filter_string: SQL-like filter, e.g. 'metrics.accuracy > 0.9' or "tags.release = 'v1.0'".
                       Multiple conditions use AND only (OR not supported).
        max_results: Maximum number of models to return (default 5).
        datasets: Filter by datasets the model was evaluated on. Each dict must include
                  'name' (str) and 'digest' (str), e.g. [{'name': 'val', 'digest': 'abc123'}].
        order_by: List of sort clauses, each a dict with 'field_name' (str) and 'ascending' (bool),
                  e.g. [{'field_name': 'metrics.accuracy', 'ascending': False}].

    Examples:
        search_logged_models(["1"], filter_string="metrics.accuracy > 0.9")
        search_logged_models(["1", "2"], order_by=[{"field_name": "metrics.f1", "ascending": False}])
    """
    logger.info(
        f"Searching logged models in experiments {experiment_ids} "
        f"(filter={filter_string!r}, max_results={max_results}, order_by={order_by})"
    )
    try:
        client = MlflowClient()
        results = client.search_logged_models(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            datasets=datasets,
            max_results=max_results,
            order_by=order_by,
        )
        logger.info(f"Found {len(results)} logged models")
        return [serialize_logged_model(m) for m in results]
    except Exception as e:
        logger.error(
            f"Error searching logged models in experiments {experiment_ids}: {e}"
        )
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_logged_model(model_id: str) -> dict:
    """Get detailed information about a specific logged model by its ID.

    Args:
        model_id: The logged model ID (obtained from search_logged_models results).
    """
    logger.info(f"Fetching logged model: {model_id}")
    try:
        client = MlflowClient()
        model = client.get_logged_model(model_id)
        logger.info(f"Retrieved logged model {model_id} (status: {model.status})")
        return serialize_logged_model(model)
    except Exception as e:
        logger.error(f"Error fetching logged model {model_id}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
def register_model(
    model_name: str,
    model_uri: str,
    tags: dict[str, Any] | None = None,
) -> dict:
    """Register a model into the model registry. Creates the registered model if it doesn't exist.

    Args:
        model_name: Name for the registered model.
        model_uri: URI of the model to register. Supports:
                   - LoggedModel: 'models:/m-abc123'
                   - Run artifact: 'runs:/run_id/artifact_path'
        tags: Optional dict of tags to set on the model version.

    Examples:
        register_model("btc-classifier", "models:/m-abc123")
        register_model("btc-classifier", "runs:/abc123/model", tags={"framework": "lightgbm"})
    """
    logger.info(f"Registering '{model_name}' from {model_uri}")
    try:
        mv = mlflow.register_model(model_uri, model_name, tags=tags)
        logger.info(f"Registered '{model_name}' v{mv.version}")
        return {
            "name": mv.name,
            "version": mv.version,
            "status": mv.status,
            "source": mv.source,
            "run_id": mv.run_id,
        }
    except Exception as e:
        logger.error(f"Error registering '{model_name}' from {model_uri}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True))
def set_registered_model_tag(name: str, key: str, value: str) -> dict:
    """Set a tag on a registered model (e.g. problem_type, team, framework).

    Args:
        name: Name of the registered model.
        key: Tag key, e.g. 'problem_type', 'team', 'framework'.
        value: Tag value.
    """
    logger.info(f"Setting tag {key}={value!r} on registered model '{name}'")
    try:
        client = MlflowClient()
        client.set_registered_model_tag(name, key, value)
        logger.info(f"Tag {key}={value!r} set on registered model '{name}'")
        return {"success": True, "name": name, "key": key, "value": value}
    except Exception as e:
        logger.error(f"Error setting tag {key} on registered model '{name}': {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True))
def set_model_alias(name: str, alias: str, version: str) -> dict:
    """Assign an alias to a specific model version (e.g. promote best model to 'champion').

    Args:
        name: Name of the registered model.
        alias: Alias to assign, e.g. 'champion', 'production', 'baseline'.
        version: Model version number to assign the alias to.

    Examples:
        set_model_alias("lightgbm", "champion", "3")
    """
    logger.info(f"Setting alias '{alias}' -> {name} v{version}")
    try:
        client = MlflowClient()
        client.set_registered_model_alias(name, alias, version)
        logger.info(f"Alias '{alias}' set to {name} v{version}")
        return {"success": True, "name": name, "alias": alias, "version": version}
    except Exception as e:
        logger.error(f"Error setting alias '{alias}' on {name} v{version}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True))
def set_run_tag(run_id: str, key: str, value: str) -> dict:
    """Set a tag on a run (e.g. annotate best model, flag for review).

    Args:
        run_id: The run ID to tag.
        key: Tag key, e.g. 'best_model', 'reviewed_by'.
        value: Tag value.
    """
    logger.info(f"Setting tag {key}={value!r} on run {run_id}")
    try:
        client = MlflowClient()
        client.set_tag(run_id, key, value)
        logger.info(f"Tag {key}={value!r} set on run {run_id}")
        return {"success": True, "run_id": run_id, "key": key, "value": value}
    except Exception as e:
        logger.error(f"Error setting tag {key} on run {run_id}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True))
def set_experiment_tag(experiment_id: str, key: str, value: str) -> dict:
    """Set a tag on an experiment.

    Args:
        experiment_id: The experiment ID to tag.
        key: Tag key, e.g. 'team', 'status'.
        value: Tag value.
    """
    logger.info(f"Setting tag {key}={value!r} on experiment {experiment_id}")
    try:
        client = MlflowClient()
        client.set_experiment_tag(experiment_id, key, value)
        logger.info(f"Tag {key}={value!r} set on experiment {experiment_id}")
        return {
            "success": True,
            "experiment_id": experiment_id,
            "key": key,
            "value": value,
        }
    except Exception as e:
        logger.error(f"Error setting tag {key} on experiment {experiment_id}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True))
def update_model_version(name: str, version: str, description: str) -> dict:
    """Update the description of a model version.

    Args:
        name: Name of the registered model.
        version: Model version number.
        description: New description text.
    """
    logger.info(f"Updating description for {name} v{version}")
    try:
        client = MlflowClient()
        mv = client.update_model_version(name, version, description=description)
        logger.info(f"Updated description for {name} v{version}")
        return {
            "name": mv.name,
            "version": mv.version,
            "description": mv.description,
            "current_stage": mv.current_stage,
            "status": mv.status,
        }
    except Exception as e:
        logger.error(f"Error updating {name} v{version}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def transition_model_version_stage(
    name: str, version: str, stage: str, archive_existing: bool = False
) -> dict:
    """Transition a model version to a new stage (Staging, Production, Archived).

    Deprecated since MLflow 2.9. Prefer aliases (set_model_alias) and copy_model_version
    for MLflow 3+ workflows. Use this only when working with legacy stage-based deployments.

    Args:
        name: Name of the registered model.
        version: Model version number.
        stage: Target stage: 'Staging', 'Production', or 'Archived'.
        archive_existing: If True, archive existing versions in the target stage.
    """
    logger.info(f"Transitioning {name} v{version} to stage '{stage}'")
    try:
        client = MlflowClient()
        mv = client.transition_model_version_stage(
            name, version, stage, archive_existing_versions=archive_existing
        )
        logger.info(f"Transitioned {name} v{version} to '{stage}'")
        return {
            "name": mv.name,
            "version": mv.version,
            "current_stage": mv.current_stage,
            "status": mv.status,
        }
    except Exception as e:
        logger.error(f"Error transitioning {name} v{version} to '{stage}': {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
def copy_model_version(
    src_model_name: str, src_version: str, dst_model_name: str
) -> dict:
    """Promote a model version to another registered model (MLflow 3 promotion pattern).
    Creates the destination model if it doesn't exist.

    Args:
        src_model_name: Source registered model name.
        src_version: Source model version number.
        dst_model_name: Destination registered model name, e.g. 'my-model-prod'.

    Examples:
        copy_model_version("my-model-dev", "3", "my-model-prod")
    """
    src_model_uri = f"models:/{src_model_name}/{src_version}"
    logger.info(f"Copying model version {src_model_uri} -> {dst_model_name}")
    try:
        client = MlflowClient()
        mv = client.copy_model_version(src_model_uri, dst_model_name)
        logger.info(f"Copied to {dst_model_name} v{mv.version}")
        return {
            "name": mv.name,
            "version": mv.version,
            "status": mv.status,
            "source": mv.source,
            "run_id": mv.run_id,
        }
    except Exception as e:
        logger.error(f"Error copying {src_model_uri} to {dst_model_name}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def delete_model_alias(name: str, alias: str) -> dict:
    """Remove an alias from a registered model (e.g. revoke 'champion'). The alias is permanently removed; the model version itself is not affected.

    Args:
        name: Name of the registered model.
        alias: Alias to remove, e.g. 'champion', 'production'.
    """
    logger.info(f"Deleting alias '{alias}' from model '{name}'")
    try:
        client = MlflowClient()
        client.delete_registered_model_alias(name, alias)
        logger.info(f"Alias '{alias}' removed from model '{name}'")
        return {"success": True, "name": name, "alias": alias}
    except Exception as e:
        logger.error(f"Error deleting alias '{alias}' from model '{name}': {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def delete_model_version(name: str, version: str) -> dict:
    """Delete a specific model version from the registry. Irreversible — the version and its metadata cannot be recovered.

    Args:
        name: Name of the registered model.
        version: Version number to delete.
    """
    logger.info(f"Deleting model version {name} v{version}")
    try:
        client = MlflowClient()
        client.delete_model_version(name, version)
        logger.info(f"Deleted model version {name} v{version}")
        return {"success": True, "name": name, "version": version}
    except Exception as e:
        logger.error(f"Error deleting model version {name} v{version}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def delete_registered_model(name: str) -> dict:
    """Delete an entire registered model and all its versions. Irreversible — all versions, aliases, and tags are permanently removed.

    Args:
        name: Name of the registered model to delete.
    """
    logger.info(f"Deleting registered model '{name}'")
    try:
        client = MlflowClient()
        client.delete_registered_model(name)
        logger.info(f"Deleted registered model '{name}'")
        return {"success": True, "name": name}
    except Exception as e:
        logger.error(f"Error deleting registered model '{name}': {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def delete_run(run_id: str) -> dict:
    """Delete a run. Moves it to the 'deleted' lifecycle stage — not shown in UI or queries, but recoverable via the MLflow API.

    Args:
        run_id: The run ID to delete.
    """
    logger.info(f"Deleting run {run_id}")
    try:
        client = MlflowClient()
        client.delete_run(run_id)
        logger.info(f"Deleted run {run_id}")
        return {"success": True, "run_id": run_id}
    except Exception as e:
        logger.error(f"Error deleting run {run_id}: {e}")
        raise


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def delete_experiment(experiment_id: str) -> dict:
    """Delete an experiment and all its runs. Moves to the 'deleted' lifecycle stage — not shown in UI or queries, but recoverable via the MLflow API.

    Args:
        experiment_id: The experiment ID to delete.
    """
    logger.info(f"Deleting experiment {experiment_id}")
    try:
        client = MlflowClient()
        client.delete_experiment(experiment_id)
        logger.info(f"Deleted experiment {experiment_id}")
        return {"success": True, "experiment_id": experiment_id}
    except Exception as e:
        logger.error(f"Error deleting experiment {experiment_id}: {e}")
        raise


@mcp.prompt()
def compare_runs_by_ids(experiment_id: str, run_ids: list[str]) -> str:
    """Compare specific runs side-by-side by their IDs.

    Args:
        experiment_id: The experiment the runs belong to.
        run_ids: List of run IDs to compare.
    """
    return f"""Compare these runs from experiment {experiment_id} side-by-side: {run_ids}.

1. Use get_run for each run ID to fetch full details.
2. Use compare_runs to get a structured side-by-side comparison of all metrics and params.
3. Highlight which metrics differ most, which params are responsible for the differences, and which run wins overall. Explain why."""


@mcp.prompt()
def find_best_run(experiment_id: str, metric: str) -> str:
    """Find and analyze the best run in an experiment by a given metric.

    Args:
        experiment_id: The experiment ID to analyze.
        metric: Metric to rank by, e.g. 'test/recall', 'test/f1'.
    """
    return f"""Find the best run in experiment {experiment_id} by {metric}.

Get the top run by that metric, then fetch the top 5 for comparison.
Compare them side-by-side and summarize: which run wins, what params made the difference, any metric tradeoffs worth noting."""


@mcp.prompt()
def promote_best_model(experiment_id: str, metric: str, model_name: str) -> str:
    """Find the best model and promote it to the registry with tags and alias.

    Args:
        experiment_id: The experiment ID to search in.
        metric: Metric to rank models by, e.g. 'test/recall', 'test/f1'.
        model_name: Name to register the model under in the registry.
    """
    return f"""Promote the best model from experiment {experiment_id} to the registry as "{model_name}".

Find the best logged model by {metric}. Register it as "{model_name}" with a selection_metric tag.
Add relevant model-level tags (problem_type, framework, asset). Tag the source run as registered.
Assign the "champion" alias to the new version.
Ask the user if they want to copy it to a separate production model entry.
Summarize what was done."""


_AUDIT_PROMPT = """You are a senior MLOps consultant auditing this MLflow deployment against Google/Databricks industry best practices.

<instructions>
  <step name="gather_data">
    Call tools in this order to collect evidence before evaluating anything.

    1. get_experiments() — list all experiments; note names, count, naming patterns
    2. For up to 3 representative experiments:
       - get_experiment_metrics(experiment_id)
       - get_experiment_params(experiment_id)
       - get_experiment_tags(experiment_id)
       - get_runs(experiment_id, limit=5)
    3. For up to 3 individual runs sampled above:
       - get_run(run_id) — inspect params, metrics, tags, artifact_uri, inputs
       - get_run_artifacts(run_id) — inspect artifact structure and file names
    4. get_registered_models()
    5. For up to 2 registered models:
       - get_registered_model(name) — check aliases
       - get_model_versions(name) — check versioning, descriptions, stages

    <note>If the instance has very few runs, treat missing data as "unknown" not "bad". Score only what you can observe.</note>
  </step>

  <step name="evaluate">
    Score each category 1–10 and fill in the structured blocks below.

    <category name="experiment_organization">
      <best_practice>Hierarchical dot-notation names (team.project.task); consistent convention; no generic names like "test", "Default", "my_experiment"; searchable by business context.</best_practice>
      <whats_good></whats_good>
      <whats_bad></whats_bad>
      <what_to_improve></what_to_improve>
      <score>X/10</score>
    </category>

    <category name="parameter_logging">
      <best_practice>All hyperparameters logged at run start via log_params(); descriptive names ("learning_rate" not "lr"); parent-child run structure for tuning sweeps.</best_practice>
      <whats_good></whats_good>
      <whats_bad></whats_bad>
      <what_to_improve></what_to_improve>
      <score>X/10</score>
    </category>

    <category name="metric_logging">
      <best_practice>train/val/test metrics logged separately with consistent prefixes; per-step logging for training progression (not just final values); business metrics alongside technical ones; no mixed naming ("acc" vs "accuracy").</best_practice>
      <whats_good></whats_good>
      <whats_bad></whats_bad>
      <what_to_improve></what_to_improve>
      <score>X/10</score>
    </category>

    <category name="tagging_strategy">
      <best_practice>Tags = mutable categorical metadata for filtering (env, model_type, dataset_version, team); params = immutable training config; never store hyperparameters as tags; consistent naming convention.</best_practice>
      <whats_good></whats_good>
      <whats_bad></whats_bad>
      <what_to_improve></what_to_improve>
      <score>X/10</score>
    </category>

    <category name="artifact_management">
      <best_practice>log_model() not just log_artifact(); artifact_uri points to cloud storage (s3://, gs://) or uses the MLflow proxy (mlflow-artifacts:/) backed by cloud — not bare file:// local paths in production; model cards, feature schemas, validation results included; organized paths (model/, data/, results/).</best_practice>
      <whats_good></whats_good>
      <whats_bad></whats_bad>
      <what_to_improve></what_to_improve>
      <score>X/10</score>
    </category>

    <category name="model_registry">
      <best_practice>Descriptive model names (churn_xgboost, not model1); version descriptions capture key differences; aliases used (champion, production) for routing; versions linked to source runs.</best_practice>
      <whats_good></whats_good>
      <whats_bad></whats_bad>
      <what_to_improve></what_to_improve>
      <score>X/10</score>
    </category>

    <category name="reproducibility">
      <best_practice>Git commit SHA in tags (mlflow.source.git.commit); random seeds logged as params; datasets tracked via mlflow.log_input() (visible in run inputs); dependency versions captured (requirements.txt in artifacts).</best_practice>
      <whats_good></whats_good>
      <whats_bad></whats_bad>
      <what_to_improve></what_to_improve>
      <score>X/10</score>
    </category>
  </step>

  <step name="report">
    Present a summary table:

    | Category | Score | Top Issue |
    |---|---|---|
    | Experiment Organization | X/10 | … |
    | Parameter Logging | X/10 | … |
    | Metric Logging | X/10 | … |
    | Tagging Strategy | X/10 | … |
    | Artifact Management | X/10 | … |
    | Model Registry | X/10 | … |
    | Reproducibility | X/10 | … |
    | **Mean Score** | **X.X/10** | |

    Then list the 3 most impactful improvements with specific actionable steps.
  </step>
</instructions>
"""


@mcp.prompt()
def audit_mlflow_setup() -> str:
    """Audit the current MLflow setup against industry best practices.

    Evaluates experiment organization, run logging quality, tagging strategy,
    artifact management, model registry usage, production workflow, and reproducibility.
    Each category is scored 1–10. Ends with a mean score and improvement roadmap.
    """
    return _AUDIT_PROMPT


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def health() -> dict:
    """Check MLflow server health and connectivity"""
    logger.info(f"Checking MLflow server health at {MLFLOW_TRACKING_URI}")
    try:
        client = MlflowClient()
        client.search_experiments(max_results=1)
        logger.info("MLflow server health check: HEALTHY")
        return {
            "status": "healthy",
            "tracking_uri": MLFLOW_TRACKING_URI,
            "message": "Successfully connected to MLflow server",
        }
    except Exception as e:
        logger.error(f"MLflow server health check: UNHEALTHY - {e}")
        return {
            "status": "unhealthy",
            "tracking_uri": MLFLOW_TRACKING_URI,
            "error": str(e),
        }


def main():
    logger.info("Starting MLflow MCP server")
    try:
        mcp.run("stdio")
    except KeyboardInterrupt:
        logger.info("MLflow MCP server stopped by user")
    except Exception as e:
        logger.error(f"MLflow MCP server error: {e}")
        raise
    finally:
        logger.info("MLflow MCP server shutdown")


if __name__ == "__main__":
    main()
