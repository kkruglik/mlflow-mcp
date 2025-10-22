import logging
import os

import mlflow
from mcp.server.fastmcp import FastMCP
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

if not MLFLOW_TRACKING_URI:
    logger.error("MLFLOW_TRACKING_URI is not set")
    exit(1)


mcp = FastMCP("mlflow")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logger.info(f"MLflow MCP server initialized with tracking URI: {MLFLOW_TRACKING_URI}")


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
def get_runs(
    experiment_id: str, limit: int = 3, offset: int = 0, order_by: str = None
) -> list[dict]:
    """Get runs for a specific experiment with full details.

    Args:
        experiment_id: The experiment ID
        limit: Maximum number of runs to return
        offset: Number of runs to skip
        order_by: Column to order by (e.g., 'metrics.rmse DESC', 'params.lr ASC')
                  Use backticks for special characters: 'metrics.`trading/loss` DESC'

    Examples:
        get_runs("1", limit=5)
        get_runs("1", order_by="metrics.accuracy DESC")
    """
    logger.info(
        f"Fetching runs for experiment {experiment_id} (limit={limit}, offset={offset}, order_by={order_by})"
    )
    try:
        client = MlflowClient()
        # Fetch offset + limit results, then slice
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=[order_by] if order_by else None,
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


@mcp.tool()
def get_run(run_id: str) -> dict:
    """Get detailed information about a specific run"""
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
        }
    except Exception as e:
        logger.error(f"Error fetching run {run_id}: {e}")
        raise


@mcp.tool()
def query_runs(
    experiment_id: str,
    query: str,
    limit: int = 3,
    offset: int = 0,
    order_by: str = None,
) -> list[dict]:
    """Query runs using MLflow's filter syntax with optional sorting.

    Args:
        experiment_id: The experiment ID
        query: MLflow filter string (e.g., 'metrics.accuracy > 0.9')
        limit: Maximum number of runs to return
        offset: Number of runs to skip
        order_by: Column to order by (e.g., 'metrics.rmse DESC', 'params.lr ASC')
                  Use backticks for special characters: 'metrics.`trading/loss` DESC'

    Examples:
        query_runs("1", "metrics.accuracy > 0", order_by="metrics.accuracy DESC")
        query_runs("1", "", order_by="metrics.`f1/score` DESC")
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
            order_by=[order_by] if order_by else None,
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


@mcp.tool()
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
                "path": artifact.path,
                "is_dir": artifact.is_dir,
                "file_size": artifact.file_size,
            }
            for artifact in artifacts
        ]
    except Exception as e:
        logger.error(f"Error listing artifacts for run {run_id} at path '{path}': {e}")
        raise


@mcp.tool()
def get_run_artifact(run_id: str, artifact_path: str) -> str:
    """Download and return the local path to a specific artifact"""
    logger.info(f"Downloading artifact {artifact_path} from run {run_id}")
    try:
        client = MlflowClient()
        local_path = client.download_artifacts(run_id, artifact_path)
        logger.info(f"Artifact downloaded to: {local_path}")
        return local_path
    except Exception as e:
        logger.error(
            f"Error downloading artifact {artifact_path} from run {run_id}: {e}"
        )
        raise


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
def compare_runs(experiment_id: str, run_ids: list[str]) -> dict:
    """Compare runs side-by-side with full metrics and params. Use get_run() for individual run details."""
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
def search_runs_by_tags(
    experiment_id: str, tags: dict, limit: int = 3, offset: int = 0
) -> list[dict]:
    """Find runs with specific tags (e.g., {'team': 'nlp', 'production': 'true'})."""
    logger.info(
        f"Searching runs by tags in experiment {experiment_id}: {tags} (limit={limit}, offset={offset})"
    )
    try:
        client = MlflowClient()

        filter_parts = [f"tags.{key} = '{value}'" for key, value in tags.items()]
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


@mcp.tool()
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


@mcp.tool()
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
