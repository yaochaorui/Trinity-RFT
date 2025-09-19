"""Monitor"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import wandb
except ImportError:
    wandb = None

try:
    import mlflow
except ImportError:
    mlflow = None
from torch.utils.tensorboard import SummaryWriter

from trinity.common.config import Config
from trinity.utils.log import get_logger
from trinity.utils.registry import Registry

MONITOR = Registry("monitor")


def gather_metrics(metric_list: List[Dict], prefix: str) -> Dict:
    df = pd.DataFrame(metric_list)
    numeric_df = df.select_dtypes(include=[np.number])
    stats_df = numeric_df.agg(["mean", "max", "min"])
    metric = {}
    for col in stats_df.columns:
        metric[f"{prefix}/{col}/mean"] = stats_df.loc["mean", col].item()
        metric[f"{prefix}/{col}/max"] = stats_df.loc["max", col].item()
        metric[f"{prefix}/{col}/min"] = stats_df.loc["min", col].item()
    return metric


class Monitor(ABC):
    """Monitor"""

    def __init__(
        self,
        project: str,
        name: str,
        role: str,
        config: Config = None,  # pass the global Config for recording
    ) -> None:
        self.project = project
        self.name = name
        self.role = role
        self.config = config

    @abstractmethod
    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        """Log a table"""

    @abstractmethod
    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""

    @abstractmethod
    def close(self) -> None:
        """Close the monitor"""

    def __del__(self) -> None:
        self.close()

    def calculate_metrics(
        self, data: dict[str, Union[List[float], float]], prefix: Optional[str] = None
    ) -> dict[str, float]:
        metrics = {}
        for key, val in data.items():
            if prefix is not None:
                key = f"{prefix}/{key}"

            if isinstance(val, List):
                if len(val) > 1:
                    metrics[f"{key}/mean"] = np.mean(val)
                    metrics[f"{key}/max"] = np.amax(val)
                    metrics[f"{key}/min"] = np.amin(val)
                elif len(val) == 1:
                    metrics[key] = val[0]
            else:
                metrics[key] = val
        return metrics

    @classmethod
    def default_args(cls) -> Dict:
        """Return default arguments for the monitor."""
        return {}


@MONITOR.register_module("tensorboard")
class TensorboardMonitor(Monitor):
    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        self.tensorboard_dir = os.path.join(config.monitor.cache_dir, "tensorboard", role)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.logger = SummaryWriter(self.tensorboard_dir)
        self.console_logger = get_logger(__name__, in_ray_actor=True)

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        pass

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        for key in data:
            self.logger.add_scalar(key, data[key], step)
        self.console_logger.info(f"Step {step}: {data}")

    def close(self) -> None:
        self.logger.close()


@MONITOR.register_module("wandb")
class WandbMonitor(Monitor):
    """Monitor with Weights & Biases.

    Args:
        base_url (`Optional[str]`): The base URL of the W&B server. If not provided, use the environment variable `WANDB_BASE_URL`.
        api_key (`Optional[str]`): The API key for W&B. If not provided, use the environment variable `WANDB_API_KEY`.
    """

    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        assert wandb is not None, "wandb is not installed. Please install it to use WandbMonitor."
        if not group:
            group = name
        monitor_args = config.monitor.monitor_args or {}
        if base_url := monitor_args.get("base_url"):
            os.environ["WANDB_BASE_URL"] = base_url
        if api_key := monitor_args.get("api_key"):
            os.environ["WANDB_API_KEY"] = api_key
        self.logger = wandb.init(
            project=project,
            group=group,
            name=f"{name}_{role}",
            tags=[role],
            config=config,
            save_code=False,
        )
        self.console_logger = get_logger(__name__, in_ray_actor=True)

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        experiences_table = wandb.Table(dataframe=experiences_table)
        self.log(data={table_name: experiences_table}, step=step)

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        self.logger.log(data, step=step, commit=commit)
        self.console_logger.info(f"Step {step}: {data}")

    def close(self) -> None:
        self.logger.finish()

    @classmethod
    def default_args(cls) -> Dict:
        """Return default arguments for the monitor."""
        return {
            "base_url": None,
            "api_key": None,
        }


@MONITOR.register_module("mlflow")
class MlflowMonitor(Monitor):
    """Monitor with MLflow.

    Args:
        uri (`Optional[str]`): The tracking server URI. If not provided, the default is `http://localhost:5000`.
        username (`Optional[str]`): The username to login. If not provided, the default is `None`.
        password (`Optional[str]`): The password to login. If not provided, the default is `None`.
    """

    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        assert (
            mlflow is not None
        ), "mlflow is not installed. Please install it to use MlflowMonitor."
        monitor_args = config.monitor.monitor_args or {}
        if username := monitor_args.get("username"):
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
        if password := monitor_args.get("password"):
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        mlflow.set_tracking_uri(config.monitor.monitor_args.get("uri", "http://localhost:5000"))
        mlflow.set_experiment(project)
        mlflow.start_run(
            run_name=f"{name}_{role}",
            tags={
                "group": group,
                "role": role,
            },
        )
        mlflow.log_params(config.flatten())
        self.console_logger = get_logger(__name__, in_ray_actor=True)

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        experiences_table["step"] = step
        mlflow.log_table(data=experiences_table, artifact_file=f"{table_name}.json")

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        mlflow.log_metrics(metrics=data, step=step)
        self.console_logger.info(f"Step {step}: {data}")

    def close(self) -> None:
        mlflow.end_run()

    @classmethod
    def default_args(cls) -> Dict:
        """Return default arguments for the monitor."""
        return {
            "uri": "http://localhost:5000",
            "username": None,
            "password": None,
        }
