from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from config import Config
from utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrackingConfig:
    experiment_name: str = Config.MLFLOW_EXPERIMENT_NAME
    run_name: str | None = None
    tracking_uri: str | None = Config.MLFLOW_TRACKING_URI
    enabled: bool = Config.MLFLOW_ENABLED
    tags: dict[str, str] = field(default_factory=dict)


class TrackingAdapter(ABC):
    @abstractmethod
    def start_run(self, config: TrackingConfig | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float | int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_tags(self, tags: dict[str, str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def end_run(self) -> None:
        raise NotImplementedError


class NoOpTrackingAdapter(TrackingAdapter):
    def start_run(self, config: TrackingConfig | None = None) -> None:
        logger.debug("NoOpTrackingAdapter.start_run called.")

    def log_params(self, params: dict[str, Any]) -> None:
        logger.debug("NoOpTrackingAdapter.log_params called with %s", params)

    def log_metrics(self, metrics: dict[str, float | int]) -> None:
        logger.debug("NoOpTrackingAdapter.log_metrics called with %s", metrics)

    def set_tags(self, tags: dict[str, str]) -> None:
        logger.debug("NoOpTrackingAdapter.set_tags called with %s", tags)

    def end_run(self) -> None:
        logger.debug("NoOpTrackingAdapter.end_run called.")


class MLflowTrackingAdapter(TrackingAdapter):
    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._client: Any = None
        self._active = False

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import mlflow

            self._client = mlflow
        except ImportError as exc:
            raise RuntimeError("MLflow is not installed. Install the optional mlflow dependency to enable tracking.") from exc

    def _start_run(self) -> None:
        if not self.config.enabled:
            logger.debug("MLflow disabled via config.")
            return
        if self._active:
            return
        self._ensure_client()
        if self.config.tracking_uri:
            self._client.set_tracking_uri(self.config.tracking_uri)
        self._client.set_experiment(self.config.experiment_name)
        self._client.start_run(run_name=self.config.run_name)
        self._active = True
        if self.config.tags:
            self._client.set_tags(self.config.tags)

    def start_run(self, config: TrackingConfig | None = None) -> None:
        if config is not None:
            object.__setattr__(self, "config", config)
        self._start_run()

    def log_params(self, params: dict[str, Any]) -> None:
        self._start_run()
        if not self._active:
            return
        self._client.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float | int]) -> None:
        self._start_run()
        if not self._active:
            return
        self._client.log_metrics({k: float(v) for k, v in metrics.items()})

    def set_tags(self, tags: dict[str, str]) -> None:
        self._start_run()
        if not self._active:
            return
        self._client.set_tags(tags)

    def end_run(self) -> None:
        if not self._active:
            return
        self._client.end_run()
        self._active = False
