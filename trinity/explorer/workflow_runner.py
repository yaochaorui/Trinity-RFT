# -*- coding: utf-8 -*-
"""The Workflow Runner Module."""
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

from trinity.buffer import get_buffer_writer
from trinity.common.config import Config
from trinity.common.experience import Experience
from trinity.common.models.model import InferenceModel, ModelWrapper
from trinity.common.workflows import Task
from trinity.utils.log import get_logger


@dataclass(frozen=True)
class Status:
    """Status of the task running result."""

    ok: bool
    metric: dict[str, float]
    message: Optional[str] = None


class WorkflowRunner:
    """A Ray remote actor to run the workflow and generate experiences."""

    def __init__(
        self,
        config: Config,
        model: InferenceModel,
        auxiliary_models: Optional[List[InferenceModel]] = None,
        runner_id: Optional[int] = None,
    ) -> None:
        self.config = config
        self.experience_buffer = get_buffer_writer(
            self.config.buffer.explorer_output,  # type: ignore
            self.config.buffer,
        )
        self.model = model
        self.model_wrapper = ModelWrapper(
            model,
            config.explorer.rollout_model.engine_type,
        )
        self.auxiliary_models = []
        if auxiliary_models is not None:
            for model in auxiliary_models:
                api_client = ModelWrapper(
                    model,
                    "vllm_async",
                ).get_openai_client()
                self.auxiliary_models.append(api_client)
        self.logger = get_logger(__name__)
        self.workflow_instance = None
        self.runner_id = runner_id
        self.return_experiences = self.config.explorer.collect_experiences

    def is_alive(self):
        return True

    def _run_task(self, task: Task) -> List[Experience]:
        """Init workflow from the task and run it."""
        if task.workflow is None:
            raise ValueError("Workflow is not set in the task.")
        if (
            self.workflow_instance is None
            or not self.workflow_instance.__class__ == task.workflow
            or not self.workflow_instance.resettable
        ):
            self.workflow_instance = task.to_workflow(self.model_wrapper, self.auxiliary_models)
        else:
            self.workflow_instance.reset(task)
        return self.workflow_instance.run()

    def run_task(self, task: Task) -> Tuple[Status, List[Experience]]:
        """Run the task and return the states."""
        # TODO: avoid sending the experiences back to the scheduler to reduce the communication overhead
        try:
            st = time.time()
            exps = self._run_task(task)
            assert exps is not None and len(exps) > 0, "An empty experience is generated"
            metrics: dict[str, List[float]] = defaultdict(list)
            # set group id
            for _, exp in enumerate(exps):
                exp.eid.batch = task.batch_id
                exp.eid.task = task.task_id
                if not hasattr(exp, "info") or exp.info is None:
                    exp.info = {}
                exp.info["model_version"] = self.model_wrapper.model_version
                exp.info["use_count"] = 0

                if not hasattr(exp, "metrics") or exp.metrics is None:
                    exp.metrics = {}
                for k, v in exp.metrics.items():
                    metrics[k].append(v)
            # We get the average of metrics into the state
            metric = {}
            metric["time_per_task"] = time.time() - st
            if metrics:
                for k, v in metrics.items():
                    metric[k] = sum(v) / len(v)  # type: ignore

            if task.is_eval:
                # If the task is an evaluation task, we do not record the experiences to the buffer
                return Status(True, metric=metric), []
            elif self.return_experiences:
                return Status(True, metric=metric), exps
            else:
                self.experience_buffer.write(exps)
                return Status(True, metric=metric), []

        except Exception as e:
            error_trace_back = traceback.format_exc()
            self.logger.error(f"WorkflowRunner run task error: {e}\nTraceback:\n{error_trace_back}")
            return Status(False, metric={"time_per_task": time.time() - st}, message=str(e)), []
