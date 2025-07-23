from typing import List

from trinity.common.workflows import WORKFLOWS, Workflow


@WORKFLOWS.register_module("my_workflow")
class MyWorkflow(Workflow):
    def __init__(self, *, task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)

    def run(self) -> List:
        return ["Hello world", "Hi"]
