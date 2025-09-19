from __future__ import annotations

from typing import Dict, List, Optional

import openai

from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow

from .react_agent import EmailSearchAgent
from .utils import AnswerModel, FinalRubric, QueryModel, judge_correctness

SYSTEM_PROMPT = """You are an email search agent. You are given a user query and a list of tools you can use to search the user's email. Use the tools to search the user's emails and find the answer to the user's query. You may take up to {max_turns} turns to find the answer, so if your first seach doesn't find the answer, you can try with different keywords.
Always describe what you see and plan your next steps clearly. When taking actions, explain what you're doing and why. When the answer to the task is found, call `generate_response` to finish the process. Only call `generate_response` when answer is found. You should not respond any next steps in `generate_response`. Complete all steps and then call `generate_response`.

User's email address is {inbox_address}
Today's date is {query_date}
"""


@WORKFLOWS.register_module("email_search_workflow")
class EmailSearchWorkflow(Workflow):
    """
    Multi-turn Email Search workflow (ReAct-style tool use).
    """

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        try:
            import agentscope
            from agentscope.service import ServiceToolkit
        except ImportError as e:
            error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            self.logger.error(error_message)
            raise ImportError(error_message)

        # get openai client from model
        self.openai_client = model.get_openai_client()
        self.model_name = self.openai_client.model_path
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

        temperature = self.rollout_args.get("temperature", 1.0)
        max_tokens = 4096

        agentscope.init(
            model_configs=[
                {
                    "model_type": "openai_chat",
                    "config_name": "react_model",
                    "model_name": self.model_name,
                    "api_key": "EMPTY",
                    "generate_args": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    "use_openai_formatter": True,
                }
            ],
            disable_saving=True,
        )

        self.toolkit = ServiceToolkit()
        self.reset(task)

    @property
    def repeatable(self) -> bool:
        return False

    @property
    def resettable(self):
        return True

    def reset(self, task: Task):
        self.query = QueryModel.model_validate(task.raw_task)
        self.task_desc = task.task_desc  # question
        self.truth = task.truth  # ground truth answer

        self.workflow_args = task.workflow_args
        self.max_turns = int(self.workflow_args.get("max_turns", 10))
        self.tool_obs_char_limit = int(self.workflow_args.get("tool_obs_char_limit", 2000))
        self.reward_fn_args = task.reward_fn_args or {}

        self.system_prompt = SYSTEM_PROMPT.format(
            max_turns=self.max_turns,
            inbox_address=self.query.inbox_address,
            query_date=self.query.query_date,
        )

        self.agent = EmailSearchAgent(
            name="react_agent",
            sys_prompt=self.system_prompt,
            model_config_name="react_model",
            service_toolkit=self.toolkit,
            max_iters=self.max_turns,
            verbose=False,
        )
        # we set the openai client to the agent's model
        self.agent.model.client = self.openai_client
        self.agent.message_id_list = []
        self.agent.ever_read_message_ids = []

    def run(self):
        # make sure that we have the correct import
        try:
            from agentscope.message import Msg
        except ImportError as e:
            error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            self.logger.error(error_message)
            raise ImportError(error_message)

        # provide the task to the react agent
        msg = Msg("user", self.task_desc, role="user")

        response = self.agent.reply(
            msg,
            structured_model=AnswerModel,
        )
        if response.metadata is None:
            answer_and_sources = {"answer": response.content, "sources": []}
        else:
            answer_and_sources = response.metadata

        experiences = self.model.extract_experience_from_history(clear_history=True)
        self.actual_turns = len(
            experiences
        )  # NOTE: this metrics works only if the agent calls model once in each turn

        reward_dict = self.calculate_reward(answer_and_sources)
        reward = sum(reward_dict.values())

        for i, experience in enumerate(experiences):
            experience.eid.step = i
            experience.reward = reward
            if experience.metrics is None:
                experience.metrics = {}
            experience.metrics.update({"actual_turns": self.actual_turns})
            experience.metrics.update(reward_dict)
        self.logger.info(
            f"return experience len: {len(experiences)}, final step reward: {experiences[-1].reward}"
        )
        return experiences

    def calculate_reward(self, answer_and_sources: Dict) -> Dict[str, float]:
        """Ref: calculate_reward in https://github.com/OpenPipe/ART/blob/main/dev/art-e/art_e/rollout.py#L64"""
        try:
            answer = answer_and_sources.get("answer", None)
            sources = answer_and_sources.get("sources", [])
        except Exception as e:
            self.logger.error(f"Error extracting answer and sources: {e}")
            result = {"accuracy": 0.0, "format": -1.0}
            return result

        if answer is None:
            result = {"accuracy": 0.0, "format": -1.0}
            return result

        if not self.reward_fn_args.get("llm_as_a_judge", True):
            result = {"accuracy": float(answer.lower() in self.truth.lower()), "format": 0.0}
            return result

        rubric = FinalRubric()
        rubric.attempted_answer = answer is not None and answer != ""
        rubric.returned_i_dont_know = answer == "I don't know"
        if len(self.query.message_ids) > 0:
            rubric.ever_found_right_email = self.query.message_ids[0] in self.agent.message_id_list
            rubric.ever_read_right_email = (
                self.query.message_ids[0] in self.agent.ever_read_message_ids
            )
            rubric.sources_correct = self.query.message_ids[0] in sources
        rubric.num_sources = len(sources)
        rubric.num_turns = self.actual_turns
        self.logger.debug(f"Rubric: {rubric.model_dump()}")

        try:
            judge_model = self.auxiliary_models[0] if self.auxiliary_models else None
            judge_response = judge_correctness(answer, self.query, judge_model)
            rubric.answer_correct = judge_response

        except Exception as e:
            self.logger.error(f"Error judging correctness: {e}")
            rubric.answer_correct = False

        # Note: make sure all possible partial rewards always sum to less than 0.5.
        partial_rewards = 0
        partial_rewards += 0.1 if rubric.ever_found_right_email else 0
        partial_rewards += 0.1 if rubric.ever_read_right_email else 0
        partial_rewards += 0.1 if rubric.sources_correct else 0

        # No formatting error, but wrong answer: reward will be -1 to 0
        if rubric.attempted_answer and not rubric.answer_correct:
            result = {"accuracy": -1.0, "format": partial_rewards}
            return result

        # Returned no answer at all: reward will be 0 to 1
        if rubric.returned_i_dont_know or rubric.ran_out_of_turns:
            result = {"accuracy": 0.0, "format": partial_rewards}
            return result

        # Answer is correct: reward will be 1 to 2
        if rubric.answer_correct:
            # Partial credit calculation is different for correct answers.

            reward = 1
            reward += 0.3 if rubric.sources_correct else 0

            # Extra credit for not including extra sources.
            reward += 0.1 / rubric.num_sources if rubric.num_sources > 0 else 0

            # Extra credit for being faster (taking fewer turns).
            reward += 0.1 * (1 - rubric.num_turns / self.max_turns)
            result = {"accuracy": 1.0, "format": reward}
            return result

        self.logger.error(f"Rubric {rubric} not handled properly")
        raise ValueError("Rubric is not handled properly")
