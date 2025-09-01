"""REC advantage computation
"""

import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from verl import DataProto

from trinity.algorithm.advantage_fn.advantage_fn import (
    ADVANTAGE_FN,
    AdvantageFn,
    GroupAdvantage,
)
from trinity.common.experience import Experience, group_by
from trinity.utils.annotations import Deprecated
from trinity.utils.monitor import gather_metrics


@Deprecated
@ADVANTAGE_FN.register_module("rec_verl")
class RECAdvantageFn(AdvantageFn):
    """REC advantage computation"""

    def __init__(
        self,
        epsilon: float = 1e-6,
        std_normalize: Optional[bool] = False,
    ) -> None:
        self.epsilon = epsilon
        self.std_normalize = std_normalize

    def __call__(
        self,
        exps: DataProto,
        **kwargs,
    ) -> Tuple[DataProto, Dict]:
        """
        Compute advantage for REC, operating only on Outcome reward
        (with only one scalar reward for each response).

            token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            eos_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            scores: `(torch.Tensor)`
                shape: (bs, response_length)
        """
        token_level_rewards = exps.batch["token_level_rewards"]
        eos_mask = exps.batch["response_mask"]
        index = exps.non_tensor_batch["uid"]
        epsilon = self.epsilon

        response_length = token_level_rewards.shape[-1]
        scores = token_level_rewards.sum(dim=-1)

        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}

        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                elif len(id2score[idx]) > 1:
                    id2mean[idx] = torch.mean(torch.tensor(id2score[idx], dtype=torch.float32))
                    id2std[idx] = torch.std(torch.tensor(id2score[idx], dtype=torch.float32))
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            for i in range(bsz):
                if self.std_normalize:
                    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                else:
                    scores[i] = scores[i] - id2mean[index[i]]
            scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

        exps.batch["advantages"] = scores
        exps.batch["returns"] = scores

        metrics = {
            # TODO: add meaningful metrics
        }

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "epsilon": 1e-6,
        }


@ADVANTAGE_FN.register_module("rec")
class RECGroupedAdvantage(GroupAdvantage):
    """An advantage class that calculates REC advantages."""

    def __init__(
        self,
        epsilon: float = 1e-6,
        std_normalize: Optional[bool] = False,
    ) -> None:
        """Initialize the REC advantage function.

        Args:
            epsilon (float): A small value to avoid division by zero.
            std_normalize (Optional[bool]): If provided, normalize the advantage with group-level reward standard deviation.

        """
        self.epsilon = epsilon

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        metrics = {}
        with torch.no_grad():
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)  # set to 1.0 to avoid division by zero
            else:
                rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)

            for exp in exps:
                if self.std_normalize:
                    score = (exp.reward - group_reward_mean) / (group_reward_std + self.epsilon)
                else:
                    score = exp.reward - group_reward_mean
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            metrics["reward_mean"] = group_reward_mean.item()
            metrics["reward_std"] = group_reward_std.item()

        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"epsilon": 1e-6}
