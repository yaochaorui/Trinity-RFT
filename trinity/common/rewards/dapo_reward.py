# -*- coding: utf-8 -*-
"""Reward Function with Overlong Reward Shaping described in DAPO (https://arxiv.org/pdf/2503.14476)"""
from typing import Optional

import torch

from trinity.common.rewards.reward_fn import REWARD_FUNCTIONS, RewardFn
from trinity.utils.eval_utils import compute_score
from trinity.utils.log import get_logger

logger = get_logger(__name__)


@REWARD_FUNCTIONS.register_module("math_dapo_reward")
class MathDAPORewardFn(RewardFn):
    """A reward function that follows the definition in DAPO for math task."""

    def __init__(
        self,
        enable_overlong_penalty: Optional[bool] = None,
        penalty_factor: Optional[float] = None,
        max_response_length: Optional[int] = None,
        cache_length: Optional[int] = None,
    ) -> None:
        self.enable_overlong_penalty = enable_overlong_penalty
        self.penalty_factor = penalty_factor
        self.max_response_length = max_response_length
        self.cache_length = cache_length

    def __call__(  # type: ignore
        self,
        response: str,
        response_token: torch.Tensor,
        truth: Optional[str] = None,
        **kwargs,
    ) -> dict[str, float]:
        accuracy_score = compute_score(response, truth)

        format_score = 0.0

        if self.enable_overlong_penalty:
            format_score = self.compute_overlong_penalty(response_token)

        return {
            "accuracy": accuracy_score,
            "format_score": format_score,
        }

    def compute_overlong_penalty(self, response_token):
        assert (
            self.max_response_length is not None
            and self.cache_length is not None
            and self.penalty_factor is not None
        ), "When enable_overlong_penalty = true, max_response_length, penalty_factor, cache_length must be set"
        assert (
            self.max_response_length > self.cache_length
        ), "max_response_length must be greater than cache_length"

        response_len = len(response_token)
        excepted_len = self.max_response_length - self.cache_length

        if response_len < excepted_len:
            return 0.0
        elif response_len > self.max_response_length:
            return -self.penalty_factor
        else:
            return (excepted_len - response_len) / self.cache_length * self.penalty_factor
