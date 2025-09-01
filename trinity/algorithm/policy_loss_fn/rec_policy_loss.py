"""REC-token policy loss function.
"""

from typing import Dict, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN, PolicyLossFn
from trinity.algorithm.utils import masked_mean


@POLICY_LOSS_FN.register_module("sppo")
class RECPolicyLossFn(PolicyLossFn):
    def __init__(
        self,
        backend: str = "verl",
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        clip_mode: str = "none",
        weight: str = "none",
    ) -> None:
        super().__init__(backend=backend)

        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        assert 0.0 < self.epsilon_low <= 1.0, f"Invalid epsilon_low: {self.epsilon_low}"
        assert 0.0 < self.epsilon_high <= 1.0, f"Invalid epsilon_high: {self.epsilon_high}"
        self.clip_mode = clip_mode
        assert self.clip_mode in [
            "none",
            "one-side",
            "two-side",
        ], f"Invalid clip_mode: {self.clip_mode}"
        self.weight = weight
        assert self.weight in ["none", "importance_sampling"], f"Invalid weight: {self.weight}"

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,  # [batch_size, seq_len]
        old_logprob: torch.Tensor,  # [batch_size, seq_len]
        action_mask: torch.Tensor,  # [batch_size, seq_len]
        advantages: torch.Tensor,  # [batch_size, seq_len]
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        """Calculate REC loss."""
        # token-wise
        ratio = torch.exp(logprob - old_logprob).detach()

        if self.clip_mode == "two-side":
            is_in_range = (ratio >= (1 - self.epsilon_low)) * (ratio <= (1 + self.epsilon_high))
        elif self.clip_mode == "one-side":
            is_in_range = (ratio <= (1 + self.epsilon_high)) * (advantages > 0) + (
                advantages < 0
            ) * (ratio >= (1 - self.epsilon_low))
        else:  # none
            is_in_range = torch.ones_like(ratio).bool()
        is_clipped_mask = ~is_in_range

        if self.weight == "importance_sampling":
            advantages = advantages * ratio  # importance sampling

        pg_losses = -advantages * (logprob - old_logprob) * is_in_range.float()
        pg_loss = masked_mean(pg_losses, action_mask)
        pg_clipfrac = masked_mean(is_clipped_mask.float(), action_mask)
        metrics = {
            "pg_clipfrac": pg_clipfrac.item(),
            "pg_loss": pg_loss.detach().item(),
        }
        return pg_loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "epsilon_low": 0.2,
            "epsilon_high": 0.2,
            "clip_mode": "none",
            "weight": "none",
        }
