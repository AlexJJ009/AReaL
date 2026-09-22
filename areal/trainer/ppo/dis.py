# SPDX-License-Identifier: Apache-2.0

"""Direct importance sampling with the approved score-function gradient."""

import math

import torch

from areal.utils import stats_tracker


class DirectDISLoss:
    """One optimizer batch's DIS loss and retained-token count.

    Engine weighting combines microbatch token means using the original action
    count. The FSDP engine uses kept_tokens to skip an entirely rejected batch,
    including Adam momentum/weight decay; reset once per train_batch invocation.
    """

    def __init__(self, epsilon_low: float = 0.3, epsilon_high: float = 5.0):
        if not math.isfinite(epsilon_low) or not 0 <= epsilon_low < 1:
            raise ValueError("DIS epsilon_low must be finite in [0, 1)")
        if not math.isfinite(epsilon_high) or epsilon_high < 0:
            raise ValueError("DIS epsilon_high must be finite and nonnegative")
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.kept_tokens = None

    def reset(self):
        self.kept_tokens = None

    def __call__(self, logprobs, entropy, input_data, **kwargs):
        if input_data.get("teacher_logp") is not None:
            raise ValueError("Direct DIS cannot include undeclared teacher/KL loss")
        mask = input_data["loss_mask"].bool()
        current = (
            logprobs.double() if logprobs.dtype == torch.float64 else logprobs.float()
        )
        behavior = input_data["logprobs"].detach().to(current.dtype)
        advantage = input_data["advantages"].detach().to(current.dtype)
        if not (current.shape == behavior.shape == advantage.shape == mask.shape):
            raise ValueError("Direct DIS inputs must have matching token shapes")
        torch._assert_async(mask.any(), "Direct DIS requires valid action tokens")
        torch._assert_async(
            torch.all(
                ~mask
                | (
                    torch.isfinite(current)
                    & torch.isfinite(behavior)
                    & torch.isfinite(advantage)
                )
            ),
            "Non-finite Direct DIS input",
        )
        log_ratio = torch.where(mask, current - behavior, 0.0)
        ratio = log_ratio.exp()
        torch._assert_async(
            torch.all(torch.isfinite(ratio)), "Non-finite Direct DIS ratio"
        )
        # Compare logs at the same dtype to make exact boundary inputs strict;
        # exp rounding near a threshold must not turn equality into acceptance.
        kept = (
            mask
            & (log_ratio > math.log1p(-self.epsilon_low))
            & (log_ratio < math.log1p(self.epsilon_high))
        )
        count = kept.sum().detach()
        self.kept_tokens = (
            count if self.kept_tokens is None else self.kept_tokens + count
        )
        weight = torch.where(kept, ratio * advantage, 0.0).detach()
        terms = -weight * torch.where(mask, current, 0.0)
        stats_tracker.denominator(dis_action_tokens=mask)
        stats_tracker.stat(
            dis_ratio=ratio.detach().float(),
            dis_kept=kept.float(),
            dis_loss=terms.detach().float(),
            denominator="dis_action_tokens",
        )
        return terms.sum() / mask.count_nonzero()
