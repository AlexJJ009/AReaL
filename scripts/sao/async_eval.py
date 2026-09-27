# SPDX-License-Identifier: Apache-2.0
"""Math-specific evidence hooks; scheduling and metrics live in AReaL."""

import os
from pathlib import Path

from areal.trainer.async_eval import (
    AsyncEvalGRPOConfig as SaoGRPOConfig,
)
from areal.trainer.async_eval import (
    AsyncEvalPPOConfig as SaoPPOConfig,
)
from areal.trainer.async_eval import (
    AsyncEvalPPOTrainer as _AsyncEvalPPOTrainer,
)

__all__ = [
    "AsyncEvalGRPOTrainer",
    "AsyncEvalPPOTrainer",
    "SaoGRPOConfig",
    "SaoPPOConfig",
]


class AsyncEvalGRPOTrainer(_AsyncEvalPPOTrainer):
    """Retain the math recipe's snapshot checks and explicit initial eval hook."""

    _snapshot_evidence = True
    _manage_eval_schedule = False

    def _is_preflight(self) -> bool:
        return os.environ.get("SAO_PREFLIGHT", "0") == "1"

    def _snapshot_eval(self, version: int) -> None:
        from scripts.sao.snapshot_eval import snapshot_eval

        snapshot_eval(
            Path(self.config.cluster.fileroot) / "evidence",
            Path(self.config.train_dataset.path),
            version,
            allowed_versions=(version,),
            n_samples=self.config.eval_gconfig.n_samples,
        )


AsyncEvalPPOTrainer = AsyncEvalGRPOTrainer
