# SPDX-License-Identifier: Apache-2.0
"""Fatal rollout failures that must cross the worker/controller boundary."""

QUOTA_EXIT_CODE = 86


class FatalRolloutError(RuntimeError):
    """Stop collection and fail the trainer, rather than reject one episode."""
