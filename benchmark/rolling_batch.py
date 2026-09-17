# SPDX-License-Identifier: Apache-2.0

"""Bounded lookahead and completion-order batch consumption for rollout benchmarks."""


class RollingBatchWindow:
    """Count pending, running and ready groups in one unconsumed window.

    Completing a group does not free a window slot until a batch consumes it.
    Each consumed batch opens exactly one batch of new submissions. After the
    requested measurements, all remaining submitted groups are drained.
    """

    def __init__(self, batch_size: int, prefetch_batches: int, measure_batches: int):
        if min(batch_size, prefetch_batches, measure_batches) < 1:
            raise ValueError(
                "batch size, prefetch and measurement counts must be positive"
            )
        self.batch_size = batch_size
        self.window_size = batch_size * prefetch_batches
        self.measure_batches = measure_batches
        self.total_groups = self.window_size + batch_size * (measure_batches - 1)
        self.submitted = 0
        self.rejected = 0
        self.consumed = 0
        self.batches = 0
        self.ready: list[int] = []
        self.completed: set[int] = set()

    def take_submission(self) -> int | None:
        if (
            self.batches == self.measure_batches
            or self.submitted - self.consumed - self.rejected >= self.window_size
        ):
            return None
        task_id = self.submitted
        self.submitted += 1
        return task_id

    def complete(self, task_id: int, accepted: bool = True) -> list[int] | None:
        if task_id < 0 or task_id >= self.submitted or task_id in self.completed:
            raise ValueError("Completion must identify a unique submitted group")
        self.completed.add(task_id)
        if not accepted:
            self.rejected += 1
            return None
        if self.batches == self.measure_batches:
            return None
        self.ready.append(task_id)
        if len(self.ready) < self.batch_size:
            return None
        batch = self.ready
        self.ready = []
        self.consumed += len(batch)
        self.batches += 1
        return batch
