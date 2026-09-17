# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmark.rolling_batch import RollingBatchWindow


def fill(window):
    result = []
    while (task_id := window.take_submission()) is not None:
        result.append(task_id)
    return result


def test_window_consumes_any_complete_groups_and_keeps_tail():
    window = RollingBatchWindow(2, 3, 3)
    assert fill(window) == list(range(6))
    assert window.complete(5) is None
    assert window.take_submission() is None
    assert window.complete(3) == [5, 3]
    assert fill(window) == [6, 7]
    assert window.complete(6) is None
    assert window.complete(1) == [6, 1]
    assert fill(window) == [8, 9]
    assert window.complete(7) is None
    assert window.complete(9) == [7, 9]
    assert fill(window) == []
    # Slow groups from the initial window are still retained for final drain.
    for task_id in [0, 2, 4, 8]:
        assert window.complete(task_id) is None
    assert len(window.completed) == 10
    assert window.consumed == 6
    assert window.batches == 3


def test_real_window_has_48_prompts_and_refills_exactly_16():
    window = RollingBatchWindow(16, 3, 10)
    assert fill(window) == list(range(48))
    for task_id in range(16, 31):
        assert window.complete(task_id) is None
        assert window.take_submission() is None
    assert window.complete(31) == list(range(16, 32))
    assert fill(window) == list(range(48, 64))
    assert window.submitted - window.consumed == 48
    assert window.total_groups == 192


@pytest.mark.parametrize("task_id", [-1, 6])
def test_window_rejects_unsubmitted_completion(task_id):
    window = RollingBatchWindow(2, 3, 1)
    fill(window)
    with pytest.raises(ValueError):
        window.complete(task_id)


def test_window_rejects_duplicate_completion():
    window = RollingBatchWindow(2, 3, 1)
    fill(window)
    window.complete(1)
    with pytest.raises(ValueError):
        window.complete(1)


def test_rejected_group_opens_one_slot_without_entering_ready_batch():
    window = RollingBatchWindow(2, 3, 2)
    assert fill(window) == list(range(6))
    assert window.complete(0, accepted=False) is None
    assert fill(window) == [6]
    assert window.complete(1) is None
    assert window.complete(2) == [1, 2]
    assert fill(window) == [7, 8]
    assert window.submitted - window.rejected - window.consumed == 6
    assert window.complete(3) is None
    assert window.complete(4) == [3, 4]
    assert window.complete(5, accepted=False) is None
    assert fill(window) == []
