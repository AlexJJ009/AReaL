# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from examples.swe.qwen38_flash_next.train_rl import validate_evaluation_only


def evaluation_config():
    return SimpleNamespace(
        total_train_steps=0,
        evaluator=SimpleNamespace(eval_before_train=True),
        recover=SimpleNamespace(mode="disabled"),
        eval_gconfig=SimpleNamespace(n_samples=1),
        valid_dataset=SimpleNamespace(shuffle=False, drop_last=False, batch_size=3),
    )


def test_evaluation_complete_selection_preserves_config():
    config = evaluation_config()
    before = repr(config)

    validate_evaluation_only(config, ["task-a", "task-b", "task-c"])

    assert repr(config) == before


@pytest.mark.parametrize(
    "section,field,value",
    [
        (None, "total_train_steps", 1),
        ("evaluator", "eval_before_train", False),
        ("recover", "mode", "auto"),
        ("eval_gconfig", "n_samples", 8),
        (None, "valid_dataset", None),
        ("valid_dataset", "shuffle", True),
        ("valid_dataset", "drop_last", True),
        ("valid_dataset", "batch_size", 2),
    ],
)
def test_evaluation_unsafe_config_rejected_before_initialization(section, field, value):
    config = evaluation_config()
    setattr(config if section is None else getattr(config, section), field, value)

    with pytest.raises(ValueError, match="swe-eval"):
        validate_evaluation_only(config, ["task-a", "task-b", "task-c"])


def test_evaluation_empty_selection_rejected():
    config = evaluation_config()
    config.valid_dataset.batch_size = 0

    with pytest.raises(ValueError, match="selected task count"):
        validate_evaluation_only(config, [])
