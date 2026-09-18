# SPDX-License-Identifier: Apache-2.0

import pytest


def test_thinking_defaults_respect_explicit_switch_without_mutation():
    from examples.swe.qwen38_flash_next.template_defaults import with_template_defaults

    assert with_template_defaults()["chat_template_kwargs"] == dict(
        enable_thinking=True, reasoning_effort="medium", thinking_option=None
    )
    body = {"chat_template_kwargs": {"thinking_option": "off"}, "other": 42}
    merged = with_template_defaults(body)
    assert merged["chat_template_kwargs"] == dict(
        thinking_option="off", reasoning_effort="medium"
    )
    assert merged["other"] == 42
    assert body == {"chat_template_kwargs": {"thinking_option": "off"}, "other": 42}


def test_external_task_selection_preserves_order_and_rejects_drift():
    from examples.swe.qwen38_flash_next.train_rl import select_task_indices

    assert select_task_indices(["a", "b", "c"], ["c", "a"]) == [2, 0]
    for selected in ([], ["a", "a"], ["missing"], "a", [None]):
        with pytest.raises(ValueError):
            select_task_indices(["a", "b"], selected)


def test_cache_isolation_preserves_original_request():
    from types import SimpleNamespace

    from examples.swe.qwen38_flash_next.proxy import wrap_isolated_cache

    original = SimpleNamespace(payload={"input_ids": [1, 2]})
    build = wrap_isolated_cache(lambda _: original)
    first, second = build(None), build(None)
    assert first.payload["cache_salt"] != second.payload["cache_salt"]
    assert first.payload["input_ids"] == original.payload["input_ids"]
    assert "cache_salt" not in original.payload
