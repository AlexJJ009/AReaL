"""Prevent inherited actor TMS hooks from changing SGLang CUDA allocations."""

import os
import re
from pathlib import Path
from typing import Any

import pytest

from areal.api.cli_args import SGLangConfig
from areal.engine import sglang_remote
from areal.engine.sglang_remote import SGLangBackend


def _preload_entries(value: str) -> list[str]:
    return [entry for entry in re.split(r"[:\s]+", value) if entry]


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_build_server_env_inherited_actor_flags_removes_flags(
    enabled: str, tmp_path: Path
):
    """Remove actor initialization flags even when their inherited value is zero."""
    env = {
        "TMS_INIT_ENABLE": enabled,
        "TMS_INIT_ENABLE_CPU_BACKUP": enabled,
        "CUDA_VISIBLE_DEVICES": "0",
        "TRITON_CACHE_PATH": str(tmp_path),
    }
    original = env.copy()

    result = SGLangBackend.build_server_env(env)

    assert "TMS_INIT_ENABLE" not in result
    assert "TMS_INIT_ENABLE_CPU_BACKUP" not in result
    assert result["CUDA_VISIBLE_DEVICES"] == "0"
    assert Path(result["TRITON_CACHE_PATH"]).parent == tmp_path
    assert result["TRITON_CACHE_PATH"] != str(tmp_path)
    assert env == original


@pytest.mark.parametrize("separator", [":", " ", "\t", ": \t"])
@pytest.mark.parametrize(
    "hook_name",
    [
        "torch_memory_saver_hook_mode_preload.so",
        "torch_memory_saver_hook_mode_preload.abi3.so",
    ],
)
def test_build_server_env_mixed_preloads_preserves_other_hooks(
    separator: str, hook_name: str, tmp_path: Path
):
    """Linux colon and whitespace syntax must retain unrelated loader hooks."""
    retained = [str(tmp_path / "libstdc++.so.6"), str(tmp_path / "malloc_hook.so")]
    hook = str(tmp_path / hook_name)
    env = {"LD_PRELOAD": separator.join([retained[0], hook, retained[1]])}
    original = env.copy()

    result = SGLangBackend.build_server_env(env)

    assert _preload_entries(result["LD_PRELOAD"]) == retained
    assert env == original


@pytest.mark.parametrize(
    "preload",
    [
        "torch_memory_saver_hook_mode_preload.abi3.so",
        "/opt/hooks/torch_memory_saver_hook_mode_preload.so",
        " :\t/opt/hooks/torch_memory_saver_hook_mode_preload.abi3.so : ",
    ],
)
def test_build_server_env_only_actor_preload_removes_empty_key(preload: str):
    """Do not pass an empty LD_PRELOAD after removing the actor's sole hook."""
    env = {"LD_PRELOAD": preload}
    original = env.copy()

    result = SGLangBackend.build_server_env(env)

    assert "LD_PRELOAD" not in result
    assert env == original


@pytest.mark.parametrize("preload", [None, "libstdc++.so.6:malloc_hook.so"])
def test_build_server_env_without_actor_tms_preserves_environment(
    preload: str | None, tmp_path: Path
):
    """An ordinary receiver keeps its environment except its unique Triton path."""
    env = {
        "CUDA_VISIBLE_DEVICES": "1",
        "NCCL_CUMEM_ENABLE": "0",
        "LD_LIBRARY_PATH": str(tmp_path / "libraries"),
        "TRITON_CACHE_PATH": str(tmp_path / "triton"),
    }
    if preload is not None:
        env["LD_PRELOAD"] = preload
    original = env.copy()

    first = SGLangBackend.build_server_env(env)
    second = SGLangBackend.build_server_env(env)

    assert {
        key: value for key, value in first.items() if key != "TRITON_CACHE_PATH"
    } == {key: value for key, value in env.items() if key != "TRITON_CACHE_PATH"}
    assert first["TRITON_CACHE_PATH"] != second["TRITON_CACHE_PATH"]
    assert Path(first["TRITON_CACHE_PATH"]).parent == Path(env["TRITON_CACHE_PATH"])
    assert "TMS_INIT_ENABLE" not in first
    assert "TMS_INIT_ENABLE_CPU_BACKUP" not in first
    assert env == original


@pytest.mark.parametrize("enable_memory_saver", [False, True])
def test_launch_server_inherited_actor_tms_sanitizes_child_only(
    monkeypatch: pytest.MonkeyPatch, enable_memory_saver: bool, tmp_path: Path
):
    """Launch uses the clean child env and preserves SGLang's own saver choice."""
    hook = str(tmp_path / "torch_memory_saver_hook_mode_preload.abi3.so")
    retained = str(tmp_path / "libstdc++.so.6")
    monkeypatch.setenv("LD_PRELOAD", f"{retained}:{hook}")
    monkeypatch.setenv("TMS_INIT_ENABLE", "1")
    monkeypatch.setenv("TMS_INIT_ENABLE_CPU_BACKUP", "1")
    monkeypatch.setenv("TRITON_CACHE_PATH", str(tmp_path / "triton"))
    monkeypatch.delenv("AWEX_META_SERVER_ADDR", raising=False)
    original = dict(os.environ)
    built_args = []
    launches = []
    process = object()

    def build_command(args: dict[str, Any]) -> list[str]:
        built_args.append(args.copy())
        command = ["python", "-m", "sglang.launch_server"]
        if args["enable_memory_saver"]:
            command.append("--enable-memory-saver")
        return command

    def popen(command: list[str], **kwargs: Any) -> object:
        launches.append((command, kwargs))
        return process

    monkeypatch.setattr(
        SGLangConfig, "build_cmd_from_args", staticmethod(build_command)
    )
    monkeypatch.setattr(sglang_remote.subprocess, "Popen", popen)

    result = SGLangBackend().launch_server({"enable_memory_saver": enable_memory_saver})

    assert result is process
    assert built_args == [{"enable_memory_saver": enable_memory_saver}]
    assert len(launches) == 1
    command, kwargs = launches[0]
    assert ("--enable-memory-saver" in command) is enable_memory_saver
    child_env = kwargs["env"]
    assert "TMS_INIT_ENABLE" not in child_env
    assert "TMS_INIT_ENABLE_CPU_BACKUP" not in child_env
    assert _preload_entries(child_env["LD_PRELOAD"]) == [retained]
    assert child_env["TRITON_CACHE_PATH"] != original["TRITON_CACHE_PATH"]
    assert dict(os.environ) == original
