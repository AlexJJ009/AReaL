# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for full-pool critic epoch shuffling (K2: 648 + 99 tasks)."""

import math
import random
from collections import Counter
from copy import deepcopy

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

from scripts.tau2 import train_critic, train_critic_head

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config


@pytest.fixture
def critic_pool(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))
    config, _ = load_expr_config(
        [
            "--config",
            "examples/tau2/config_critic_production.yaml",
            "econfig.benchmark_variant=fixed",
            f"critic_task_pack={tmp_path / 'pack'}",
            "critic_head_bias_init=0.6204819277108434",
        ],
        train_critic.Tau2PPOConfig,
    )

    def rows(counts, split, pack=None):
        return [
            {
                "domain": domain,
                "task_id": f"{split}-{domain}-{i}",
                "split": split,
                "task_pack": pack,
            }
            for domain, count in counts.items()
            for i in range(count)
        ]

    official = rows({"airline": 25, "retail": 74}, "train")
    valid = rows({"airline": 20, "retail": 40}, "test")
    pack = rows({"airline": 380, "retail": 268}, "pack", config.critic_task_pack)
    random.Random(f"{config.seed}:task_pack").shuffle(pack)
    monkeypatch.setattr(
        train_critic.benchmark_variant, "validate_task_files", lambda *a: None
    )
    monkeypatch.setattr(
        train_critic,
        "get_tau2_dataset",
        lambda **kw: Dataset.from_list(official if kw["split"] == "train" else valid),
    )
    monkeypatch.setattr(
        train_critic, "_task_pack_rows", lambda *a, **kw: deepcopy(pack)
    )
    monkeypatch.setattr(
        train_critic.task_pack, "load_manifest", lambda _: {"name": "test-pack"}
    )
    train, validation, _ = train_critic.build_tau2_critic_datasets(config)
    train_critic.fill_derived_train_steps(config, train_rows=len(train))

    def init(self, config, *a, **kw):
        self.config = config
        self.recover_info = None
        self.rollout = type("Rollout", (), {})()
        self.train_dataloader = self._create_dataloader(
            train, config.train_dataset, rank=0, world_size=1
        )

    monkeypatch.setattr(PPOTrainer, "_init_impl", init)
    return config, train, validation


def _trainer(config, cls=train_critic.Tau2CriticOnlineTrainer):
    trainer = object.__new__(cls)
    trainer._init_impl(config)
    return trainer


def _epoch(trainer):
    return [
        trainer._next_strict_rows(trainer.train_dataloader)
        for _ in range(len(trainer.train_dataloader))
    ]


def _identities(batches):
    return [
        (r["task_pack"], r["domain"], r["task_id"]) for batch in batches for r in batch
    ]


@pytest.mark.parametrize(
    "cls",
    [train_critic.Tau2CriticOnlineTrainer, train_critic_head.Tau2HeadCriticTrainer],
)
def test_epoch_edges_747_pool_mix_official_pack_and_domains(critic_pool, cls):
    """Both epoch edges must lose the old pack-then-official boundary."""
    config, _, _ = critic_pool
    trainer = _trainer(config, cls)
    for _ in range(2):
        batches = _epoch(trainer)
        for batch in (batches[0], batches[-1]):
            assert {bool(r["task_pack"]) for r in batch} == {True, False}
            assert {r["domain"] for r in batch} == {"airline", "retail"}


def test_epoch_747_pool_covers_each_task_once_and_keeps_tail(critic_pool):
    """The logical 11-task tail is neither dropped nor padded."""
    config, train, _ = critic_pool
    trainer = _trainer(config)
    expected = Counter(_identities([list(train)]))
    assert len(expected) == 747
    for _ in range(2):
        batches = _epoch(trainer)
        assert len(batches) == math.ceil(747 / 16)
        assert len(batches[-1]) == 11
        assert Counter(_identities(batches)) == expected
        assert set(Counter(_identities(batches)).values()) == {1}


def test_epoch_permutation_changes_and_is_seed_deterministic(critic_pool):
    """Epoch and config seed control the real stateful sampler's order."""
    config, _, _ = critic_pool
    trainer = _trainer(config)
    first, second = _identities(_epoch(trainer)), _identities(_epoch(trainer))
    assert first != second
    repeated = _trainer(config)
    assert _identities(_epoch(repeated)) == first
    assert _identities(_epoch(repeated)) == second
    changed = deepcopy(config)
    changed.seed += 1
    assert _identities(_epoch(_trainer(changed))) != first


def test_epoch_iteration_preserves_pool_validation_and_head_prior(critic_pool):
    """Shuffling samples cannot alter the datasets or configured train prior."""
    config, train, validation = critic_pool
    before_train, before_validation = list(train), list(validation)
    prior = config.critic_head_bias_init
    trainer = _trainer(config)
    _epoch(trainer)
    _epoch(trainer)
    rebuilt_train, rebuilt_validation, _ = train_critic.build_tau2_critic_datasets(
        config
    )
    assert list(train) == list(rebuilt_train) == before_train
    assert list(validation) == list(rebuilt_validation) == before_validation
    assert config.critic_head_bias_init == prior


@pytest.mark.parametrize("mode", ["formal", "tune"])
@pytest.mark.parametrize("entry", ["validate", "trainer"])
def test_critic_unshuffled_config_refuses_before_worker_init(critic_pool, mode, entry):
    """A CLI override or direct trainer call cannot silently disable shuffle."""
    config, _, _ = critic_pool
    config.experiment_mode = mode
    config.train_dataset.shuffle = False
    with pytest.raises(ValueError, match="shuffle"):
        if entry == "validate":
            train_critic.validate_tau2_critic_config(config)
        else:
            _trainer(config)


def test_check_config_reports_actual_epoch_edge_composition(critic_pool, capsys):
    """Preview counts agree with the runtime iterator for both K2 epochs."""
    config, _, validation = critic_pool
    before = list(validation)
    train_critic_head.main(
        [
            "--config",
            "examples/tau2/config_critic_production.yaml",
            "--check-config",
            "econfig.benchmark_variant=fixed",
            f"critic_task_pack={config.critic_task_pack}",
            f"critic_head_bias_init={config.critic_head_bias_init}",
        ]
    )
    output = OmegaConf.to_container(OmegaConf.create(capsys.readouterr().out))
    preview = output["train_order"]
    assert preview["seed"] == config.seed
    trainer = _trainer(config)
    for epoch, report in enumerate(preview["epochs"]):
        batches = _epoch(trainer)
        assert report["epoch"] == epoch
        assert [b["batch"] for b in report["batches"]] == [0, 1, 2, 44, 45, 46]
        for batch in report["batches"]:
            counts = Counter(
                ("pack" if r["task_pack"] else "official", r["domain"])
                for r in batches[batch["batch"]]
            )
            assert batch["composition"] == [
                {"source": source, "domain": domain, "count": count}
                for (source, domain), count in sorted(counts.items())
            ]
    assert list(validation) == before
    assert output["config"]["critic_head_bias_init"] == config.critic_head_bias_init
