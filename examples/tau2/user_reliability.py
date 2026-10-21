# SPDX-License-Identifier: Apache-2.0
"""Past-only per-task reliability for the opt-in Tau2 PPO recipe.

All state lives on the trainer/controller, using CPU rollout tensors. Epochs
here are one-based; finite rollout epochs are converted at the recipe boundary.
"""

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


def scenario_key(row: Mapping[str, Any]) -> tuple[str, str, str | None]:
    """One task per scenario, including Telecom and distinct task packs."""
    return str(row["domain"]), str(row["task_id"]), row.get("task_pack")


def scenario_id(row: Mapping[str, Any]) -> int:
    """Stable tensor-transport identity; collisions are rejected against the pool."""
    payload = json.dumps(scenario_key(row), ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % 2**63


class UserReliability:
    """Freeze epoch weights before observing outcomes; checkpoint the full history."""

    def __init__(
        self,
        scheduled: Iterable[Mapping[str, Any]],
        shrink_k: float = 2.0,
        warmup_epochs: int = 2,
    ) -> None:
        if not math.isfinite(shrink_k) or shrink_k < 0:
            raise ValueError("reliability shrink_k must be finite and non-negative")
        if (
            isinstance(warmup_epochs, bool)
            or not isinstance(warmup_epochs, int)
            or warmup_epochs < 0
        ):
            raise ValueError("reliability warmup_epochs must be a non-negative integer")
        self.scenarios: dict[int, tuple[str, str, str | None]] = {}
        for row in scheduled:
            ident, key = scenario_id(row), scenario_key(row)
            if ident in self.scenarios and self.scenarios[ident] != key:
                raise ValueError("Scenario identity hash collision")
            self.scenarios[ident] = key
        self.shrink_k = float(shrink_k)
        self.warmup_epochs = warmup_epochs
        self.records: list[list[int | float]] = []
        self.epoch = 0
        self.table: list[dict[str, Any]] = []

    def record(self, epoch: int, ident: int, surprise: float) -> None:
        if ident not in self.scenarios or epoch < 1 or not math.isfinite(surprise):
            raise ValueError("Invalid reliability observation")
        self.records.append([epoch, ident, float(surprise)])

    def begin_epoch(
        self, epoch: int, log_dir: str | Path | None = None
    ) -> list[dict[str, Any]]:
        if epoch == self.epoch:
            self._write_table(log_dir)
            return self.table
        if epoch < self.epoch or epoch < 1:
            raise ValueError("Reliability epochs must advance monotonically")
        history = defaultdict(list)
        for past_epoch, ident, surprise in self.records:
            if past_epoch < epoch:
                history[ident].append(surprise)
        # Sort by the complete task identity, independent of dataset shuffle/order.
        identities = sorted(
            self.scenarios, key=lambda i: tuple(s or "" for s in self.scenarios[i])
        )
        variances = {
            i: statistics.variance(history[i]) if len(history[i]) >= 2 else None
            for i in identities
        }
        by_domain = defaultdict(list)
        for ident in identities:
            by_domain[self.scenarios[ident][0]].append(ident)
        table = []
        for domain, ids in sorted(by_domain.items()):
            samples = [variances[i] for i in ids if variances[i] is not None]
            prior = statistics.mean(samples) if samples else None
            sigma = {}
            for i in ids:
                n, s2 = len(history[i]), variances[i]
                sigma[i] = (
                    (n * s2 + self.shrink_k * prior) / (n + self.shrink_k)
                    if s2 is not None
                    else prior
                )
            tau = statistics.mean(sigma.values()) if prior is not None else None
            use_weights = epoch > self.warmup_epochs and tau is not None and tau > 0
            raw = {i: tau / (tau + sigma[i]) if use_weights else 1.0 for i in ids}
            normalizer = statistics.mean(raw.values())
            for i in ids:
                table.append(
                    {
                        "epoch": epoch,
                        "scenario": list(self.scenarios[i]),
                        "id": i,
                        "n": len(history[i]),
                        "s2": variances[i],
                        "prior": prior,
                        "sigma2": sigma[i],
                        "tau": tau,
                        "raw_w": raw[i],
                        "weight": raw[i] / normalizer,
                    }
                )
        self.epoch, self.table = epoch, table
        self._write_table(log_dir)
        return table

    def _write_table(self, log_dir: str | Path | None) -> None:
        if log_dir is not None:
            path = Path(log_dir) / "user-reliability" / f"epoch-{self.epoch:04d}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = "".join(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                for row in self.table
            )
            if not path.exists() or path.read_text(encoding="utf-8") != payload:
                path.write_text(payload, encoding="utf-8")

    def apply(self, groups: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        """Attach token-aligned weights before DP padding; report real batch stats."""
        weights = {row["id"]: row["weight"] for row in self.table}
        applied = defaultdict(list)
        for group in groups:
            ids = group["user_reliability_id"].tolist()
            ws = [weights[i] for i in ids]
            group["user_reliability_weight"] = (
                torch.tensor(ws, dtype=torch.float32)
                .unsqueeze(1)
                .expand_as(group["input_ids"])
                .clone()
            )
            for i, w in zip(ids, ws, strict=True):
                applied[self.scenarios[i][0]].append(w)
        stats = {}
        for domain in sorted({key[0] for key in self.scenarios.values()}):
            ws = applied[domain]
            prefix = f"user_reliability/{domain}"
            stats[f"{prefix}/scenarios_with_history"] = sum(
                row["n"] > 0 for row in self.table if row["scenario"][0] == domain
            )
            # All domains are logged at every step, including an empty-domain tail.
            stats[f"{prefix}/episodes"] = len(ws)
            for name, value in (
                ("mean", statistics.mean(ws) if ws else 1.0),
                ("min", min(ws) if ws else 1.0),
                ("max", max(ws) if ws else 1.0),
                ("std", statistics.pstdev(ws) if ws else 0.0),
            ):
                stats[f"{prefix}/weight_{name}"] = value
        return stats

    def observe(self, groups: list[dict[str, torch.Tensor]]) -> None:
        """Reuse type=3 at b, reading fresh U before GAE/critic fitting alters data."""
        for group in groups:
            terminal = group["user_boundary_types"] == 3
            padding = group.get("is_padding", torch.zeros_like(terminal)).bool().any(-1)
            eligible = (
                group["terminated"] & ~group["truncated"] & ~padding & terminal.any(-1)
            )
            if torch.any(terminal.sum(-1) > 1):
                raise ValueError("Multiple terminal user boundaries")
            positions = terminal.to(torch.int64).argmax(-1)
            surprise = group["rewards"] - group["values"].gather(
                1, positions.unsqueeze(1)
            ).squeeze(1)
            for ident, x in zip(
                group["user_reliability_id"][eligible].tolist(),
                surprise[eligible].tolist(),
                strict=True,
            ):
                self.record(self.epoch, ident, x)

    def state_dict(self) -> dict[str, Any]:
        return deepcopy(
            {
                "version": 1,
                "shrink_k": self.shrink_k,
                "warmup_epochs": self.warmup_epochs,
                "scenarios": [
                    [i, list(key)] for i, key in sorted(self.scenarios.items())
                ],
                "records": self.records,
                "epoch": self.epoch,
                "table": self.table,
            }
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict()
        for name in ("version", "shrink_k", "warmup_epochs", "scenarios"):
            if state.get(name) != expected[name]:
                raise ValueError(f"Incompatible reliability recovery: {name}")
        self.records = []
        for epoch, ident, surprise in state["records"]:
            self.record(epoch, ident, surprise)
        self.epoch, self.table = state["epoch"], deepcopy(state["table"])
