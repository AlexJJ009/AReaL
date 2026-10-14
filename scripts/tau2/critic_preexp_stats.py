# SPDX-License-Identifier: Apache-2.0
"""Paired 60-task bootstrap using the production metric reducer."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from scripts.qualification.critic_pretrain import aggregate_validation

SEED = 20261014
DRAWS = 20000
FIELDS = ("n_tokens", "sse", "error_sum", "target_sum", "target2_sum")


class TaskBootstrap:
    """One shared draw matrix for every comparison, including domain subsets."""

    def __init__(self, rows: list[dict[str, Any]], *, require_60: bool = True):
        self.keys = sorted({(str(r["domain"]), str(r["task_id"])) for r in rows})
        if require_60 and len(self.keys) != 60:
            raise ValueError(f"Expected 60 validation tasks, got {len(self.keys)}")
        self.identities = {self.identity(r): (r["n_tokens"], r["target"]) for r in rows}
        if len(self.identities) != len(rows):
            raise ValueError("Duplicate validation episode identities")
        generator = torch.Generator().manual_seed(SEED)
        sampled = torch.randint(
            len(self.keys), (DRAWS, len(self.keys)), generator=generator
        )
        self.counts = torch.zeros(DRAWS, len(self.keys), dtype=torch.float64)
        self.counts.scatter_add_(
            1, sampled, torch.ones_like(sampled, dtype=torch.float64)
        )
        self.memo: dict[Any, Any] = {}

    @staticmethod
    def identity(row):
        return (
            str(row.get("run")),
            str(row["episode_id"]),
            str(row["domain"]),
            str(row["task_id"]),
        )

    def values(self, rows, domain):
        signatures = tuple((self.identity(r), *(r[k] for k in FIELDS)) for r in rows)
        cache_key = (domain, signatures)
        if cache_key in self.memo:
            return self.memo[cache_key]
        if {
            self.identity(r): (r["n_tokens"], r["target"]) for r in rows
        } != self.identities:
            raise ValueError(
                "Bootstrap requires identical episodes, targets and token counts"
            )
        grouped = defaultdict(list)
        for row in rows:
            if domain == "overall" or row["domain"] == domain:
                grouped[(str(row["domain"]), str(row["task_id"]))].append(row)
        totals = torch.tensor(
            [
                [sum(float(r[k]) for r in grouped[key]) for k in FIELDS]
                for key in self.keys
            ],
            dtype=torch.float64,
        )
        sampled = (self.counts @ totals).tolist()
        values = []
        for sums in sampled:
            if sums[0] == 0:
                values.append(float("nan"))
                continue
            row = dict(zip(FIELDS, sums, strict=True))
            row.update(target=0.0, truncated=False)
            value = aggregate_validation([row])["explained_variance"]
            values.append(float("nan") if value is None else value)
        result = torch.tensor(values, dtype=torch.float64)
        self.memo[cache_key] = result
        return result

    def auc_values(self, rows, domain):
        selected = [r for r in rows if domain == "overall" or r["domain"] == domain]
        if any(r["target"] not in (0, 1) for r in selected):
            return torch.full((DRAWS,), float("nan"), dtype=torch.float64)
        groups = defaultdict(list)
        for row in selected:
            groups[row["prediction_mean"]].append(row)
        key_index = {key: i for i, key in enumerate(self.keys)}
        positive = torch.zeros(DRAWS, dtype=torch.float64)
        negative = torch.zeros_like(positive)
        numerator = torch.zeros_like(positive)
        for prediction in sorted(groups):
            pos = torch.zeros_like(positive)
            neg = torch.zeros_like(positive)
            for row in groups[prediction]:
                counts = self.counts[
                    :, key_index[(str(row["domain"]), str(row["task_id"]))]
                ]
                if row["target"] == 1:
                    pos += counts
                else:
                    neg += counts
            numerator += pos * (negative + 0.5 * neg)
            positive += pos
            negative += neg
        denominator = positive * negative
        return torch.where(denominator > 0, numerator / denominator, float("nan"))

    def report(self, rows, reference=None):
        output = {}
        for domain in ["overall", *sorted({r["domain"] for r in rows})]:
            values = self.values(rows, domain)
            if reference is not None:
                values = values - self.values(reference, domain)
            values = values[torch.isfinite(values)]
            aucs = self.auc_values(rows, domain)
            if reference is not None:
                aucs = aucs - self.auc_values(reference, domain)
            aucs = aucs[torch.isfinite(aucs)]
            output[domain] = {
                "auc_95ci": torch.quantile(
                    aucs, torch.tensor([0.025, 0.975], dtype=torch.float64)
                ).tolist()
                if len(aucs)
                else None,
                "defined_auc_draws": len(aucs),
                "ev_95ci": torch.quantile(
                    values, torch.tensor([0.025, 0.975], dtype=torch.float64)
                ).tolist()
                if len(values)
                else None,
                "defined_draws": len(values),
            }
        return {
            "draws": DRAWS,
            "tasks_per_draw": len(self.keys),
            "seed": SEED,
            "paired_difference": reference is not None,
            "by_domain": output,
        }
