# SPDX-License-Identifier: Apache-2.0
"""CPU probes and ordered AdamW replay on saved causal value-head features.

Feature shards are hash-checked and loaded once per store, then kept in CPU
memory for training and validation through the production critic metric code.
Sampling weights preserve episode token mass, but K sampled prefixes do not
reproduce full-token gradients exactly.
The backbone is fixed: this is an optimisation diagnostic, not a reproduction of
the original run's backbone updates or its distributed numerical arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def canonical_rollout_task_id(row: dict[str, Any]) -> int:
    """Remove the production retry offset, preserving original submission ID."""
    from scripts.tau2.train_critic import _REPLAY_TASK_ID_OFFSET

    value = int(row["rollout_task_id"])
    if value >= _REPLAY_TASK_ID_OFFSET:
        value -= _REPLAY_TASK_ID_OFFSET
    if value < 0 or value >= _REPLAY_TASK_ID_OFFSET:
        raise ValueError(f"Invalid canonical rollout task ID: {value}")
    return value


def recorded_replay_rows(
    rows: list[dict[str, Any]], *, batch_size: int = 16
) -> list[dict[str, Any]]:
    """Derive logical step/order from real rollout IDs, including retry rows."""
    output = []
    for row in rows:
        canonical = canonical_rollout_task_id(row)
        output.append(
            {
                **row,
                "trajectory_id": str(
                    row.get("trajectory_id", row.get("episode_id", canonical))
                ),
                "step": canonical // batch_size,
                "order": canonical % batch_size,
                "canonical_rollout_task_id": canonical,
            }
        )
    return output


def replay_batches(
    rows: list[dict[str, Any]],
    *,
    batch_size: int = 16,
    total_steps: int = 80,
    allow_tail: bool = False,
) -> list[list[dict[str, Any]]]:
    """Replay recorded rollout step/order; never invent a shuffle from a seed.

    Both zero-based and one-based *complete* step sequences are accepted. Each
    step must be full unless a finite-dataset tail is explicitly enabled for
    the final step. Duplicates, missing orders, and missing steps fail.
    """
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        identity = str(row["trajectory_id"])
        if identity in seen:
            raise ValueError(f"Duplicate replay trajectory: {identity}")
        seen.add(identity)
        grouped[int(row["step"])].append(row)
    offset = 0 if 0 in grouped else 1
    expected = set(range(offset, offset + total_steps))
    if set(grouped) != expected:
        raise ValueError(
            f"Incomplete replay steps: missing={sorted(expected - set(grouped))}, "
            f"extra={sorted(set(grouped) - expected)}"
        )
    batches = []
    for step in sorted(grouped):
        batch = sorted(grouped[step], key=lambda row: int(row["order"]))
        if not 0 < len(batch) <= batch_size:
            raise ValueError(f"Replay step {step} has {len(batch)} episodes")
        if len(batch) != batch_size and not (allow_tail and step == max(grouped)):
            raise ValueError(
                f"Incomplete replay batch at step {step}: {len(batch)} != {batch_size}"
            )
        orders = [int(row["order"]) for row in batch]
        if len(set(orders)) != len(orders):
            raise ValueError(f"Duplicate replay order at step {step}: {orders}")
        canonical_start = (step - offset) * batch_size
        canonical_orders = list(range(canonical_start, canonical_start + len(batch)))
        if orders != list(range(len(batch))) and orders != canonical_orders:
            raise ValueError(f"Missing replay order at step {step}: {orders}")
        batches.append(batch)
    return batches


def official_metric_row(
    predictions: torch.Tensor, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Use the online evaluator's shifted-mask row, with already selected prefixes."""
    from scripts.tau2.train_critic import _metric_row

    n = predictions.numel()
    # Prefixes were selected at original action token t-1. A synthetic row with
    # action mask [0,1,...,1] reproduces that same selection without token IDs.
    row = {
        **metadata,
        "loss_mask": torch.tensor([False, *([True] * n)]),
        "reward": float(metadata["reward"]),
    }
    result = _metric_row(row, torch.cat([predictions.float(), torch.zeros(1)]))
    result["run"] = str(metadata.get("run", "unknown"))
    return result


def auc(rows: list[dict[str, Any]]) -> float | None:
    """Trajectory-mean success/failure AUC, giving exact ties half credit."""
    positive = sorted(row["prediction_mean"] for row in rows if row["target"] == 1)
    negative = sorted(row["prediction_mean"] for row in rows if row["target"] == 0)
    if not positive or not negative:
        return None
    import bisect

    score = sum(
        bisect.bisect_left(negative, value)
        + 0.5
        * (bisect.bisect_right(negative, value) - bisect.bisect_left(negative, value))
        for value in positive
    )
    return score / (len(positive) * len(negative))


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scripts.qualification.critic_pretrain import (
        aggregate_by_key,
        aggregate_validation,
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row)
        run_grouped[str(row.get("run", "unknown"))].append(row)
    return {
        "summary": aggregate_validation(rows),
        "by_domain": aggregate_by_key(rows, "domain"),
        "auc": {domain: auc(group) for domain, group in grouped.items()},
        "by_run_domain": {
            run: aggregate_by_key(group, "domain") for run, group in run_grouped.items()
        },
        "rows": rows,
    }


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    reference: list[dict[str, Any]] | None = None,
    draws: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired resampling of task clusters, retaining all runs of each task.

    A model comparison uses identical task draws and requires the same episode
    identities, token counts and targets. No validation labels fit a model here.
    """
    from scripts.qualification.critic_pretrain import aggregate_validation

    def identity(row):
        return (
            row.get("run", "unknown"),
            row["episode_id"],
            row["domain"],
            row["task_id"],
        )

    if reference is not None:
        ref_map = {identity(row): row for row in reference}
        row_map = {identity(row): row for row in rows}
        if len(row_map) != len(rows) or len(ref_map) != len(reference):
            raise ValueError("Paired bootstrap has duplicate episode identities")
        if set(row_map) != set(ref_map):
            raise ValueError("Paired bootstrap episode identities differ")
        for key, row in row_map.items():
            if (row["n_tokens"], row["target"]) != (
                ref_map[key]["n_tokens"],
                ref_map[key]["target"],
            ):
                raise ValueError("Paired bootstrap targets or token counts differ")
    else:
        ref_map = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["task_id"])].append(row)
    output = {}
    generator = torch.Generator().manual_seed(seed)
    for domain in sorted({row["domain"] for row in rows}):
        clusters = [group for (d, _), group in sorted(grouped.items()) if d == domain]
        values, aucs = [], []
        for _ in range(draws):
            sampled = [
                row
                for index in torch.randint(
                    len(clusters), (len(clusters),), generator=generator
                )
                for row in clusters[int(index)]
            ]
            ev = aggregate_validation(sampled)["explained_variance"]
            a = auc(sampled)
            if reference is not None:
                paired = [ref_map[identity(row)] for row in sampled]
                reference_ev = aggregate_validation(paired)["explained_variance"]
                reference_auc = auc(paired)
                ev = (
                    ev - reference_ev
                    if ev is not None and reference_ev is not None
                    else None
                )
                a = (
                    a - reference_auc
                    if a is not None and reference_auc is not None
                    else None
                )
            if ev is not None:
                values.append(ev)
            if a is not None:
                aucs.append(a)

        def interval(samples):
            if not samples:
                return None
            return torch.quantile(
                torch.tensor(samples, dtype=torch.float64),
                torch.tensor([0.025, 0.975], dtype=torch.float64),
            ).tolist()

        output[domain] = {
            "ev_95ci": interval(values),
            "auc_95ci": interval(aucs),
            "defined_ev_draws": len(values),
            "defined_auc_draws": len(aucs),
            "task_clusters": len(clusters),
        }
    return {
        "paired_difference": reference is not None,
        "draws": draws,
        "seed": seed,
        "by_domain": output,
    }


class FeatureStore:
    """Cache each CPU feature shard on first access for all subsequent passes."""

    _feature_semantics = ("score_input_final_norm_shifted_action_prefix",)

    def __init__(self, manifest: Path):
        self.path = Path(manifest)
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("schema") != "tau2-critic-features-v1":
            raise ValueError("Unsupported feature manifest schema")
        if self.manifest.get("feature_semantics") not in self._feature_semantics:
            raise ValueError("Features must be causal shifted-action score inputs")
        self.hidden_size = int(self.manifest["hidden_size"])
        self.episodes: list[dict[str, Any]] = []
        self.sampled_training = False
        self._train: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]] | None
        ) = None
        self._shards: dict[int, tuple[torch.Tensor, dict[str, Any]]] = {}

    def shards(
        self, split: str | None = None
    ) -> Iterator[tuple[torch.Tensor, dict[str, Any], torch.Tensor]]:
        for index, record in enumerate(self.manifest["shards"]):
            if index not in self._shards:
                path = Path(record["path"])
                if not path.is_absolute():
                    path = self.path.parent / path
                if "sha256" in record:
                    hasher = hashlib.sha256()
                    with path.open("rb") as source:
                        while data := source.read(1024 * 1024):
                            hasher.update(data)
                    if hasher.hexdigest() != record["sha256"]:
                        raise ValueError(f"Feature shard hash mismatch: {path}")
                shard = torch.load(path, map_location="cpu", weights_only=True)
                x = shard["features"].float()
                if x.ndim != 2 or x.shape[1] != self.hidden_size:
                    raise ValueError(f"Incorrect feature shape in {path}")
                self._shards[index] = x, shard
            x, shard = self._shards[index]
            if split is None:
                indices = torch.arange(len(x))
            else:
                indices = torch.tensor(
                    [i for i, value in enumerate(shard["split"]) if value == split],
                    dtype=torch.long,
                )
            if indices.numel():
                yield x[indices], shard, indices

    def training(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        if self._train is not None:
            return self._train
        features, rewards, weights, identities = [], [], [], []
        episodes = {}
        for x, shard, indices in self.shards("train"):
            features.append(x)
            rewards.append(torch.as_tensor(shard["reward"])[indices].float())
            weights.append(torch.as_tensor(shard["sampling_weight"])[indices].float())
            self.sampled_training |= bool(
                torch.as_tensor(shard["sampling_weight"])[indices].ne(1).any()
            )
            for i in indices.tolist():
                identity = str(shard["trajectory_id"][i])
                identities.append(identity)
                metadata = self.metadata(shard, i)
                if identity in episodes and episodes[identity] != metadata:
                    raise ValueError(f"Inconsistent trajectory metadata: {identity}")
                episodes[identity] = metadata
        if not features:
            raise ValueError("No training features")
        self.episodes = list(episodes.values())
        self._train = (
            torch.cat(features),
            torch.cat(rewards),
            torch.cat(weights),
            identities,
        )
        return self._train

    @staticmethod
    def metadata(shard: dict[str, Any], i: int) -> dict[str, Any]:
        result = {}
        for name in (
            "trajectory_id",
            "step",
            "order",
            "domain",
            "reward",
            "task_id",
            "run",
            "original_action_count",
            "truncated",
        ):
            if name in shard:
                value = shard[name][i]
                result[name] = (
                    value.item() if isinstance(value, torch.Tensor) else value
                )
        result["episode_id"] = str(result["trajectory_id"])
        return result

    @torch.no_grad()
    def evaluate(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | float,
        *,
        logistic: bool = False,
        split: str = "validation",
    ) -> dict[str, Any]:
        # Accumulate per-episode production rows, preserving shard order even
        # when an episode spans multiple cached shards.
        accumulated: dict[str, dict[str, Any]] = {}
        expected_counts = {}
        for x, shard, indices in self.shards(split):
            if torch.as_tensor(shard["sampling_weight"])[indices].ne(1).any():
                raise ValueError(
                    "Validation requires every action prefix; sampled EV is an approximation"
                )
            prediction = x @ weight.float() + bias
            if logistic:
                prediction = prediction.sigmoid()
            grouped: dict[str, list[int]] = defaultdict(list)
            for local, original in enumerate(indices.tolist()):
                grouped[str(shard["trajectory_id"][original])].append(local)
            for identity, selected in grouped.items():
                metadata = self.metadata(shard, int(indices[selected[0]]))
                expected_counts[identity] = int(metadata["original_action_count"])
                metric = official_metric_row(prediction[selected], metadata)
                if identity not in accumulated:
                    accumulated[identity] = metric
                else:
                    old = accumulated[identity]
                    n_old, n_new = old["n_tokens"], metric["n_tokens"]
                    old["prediction_mean"] = (
                        old["prediction_mean"] * n_old
                        + metric["prediction_mean"] * n_new
                    ) / (n_old + n_new)
                    old["prediction_min"] = min(
                        old["prediction_min"], metric["prediction_min"]
                    )
                    old["prediction_max"] = max(
                        old["prediction_max"], metric["prediction_max"]
                    )
                    for key in (
                        "n_tokens",
                        "error_sum",
                        "sse",
                        "target_sum",
                        "target2_sum",
                    ):
                        old[key] += metric[key]
        if not accumulated:
            raise ValueError(f"No {split} features")
        for identity, metric in accumulated.items():
            if metric["n_tokens"] != expected_counts[identity]:
                raise ValueError(
                    f"Incomplete validation prefixes for {identity}: {metric['n_tokens']} != {expected_counts[identity]}"
                )
        return aggregate_metrics(list(accumulated.values()))


def weighted_mse(
    x: torch.Tensor,
    y: torch.Tensor,
    mass: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Production eps_clip=None loss: squared error, weighted token mean."""
    return ((x @ weight + bias - y).square() * mass).sum() / mass.sum()


def fit_ridge(
    x: torch.Tensor, y: torch.Tensor, mass: torch.Tensor, penalty: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve weighted ridge; the intercept is unpenalised."""
    dimension = x.shape[1] + 1
    gram = torch.zeros((dimension, dimension), dtype=torch.float64)
    rhs = torch.zeros(dimension, dtype=torch.float64)
    total_mass = mass.double().sum()
    for start in range(0, len(x), 8192):
        sl = slice(start, start + 8192)
        design = torch.cat(
            [x[sl].double(), torch.ones((len(x[sl]), 1), dtype=torch.float64)], dim=1
        )
        scaled_mass = mass[sl].double() / total_mass
        gram.add_(design.T @ (design * scaled_mass[:, None]))
        rhs.add_(design.T @ (y[sl].double() * scaled_mass))
    regularizer = torch.eye(gram.shape[0], dtype=torch.float64) * penalty
    regularizer[-1, -1] = 0
    coefficients = torch.linalg.solve(gram + regularizer, rhs)
    return coefficients[:-1].float(), coefficients[-1].float()


def fit_logistic(
    x: torch.Tensor,
    y: torch.Tensor,
    mass: torch.Tensor,
    penalty: float,
    *,
    iterations: int = 60,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train-only weighted BCE probe using CPU L-BFGS, with no validation fitting."""
    weight = torch.zeros(x.shape[1], requires_grad=True)
    prior = ((mass * y).sum() / mass.sum()).clamp(1e-4, 1 - 1e-4)
    bias = prior.logit().detach().requires_grad_()
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=iterations, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        # Chunked backward prevents validation-size-like activation memory.
        objective = torch.zeros(())
        for start in range(0, len(x), 8192):
            sl = slice(start, start + 8192)
            loss = (
                F.binary_cross_entropy_with_logits(
                    x[sl] @ weight + bias, y[sl], reduction="none"
                )
                * mass[sl]
            ).sum() / mass.sum()
            loss.backward()
            objective += loss.detach()
        regularization = penalty * weight.square().sum() / 2
        regularization.backward()
        return objective + regularization.detach()

    optimizer.step(closure)
    return weight.detach(), bias.detach()


def train_only_probes(
    store: FeatureStore,
    *,
    penalties: tuple[float, ...] = (0.01, 0.1, 1.0),
    export_dir: Path | None = None,
) -> dict[str, Any]:
    """Choose regularisation on a deterministic task-grouped training holdout."""
    x, y, mass, identities = store.training()
    metadata = {row["trajectory_id"]: row for row in store.episodes}
    holdout = torch.tensor(
        [
            int(
                hashlib.sha256(
                    f"{metadata[i]['domain']}:{metadata[i]['task_id']}".encode()
                ).hexdigest()[:8],
                16,
            )
            % 5
            == 0
            for i in identities
        ]
    )
    if not holdout.any() or holdout.all():
        raise ValueError("Training task holdout must contain fitting and tuning tasks")
    output = {}
    for kind, fitter in (("ridge", fit_ridge), ("logistic", fit_logistic)):
        candidates = []
        for penalty in penalties:
            weight, bias = fitter(x[~holdout], y[~holdout], mass[~holdout], penalty)
            prediction = x[holdout] @ weight + bias
            if kind == "logistic":
                prediction = prediction.sigmoid()
            score = float(
                ((prediction - y[holdout]).square() * mass[holdout]).sum()
                / mass[holdout].sum()
            )
            candidates.append({"penalty": penalty, "training_holdout_mse": score})
        selected = min(
            candidates, key=lambda candidate: candidate["training_holdout_mse"]
        )["penalty"]
        weight, bias = fitter(x, y, mass, selected)
        head_path = None
        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            head_path = export_dir / f"{kind}-head.pt"
            torch.save(
                {
                    "base_checkpoint": store.manifest.get(
                        "base_checkpoint", store.manifest.get("checkpoint")
                    ),
                    "state_dict": {
                        "score.weight": weight[None],
                        "score.bias": bias.reshape(1),
                    },
                    "feature_manifest": str(store.path.resolve()),
                    "feature_manifest_sha256": hashlib.sha256(
                        store.path.read_bytes()
                    ).hexdigest(),
                    "feature_semantics": store.manifest["feature_semantics"],
                    "fit": kind,
                    "selected_penalty": selected,
                    "requires_sigmoid": kind == "logistic",
                },
                head_path,
            )
        output[kind] = {
            "selection": "task-grouped training holdout only; validation never fitted",
            "candidates": candidates,
            "selected_penalty": selected,
            "weight_norm": float(weight.norm()),
            "bias": float(bias),
            "head_path": str(head_path.resolve()) if head_path is not None else None,
            "validation": store.evaluate(weight, bias, logistic=kind == "logistic"),
        }
    return output


def simulate_adam(
    store: FeatureStore,
    *,
    lr: float,
    frozen_bias: bool = False,
    warmup_steps: int = 0,
    total_steps: int = 80,
    eval_every: int = 5,
    bias_init: float = 0.78,
    replay_run: str = "r510",
    batch_size: int = 16,
) -> dict[str, Any]:
    """Original recorded episode order with head-only AdamW and token mass."""
    x, y, mass, identities = store.training()
    replay_rows = [row for row in store.episodes if row.get("run") == replay_run]
    if not replay_rows:
        raise ValueError(f"No saved training episodes for replay run {replay_run}")
    batches = replay_batches(
        replay_rows, batch_size=batch_size, total_steps=total_steps
    )
    indices: dict[str, list[int]] = defaultdict(list)
    for i, identity in enumerate(identities):
        indices[identity].append(i)
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.tensor(bias_init, requires_grad=not frozen_bias)
    parameters = [weight] if frozen_bias else [weight, bias]
    optimizer = torch.optim.AdamW(
        parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    baseline = store.evaluate(weight, bias)
    evaluations = [
        {
            "step": 0,
            "weight_norm": 0.0,
            "bias": float(bias.detach()),
            "validation": baseline,
        }
    ]
    steps, gradients, initial_gradients = [], [], []
    for step, batch in enumerate(batches):
        selected = torch.tensor(
            [i for row in batch for i in indices[row["trajectory_id"]]],
            dtype=torch.long,
        )
        # Transformers constant-with-warmup LambdaLR starts at multiplier 0;
        # engine advances scheduler after the optimizer step.
        consumed_lr = lr * min(1.0, step / warmup_steps) if warmup_steps else lr
        optimizer.param_groups[0]["lr"] = consumed_lr
        optimizer.zero_grad()
        initial_gradients.append(
            (
                2 * ((bias_init - y[selected]) * mass[selected])[:, None] * x[selected]
            ).sum(0)
            / mass[selected].sum()
        )
        loss = weighted_mse(x[selected], y[selected], mass[selected], weight, bias)
        loss.backward()
        gradients.append(weight.grad.detach().clone())
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        steps.append(
            {
                "step": step + 1,
                "recorded_step": int(batch[0]["step"]),
                "trajectory_ids": [row["trajectory_id"] for row in batch],
                "episodes": len(batch),
                "sampled_tokens": len(selected),
                "original_token_mass": float(mass[selected].sum()),
                "loss": float(loss.detach()),
                "lr": consumed_lr,
                "preclip_grad_norm": float(norm),
                "weight_norm": float(weight.detach().norm()),
                "bias": float(bias.detach()),
            }
        )
        if (step + 1) % eval_every == 0 or step + 1 == total_steps:
            evaluations.append(
                {
                    "step": step + 1,
                    "weight_norm": float(weight.detach().norm()),
                    "bias": float(bias.detach()),
                    "validation": store.evaluate(weight, bias),
                }
            )
    gradient = torch.stack(gradients)
    signs = gradient.sign()
    active = signs.ne(0).sum(0)
    consistency = signs.sum(0).abs() / active.clamp_min(1)
    majority_fraction = (1 + consistency) / 2
    adjacent_active = signs[1:].ne(0) & signs[:-1].ne(0)
    adjacent_sign_agreement = (
        signs[1:].eq(signs[:-1]) & adjacent_active
    ).sum() / adjacent_active.sum().clamp_min(1)
    rms = gradient.square().mean(0).sqrt()
    adjacent = (gradient[1:] * gradient[:-1]).sum(1) / (
        gradient[1:].norm(dim=1) * gradient[:-1].norm(dim=1)
    ).clamp_min(1e-30)
    fixed_sign = torch.stack(initial_gradients).sign()
    fixed_consistency = fixed_sign.sum(0).abs() / fixed_sign.ne(0).sum(0).clamp_min(1)
    final_domains = evaluations[-1]["validation"]["rows"]
    r510_rows = [row for row in final_domains if row["run"] == replay_run]
    r510_metrics = aggregate_metrics(r510_rows) if r510_rows else None
    norm_ratio = float(weight.detach().norm()) / 0.00240323646
    near_norm = 0.5 <= norm_ratio <= 2.0
    near_ev = r510_metrics is not None and all(
        summary["explained_variance"] is not None
        and abs(summary["explained_variance"]) <= 0.05
        for summary in r510_metrics["by_domain"].values()
    )
    return {
        "lr": lr,
        "frozen_bias": frozen_bias,
        "warmup_steps": warmup_steps,
        "bias_init": bias_init,
        "replay_run": replay_run,
        "steps": steps,
        "evaluations": evaluations,
        "gradient_sign_consistency": {
            "definition": "abs(sum(sign(g)))/nonzero_steps per weight coordinate, preclip; gradients at the changing trained head",
            "mean": float(consistency.mean()),
            "median": float(consistency.median()),
            "gradient_rms_weighted_mean": float(
                (consistency * rms).sum() / rms.sum().clamp_min(1e-30)
            ),
            "fraction_at_least_0_8": float((consistency >= 0.8).float().mean()),
            "adjacent_gradient_cosines": adjacent.tolist(),
            "per_coordinate": consistency.tolist(),
            "majority_sign_fraction_mean_active_coordinates": float(
                majority_fraction[active > 0].mean()
            )
            if (active > 0).any()
            else None,
            "adjacent_sign_agreement_fraction": float(adjacent_sign_agreement),
            "fixed_initial_head": {
                "mean": float(fixed_consistency.mean()),
                "median": float(fixed_consistency.median()),
                "fraction_at_least_0_8": float(
                    (fixed_consistency >= 0.8).float().mean()
                ),
                "per_coordinate": fixed_consistency.tolist(),
            },
        },
        "sanity": {
            "real_r510_head_norm_reference": 0.00240323646,
            "real_r510_final_bias_reference": 0.78125,
            "simulation_head_norm": float(weight.detach().norm()),
            "head_norm_ratio_to_real": norm_ratio,
            "norm_near_reference": near_norm,
            "norm_tolerance": "between 0.5x and 2x final real norm, declared diagnostic tolerance",
            "r510_only_final_metrics": r510_metrics,
            "ev_near_zero": near_ev,
            "ev_tolerance": "absolute domain EV <= 0.05",
            "real_r510_final_ev_reference": {"airline": -0.005, "retail": 0.001},
            "status": "near_reference" if near_norm and near_ev else "discrepancy",
            "interpretation": "frozen-backbone diagnostic; proximity does not verify original changing-backbone reproduction",
            "feature_sampling": "sampled training: mass-weighted approximation"
            if store.sampled_training
            else "all saved training action prefixes",
            "arithmetic": "CPU float32 head versus distributed backbone computation and real bf16 parameter rounding",
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--replay-run", default="r510")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    store = FeatureStore(args.features)
    output: dict[str, Any] = {
        "feature_manifest": str(args.features.resolve()),
        "feature_manifest_sha256": hashlib.sha256(
            args.features.read_bytes()
        ).hexdigest(),
        "backbone": "frozen saved features",
        "loss": "original unhalved token_mean MSE; no value clipping",
        "validation": "official _metric_row and aggregate_validation; all saved validation prefixes",
        "selection_warning": "validation is reported at each step; no best-validation step is selected",
    }
    if not args.skip_probes:
        output["probes"] = train_only_probes(store, export_dir=args.out)
        for probe in output["probes"].values():
            probe["validation"]["bootstrap"] = cluster_bootstrap(
                probe["validation"]["rows"], draws=args.bootstrap_draws
            )
        output["probes"]["logistic"]["paired_vs_ridge"] = cluster_bootstrap(
            output["probes"]["logistic"]["validation"]["rows"],
            reference=output["probes"]["ridge"]["validation"]["rows"],
            draws=args.bootstrap_draws,
        )
        (args.out / "probes.json").write_text(json.dumps(output, indent=2) + "\n")
    arms = []
    reference = None
    for lr in (5e-6, 2e-5, 1e-4, 1e-3):
        for frozen_bias in (False, True):
            for warmup_steps in (0, 10):
                result = simulate_adam(
                    store,
                    lr=lr,
                    frozen_bias=frozen_bias,
                    warmup_steps=warmup_steps,
                    total_steps=args.steps,
                    replay_run=args.replay_run,
                )
                final_rows = result["evaluations"][-1]["validation"]["rows"]
                if reference is None:
                    reference = final_rows
                result["bootstrap"] = cluster_bootstrap(
                    final_rows, draws=args.bootstrap_draws
                )
                result["paired_vs_original_head_arm"] = cluster_bootstrap(
                    final_rows, reference=reference, draws=args.bootstrap_draws
                )
                name = f"adam-lr{lr:g}-bias{'frozen' if frozen_bias else 'learned'}-warmup{warmup_steps}.json"
                (args.out / name).write_text(json.dumps(result, indent=2) + "\n")
                arms.append(
                    {
                        "path": name,
                        "lr": lr,
                        "frozen_bias": frozen_bias,
                        "warmup_steps": warmup_steps,
                        "final_weight_norm": result["steps"][-1]["weight_norm"],
                        "final_by_domain": result["evaluations"][-1]["validation"][
                            "by_domain"
                        ],
                        "bootstrap": result["bootstrap"],
                        "paired_vs_original_head_arm": result[
                            "paired_vs_original_head_arm"
                        ],
                    }
                )
    output["adam_arms"] = arms
    (args.out / "simulation.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
