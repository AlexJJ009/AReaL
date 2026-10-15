# SPDX-License-Identifier: Apache-2.0
"""Protocol E2–E9 CPU experiments; synthetic outputs are execution checks only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.tau2.critic_head_sim import (
    FeatureStore,
    aggregate_metrics,
    auc,
    fit_logistic,
    fit_ridge,
    official_metric_row,
    replay_batches,
    weighted_mse,
)
from scripts.tau2.critic_preexp_stats import SEED, TaskBootstrap

LRS = (5e-6, 2e-5, 1e-4, 3e-4, 1e-3)
PENALTIES = (1e-5, 1e-3, 0.01, 0.1, 1.0, 10.0)
SEEDS = (SEED, SEED + 1, SEED + 2)


QWEN3_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"

PINNED_BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def checkpoint_fingerprint(manifest, *, synthetic=False):
    """Require checkpoint identity plus configuration and weight-file binding."""
    checkpoint = manifest.get("checkpoint")
    if not checkpoint:
        raise ValueError("Cache checkpoint identity is missing")
    config_hash = manifest.get("checkpoint_config_sha256")
    weights = manifest.get("checkpoint_files_sha256")
    if not config_hash or not weights:
        if not synthetic or bool(config_hash) != bool(weights):
            raise ValueError(
                "Cache requires nonempty checkpoint config and weights SHA256 fingerprints"
            )
    else:
        hashes = [config_hash, *weights.values()] if isinstance(weights, dict) else []
        if not hashes or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
            for value in hashes
        ):
            raise ValueError("Cache checkpoint SHA256 fingerprints are invalid")
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_config_sha256": config_hash,
        "checkpoint_files_sha256": weights,
    }


def validate_primary_cache(manifest, *, synthetic=False):
    if (
        manifest.get("layer") != "final"
        or manifest.get("feature_semantics")
        != "score_input_final_norm_shifted_action_prefix"
    ):
        raise ValueError(
            "Primary features must be the base model final score-input layer"
        )
    fingerprint = checkpoint_fingerprint(manifest, synthetic=synthetic)
    revision = (
        QWEN3_REVISION
        if manifest.get("protocol") == "qwen3-20261015"
        else PINNED_BASE_REVISION
    )
    if not synthetic and revision not in Path(fingerprint["checkpoint"]).parts:
        raise ValueError(
            "Primary checkpoint must be the registered pinned base snapshot"
        )
    return fingerprint


def validate_e7_cache_alias(alias, manifest, base_manifest, *, synthetic=False):
    """Bind an E7 arm name to its actual layer and registered backbone identity."""
    from scripts.tau2.critic_preexp_extract import layer_indices

    base = validate_primary_cache(base_manifest, synthetic=synthetic)
    if bool(manifest.get("synthetic", False)) != synthetic:
        raise ValueError("E7 cache synthetic provenance mismatch")
    fingerprint = checkpoint_fingerprint(manifest, synthetic=synthetic)
    if manifest.get("depth") != base_manifest.get("depth") or manifest.get(
        "hidden_size"
    ) != base_manifest.get("hidden_size"):
        raise ValueError("E7 cache backbone depth/hidden size differs from base")
    expected = {
        f"base-layer-{index}": str(index)
        for index in layer_indices(int(base_manifest["depth"]))
    }
    if alias in expected:
        if (
            str(manifest.get("layer")) != expected[alias]
            or manifest.get("feature_semantics")
            != "intermediate_layer_output_shifted_action_prefix"
        ):
            raise ValueError("E7 intermediate alias does not match the cached layer")
        if fingerprint != base:
            raise ValueError(
                "E7 intermediate layer checkpoint/fingerprints differ from base"
            )
    elif alias == (
        "r480-final"
        if base_manifest.get("protocol") == "qwen3-20261015"
        else "r510-final"
    ):
        if (
            manifest.get("layer") != "final"
            or manifest.get("feature_semantics")
            != "score_input_final_norm_shifted_action_prefix"
        ):
            raise ValueError("E7 r510-final alias requires final score-input features")
        if fingerprint["checkpoint"] == base["checkpoint"]:
            raise ValueError(
                "E7 r510-final checkpoint must differ from the base checkpoint"
            )
        if not synthetic:
            checkpoint = Path(fingerprint["checkpoint"])
            if (
                (
                    "critic-instruct-formal-v1"
                    if alias == "r480-final"
                    else "critic-head-qwen35-mix-r2"
                )
                not in checkpoint.parts
                or checkpoint.name
                != (
                    "epoch1epochstep46globalstep93"
                    if alias == "r480-final"
                    else "epoch2epochstep14globalstep54"
                )
                or checkpoint.parent.name != "critic"
            ):
                raise ValueError(
                    "E7 r510-final requires the registered r510 step55 critic checkpoint"
                )
            if fingerprint["checkpoint_files_sha256"] == base[
                "checkpoint_files_sha256"
            ] or set(fingerprint["checkpoint_files_sha256"].values()) == set(
                base["checkpoint_files_sha256"].values()
            ):
                raise ValueError(
                    "E7 r510-final weight fingerprints must differ from the base checkpoint"
                )
    else:
        raise ValueError(
            "E7 alias is not one of the three registered quarter-depth layers or r510-final"
        )
    return {
        **fingerprint,
        "layer": manifest["layer"],
        "depth": manifest["depth"],
        "hidden_size": manifest["hidden_size"],
    }


def grouped_folds(tasks, seed=SEED):
    keys = sorted(set(tasks))
    if len(keys) < 5:
        raise ValueError("Five-fold CV requires at least five training tasks")
    permutation = torch.randperm(
        len(keys), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    mapping = {keys[index]: rank % 5 for rank, index in enumerate(permutation)}
    return torch.tensor([mapping[key] for key in tasks])


def soft_labels(y, tasks):
    values = defaultdict(list)
    for label, task in zip(y.tolist(), tasks, strict=True):
        values[task].append(label)
    means = {key: sum(labels) / len(labels) for key, labels in values.items()}
    return torch.tensor([means[key] for key in tasks], dtype=y.dtype)


def e3_grid():
    return [
        dict(lr=lr, frozen_bias=frozen, warmup_steps=warmup)
        for lr in LRS
        for frozen in (False, True)
        for warmup in (0, 10)
    ]


def h2_status(norm, sim_airline_ev, reference_ev, reference_step=80):
    if reference_step != 80 or reference_ev is None or sim_airline_ev is None:
        raise ValueError("H2 needs real r510 Airline EV at the same completed step 80")
    delta = abs(sim_airline_ev - reference_ev)
    norm_ok = (
        norm > 0.0012 or math.isclose(norm, 0.0012, rel_tol=0, abs_tol=1e-12)
    ) and (norm < 0.005 or math.isclose(norm, 0.005, rel_tol=0, abs_tol=1e-12))
    delta_ok = delta < 0.1 or math.isclose(delta, 0.1, rel_tol=0, abs_tol=1e-12)
    return {
        "status": "成立" if norm_ok and delta_ok else "不成立",
        "weight_norm": norm,
        "norm_interval": [0.0012, 0.005],
        "simulation_r510_airline_ev": sim_airline_ev,
        "real_r510_airline_ev": reference_ev,
        "airline_ev_absolute_difference": delta,
        "ev_tolerance": 0.1,
        "completed_step": 80,
    }


def e4_budgets(batch):
    if batch not in (16, 64, 128):
        raise ValueError("Protocol batch must be16,64,128")
    return {"A": 1280 // batch, "B": 80}


def gradient_summary(gradients, full_gradients):
    gradient = torch.stack(gradients) if isinstance(gradients, list) else gradients
    full = (
        torch.stack(full_gradients)
        if isinstance(full_gradients, list)
        else full_gradients
    )
    signs = gradient.sign()
    active = signs.ne(0).sum(0)
    consistency = signs.sum(0).abs() / active.clamp_min(1)
    adjacent = F.cosine_similarity(gradient[1:], gradient[:-1], dim=1)
    whole = F.cosine_similarity(gradient, full, dim=1)
    return {
        "adjacent_cosines": adjacent.tolist(),
        "full_data_cosines": whole.tolist(),
        "mean_full_data_cosine": float(whole.mean()),
        "coordinate_sign_consistency": consistency.tolist(),
        "coordinate_majority_sign_fraction": ((1 + consistency) / 2).tolist(),
        "fraction_coordinates_same_sign_all_steps": float(
            ((consistency == 1) & (active == len(gradient))).float().mean()
        ),
        "sign_definition": "abs(sum(sign(g)))/number_of_nonzero_steps; changing trained head, before clipping",
    }


def position_group(progress, first=False):
    if first:
        return "first_action_token"
    if not 0 <= progress <= 1:
        raise ValueError("Progress must be between zero and one")
    index = min(4, int(progress * 5))
    return f"{index * 20}-{(index + 1) * 20}%"


def validate_planted_prevalence(labels, mass, domains=None):
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Planted prefix labels must be binary")
    fraction = float((labels * mass).sum() / mass.sum())
    if not (0.2 <= fraction <= 0.6 and 0.2 <= float(labels.mean()) <= 0.6):
        raise ValueError("Planted signal prevalence outside pooled 20–60%")
    if domains is not None:
        if len(domains) != len(labels):
            raise ValueError("Planted label domains are not aligned")
        for domain in sorted(set(domains)):
            selected = torch.tensor([value == domain for value in domains])
            weighted = float(
                (labels[selected] * mass[selected]).sum() / mass[selected].sum()
            )
            sampled = float(labels[selected].mean())
            if not (0.05 <= weighted <= 0.95 and 0.05 <= sampled <= 0.95):
                raise ValueError(
                    f"Planted signal prevalence for domain {domain} outside 5–95%"
                )
    return fraction


def export_head(store, weight, bias, out, penalty):
    path = Path(out)
    artifact = {
        "base_checkpoint": store.manifest.get(
            "base_checkpoint", store.manifest.get("checkpoint")
        ),
        "state_dict": {
            "score.weight": weight.detach().cpu()[None],
            "score.bias": torch.as_tensor(bias).detach().cpu().reshape(1),
        },
        "feature_manifest": str(store.path.resolve()),
        "feature_manifest_sha256": hashlib.sha256(store.path.read_bytes()).hexdigest(),
        "pool_manifest_sha256": store.manifest.get("pool_manifest_sha256"),
        "feature_semantics": store.manifest["feature_semantics"],
        "fit": "ridge",
        "selected_penalty": penalty,
        "requires_sigmoid": False,
        "selection": "five-fold grouped training CV only",
    }
    torch.save(artifact, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    for key, value in artifact["state_dict"].items():
        torch.testing.assert_close(value, loaded["state_dict"][key], rtol=0, atol=0)
    return path


class PreexpStore(FeatureStore):
    def __init__(self, manifest, *, resident=True):
        # Intermediate hidden states deliberately differ from final score input.
        self.path = Path(manifest)
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("schema") != "tau2-critic-features-v1":
            raise ValueError("Unsupported feature manifest")
        semantics = self.manifest.get("feature_semantics", "")
        if semantics not in (
            "score_input_final_norm_shifted_action_prefix",
            "intermediate_layer_output_shifted_action_prefix",
        ):
            raise ValueError("Unknown causal prefix feature semantics")
        self.hidden_size = int(self.manifest["hidden_size"])
        self.episodes = []
        self.sampled_training = False
        self._train = None
        self.resident = resident
        self._resident_shards = None

    def shards(self, split=None):
        """Hash/load each feature shard once; keep arithmetic and row order intact."""
        if not self.resident:
            yield from super().shards(split)
            return
        if self._resident_shards is None:
            self._resident_shards = list(super().shards())
        for x, shard, indices in self._resident_shards:
            if split is not None:
                indices = torch.tensor(
                    [i for i, value in enumerate(shard["split"]) if value == split],
                    dtype=torch.long,
                )
            if len(indices):
                yield x[indices], shard, indices

    @staticmethod
    def metadata(shard, index):
        row = FeatureStore.metadata(shard, index)
        if "source_id" in shard:
            row["source_id"] = str(shard["source_id"][index])
        return row

    def token_fields(self, split="train"):
        fields = defaultdict(list)
        for _, shard, indices in self.shards(split):
            for index in indices.tolist():
                for key in (
                    "domain",
                    "task_id",
                    "trajectory_id",
                    "progress",
                    "turn",
                    "planted_label",
                    "source_id",
                ):
                    if key not in shard:
                        raise ValueError(f"Feature cache missing {key}")
                    value = shard[key][index]
                    fields[key].append(
                        value.item() if isinstance(value, torch.Tensor) else value
                    )
        return dict(fields)

    @torch.no_grad()
    def evaluate_targets(self, models, *, logistic=False, planted=False, position=None):
        accumulated = {}
        expected = {}
        for x, shard, indices in self.shards("validation"):
            if torch.as_tensor(shard["sampling_weight"])[indices].ne(1).any():
                raise ValueError("Validation requires all prefixes")
            groups = defaultdict(list)
            for local, index in enumerate(indices.tolist()):
                identity = str(shard["trajectory_id"][index])
                count = int(shard["original_action_count"][index])
                if identity in expected and expected[identity] != count:
                    raise ValueError("Inconsistent original action count")
                expected[identity] = count
                if position is not None:
                    progress = float(shard["progress"][index])
                    if position == "first_action_token":
                        if progress != 0:
                            continue
                    elif position_group(progress) != position:
                        continue
                target = (
                    float(shard["planted_label"][index])
                    if planted
                    else float(shard["reward"][index])
                )
                groups[(identity, target)].append((local, index))
            for (identity, target), selected in groups.items():
                local, index = zip(*selected, strict=True)
                metadata = self.metadata(shard, index[0])
                domain = metadata["domain"]
                weight, bias = models.get(domain, models.get("shared"))
                prediction = x[list(local)] @ weight + bias
                if logistic:
                    prediction = prediction.sigmoid()
                metadata["reward"] = target
                row = official_metric_row(prediction, metadata)
                row["source_id"] = metadata.get("source_id")
                if identity not in accumulated:
                    accumulated[identity] = row
                else:
                    old = accumulated[identity]
                    count = old["n_tokens"] + row["n_tokens"]
                    old["prediction_mean"] = (
                        old["prediction_mean"] * old["n_tokens"]
                        + row["prediction_mean"] * row["n_tokens"]
                    ) / count
                    old["prediction_min"] = min(
                        old["prediction_min"], row["prediction_min"]
                    )
                    old["prediction_max"] = max(
                        old["prediction_max"], row["prediction_max"]
                    )
                    for key in (
                        "n_tokens",
                        "sse",
                        "error_sum",
                        "target_sum",
                        "target2_sum",
                    ):
                        old[key] += row[key]
                accumulated[identity]["target"] = (
                    accumulated[identity]["target_sum"]
                    / accumulated[identity]["n_tokens"]
                )
        if not accumulated:
            raise ValueError("No selected validation prefixes")
        if position is None:
            for identity, row in accumulated.items():
                if row["n_tokens"] != expected[identity]:
                    raise ValueError("Validation episode missing cached prefixes")
                # Also enforce original token count, not merely cached count.
                for record in self.manifest.get("episodes", []):
                    if (
                        record.get("trajectory_id") == identity
                        and row["n_tokens"] != record["original_action_count"]
                    ):
                        raise ValueError("Validation original token count mismatch")
        result = aggregate_metrics(list(accumulated.values()))
        result["auc"]["overall"] = auc(result["rows"])
        result["by_run_auc"] = {
            run: {
                "overall": auc([r for r in result["rows"] if r["run"] == run]),
                **{
                    domain: auc(
                        [
                            r
                            for r in result["rows"]
                            if r["run"] == run and r["domain"] == domain
                        ]
                    )
                    for domain in result["by_domain"]
                },
            }
            for run in result["by_run_domain"]
        }
        if planted and any(row["target"] not in (0, 1) for row in result["rows"]):
            result["auc"] = {key: None for key in result["auc"]}
            result["by_run_auc"] = {
                run: {key: None for key in values}
                for run, values in result["by_run_auc"].items()
            }
            result["auc_undefined_reason"] = (
                "Prefix-local labels vary within a trajectory; no binary trajectory AUC target exists"
            )
        return result


class Preexperiment:
    seeds = SEEDS
    seed = SEED

    def __init__(
        self,
        features,
        out,
        *,
        pool_manifest=None,
        synthetic=False,
        r510_reference=None,
        seed=SEED,
    ):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.synthetic = synthetic
        self.seed = seed
        self.seeds = (seed, seed + 1, seed + 2)
        self.store = PreexpStore(features)
        self.reference_path = r510_reference
        validate_primary_cache(self.store.manifest, synthetic=synthetic)
        self.pool_manifest_path = Path(pool_manifest) if pool_manifest else None
        if pool_manifest is None:
            raise ValueError("Every analysis requires a verified pool manifest")
        from scripts.tau2.critic_preexp_data import load_pool
        from scripts.tau2.critic_preexp_extract import verify_cache

        self.pool_metadata, self.pool = load_pool(
            Path(pool_manifest), synthetic=synthetic
        )
        if (
            bool(self.pool_metadata.get("synthetic", False)) != synthetic
            or bool(self.store.manifest.get("synthetic", False)) != synthetic
        ):
            raise ValueError("Synthetic mode must match both pool and cache provenance")
        verify_cache(features, self.pool, self.pool_metadata, synthetic=synthetic)
        pool_hash = hashlib.sha256(Path(pool_manifest).read_bytes()).hexdigest()
        if self.store.manifest.get("pool_manifest_sha256") != pool_hash:
            raise ValueError("Cache is not bound to the verified pool manifest")
        cache_ids = set()
        for _, shard, indices in self.store.shards():
            cache_ids.update(str(shard["trajectory_id"][i]) for i in indices.tolist())
        if cache_ids != {str(row["_trajectory_id"]) for row in self.pool}:
            raise ValueError("Cache trajectories differ from manifest pool")
        self.x, self.y, self.mass, self.identities = self.store.training()
        self.fields = self.store.token_fields()
        self.tasks = list(
            zip(self.fields["domain"], self.fields["task_id"], strict=True)
        )
        self.folds = grouped_folds(self.tasks, seed=self.seed)
        self.episode_map = {row["trajectory_id"]: row for row in self.store.episodes}
        self.indices = defaultdict(list)
        for index, identity in enumerate(self.identities):
            self.indices[identity].append(index)
        initial = self.store.evaluate_targets(
            {"shared": (torch.zeros(self.x.shape[1]), torch.tensor(0.78))}
        )
        self.bootstrap = TaskBootstrap(initial["rows"], seed=self.seed)
        self.provenance = {
            "synthetic": synthetic,
            "scientific_interpretation": "disabled: synthetic execution check"
            if synthetic
            else "real frozen-feature offline experiment",
            "pool_manifest": str(Path(pool_manifest).resolve()),
            "pool_manifest_sha256": pool_hash,
            "feature_manifest": str(self.store.path.resolve()),
            "feature_manifest_sha256": hashlib.sha256(
                self.store.path.read_bytes()
            ).hexdigest(),
            "seeds": list(self.seeds),
            "bootstrap_draws": 20000,
            "bootstrap_tasks": 60,
            "cv": "five folds grouped by domain/task; training only",
            "loss": "production unhalved token-weighted MSE; max-grad-norm1; AdamW weight_decay0",
        }
        self.results = {}
        self.ridge_models = None
        self.lr_cv = None
        self.lr_cv_cache = {}

    def save(self, name, value):
        self.results[name] = value
        (self.out / f"{name}.json").write_text(
            json.dumps(
                {"experiment": name, **self.provenance, **value},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        return value

    def cv_probe(self, *, kind="ridge", y=None, subset=None, score_y=None):
        y = self.y if y is None else y
        score_y = self.y if score_y is None else score_y
        subset = torch.ones(len(y), dtype=torch.bool) if subset is None else subset
        fitter = fit_ridge if kind == "ridge" else fit_logistic
        candidates = []
        for penalty in PENALTIES:
            losses, seed_mse = [], {}
            for seed in self.seeds:
                folds = grouped_folds(self.tasks, seed=seed)
                seed_losses = []
                for fold in range(5):
                    train, held = subset & (folds != fold), subset & (folds == fold)
                    if not train.any() or not held.any():
                        raise ValueError("Empty grouped training CV fold")
                    weight, bias = fitter(
                        self.x[train], y[train], self.mass[train], penalty
                    )
                    prediction = self.x[held] @ weight + bias
                    if kind == "logistic":
                        prediction = prediction.sigmoid()
                    seed_losses.append(
                        float(
                            (
                                (prediction - score_y[held]).square() * self.mass[held]
                            ).sum()
                            / self.mass[held].sum()
                        )
                    )
                losses.extend(seed_losses)
                seed_mse[str(seed)] = sum(seed_losses) / 5
            candidates.append(
                {
                    "penalty": penalty,
                    "fold_mse": losses,
                    "seed_mse": seed_mse,
                    "mean_mse": sum(losses) / len(losses),
                }
            )
        selected = min(candidates, key=lambda item: item["mean_mse"])
        weight, bias = fitter(
            self.x[subset], y[subset], self.mass[subset], selected["penalty"]
        )
        return (weight, bias), {
            "candidates": candidates,
            "selected_penalty": selected["penalty"],
            "selected_training_cv_mse": selected["mean_mse"],
            "weight_norm": float(weight.norm()),
            "bias": float(bias),
        }

    def seed_probe_refits(
        self,
        selection,
        *,
        kind="ridge",
        y=None,
        subset=None,
        planted=False,
        bootstrap=None,
    ):
        """Refit each seed's train-CV winner, preserving any training stratum."""
        labels = self.y if y is None else y
        bootstrap = self.bootstrap if bootstrap is None else bootstrap
        fitter = fit_ridge if kind == "ridge" else fit_logistic
        output = []
        for seed in self.seeds:
            models, penalties, heads = {}, {}, {}
            for domain, choices in selection.items():
                selected = min(
                    choices["candidates"], key=lambda item: item["seed_mse"][str(seed)]
                )
                mask = (
                    torch.ones(len(labels), dtype=torch.bool)
                    if subset is None
                    else subset.clone()
                )
                if domain != "shared":
                    mask &= torch.tensor([d == domain for d in self.fields["domain"]])
                if not mask.any():
                    raise ValueError("Seed probe refit has no training tokens")
                weight, bias = fitter(
                    self.x[mask], labels[mask], self.mass[mask], selected["penalty"]
                )
                models[domain] = (weight, bias)
                penalties[domain] = selected["penalty"]
                heads[domain] = {
                    "weight_norm": float(weight.norm()),
                    "bias": float(bias),
                }
            validation = self.store.evaluate_targets(
                models, logistic=kind == "logistic", planted=planted
            )
            output.append(
                {
                    "seed": seed,
                    "selected_penalties": penalties,
                    "heads": heads,
                    "validation": validation,
                    "bootstrap": bootstrap.report(validation["rows"]),
                }
            )
        return output

    def probes(self, y=None, *, planted=False, kinds=("ridge", "logistic")):
        results, fitted = {}, {}
        for kind in kinds:
            for scope in ("shared", "domain"):
                models, selection = {}, {}
                domains = (
                    ["shared"]
                    if scope == "shared"
                    else sorted(set(self.fields["domain"]))
                )
                for domain in domains:
                    subset = (
                        None
                        if domain == "shared"
                        else torch.tensor([d == domain for d in self.fields["domain"]])
                    )
                    models[domain], selection[domain] = self.cv_probe(
                        kind=kind, y=y, subset=subset, score_y=y if planted else self.y
                    )
                validation = self.store.evaluate_targets(
                    models, logistic=kind == "logistic", planted=planted
                )
                bootstrap = (
                    self.bootstrap
                    if not planted
                    else TaskBootstrap(validation["rows"], seed=self.seed)
                )
                results[f"{kind}_{scope}"] = {
                    "selection": selection,
                    "validation": validation,
                    "bootstrap": bootstrap.report(validation["rows"]),
                }
                fitted[f"{kind}_{scope}"] = models
                seed_results = self.seed_probe_refits(
                    selection, kind=kind, y=y, planted=planted, bootstrap=bootstrap
                )
                results[f"{kind}_{scope}"]["seeds"] = seed_results
                results[f"{kind}_{scope}"]["summary"] = summarize_validations(
                    [r["validation"] for r in seed_results]
                )
        for kind in kinds:
            results[f"{kind}_domain"]["paired_vs_shared"] = bootstrap.report(
                results[f"{kind}_domain"]["validation"]["rows"],
                results[f"{kind}_shared"]["validation"]["rows"],
            )
        return results, fitted

    def run_E2(self):
        results, self.ridge_models = self.probes()
        results["logistic_shared"]["paired_vs_ridge"] = self.bootstrap.report(
            results["logistic_shared"]["validation"]["rows"],
            results["ridge_shared"]["validation"]["rows"],
        )
        return self.save(
            "E2",
            {
                "probes": results,
                "real_weight_norm_references": {"r499": 0.0015, "r510": 0.0024},
            },
        )

    def batches(self, batch, steps, seed, *, subset=None, replay=False):
        if replay:
            rows = [row for row in self.store.episodes if row["run"] == "r510"]
            return replay_batches(rows, batch_size=16, total_steps=80)
        rows = [
            row
            for row in self.store.episodes
            if subset is None or subset[self.indices[row["trajectory_id"]][0]]
        ]
        if not rows:
            raise ValueError("No episodes for training")
        generator = torch.Generator().manual_seed(seed)
        ordered = []
        while len(ordered) < batch * steps:
            ordered.extend(
                rows[i] for i in torch.randperm(len(rows), generator=generator).tolist()
            )
        return [
            ordered[start : start + batch] for start in range(0, batch * steps, batch)
        ]

    def simulate(
        self,
        *,
        lr,
        batch=16,
        steps=80,
        seed=SEED,
        frozen_bias=False,
        warmup_steps=0,
        replay=False,
        labels=None,
        held=None,
        diagnostics=False,
        evaluate=True,
        planted=False,
        training_subset=None,
        initial_head=None,
        bias_init=0.78,
    ):
        labels = self.y if labels is None else labels
        weight = torch.zeros(self.x.shape[1], requires_grad=True)
        bias = torch.tensor(bias_init, requires_grad=not frozen_bias)
        if initial_head is not None:
            with torch.no_grad():
                weight.copy_(initial_head["weight"].reshape_as(weight))
                bias.copy_(initial_head["bias"].reshape_as(bias))
        parameters = [weight] if frozen_bias else [weight, bias]
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0)
        train_mask = training_subset if held is None else ~held
        if training_subset is not None and held is not None:
            train_mask = train_mask & training_subset
        full_indices = (
            torch.arange(len(self.x))
            if train_mask is None
            else torch.where(train_mask)[0]
        )
        batches = self.batches(batch, steps, seed, subset=train_mask, replay=replay)
        if diagnostics:
            gram = torch.zeros(self.x.shape[1], self.x.shape[1], dtype=torch.float64)
            feature_sum = torch.zeros(self.x.shape[1], dtype=torch.float64)
            target_feature_sum = torch.zeros_like(feature_sum)
            target_mean = torch.zeros((), dtype=torch.float64)
            total_mass = self.mass[full_indices].double().sum()
            for start in range(0, len(full_indices), 4096):
                indices = full_indices[start : start + 4096]
                x = self.x[indices].double()
                mass = self.mass[indices].double()
                gram += x.T @ (x * mass[:, None]) / total_mass
                feature_sum += (x * mass[:, None]).sum(0) / total_mass
                target_feature_sum += (x * (mass * labels[indices])[:, None]).sum(
                    0
                ) / total_mass
                target_mean += (mass * labels[indices]).sum() / total_mass
        evaluations, gradient, full, consumed = [], [], [], []
        bias_gradients, full_bias_gradients = [], []

        def checkpoint(step):
            return {
                "step": step,
                "weight_norm": float(weight.detach().norm()),
                "bias": float(bias.detach()),
                "validation": self.store.evaluate_targets(
                    {"shared": (weight.detach(), bias.detach())}, planted=planted
                ),
            }

        if evaluate:
            evaluations.append(checkpoint(0))
        for step, rows in enumerate(batches):
            selected = torch.tensor(
                [i for row in rows for i in self.indices[row["trajectory_id"]]]
            )
            optimizer.param_groups[0]["lr"] = (
                lr * min(1.0, step / warmup_steps) if warmup_steps else lr
            )
            optimizer.zero_grad()
            loss = weighted_mse(
                self.x[selected], labels[selected], self.mass[selected], weight, bias
            )
            loss.backward()
            if diagnostics:
                gradient.append(weight.grad.detach().clone())
                full.append(
                    (
                        2
                        * (
                            gram @ weight.detach().double()
                            + feature_sum * bias.detach().double()
                            - target_feature_sum
                        )
                    ).float()
                )
                bias_gradients.append(
                    torch.zeros(1)
                    if bias.grad is None
                    else bias.grad.detach().reshape(1).clone()
                )
                full_bias_gradients.append(
                    (
                        2
                        * (
                            feature_sum @ weight.detach().double()
                            + bias.detach().double()
                            - target_mean
                        )
                    )
                    .float()
                    .reshape(1)
                )
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            consumed.append(
                {
                    "step": step + 1,
                    "episodes": len(rows),
                    "loss": float(loss.detach()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "weight_norm": float(weight.detach().norm()),
                    "bias": float(bias.detach()),
                }
            )
            if evaluate and ((step + 1) % 5 == 0 or step + 1 == steps):
                evaluations.append(checkpoint(step + 1))
        output = {
            "lr": lr,
            "batch": batch,
            "seed": seed,
            "total_steps": steps,
            "frozen_bias": frozen_bias,
            "warmup_steps": warmup_steps,
            "bias_init": float(initial_head["bias"].reshape(()))
            if initial_head is not None
            else bias_init,
            "episodes_seen": sum(len(rows) for rows in batches),
            "steps": consumed,
            "evaluations": evaluations,
        }
        if evaluate:
            rows = evaluations[-1]["validation"]["rows"]
            bootstrap = (
                self.bootstrap if not planted else TaskBootstrap(rows, seed=self.seed)
            )
            output["bootstrap"] = bootstrap.report(rows)
            output["maximum_adjacent_domain_bias_jump"] = {
                domain: max(
                    abs(
                        right["validation"]["by_domain"][domain]["bias"]
                        - left["validation"]["by_domain"][domain]["bias"]
                    )
                    for left, right in zip(evaluations, evaluations[1:], strict=False)
                )
                for domain in evaluations[0]["validation"]["by_domain"]
            }
            output["maximum_adjacent_bias_jump_definition"] = (
                "Overall token-weighted residual offset, including initial step0; domain offsets also reported"
            )
            output["maximum_adjacent_bias_jump"] = max(
                abs(
                    right["validation"]["summary"]["bias"]
                    - left["validation"]["summary"]["bias"]
                )
                for left, right in zip(evaluations, evaluations[1:], strict=False)
            )
        if diagnostics:
            output["gradient_weight_only"] = gradient_summary(gradient, full)
            output["gradient_weight_only"]["coordinates"] = "weight coordinates only"
            output["gradient"] = gradient_summary(
                [
                    torch.cat([g, b])
                    for g, b in zip(gradient, bias_gradients, strict=True)
                ],
                [
                    torch.cat([g, b])
                    for g, b in zip(full, full_bias_gradients, strict=True)
                ],
            )
            output["gradient"]["coordinates"] = (
                "all trainable head coordinates: weight followed by bias"
            )
            if frozen_bias:
                output["gradient"] = dict(output["gradient_weight_only"])
            output["bias_gradients"] = [float(value[0]) for value in bias_gradients]
        return output, (weight.detach(), bias.detach())

    def select_lrs(self, labels=None, *, batch=16, steps=80):
        labels = self.y if labels is None else labels
        candidates = []
        for lr in LRS:
            losses, per_seed = [], {}
            for seed in self.seeds:
                folds = grouped_folds(self.tasks, seed=seed)
                values = []
                for fold in range(5):
                    held = folds == fold
                    _, (weight, bias) = self.simulate(
                        lr=lr,
                        batch=batch,
                        steps=steps,
                        seed=seed,
                        labels=labels,
                        held=held,
                        evaluate=False,
                    )
                    values.append(
                        float(
                            weighted_mse(
                                self.x[held],
                                self.y[held],
                                self.mass[held],
                                weight,
                                bias,
                            )
                        )
                    )
                losses.extend(values)
                per_seed[str(seed)] = sum(values) / 5
            candidates.append(
                {
                    "lr": lr,
                    "fold_mse": losses,
                    "seed_mse": per_seed,
                    "mean_mse": sum(losses) / len(losses),
                }
            )
        return sorted(candidates, key=lambda candidate: candidate["mean_mse"])

    def run_E3(self):
        if self.reference_path is None:
            raise ValueError(
                "E3 requires --r510-reference-json; cannot invent same-step EV"
            )
        real = json.loads(Path(self.reference_path).read_text())
        if real.get("completed_step") != 80:
            raise ValueError("H2 reference must be completed_step80")
        arms = [self.simulate(**arm, replay=True)[0] for arm in e3_grid()]
        baseline = arms[0]
        simulated = baseline["evaluations"][-1]["validation"]["by_run_domain"]["r510"][
            "airline"
        ]["explained_variance"]
        h2 = h2_status(
            baseline["steps"][-1]["weight_norm"],
            simulated,
            real["by_domain"]["airline"]["explained_variance"],
        )
        h2["reference_path"] = str(Path(self.reference_path).resolve())
        h2["reference_sha256"] = hashlib.sha256(
            Path(self.reference_path).read_bytes()
        ).hexdigest()
        real_rows = {(r["domain"], str(r["task_id"])): r for r in real["rows"]}
        simulated_rows = [
            r
            for r in baseline["evaluations"][-1]["validation"]["rows"]
            if r["run"] == "r510"
        ]
        if {(r["domain"], str(r["task_id"])) for r in simulated_rows} != set(real_rows):
            raise ValueError("H2 r510 reference tasks differ")
        for row in simulated_rows:
            ref = real_rows[(row["domain"], str(row["task_id"]))]
            if (
                ref.get("source_id") is not None
                and row.get("source_id") != ref["source_id"]
            ):
                raise ValueError("H2 reference source trajectory identity differs")
            if (row["n_tokens"], row["target"]) != (ref["n_tokens"], ref["target"]):
                raise ValueError(
                    "H2 reference trajectories differ in labels or token counts"
                )
        probe_ev = (
            self.results.get("E2", {})
            .get("probes", {})
            .get("ridge_shared", {})
            .get("validation", {})
            .get("by_domain", {})
            .get("airline", {})
            .get("explained_variance")
        )
        for arm in arms:
            final_ev = arm["evaluations"][-1]["validation"]["by_domain"]["airline"][
                "explained_variance"
            ]
            arm["H3_individual"] = {
                "probe_ev": probe_ev,
                "final_airline_ev": final_ev,
                "reaches_70pct": probe_ev is not None
                and final_ev is not None
                and final_ev >= 0.7 * probe_ev,
                "reaches_40pct": probe_ev is not None
                and final_ev is not None
                and final_ev >= 0.4 * probe_ev,
                "bias_jump_at_most_0_2": arm["maximum_adjacent_bias_jump"] <= 0.2,
                "requires_H2_pass": True,
            }
            arm["paired_vs_baseline"] = self.bootstrap.report(
                arm["evaluations"][-1]["validation"]["rows"],
                baseline["evaluations"][-1]["validation"]["rows"],
            )
        self.lr_cv = self.select_lrs()
        return self.save(
            "E3",
            {
                "arms": arms,
                "H2": h2,
                "training_only_lr_cv": self.lr_cv,
                "interpretation": "reference only if H2 fails; frozen backbone and sampled training features differ from real training",
            },
        )

    def run_E4(self):
        self.lr_cv = self.lr_cv or self.select_lrs()
        lrs = list(dict.fromkeys([5e-6, *[item["lr"] for item in self.lr_cv[:2]]]))
        arms, selections = [], {}
        for batch in (16, 64, 128):
            for budget, steps in e4_budgets(batch).items():
                cv = self.select_lrs(batch=batch, steps=steps)
                selections[f"{batch}/{budget}"] = cv
                arm_lrs = list(dict.fromkeys([5e-6, *[item["lr"] for item in cv[:2]]]))
                for lr in arm_lrs:
                    seeds = []
                    for seed in self.seeds:
                        arm, (weight, bias) = self.simulate(
                            lr=lr, batch=batch, steps=steps, seed=seed
                        )
                        if budget == "B":
                            prediction = self.x @ weight + bias
                            arm["training_mse"] = float(
                                ((prediction - self.y).square() * self.mass).sum()
                                / self.mass.sum()
                            )
                            from scripts.qualification.critic_pretrain import (
                                aggregate_validation,
                            )

                            train_rows = []
                            for identity, indices in self.indices.items():
                                row = official_metric_row(
                                    prediction[indices], self.episode_map[identity]
                                )
                                exact_count = int(
                                    self.episode_map[identity]["original_action_count"]
                                )
                                factor = exact_count / len(indices)
                                for key in (
                                    "sse",
                                    "error_sum",
                                    "target_sum",
                                    "target2_sum",
                                ):
                                    row[key] *= factor
                                row["n_tokens"] = exact_count
                                train_rows.append(row)
                            arm["training_ev"] = aggregate_validation(train_rows)[
                                "explained_variance"
                            ]
                            arm["training_ev_approximate"] = self.store.sampled_training
                        seeds.append(arm)
                    arms.append(
                        {
                            "batch": batch,
                            "budget": budget,
                            "lr": lr,
                            "seeds": seeds,
                            "summary": summarize_seeds(seeds),
                        }
                    )
        for arm in arms:
            reference_lr = selections[f"16/{arm['budget']}"][0]["lr"]
            reference = next(
                item
                for item in arms
                if item["batch"] == 16
                and item["budget"] == arm["budget"]
                and item["lr"] == reference_lr
            )
            for value, ref in zip(arm["seeds"], reference["seeds"], strict=True):
                value["paired_vs_batch16_cv_best"] = self.bootstrap.report(
                    value["evaluations"][-1]["validation"]["rows"],
                    ref["evaluations"][-1]["validation"]["rows"],
                )
        return self.save(
            "E4",
            {
                "training_only_lr_cv": self.lr_cv,
                "selected_lrs": lrs,
                "batch_budget_training_cv": selections,
                "arms": arms,
            },
        )

    def run_E5(self):
        arms = []
        r510 = torch.tensor(
            [
                self.episode_map[identity]["run"] == "r510"
                for identity in self.identities
            ]
        )
        for batch in (16, 64, 128):
            seeds = [
                self.simulate(
                    lr=5e-6,
                    batch=batch,
                    seed=seed,
                    diagnostics=True,
                    evaluate=False,
                    replay=batch == 16,
                    training_subset=r510,
                )[0]
                for seed in (self.seeds[:1] if batch == 16 else self.seeds)
            ]
            arms.append(
                {
                    "batch": batch,
                    "seeds": seeds,
                    "mean_full_data_cosine": sum(
                        arm["gradient"]["mean_full_data_cosine"] for arm in seeds
                    )
                    / len(seeds),
                }
            )
        return self.save(
            "E5",
            {
                "arms": arms,
                "noise_threshold_batch16": 0.1,
                "support_threshold_batch128": 0.3,
            },
        )

    def run_E6(self):
        episodes = list(self.episode_map.values())
        task_keys = [(row["domain"], str(row["task_id"])) for row in episodes]
        soft = soft_labels(torch.tensor([row["reward"] for row in episodes]), task_keys)
        lookup = {
            row["trajectory_id"]: soft[index] for index, row in enumerate(episodes)
        }
        labels = torch.stack([lookup[identity] for identity in self.identities])
        probes, _ = self.probes(labels)
        hard_probes = self.results.get("E2", {}).get("probes")
        if hard_probes is None:
            hard_probes, _ = self.probes()
        for key, probe in probes.items():
            probe["paired_vs_hard_labels"] = self.bootstrap.report(
                probe["validation"]["rows"], hard_probes[key]["validation"]["rows"]
            )
        lr_cv = self.select_lrs(labels)
        simulations = [
            self.simulate(lr=lr_cv[0]["lr"], labels=labels, seed=seed)[0]
            for seed in SEEDS
        ]
        counts = defaultdict(int)
        for key in task_keys:
            counts[key] += 1
        strata = {}
        for count in (2, 6):
            subset = torch.tensor(
                [counts[(domain, str(task))] == count for domain, task in self.tasks]
            )
            if not subset.any():
                raise ValueError(f"No {count}-trajectory training tasks")
            fitted, selection = self.cv_probe(y=labels, subset=subset)
            validation = self.store.evaluate_targets({"shared": fitted})
            hard_fitted, hard_selection = self.cv_probe(subset=subset)
            hard_validation = self.store.evaluate_targets({"shared": hard_fitted})
            soft_seeds = self.seed_probe_refits(
                {"shared": selection}, y=labels, subset=subset
            )
            hard_seeds = self.seed_probe_refits(
                {"shared": hard_selection}, subset=subset
            )
            for soft_seed, hard_seed in zip(soft_seeds, hard_seeds, strict=True):
                soft_seed["paired_vs_hard_labels"] = self.bootstrap.report(
                    soft_seed["validation"]["rows"], hard_seed["validation"]["rows"]
                )
            strata[str(count)] = {
                "seeds": soft_seeds,
                "summary": summarize_validations([r["validation"] for r in soft_seeds]),
                "hard_label_seeds": hard_seeds,
                "hard_label_summary": summarize_validations(
                    [r["validation"] for r in hard_seeds]
                ),
                "hard_label_selection": hard_selection,
                "hard_label_validation": hard_validation,
                "paired_vs_hard_labels": self.bootstrap.report(
                    validation["rows"], hard_validation["rows"]
                ),
                "training_tasks": sum(value == count for value in counts.values()),
                "selection": selection,
                "validation": validation,
                "bootstrap": self.bootstrap.report(validation["rows"]),
            }
        return self.save(
            "E6",
            {
                "soft_label_source": "mean reward across training episodes of each task; no validation labels",
                "probes": probes,
                "training_only_lr_cv": lr_cv,
                "simulations": simulations,
                "summary": summarize_seeds(simulations),
                "training_task_count_strata": strata,
            },
        )

    def run_E7(self, caches):
        from scripts.tau2.critic_preexp_extract import layer_indices

        expected_aliases = {
            f"base-layer-{index}"
            for index in layer_indices(int(self.store.manifest["depth"]))
        } | {
            "r480-final"
            if self.store.manifest.get("protocol") == "qwen3-20261015"
            else "r510-final"
        }
        if set(caches) != expected_aliases:
            raise ValueError(
                "E7 requires exactly the three registered quarter-depth base layers and r510-final"
            )
        provenance = {}
        for alias, path in caches.items():
            manifest_path = Path(path)
            metadata = json.loads(manifest_path.read_text())
            provenance[alias] = {
                **validate_e7_cache_alias(
                    alias, metadata, self.store.manifest, synthetic=self.synthetic
                ),
                "feature_manifest": str(manifest_path.resolve()),
                "feature_manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            }
        probes = {}
        original_store, original_x, original_fields = self.store, self.x, self.fields
        original_y, original_mass, original_ids = self.y, self.mass, self.identities
        for name, manifest in caches.items():
            from scripts.tau2.critic_preexp_extract import verify_cache

            verify_cache(
                manifest, self.pool, self.pool_metadata, synthetic=self.synthetic
            )
            store = PreexpStore(manifest)
            if (
                store.manifest.get("pool_manifest_sha256")
                != self.provenance["pool_manifest_sha256"]
            ):
                raise ValueError("E7 cache pool binding differs")
            self.store = store
            self.x, self.y, self.mass, ids = store.training()
            if (
                ids != original_ids
                or not torch.equal(self.y, original_y)
                or not torch.equal(self.mass, original_mass)
            ):
                raise ValueError(
                    "E7 caches differ in aligned training token identity/labels/mass"
                )
            (weight, bias), selection = self.cv_probe()
            validation = store.evaluate_targets({"shared": (weight, bias)})
            seed_results = self.seed_probe_refits({"shared": selection})
            probes[name] = {
                "cache_provenance": provenance[name],
                "seeds": seed_results,
                "summary": summarize_validations(
                    [r["validation"] for r in seed_results]
                ),
                "selection": selection,
                "validation": validation,
                "bootstrap": self.bootstrap.report(validation["rows"]),
            }
        self.store, self.x, self.fields = original_store, original_x, original_fields
        self.y, self.mass, self.identities = original_y, original_mass, original_ids
        models = (
            self.ridge_models["ridge_shared"]
            if self.ridge_models
            else {"shared": self.cv_probe()[0]}
        )
        positions = {
            name: self.store.evaluate_targets(models, position=name)
            for name in [
                "first_action_token",
                "0-20%",
                "20-40%",
                "40-60%",
                "60-80%",
                "80-100%",
            ]
        }
        position_bootstrap = {}
        for name, validation in positions.items():
            tasks = {
                (str(row["domain"]), str(row["task_id"])) for row in validation["rows"]
            }
            if len(tasks) == 60:
                position_bootstrap[name] = TaskBootstrap(
                    validation["rows"], seed=self.seed
                ).report(validation["rows"])
            else:
                position_bootstrap[name] = {
                    "draws": 20000,
                    "tasks_per_draw": 60,
                    "defined": False,
                    "observed_task_clusters": len(tasks),
                    "undefined_reason": "Position has fewer than the registered 60 validation task clusters; no reduced-cluster interval reported",
                }
        base = (
            self.results["E2"]["probes"]["ridge_shared"]["validation"]["rows"]
            if "E2" in self.results
            else self.store.evaluate_targets(models)["rows"]
        )
        for result in probes.values():
            result["paired_vs_base_final"] = self.bootstrap.report(
                result["validation"]["rows"], base
            )
        selected_intermediate = min(
            (key for key in probes if key.startswith("base-layer-")),
            key=lambda key: probes[key]["selection"]["selected_training_cv_mse"],
        )
        return self.save(
            "E7",
            {
                "layer_backbone_probes": probes,
                "base_final_cache_provenance": {
                    **validate_primary_cache(
                        self.store.manifest, synthetic=self.synthetic
                    ),
                    "layer": self.store.manifest["layer"],
                    "depth": self.store.manifest["depth"],
                    "hidden_size": self.store.hidden_size,
                    "feature_manifest": str(self.store.path.resolve()),
                    "feature_manifest_sha256": hashlib.sha256(
                        self.store.path.read_bytes()
                    ).hexdigest(),
                },
                "base_final_positions": positions,
                "base_final_position_bootstrap": position_bootstrap,
                "selected_intermediate_training_cv": selected_intermediate,
            },
        )

    def run_E8(self):
        labels = torch.tensor(self.fields["planted_label"]).float()
        fraction = validate_planted_prevalence(labels, self.mass, self.fields["domain"])
        probes, _ = self.probes(labels, planted=True, kinds=("ridge",))
        simulations = [
            self.simulate(lr=5e-6, labels=labels, replay=True, planted=True)[0]
        ]
        return self.save(
            "E8",
            {
                "rule": self.store.manifest.get(
                    "planted_rule",
                    self.store.manifest.get("extraction", {}).get("planted_rule"),
                ),
                "token_mass_weighted_training_prevalence": fraction,
                "sampled_prefix_training_prevalence": float(labels.mean()),
                "target_scope": "prefix-local rule; validation uses its own prefix labels",
                "probes": probes,
                "fixed_r510_replay": simulations,
                "summary": summarize_seeds(simulations),
            },
        )

    def run_E9(self):
        models, selection = self.cv_probe()
        path = export_head(
            self.store,
            *models,
            self.out / "ridge-head.pt",
            selection["selected_penalty"],
        )
        from scripts.tau2.critic_preexp_delivery import finish_e9_export

        consumers = finish_e9_export(self.store, *models, path)
        return self.save(
            "E9",
            {
                "consumers": consumers,
                "head_path": str(path.resolve()),
                "selection": selection,
                "readback_exact": True,
                "weight_shape": [1, self.x.shape[1]],
                "bias_shape": [1],
                "consumer_validation": "Both offline_critic_fit head loader and SAO checker executed on CPU; no training",
            },
        )


def summarize_validations(reports):
    values = defaultdict(list)
    for report in reports:
        for domain, result in report["by_domain"].items():
            values[domain]
            if result["explained_variance"] is not None:
                values[domain].append(result["explained_variance"])
    return {
        domain: {
            "mean_ev": sum(numbers) / len(numbers) if numbers else None,
            "range_ev": [min(numbers), max(numbers)] if numbers else None,
            "seeds": len(reports),
            "defined_ev_seeds": len(numbers),
            "undefined_reason": None
            if numbers
            else "Target variance is zero in every seed",
        }
        for domain, numbers in values.items()
    }


def summarize_seeds(arms):
    return summarize_validations(
        [arm["evaluations"][-1]["validation"] for arm in arms if arm["evaluations"]]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--pool-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
    )
    parser.add_argument(
        "--layer-cache",
        action="append",
        default=[],
        help="NAME=MANIFEST; three base-layer-N and r510-final",
    )
    parser.add_argument("--r510-reference-json", type=Path)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Only accept synthetic pool; no scientific conclusions",
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    experiment = Preexperiment(
        args.features,
        args.out,
        pool_manifest=args.pool_manifest,
        synthetic=args.synthetic,
        r510_reference=args.r510_reference_json,
    )
    caches = dict(entry.split("=", 1) for entry in args.layer_cache)
    for name in args.experiments:
        if name not in ("E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"):
            parser.error(f"Unknown experiment {name}")
        if name == "E7":
            experiment.run_E7(caches)
        else:
            getattr(experiment, f"run_{name}")()
    from scripts.tau2.critic_preexp_delivery import write_results_table

    write_results_table(args.out)


if __name__ == "__main__":
    main()
