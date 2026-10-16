# SPDX-License-Identifier: Apache-2.0
"""Qwen3 E2/E3/E7, reusing the audited frozen-feature pre-experiment toolkit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from scripts.tau2.critic_preexp import (
    LRS,
    Preexperiment,
    summarize_seeds,
)
from scripts.tau2.critic_preexp_data import QWEN3_PROTOCOL, sha256

SEED = 20261015
PRIOR = 927 / 1494


def read_initial_head(checkpoint: Path) -> tuple[dict, dict]:
    """Read saved tensors exactly, never regenerate the random Q2 initialization."""
    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    if checkpoint.parent.name != "critic-initial" or not checkpoint.name.endswith(
        "globalstep-1"
    ):
        raise ValueError("Q2 requires the saved critic-initial globalstep-1 checkpoint")
    tensors, sources = {}, {}
    for path in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in ("weight", "bias"):
                key = f"score.{name}"
                if key in handle.keys():
                    if name in tensors:
                        raise ValueError("Duplicate initial score tensor")
                    tensors[name] = handle.get_tensor(key)
                    sources[path.name] = sha256(path)
    if (
        set(tensors) != {"weight", "bias"}
        or tensors["weight"].shape != (1, 2560)
        or tensors["bias"].shape != (1,)
    ):
        raise ValueError(
            "Initial head must contain score.weight[1,2560] and score.bias[1]"
        )
    if not all(torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError("Nonfinite initial head")
    return tensors, {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_files_sha256": sources,
        "config_sha256": sha256(checkpoint / "config.json"),
        "weight_norm": float(tensors["weight"].float().norm()),
        "bias": float(tensors["bias"][0]),
        "dtype": {k: str(v.dtype) for k, v in tensors.items()},
        "conversion": "saved tensors preserved; simulation copies exactly representable values into float32",
    }


def q2_status(simulated, real):
    """Use both domains at step 94; no Qwen3.5 norm criterion applies."""
    differences = {}
    for domain in ("airline", "retail"):
        a, b = (
            simulated[domain]["explained_variance"],
            real[domain]["explained_variance"],
        )
        if a is None or b is None:
            raise ValueError("Q2 requires defined same-step EV for both domains")
        differences[domain] = abs(a - b)
    return {
        "status": "成立"
        if all(v <= 0.1 + 1e-12 for v in differences.values())
        else "不成立",
        "completed_step": 94,
        "domain_ev_absolute_difference": differences,
        "tolerance": 0.1,
    }


def q3_status(arms, probe, q2, all_arms=None):
    """Use seed means and a train-CV-selected rate; disclose ranges separately."""
    output = {}
    for domain in ("airline", "retail"):
        ceiling = probe[domain]["explained_variance"]
        if ceiling is None or ceiling <= 0:
            output[domain] = {
                "status": "不确定",
                "reason": "ridge EV <= 0 or undefined; ratio inapplicable",
            }
            continue
        values = [
            arm["evaluations"][-1]["validation"]["by_domain"][domain][
                "explained_variance"
            ]
            for arm in arms
        ]
        if not arms:
            raise ValueError("Q3 needs the train-selected rate's seed arms")

        def curve(settings):
            by_step = defaultdict(list)
            for arm in settings:
                for point in arm["evaluations"]:
                    by_step[point.get("step", 94)].append(
                        point["validation"]["by_domain"][domain]["explained_variance"]
                    )
            return {
                step: sum(vs) / len(vs)
                if len(vs) == len(settings) and all(v is not None for v in vs)
                else None
                for step, vs in sorted(by_step.items())
                if step > 0
            }

        selected_curve = curve(arms)
        reaching = [
            step
            for step, value in selected_curve.items()
            if value is not None and value >= 0.7 * ceiling
        ]
        mean = sum(values) / len(values) if all(v is not None for v in values) else None
        jumps = [arm["maximum_adjacent_domain_bias_jump"][domain] for arm in arms]
        stable = max(jumps) <= 0.2 + 1e-12
        by_rate = defaultdict(list)
        for arm in all_arms or arms:
            by_rate[arm["lr"]].append(arm)
        all_below = all(
            all(v is not None and v < 0.4 * ceiling for v in curve(settings).values())
            for settings in by_rate.values()
        )
        status = "不成立" if all_below else "不确定"
        if q2["status"] == "成立" and bool(reaching) and stable:
            status = "成立"
        output[domain] = {
            "status": status,
            "probe_ev": ceiling,
            "selected_lr_final_ev_seeds": values,
            "selected_lr_mean_ev": mean,
            "selected_lr_seed_mean_curve": selected_curve,
            "first_step_reaching_70pct": reaching[0] if reaching else None,
            "maximum_adjacent_bias_jump_seeds": jumps,
            "all_lr_seed_means_below_40pct": all_below,
            "interpretation": "reference only"
            if q2["status"] != "成立"
            else "Q2 passed",
        }
    return output


class Qwen3Preexperiment(Preexperiment):
    def __init__(
        self,
        features,
        out,
        *,
        pool_manifest,
        synthetic=False,
        initial_checkpoint=None,
        reference=None,
    ):
        super().__init__(
            features, out, pool_manifest=pool_manifest, synthetic=synthetic, seed=SEED
        )
        if self.pool_metadata.get("protocol") != QWEN3_PROTOCOL:
            raise ValueError("Qwen3 experiments require the registered r480 pool")
        self.initial_checkpoint = initial_checkpoint
        self.reference_path = reference
        self.provenance.update(
            protocol=QWEN3_PROTOCOL, bootstrap_tasks=60, resident_features=True
        )

    def batches(self, batch, steps, seed, *, subset=None, replay=False):
        if batch != 16 or steps != 94:
            raise ValueError("Qwen3 E3 requires batch16, 94 steps")
        if replay:
            if subset is not None:
                raise ValueError("Q2 replay cannot omit training episodes")
            lookup = {
                row["_episode_order"]: row["_trajectory_id"]
                for row in self.pool
                if row["_split"] == "train"
            }
            return [
                [self.episode_map[lookup[i]] for i in b["episode_order"]]
                for b in self.pool_metadata["replay_batches"]
            ]
        # Shuffle whole task epochs; both saved trajectories for a task stay in
        # their original epoch. Preserve the 11-row tail instead of mixing epochs.
        epochs = defaultdict(list)
        for row in self.store.episodes:
            if subset is None or bool(subset[self.indices[row["trajectory_id"]][0]]):
                epochs[int(row["step"]) // 47].append(row)
        if not epochs:
            raise ValueError("No training episodes")
        generator = torch.Generator().manual_seed(seed)
        batches = []
        while len(batches) < steps:
            for epoch in sorted(epochs):
                rows = epochs[epoch]
                if not rows:
                    continue
                shuffled = [
                    rows[i]
                    for i in torch.randperm(len(rows), generator=generator).tolist()
                ]
                batches.extend(
                    shuffled[i : i + batch] for i in range(0, len(shuffled), batch)
                )
        if not batches:
            raise ValueError("No training episodes")
        return batches[:steps]

    def simulate(self, *, steps=94, seed=SEED, bias_init=PRIOR, **kwargs):
        return super().simulate(steps=steps, seed=seed, bias_init=bias_init, **kwargs)

    def run_E2(self):
        result = super().run_E2()
        result["real_weight_norm_references"] = {"r480_initial": None}
        if self.initial_checkpoint is not None:
            _, provenance = read_initial_head(self.initial_checkpoint)
            result["real_weight_norm_references"] = {
                "r480_initial": provenance["weight_norm"]
            }
        domains = result["probes"]["ridge_shared"]["validation"]["by_domain"]
        intervals = result["probes"]["ridge_shared"]["bootstrap"]
        result["Q1_domain_ev"] = {
            d: v["explained_variance"] for d, v in domains.items()
        }
        result["Q1_bootstrap"] = intervals
        passing = any(
            v["explained_variance"] is not None
            and v["explained_variance"] >= 0.15
            and intervals["by_domain"][d]["ev_95ci"] is not None
            and intervals["by_domain"][d]["ev_95ci"][0] > 0
            for d, v in domains.items()
        )
        failing = all(
            v["explained_variance"] is not None and v["explained_variance"] < 0.05
            for v in domains.values()
        )
        result["Q1"] = {
            "status": "成立" if passing else "不成立" if failing else "不确定"
        }
        return self.save("E2", result)

    def run_E3(self):
        if self.initial_checkpoint is None or self.reference_path is None:
            raise ValueError("Q2 requires saved initial head and step94 reference")
        initial, initial_provenance = read_initial_head(self.initial_checkpoint)
        real = json.loads(Path(self.reference_path).read_text())
        if real.get("completed_step") != 94:
            raise ValueError("Q2 requires completed_step94 reference")
        baseline = self.simulate(lr=5e-6, replay=True, initial_head=initial)[0]
        final = baseline["evaluations"][-1]["validation"]
        actual = {(r["domain"], str(r["task_id"])): r for r in real["rows"]}
        predicted = {(r["domain"], str(r["task_id"])): r for r in final["rows"]}
        if set(actual) != set(predicted) or any(
            (r["source_id"], r["n_tokens"], r["target"])
            != (
                predicted[k]["source_id"],
                predicted[k]["n_tokens"],
                predicted[k]["target"],
            )
            for k, r in actual.items()
        ):
            raise ValueError("Q2 validation trajectories differ from reference")
        q2 = q2_status(final["by_domain"], real["by_domain"])
        q2.update(
            reference_path=str(Path(self.reference_path).resolve()),
            reference_sha256=sha256(self.reference_path),
        )
        self.save(
            "E3-Q2",
            {"Q2": q2, "replication": baseline, "initial_head": initial_provenance},
        )
        # Retain all registered rates and all seeds, but choose a rate using only
        # grouped training CV. No bias/warmup sweep belongs to this protocol.
        self.lr_cv = self.select_lrs(steps=94)
        selected = self.lr_cv[0]["lr"]
        arms = [self.simulate(lr=lr, seed=seed)[0] for lr in LRS for seed in self.seeds]
        selected_arms = [arm for arm in arms if arm["lr"] == selected]
        if "E2" not in self.results:
            raise ValueError("E3 requires E2's train-selected shared ridge comparison")
        probe = self.results["E2"]["probes"]["ridge_shared"]["validation"]["by_domain"]
        q3 = q3_status(selected_arms, probe, q2, all_arms=arms)
        return self.save(
            "E3",
            {
                "Q2": q2,
                "replication": baseline,
                "initial_head": initial_provenance,
                "arms": arms,
                "training_only_lr_cv": self.lr_cv,
                "selected_lr": selected,
                "selected_lr_summary": summarize_seeds(selected_arms),
                "Q3": q3,
                "interpretation": "reference only if Q2 fails; frozen backbone, sampled tokens and float32 head differ from the full bf16 training run",
            },
        )

    def run_E7(self, caches):
        result = super().run_E7(caches)
        base = self.results["E2"]["probes"]["ridge_shared"]["validation"]["by_domain"]
        selected = result["selected_intermediate_training_cv"]
        comparisons = {}
        for label, name in (("Q4", selected), ("Q5", "r480-final")):
            other = result["layer_backbone_probes"][name]["validation"]["by_domain"]
            deltas = {
                d: other[d]["explained_variance"] - base[d]["explained_variance"]
                if other[d]["explained_variance"] is not None
                and base[d]["explained_variance"] is not None
                else None
                for d in base
            }
            defined = all(v is not None for v in deltas.values())
            comparisons[label] = {
                "status": "成立"
                if any(v is not None and v >= 0.05 for v in deltas.values())
                else "不成立"
                if defined and all(v < 0.02 for v in deltas.values())
                else "不确定",
                "domain_ev_difference": deltas,
                "cache": name,
            }
        result.update(comparisons)
        return self.save("E7", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["E2", "E3", "E7"],
        default=["E2", "E3", "E7"],
    )
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--layer-cache", action="append", default=[])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    experiment = Qwen3Preexperiment(
        args.features,
        args.out,
        pool_manifest=args.pool_manifest,
        synthetic=args.synthetic,
        initial_checkpoint=args.initial_checkpoint,
        reference=args.reference,
    )
    if "E2" in args.experiments:
        experiment.run_E2()
    elif (args.out / "E2.json").exists():
        cached = json.loads((args.out / "E2.json").read_text())
        expected = {
            "protocol": QWEN3_PROTOCOL,
            "pool_manifest_sha256": sha256(args.pool_manifest),
            "feature_manifest_sha256": sha256(args.features),
            "synthetic": args.synthetic,
        }
        if any(cached.get(key) != value for key, value in expected.items()):
            raise ValueError("Saved E2 provenance differs from this Qwen3 experiment")
        experiment.results["E2"] = cached
    elif "E3" in args.experiments or "E7" in args.experiments:
        raise ValueError("Run E2 before E3/E7 in the same output directory")
    if "E3" in args.experiments:
        experiment.run_E3()
    if "E7" in args.experiments:
        experiment.run_E7(dict(value.split("=", 1) for value in args.layer_cache))


if __name__ == "__main__":
    main()
