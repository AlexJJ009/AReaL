# SPDX-License-Identifier: Apache-2.0
"""CPU consumer verification and protocol section 10 draft tables."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def finish_e9_export(store, weight, bias, path: Path) -> dict:
    """Verify both public CLIs, preserving their actual stdout/stderr and exit codes."""
    path = Path(path)
    bundle = path.parent / "ridge-head-sao"
    bundle.mkdir(exist_ok=True)
    state = {
        "score.weight": weight.detach().cpu().float().reshape(1, -1).contiguous(),
        "score.bias": torch.as_tensor(bias)
        .detach()
        .cpu()
        .float()
        .reshape(1)
        .contiguous(),
    }
    # This is explicitly a head initialization bundle, never a full backbone.
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen35TokenCriticForCausalBackbone"],
        "hidden_size": weight.numel(),
        "text_config": {"hidden_size": weight.numel()},
        "areal_value_head_bias": True,
        "head_only_initialization": True,
        "base_checkpoint": store.manifest["checkpoint"],
    }
    (bundle / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(state, str(bundle / "model.safetensors"))
    loaded = load_file(str(bundle / "model.safetensors"), device="cpu")
    for key, value in state.items():
        torch.testing.assert_close(loaded[key], value, rtol=0, atol=0)
    repo = Path(__file__).resolve().parents[2]
    environment = dict(
        os.environ,
        PYTHONPATH=str(repo),
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    commands = {
        "offline_fit": [
            sys.executable,
            "-m",
            "scripts.tau2.offline_critic_fit",
            "--check-head-init",
            "--head-init",
            str(path.resolve()),
            "--checkpoint",
            store.manifest["checkpoint"],
        ],
        "sao_guard": [
            sys.executable,
            "-m",
            "scripts.sao.check_critic_head",
            "--checkpoint",
            str(bundle.resolve()),
            "--output",
            str((path.parent / "E9-sao-readback.json").resolve()),
        ],
    }
    reports = {}
    for name, command in commands.items():
        result = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        (path.parent / f"E9-{name}.stdout.log").write_text(result.stdout)
        (path.parent / f"E9-{name}.stderr.log").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(
                f"E9 {name} failed ({result.returncode}); see E9-{name}.stderr.log"
            )
        reports[name] = {"command": command, "exit_code": result.returncode}
        if name == "offline_fit":
            reports[name]["readback"] = json.loads(result.stdout)
            if not reports[name]["readback"].get("passed") or not reports[name][
                "readback"
            ].get("readback_equal"):
                raise ValueError("E9 offline loader did not attest exact readback")
    reports["sao_guard"]["readback"] = json.loads(
        (path.parent / "E9-sao-readback.json").read_text()
    )
    if not reports["sao_guard"]["readback"].get("passed"):
        raise ValueError("E9 SAO guard did not pass")
    return {
        "consumers": reports,
        "safetensors_readback_exact": True,
        "sao_head_bundle": str(bundle.resolve()),
        "bundle_scope": "Head initialization only; combine with the provenance-matched backbone through offline_critic_fit --head-init. Do not pass this head-only directory as a full model.",
    }


def _ev(report: dict, domain: str = "airline"):
    return report["by_domain"].get(domain, {}).get("explained_variance")


def _number(value):
    return "undefined" if value is None else f"{value:.6g}"


def _at_least(value, threshold):
    return value > threshold or math.isclose(
        value, threshold, rel_tol=1e-12, abs_tol=1e-12
    )


def _gap_status(gap):
    return (
        "不确定"
        if gap is None
        else "成立"
        if _at_least(gap, 0.05)
        else "不成立"
        if not _at_least(gap, 0.02)
        else "不确定"
    )


def write_results_table(output_dir: Path) -> Path:
    """Create reviewable per-experiment tables without tuning on validation."""
    output_dir = Path(output_dir)
    results = {
        f"E{i}": json.loads((output_dir / f"E{i}.json").read_text())
        for i in range(2, 10)
        if (output_dir / f"E{i}.json").exists()
    }
    synthetic = any(result.get("synthetic") for result in results.values())
    lines = [
        "# 预实验第 10 节结果草稿",
        "",
        "**合成数据流程检查；以下数字不是实际实验结果，H1–H8 均不作科学判定。**"
        if synthetic
        else "以下为冻结协议判据的自动草稿；需核对各 JSON 中的置信区间、CV 选择和偏离项。",
        "",
    ]
    for name, result in results.items():
        lines.extend(
            [
                f"## {name}",
                "",
                "| 设置 | Airline EV | Retail EV | 说明 |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        if name in ("E2", "E6", "E8"):
            for arm, values in result.get("probes", {}).items():
                validation = values.get("validation")
                if validation:
                    lines.append(
                        f"| {arm} | {_number(_ev(validation))} | {_number(_ev(validation, 'retail'))} | 训练题 5 折选择正则；完整区间见 JSON |"
                    )
        elif name == "E3":
            for arm in result["arms"]:
                v = arm["evaluations"][-1]["validation"]
                lines.append(
                    f"| lr={arm['lr']:g}, frozen={arm['frozen_bias']}, warmup={arm['warmup_steps']} | {_number(_ev(v))} | {_number(_ev(v, 'retail'))} | 第 80 步，norm={arm['steps'][-1]['weight_norm']:.6g} |"
                )
        elif name == "E4":
            for arm in result["arms"]:
                a, r = (
                    arm["summary"].get("airline", {}),
                    arm["summary"].get("retail", {}),
                )
                lines.append(
                    f"| batch={arm['batch']}, budget={arm['budget']}, lr={arm['lr']:g} | {_number(a.get('mean_ev'))} | {_number(r.get('mean_ev'))} | 三种子；范围 A={a.get('range_ev')}, R={r.get('range_ev')} |"
                )
        elif name == "E5":
            for arm in result["arms"]:
                lines.append(
                    f"| batch={arm['batch']} | — | — | 与全量梯度平均余弦={arm['mean_full_data_cosine']:.6g} |"
                )
        elif name == "E7":
            for arm, values in result["layer_backbone_probes"].items():
                v = values["validation"]
                lines.append(
                    f"| {arm} | {_number(_ev(v))} | {_number(_ev(v, 'retail'))} | 层/骨干探针 |"
                )
            for arm, v in result["base_final_positions"].items():
                lines.append(
                    f"| {arm} | {_number(_ev(v))} | {_number(_ev(v, 'retail'))} | 最终层位置分组 |"
                )
        elif name == "E9":
            lines.append(
                f"| ridge head 导出 | — | — | readback={result.get('readback_exact')}; 双 CLI 验证见 E9.json |"
            )
        lines.extend(
            [
                "",
                f"这些数值用于检查 {name} 的输出结构。"
                if synthetic
                else f"读法与证据见 [{name}.json]({name}.json)；验证集不用于选择超参数。",
                "",
            ]
        )
    lines.extend(
        ["## H1–H8 固定判据", "", "| 假设 | 判定 | 判据对应值 |", "| --- | --- | --- |"]
    )
    decisions = assess_hypotheses(results)
    (output_dir / "hypothesis-decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n"
    )
    for name, decision in decisions.items():
        numeric = {
            key: value
            for key, value in decision.items()
            if key not in ("arms", "scientific_interpretation")
        }
        lines.append(
            f"| {name} | {decision['status']}{'（合成数据，未检验）' if synthetic else ''} | {json.dumps(numeric, ensure_ascii=False)} |"
        )
    path = output_dir / "results-table-draft.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def assess_hypotheses(results: dict) -> dict:
    """Apply frozen numerical cutoffs, retaining synthetic/nonpositive-probe caveats."""
    decisions = {
        f"H{i}": {"status": "不确定", "reason": "Required experiment absent"}
        for i in range(1, 9)
    }
    e2, e3, e4, e6, e7, e8 = (results.get(f"E{i}") for i in (2, 3, 4, 6, 7, 8))
    primary = e2["probes"]["ridge_shared"] if e2 else None
    probe_ev = _ev(primary["validation"]) if primary else None
    if primary:
        scopes = {}
        for scope in ("shared", "domain"):
            probe = e2["probes"][f"ridge_{scope}"]
            value = _ev(probe["validation"])
            interval = probe["bootstrap"]["by_domain"]["airline"]["ev_95ci"]
            status = "不确定"
            if value is not None:
                if value >= 0.15 and interval and interval[0] > 0:
                    status = "成立"
                elif value < 0.05:
                    status = "不成立"
            scopes[scope] = {"status": status, "airline_ev": value, "ev_95ci": interval}
        decisions["H1"] = {
            **scopes["shared"],
            "all_scopes": scopes,
            "interpretation": "Shared head is the declared primary; both prespecified scopes reported, never validation-selected.",
        }
    if e3:
        decisions["H2"] = dict(e3["H2"])
        h3 = []
        for arm in e3["arms"]:
            points = [
                (_ev(point["validation"]), point["step"])
                for point in arm["evaluations"]
            ]
            # H3 explicitly asks whether any <=80-step setting reaches the bar.
            reached = [
                step
                for ev, step in points
                if ev is not None
                and probe_ev is not None
                and _at_least(ev, 0.7 * probe_ev)
            ]
            h3.append(
                {
                    "lr": arm["lr"],
                    "frozen_bias": arm["frozen_bias"],
                    "warmup_steps": arm["warmup_steps"],
                    "steps_reaching_70pct": reached,
                    "maximum_bias_jump": arm["maximum_adjacent_bias_jump"],
                    "all_points_below_40pct": all(
                        ev is None or not _at_least(ev, 0.4 * probe_ev)
                        for ev, _ in points
                    )
                    if probe_ev is not None
                    else False,
                }
            )
        status = "不确定"
        if probe_ev is not None and probe_ev > 0:
            if decisions["H2"]["status"] == "成立" and any(
                arm["steps_reaching_70pct"] and arm["maximum_bias_jump"] <= 0.2
                for arm in h3
            ):
                status = "成立"
            elif all(arm["all_points_below_40pct"] for arm in h3):
                status = "不成立"
        decisions["H3"] = {
            "status": status,
            "probe_airline_ev": probe_ev,
            "arms": h3,
            "interpretation": "Prespecified existence test across arms, not validation-based learning-rate selection; nonpositive probe EV leaves ratio criterion uncertain. H2 failure makes simulations reference-only.",
        }
    if e4:
        chosen = {}
        for batch in (16, 128):
            lr = e4["batch_budget_training_cv"][f"{batch}/B"][0]["lr"]
            arm = next(
                arm
                for arm in e4["arms"]
                if arm["batch"] == batch and arm["budget"] == "B" and arm["lr"] == lr
            )
            chosen[str(batch)] = {"lr": lr, **arm["summary"]["airline"]}
        gap = chosen["128"]["mean_ev"] - chosen["16"]["mean_ev"]
        decisions["H4"] = {
            "status": _gap_status(gap),
            "airline_ev_gap_128_minus_16": gap,
            "training_cv_selected_80_step_arms": chosen,
        }
    if e6 and primary:
        soft_ev = _ev(e6["probes"]["ridge_shared"]["validation"])
        gap = (
            soft_ev - probe_ev if soft_ev is not None and probe_ev is not None else None
        )
        decisions["H5"] = {
            "status": _gap_status(gap),
            "airline_ev_soft": soft_ev,
            "airline_ev_binary": probe_ev,
            "gap": gap,
            "primary": "shared ridge; simulations and domain heads remain separately reported",
        }
    if e7 and primary:
        selected = e7["selected_intermediate_training_cv"]
        for hypothesis, name in [("H6", selected), ("H7", "r510-final")]:
            value = _ev(e7["layer_backbone_probes"][name]["validation"])
            gap = (
                value - probe_ev if value is not None and probe_ev is not None else None
            )
            decisions[hypothesis] = {
                "status": _gap_status(gap),
                "selected_cache": name,
                "airline_ev": value,
                "base_final_airline_ev": probe_ev,
                "gap": gap,
            }
    if e8:
        ridge = e8["probes"]["ridge_shared"]["validation"]["summary"][
            "explained_variance"
        ]
        replay = e8["fixed_r510_replay"][0]["evaluations"][-1]["validation"]["summary"][
            "explained_variance"
        ]
        status = "不确定"
        if ridge is not None and replay is not None:
            if ridge >= 0.8 and replay < 0.3:
                status = "成立"
            elif replay >= 0.6:
                status = "不成立"
        decisions["H8"] = {
            "status": status,
            "ridge_ev": ridge,
            "replay_ev": replay,
            "primary": "overall token EV on planted validation labels",
            "training_prevalence": e8.get("sampled_prefix_training_prevalence"),
        }
    if any(result.get("synthetic") for result in results.values()):
        for decision in decisions.values():
            decision["synthetic_cutoff_check"] = decision["status"]
            decision["status"] = "不确定"
            decision["scientific_interpretation"] = (
                "Synthetic fixture only; hypothesis not tested on real features."
            )
    return decisions
