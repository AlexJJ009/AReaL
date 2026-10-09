# ruff: noqa: E731, E741, UP031  # archived audit script, kept as run
import json
import pathlib
from multiprocessing import Pool

B = pathlib.Path(
    "/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking"
)
OUT = pathlib.Path(
    "/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/_local/scratch/claude-verify-20261009/stats"
)


def rd(f):
    res = []
    for i, l in enumerate(open(f)):
        d = json.loads(l)
        res.append(
            dict(
                file=f.name,
                line=i,
                domain=d["domain"],
                task_id=str(d["task_id"]),
                idx=d["sample_idx"],
                score=d["official_score"],
                reward=d["reward"],
                hv=d["head_version"],
                tv=d["tail_version"],
                episode_id=d.get("episode_id"),
                attempt_id=d.get("attempt_id"),
            )
        )
    return res


srcs = {}
for alg, run, vers in [
    ("GRPO", "grpo-formal-v2", [0, 5, 35]),
    ("SAO", "sao-mixed-v1", [0]),
    ("SAO", "sao-mixed-v4-disk", [5, 10]),
]:
    for v in vers:
        srcs[(alg, v)] = (
            "inline",
            sorted((B / run).glob(f"logs/root/*/*-dedicated-eval/eval-rollout/{v}"))[0],
        )
for alg, v, a in [
    ("GRPO", 10, 1),
    ("GRPO", 15, 1),
    ("GRPO", 20, 1),
    ("GRPO", 25, 2),
    ("GRPO", 30, 1),
    ("SAO", 15, 2),
    ("SAO", 20, 1),
    ("SAO", 25, 1),
]:
    srcs[(alg, v)] = (
        "detached",
        B / f"{alg.lower()}-detached-eval-v{v}-8gpu-attempt{a}" / "episodes.jsonl",
    )
allrec = {}
with Pool(16) as p:
    for k, (kind, path) in srcs.items():
        if kind == "inline":
            files = sorted(path.glob("*.jsonl"), key=lambda f: int(f.stem))
            recs = [r for rs in p.map(rd, files) for r in rs]
        else:
            recs = []
            for i, l in enumerate(open(path)):
                d = json.loads(l)
                recs.append(
                    dict(
                        file=path.name,
                        line=i,
                        domain=d["domain"],
                        task_id=str(d["task_id"]),
                        idx=d["trial"],
                        score=d["official_score"],
                        reward=d["reward"],
                        status=d.get("status"),
                        attempt_id=d.get("attempt_id"),
                    )
                )
        allrec[f"{k[0]}:{k[1]}"] = dict(source=str(path), records=recs)
        print(k, path, len(recs), flush=True)
json.dump(allrec, open(OUT / "eval_records.json", "w"))
