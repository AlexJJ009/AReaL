# ruff: noqa: E731, E741, UP031  # archived audit script, kept as run
import json
import pathlib
from multiprocessing import Pool

B = pathlib.Path(
    "/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking"
)


def rd(f):
    out = []
    for i, l in enumerate(open(f)):
        d = json.loads(l)
        out.append(
            dict(
                dir=f.parent.name,
                file=f.name,
                line=i,
                domain=d.get("domain"),
                task_id=str(d.get("task_id")),
                idx=d.get("sample_idx"),
                reward=d.get("reward"),
                official=d.get("official_score"),
                hv=d.get("head_version"),
                tv=d.get("tail_version"),
                rtid=d.get("rollout_task_id"),
                ep=d.get("episode_id"),
                split=d.get("split"),
                tbf=d.get("task_budget_failure"),
            )
        )
    return out


res = {}
for name, p in [
    (
        "grpo",
        "grpo-formal-v2/logs/root/tau2-grpo/tau2-grpo-nonthinking-formal-v2/rollout",
    ),
    (
        "sao",
        "sao-mixed-v4-disk/logs/root/tau2-sao/tau2-sao-nonthinking-mixed-v4-disk/rollout",
    ),
]:
    files = sorted((B / p).glob("*/*.jsonl"))
    with Pool(16) as pool:
        recs = [r for rs in pool.map(rd, files) for r in rs]
    res[name] = dict(source=str(B / p), records=recs)
    print(name, len(files), len(recs), flush=True)
json.dump(
    res,
    open(
        "/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/_local/scratch/claude-verify-20261009/stats/train_records.json",
        "w",
    ),
)
