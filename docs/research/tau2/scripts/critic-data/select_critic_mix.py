"""Build the critic training mix from AReaL pack v2 (docs/research/tau2/critic/data-mix.md).

Keeps every pack task with a learning signal and a stratified random sample of
the easy ones, then writes a subset task pack that ``critic_task_pack`` loads.

- easy: correct in all 6 Qwen3.5-4B episodes we have, 4 at temperature 0
  (13.3 pack eval) and 2 at temperature 1 (critic run 479);
- informative: every other pack task;
- easy sample: ``--n-easy`` tasks, allocated to (domain, keep/fix) strata in
  proportion to their easy counts (largest remainder), random within a stratum.

Usage: select_critic_mix.py SRC_PACK T0_EPISODES T1_ROLLOUT_DIR OUT_PACK
       [--n-easy 99] [--seed 20261013]
The official train tasks are not part of the pack; the critic trainer appends
them. Their 479 success rate is used only for the bias-init estimate.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

_FIELDS = re.compile(r'"(task_id|source_id|domain|reward)": ("[^"]*"|[-0-9.eE]+)')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_t1(rollout_dir: str) -> dict[tuple[str, str, str], list[float]]:
    """(source, domain, task_id) -> rewards. Regex only: rows carry token arrays."""
    out: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    for name in os.listdir(rollout_dir):
        with open(os.path.join(rollout_dir, name)) as f:
            for line in f:
                kv: dict[str, str] = {}
                for m in _FIELDS.finditer(line):
                    kv.setdefault(m.group(1), m.group(2).strip('"'))
                src = "pack" if kv["source_id"].startswith("pack:") else "official"
                out[(src, kv["domain"], kv["task_id"])].append(float(kv["reward"]))
    return out


def read_t0(path: str) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for line in open(path):
        d = json.loads(line)
        out[(d["domain"], d["task_id"])].append(float(d["reward"]))
    return out


def allocate(counts: dict[tuple[str, str], int], n: int) -> dict[tuple[str, str], int]:
    total = sum(counts.values())
    quota = {k: n * c / total for k, c in counts.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    rest = sorted(counts, key=lambda k: (alloc[k] - quota[k], k))
    for k in rest[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_pack")
    ap.add_argument("t0_episodes")
    ap.add_argument("t1_rollouts")
    ap.add_argument("out_pack")
    ap.add_argument("--n-easy", type=int, default=99)
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()

    src = Path(args.src_pack).resolve()
    manifest = json.loads((src / "manifest.json").read_text())
    splits = manifest["task_splits"]  # "domain:task_id" -> keep/fix
    t0, t1 = read_t0(args.t0_episodes), read_t1(args.t1_rollouts)

    tasks = sorted(k.split(":", 1) for k in splits)
    easy, informative = [], []
    for domain, tid in tasks:
        r0, r1 = t0.get((domain, tid), []), t1.get(("pack", domain, tid), [])
        if len(r0) != 4 or len(r1) != 2:
            raise SystemExit(
                f"{domain}:{tid}: expected 4 T0 + 2 T1 episodes, got {len(r0)}+{len(r1)}"
            )
        (easy if all(r == 1.0 for r in r0 + r1) else informative).append((domain, tid))

    strata: dict[tuple[str, str], list[tuple[str, str]]] = collections.defaultdict(list)
    for domain, tid in easy:
        strata[(domain, splits[f"{domain}:{tid}"])].append((domain, tid))
    alloc = allocate({k: len(v) for k, v in strata.items()}, args.n_easy)
    rng = random.Random(args.seed)
    easy_kept = []
    for key in sorted(strata):
        easy_kept += rng.sample(sorted(strata[key]), alloc[key])
    kept = set(informative) | set(easy_kept)

    out = Path(args.out_pack)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src / "tau2_rl_database", tmp / "tau2_rl_database")
    shutil.copy2(src / "decisions.json", tmp / "decisions.json")
    counts = {}
    for domain in sorted({d for d, _ in tasks}):
        all_tasks = json.loads((src / domain / "tasks.json").read_text())
        task_db = json.loads((src / domain / "task_db.json").read_text())
        sub = [t for t in all_tasks if (domain, str(t["id"])) in kept]
        (tmp / domain).mkdir()
        (tmp / domain / "tasks.json").write_text(
            json.dumps(sub, indent=1, ensure_ascii=False)
        )
        (tmp / domain / "task_db.json").write_text(
            json.dumps({str(t["id"]): task_db[str(t["id"])] for t in sub}, indent=1)
        )
        counts[domain] = len(sub)

    def rate(keys, src_name):
        rs = [r for d, t in keys for r in t1[(src_name, d, t)]]
        return sum(rs) / len(rs), len(rs)

    p_inf, _ = rate(informative, "pack")
    p_easy, _ = rate(easy_kept, "pack")
    official = [(d, t) for s, d, t in t1 if s == "official"]
    p_off, _ = rate(official, "official")
    n = len(informative) + len(easy_kept) + len(official)
    bias = (
        len(informative) * p_inf + len(easy_kept) * p_easy + len(official) * p_off
    ) / n
    selection = {
        "source_pack": str(src),
        "source_pack_name": manifest["name"],
        "rule": "easy = 6/6 correct (4 x T0 from 13.3 pack eval + 2 x T1 from critic run 479); "
        "keep all informative + stratified random sample of easy",
        "t0_episodes": os.path.abspath(args.t0_episodes),
        "t1_rollouts": os.path.abspath(args.t1_rollouts),
        "seed": args.seed,
        "n_easy_requested": args.n_easy,
        "easy_total": len(easy),
        "informative_total": len(informative),
        "easy_allocation": {
            f"{d}:{s}": [alloc[(d, s)], len(strata[(d, s)])] for d, s in sorted(strata)
        },
        "informative_by_domain": dict(collections.Counter(d for d, _ in informative)),
        "t1_success": {
            "informative": p_inf,
            "easy_kept": p_easy,
            "official_train": p_off,
        },
        "official_train_tasks": len(official),
        "train_rows_with_official": n,
        "expected_label_mean": bias,
        "informative": sorted(f"{d}:{t}" for d, t in informative),
        "easy_kept": sorted(f"{d}:{t}" for d, t in easy_kept),
    }
    (tmp / "selection.json").write_text(json.dumps(selection, indent=1))
    files = sorted(
        str(p.relative_to(tmp))
        for p in tmp.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    )
    new_manifest = {
        "name": out.name,
        "source": f"subset of {manifest['name']} for critic fitting (selection.json)",
        "counts": counts,
        "task_splits": {
            k: v for k, v in splits.items() if tuple(k.split(":", 1)) in kept
        },
        "files": {rel: _sha256(tmp / rel) for rel in files},
    }
    (tmp / "manifest.json").write_text(json.dumps(new_manifest, indent=1))
    tmp.rename(out)
    for p in [out, *out.rglob("*")]:
        p.chmod(p.stat().st_mode & ~0o222)
    print(
        json.dumps(
            {
                k: v
                for k, v in selection.items()
                if k not in ("informative", "easy_kept")
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
