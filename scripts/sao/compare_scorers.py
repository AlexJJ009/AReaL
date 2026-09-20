# SPDX-License-Identifier: Apache-2.0
"""Differential check against the actual previous PRD scorer in its own environment."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from areal.reward.math_prd import score_math_prd

CASES = [
    (r"\boxed{33}", "033"),
    (r"\boxed{43}", "043"),
    (r"\boxed{2/4}", r"\frac{1}{2}"),
    (r"\boxed{0.5}", r"\frac{1}{2}"),
    (r"\boxed{2\sqrt{2}}", r"\sqrt{8}"),
    (r"\boxed{even}", r"\text{even}"),
    (r"\fbox{42}", "42"),
    ("Answer: 42", "42"),
    (r"\boxed{42}, then \boxed{41}", "42"),
    (r"\boxed{42}, then \boxed{41", "42"),
    (r"\boxed{X}", "x"),
    (r"\boxed{42 or 41}", "42"),
    (r"\boxed{42,41}", "42"),
    (r"\boxed{0.5}", r"50\%"),
    (r"\boxed{10}", "10_2"),
    (r"\boxed{10_3}", "10_2"),
    (r"\boxed{1,-2,-1/2}", "1,1,-2,-1/2"),
    (r"\boxed{7}", r"7\text{ hours}."),
    (r"\boxed{m+n=15+3=18}", "18"),
    (r"\boxed{a+b+c}", "a + b + c"),
    (r"\boxed{\sum_{k=1}^{3} k}", "6"),
    (r"\boxed{\sum_{k=1}^{3} t}", "-1"),
    (r"\boxed{\frac{\sum_{k=1}^{7671}\varphi(k)-1}{2}}", "14711060"),
    (r"\boxed{7:05}", r"7\!:\!05"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--legacy-python", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    code = """import json,sys
sys.path.insert(0, sys.argv[1])
from examples.on_policy_distillation_trainer.math_prd_reward import compute_score
cases=json.load(sys.stdin)
results=[compute_score('math_dapo',pred,gold) for pred,gold in cases]
print('SCORER_RESULT='+json.dumps(results))
"""
    proc = subprocess.run(
        [args.legacy_python, "-c", code, args.legacy_root],
        input=json.dumps(CASES),
        text=True,
        capture_output=True,
        timeout=300,
        check=True,
        cwd=args.legacy_root,
    )
    legacy = json.loads(
        next(
            line.removeprefix("SCORER_RESULT=")
            for line in proc.stdout.splitlines()
            if line.startswith("SCORER_RESULT=")
        )
    )
    actual = [score_math_prd(pred, gold) for pred, gold in CASES]
    rows = [
        {
            "completion": pair[0],
            "gold": pair[1],
            "legacy": old,
            "new": new,
            "matches": old == new,
        }
        for pair, old, new in zip(CASES, legacy, actual, strict=True)
    ]
    source = (
        Path(args.legacy_root)
        / "examples/on_policy_distillation_trainer/math_prd_reward.py"
    )
    result = {
        "passed": all(row["matches"] for row in rows),
        "checked_cases": len(rows),
        "legacy_source": str(source),
        "legacy_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "legacy_python": args.legacy_python,
        "comparisons": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checked_cases": len(rows),
                "mismatches": [r for r in rows if not r["matches"]],
            }
        )
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
