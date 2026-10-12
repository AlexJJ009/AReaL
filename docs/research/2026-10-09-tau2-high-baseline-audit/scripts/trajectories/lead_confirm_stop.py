"""Correlational probe: does the user's STOP-right-after-confirmation depend on agent phrasing?

For every agent text turn that asks for confirmation and is answered affirmatively by the user,
record whether that user reply also contains ###STOP###, split by phrasing features of the
agent message. Usage: python lead_confirm_stop.py <run_dir> [<run_dir> ...]
"""

import collections
import glob
import json
import re
import sys

ASK = re.compile(
    r"\b(confirm|proceed\?|shall i|would you like me to|do you want me to|please reply|yes\b)",
    re.I,
)
YES = re.compile(
    r"^\W*(yes|yeah|yep|sure|ok|okay|confirm|i confirm|go ahead|please proceed|that'?s correct|correct)",
    re.I,
)
FEATS = {
    "says_will_execute_after": re.compile(
        r"(once you confirm|after you confirm|upon (your )?confirmation|i will (then )?proceed|i'll (then )?proceed|i will process|i'll process)",
        re.I,
    ),
    "explicit_yes_instruction": re.compile(
        r"(reply|respond|answer|type)\s+['\"“]?yes", re.I
    ),
    "multiple_actions_listed": re.compile(r"(\n\s*(\d+\.|-)\s.*){3,}"),
}
for run in sys.argv[1:]:
    c = collections.defaultdict(lambda: [0, 0])
    for f in glob.glob(f"{run}/trajectories/retail:*/*.json"):
        d = json.load(open(f))
        ms = d["messages"]
        for i in range(len(ms) - 1):
            a, u = ms[i], ms[i + 1]
            if a["role"] != "assistant" or a.get("tool_calls") or u["role"] != "user":
                continue
            at, ut = a.get("content") or "", u.get("content") or ""
            if not ASK.search(at[-400:]) or not YES.search(ut):
                continue
            stop = "###STOP###" in ut
            c["all"][0] += stop
            c["all"][1] += 1
            for k, rx in FEATS.items():
                key = f"{k}={bool(rx.search(at))}"
                c[key][0] += stop
                c[key][1] += 1
    print(run.split("/")[-1])
    for k in sorted(c):
        s, n = c[k]
        print(f"  {k:40s} stop {s:4d}/{n:4d} = {s / max(n, 1):.3f}")
