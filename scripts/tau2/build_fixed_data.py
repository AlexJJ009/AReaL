# SPDX-License-Identifier: Apache-2.0
"""Build the fixed τ² data directory from the pinned official one.

Usage: python scripts/tau2/build_fixed_data.py OFFICIAL_DATA_DIR OUT_DATA_DIR
"""

import json
import sys

from examples.tau2.benchmark_variant import TASK_FILE_SHA256, build_fixed_data

if __name__ == "__main__":
    digests = build_fixed_data(sys.argv[1], sys.argv[2])
    if digests != TASK_FILE_SHA256["fixed"]:
        raise SystemExit(f"Built task files do not match the pinned digests: {digests}")
    print(json.dumps(digests, indent=1))
