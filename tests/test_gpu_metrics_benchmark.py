# SPDX-License-Identifier: Apache-2.0

from benchmark.gpu_metrics import parse_metrics


def test_gpu_metrics_preserves_units_and_rejects_invalid_values():
    result = parse_metrics("""# HELP ignored
DCGM_FI_PROF_SM_ACTIVE{gpu="0",UUID="GPU-a"} 0.25
DCGM_FI_DEV_GPU_UTIL{gpu="0"} 75
DCGM_FI_DEV_GPU_UTIL{gpu="1"} 7407936
DCGM_FI_PROF_SM_OCCUPANCY{gpu="1"} NaN
DCGM_FI_PROF_DRAM_ACTIVE{gpu="2"} 9.223372036854776e+18
DCGM_FI_DEV_FB_USED{gpu="0"} 2048
unrelated{gpu="0"} 42
""")
    assert len(result) == 6
    assert result[0]["value"] == 0.25 and result[0]["unit"] == "ratio"
    assert result[1]["value"] == 75 and result[1]["unit"] == "percent"
    assert all(not item["valid"] and item["value"] is None for item in result[2:5])
    assert result[5]["value"] == 2048 and result[5]["unit"] == "MiB"
