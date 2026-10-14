#!/usr/bin/env bash
# GPUs 0-3: Qwen3.5-4B telecom (same protocol, for comparison), then Instruct fixed and official.
cd "$(dirname "$0")"
./launch_eval.sh qwen35-official-telecom /data_storage/yl_test/lgx/data-1/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a 0,1,2,3 official telecom 0 qwen3_coder
until [ -f /data_storage/yl_test/lgx/data-1/.cache/models/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554/fetch_manifest.json ]; do sleep 30; done
./launch_eval.sh instruct-fixed-airline-retail /data_storage/yl_test/lgx/data-1/.cache/models/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 0,1,2,3 fixed airline,retail 0 qwen25
./launch_eval.sh instruct-official-all /data_storage/yl_test/lgx/data-1/.cache/models/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 0,1,2,3 official airline,retail,telecom 0 qwen25
echo LANE_A_DONE > lane_a.done
