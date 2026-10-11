#!/usr/bin/env bash
# GPUs 4-7: Thinking fixed, then Thinking official. Waits for the verified download.
cd "$(dirname "$0")"
until [ -f /data_storage/yl_test/lgx/data-1/.cache/models/Qwen3-4B-Thinking-2507@768f209d9ea81521153ed38c47d515654e938aea/fetch_manifest.json ]; do sleep 30; done
./launch_eval.sh thinking-fixed-airline-retail /data_storage/yl_test/lgx/data-1/.cache/models/Qwen3-4B-Thinking-2507@768f209d9ea81521153ed38c47d515654e938aea 4,5,6,7 fixed airline,retail 1 qwen25
./launch_eval.sh thinking-official-all /data_storage/yl_test/lgx/data-1/.cache/models/Qwen3-4B-Thinking-2507@768f209d9ea81521153ed38c47d515654e938aea 4,5,6,7 official airline,retail,telecom 1 qwen25
echo LANE_B_DONE > lane_b.done
