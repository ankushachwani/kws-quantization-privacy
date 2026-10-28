#!/bin/bash
# Raw data -> tables and figures. Run from anywhere; needs ~6 GB disk for data/ and cache/.
#   data/speech_commands_v0.02.tar.gz   2,428,923,189 bytes, http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
#   data/test_set/                      official test set (12 class folders), from
#                                       http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz
set -euo pipefail
cd "$(dirname "$0")"
SEEDS=${SEEDS:-10}; BOOT=${BOOT:-1000}

[ -d .pylib/torchaudio ] || pip install --target .pylib --no-deps torchaudio==2.11.0

TAR=data/speech_commands_v0.02.tar.gz
[ "$(stat -f%z "$TAR" 2>/dev/null || stat -c%s "$TAR")" = 2428923189 ] || { echo "$TAR missing or incomplete"; exit 1; }
gzip -t "$TAR"
[ -f data/sc_v2/testing_list.txt ] || { mkdir -p data/sc_v2; tar -xzf "$TAR" -C data/sc_v2; }
[ -d data/test_set/_silence_ ] || { echo "put the official test set in data/test_set"; exit 1; }

cd scripts
python3 01_build_pool.py
python3 02_logits.py --seeds "$SEEDS"
python3 03_attack.py --boot "$BOOT" --seeds "$SEEDS"
python3 04_tables_figs.py
