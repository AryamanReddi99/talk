#!/usr/bin/env bash
sleep 600

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${SCRIPT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

for num_agents in 2 3 4 5 6 7 8 9 10; do
  echo "=== num_agents=${num_agents} ==="
  conda run -n talk python mappo_att_grucomm_dub_state_in.py \
    "num_agents=${num_agents}" \
    "custom_name=_stop_state_msg_gradient" \
    "stop_neighbor_msg_grad=true"
done

echo "Sweep complete."
