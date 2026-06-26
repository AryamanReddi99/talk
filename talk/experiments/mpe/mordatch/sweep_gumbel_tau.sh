for gumbel_tau in 0.1 0.2 0.5 0.8 1.0 1.5 2.0; do
  echo "=== gumbel_tau=${gumbel_tau} ==="
  python mappo_mordatch.py \
    "gumbel_tau=${gumbel_tau}" \
    "sight_range=1.5" \
    "talk_config=_1.5_sr_${gumbel_tau}_gt"
done
echo "Sweep complete."
