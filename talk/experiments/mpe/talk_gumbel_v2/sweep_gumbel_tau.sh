for gumbel_tau in 0.01 0.1 0.2 0.5 0.8 1.0 1.5 2.0; do
  echo "=== gumbel_tau=${gumbel_tau} ==="
  python mappo_talk_v2.py \
    "gumbel_tau=${gumbel_tau}" \
    "codebook_size=8" \
    "aux_coef=0.01" \
    "sight_range=1.5" \
    "talk_config=_8_code_0.01_h_aux_1.5_sr_${gumbel_tau}_gt"
done
echo "Sweep complete."